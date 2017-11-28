#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''fakelcrecovery - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This is a companion module for fakelcgen.py. It runs LCs generated using
functions in that module through variable star detection and classification to
see how well they are recovered.

TODO: implement below

calculate the various non-periodic variability indices for these sim LCs and
test recall and precision. get the PCA PC1 of this for a combined var index.

tune for a true positive and false positive rate using the ROC curve and set the
sigma limit for confirmed variability per magnitude bin.

afterwards check if successful by cross validation.

run period searches and see recovery rate by period, amplitude, magnitude,
number of observations, etc.

'''
import os
import os.path
import pickle

import multiprocessing as mp
import logging
from datetime import datetime
from traceback import format_exc
from concurrent.futures import ProcessPoolExecutor
from hashlib import md5

from math import sqrt as msqrt

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem
def dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)

import numpy as np
import numpy.random as npr
# seed the numpy random generator
npr.seed(0xdecaff)

import scipy.stats as sps
import scipy.interpolate as spi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mpc

from tqdm import tqdm

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.fakelcrecovery' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


###################
## LOCAL IMPORTS ##
###################

import astrobase.lcproc as lcproc
lcproc.set_logger_parent(__name__)

import astrobase.varbase.features as vfeatures


#######################
## LC FORMATS SET UP ##
#######################

def read_fakelc(fakelcfile):
    '''
    This just reads a pickled fake LC.

    '''

    try:
        with open(lcfile,'rb') as infd:
            lcdict = pickle.load(infd)
    except UnicodeDecodeError:
        with open(lcfile,'rb') as infd:
            lcdict = pickle.load(infd, encoding='latin1')

    return lcdict



#######################
## UTILITY FUNCTIONS ##
#######################

def get_varfeatures(simbasedir,
                    nworkers=None):
    '''
    This runs lcproc.parallel_varfeatures on light curves in simbasedir.

    '''

    # get the info from the simbasedir
    with open(os.path.join(simbasedir, 'fakelcs-info.pkl'),'rb') as infd:
        siminfo = pickle.load(infd)

    lcfpaths = siminfo['lcfpath']
    varfeaturedir = os.path.join(simbasedir,'varfeatures')

    # get the column defs for the fakelcs
    timecols = siminfo['timecols']
    magcols = siminfo['magcols']
    errcols = siminfo['errcols']

    # register the fakelc pklc as a custom lcproc format
    # now we should be able to use all lcproc functions correctly
    if 'fakelc' not in lcproc.LCFORM:

        lcproc.register_custom_lcformat(
            'fakelc',
            '*-fakelc.pkl',
            lcproc.read_pklc,
            timecols,
            magcols,
            errcols,
            magsarefluxes=False,
            specialnormfunc=None
        )

    # now we can use lcproc.parallel_varfeatures directly
    varinfo = lcproc.parallel_varfeatures(lcfpaths,
                                          varfeaturedir,
                                          lcformat='fakelc',
                                          nworkers=nworkers)

    with open(os.path.join(simbasedir,'fakelc-varfeatures.pkl'),'wb') as outfd:
        pickle.dump(varinfo, outfd, pickle.HIGHEST_PROTOCOL)

    return os.path.join(simbasedir,'fakelc-varfeatures.pkl')



def precision(ntp, nfp):
    '''
    This calculates the precision.

    '''

    if (ntp+nfp) > 0:
        return ntp/(ntp+nfp)
    else:
        return np.nan



def recall(ntp, nfn):
    '''
    This calculates the recall.

    '''

    if (ntp+nfn) > 0:
        return ntp/(ntp+nfn)
    else:
        return np.nan



def matthews_correl_coeff(ntp, ntn, nfp, nfn):
    '''
    This calculates the Matthews correlation coefficent.

    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    '''

    mcc_top = (ntp*ntn - nfp*nfn)
    mcc_bot = msqrt((ntp + nfp)*(ntp + nfn)*(ntn + nfp)*(ntn + nfn))

    if mcc_bot > 0:
        return mcc_top/mcc_bot
    else:
        return np.nan



####################################
## VARIABILITY RECOVERY (OVERALL) ##
####################################

def get_overall_recovered_variables(simbasedir,
                                    stetson_stdev_min=2.0,
                                    inveta_stdev_min=2.0,
                                    statsonly=True):
    '''This runs variability selection for LCs in simbasedir and gets recovery
    stats for the overall sample (i.e. over all magbins).

    returns:

    {'stetson_stdev_min', 'inveta_stdev_min', 'recovered_varlcs',
     'truepositives', 'truenegatives', 'falsepositives', 'falsenegatives',
     'matthewscorrcoeff', 'precision', 'recall'}

    '''

    # get the info from the simbasedir
    with open(os.path.join(simbasedir, 'fakelcs-info.pkl'),'rb') as infd:
        siminfo = pickle.load(infd)

    lcfpaths = siminfo['lcfpath']
    objectids = siminfo['objectid']
    varflags = siminfo['isvariable']
    sdssr = siminfo['sdssr']
    ndet = siminfo['ndet']

    # get the column defs for the fakelcs
    timecols = siminfo['timecols']
    magcols = siminfo['magcols']
    errcols = siminfo['errcols']

    # register the fakelc pklc as a custom lcproc format
    # now we should be able to use all lcproc functions correctly
    if 'fakelc' not in lcproc.LCFORM:

        lcproc.register_custom_lcformat(
            'fakelc',
            '*-fakelc.pkl',
            lcproc.read_pklc,
            timecols,
            magcols,
            errcols,
            magsarefluxes=False,
            specialnormfunc=None
        )

    # make the output directory if it doesn't exit
    outdir = os.path.join(simbasedir, 'recvar-threshold-pkls')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # run the variability search using the results of get_varfeatures
    varfeaturedir = os.path.join(simbasedir, 'varfeatures')
    varthreshinfof = os.path.join(
        outdir,
        'varthresh-stet%.2f-inveta%.2f.pkl' % (stetson_stdev_min,
                                               inveta_stdev_min)
    )
    varthresh = lcproc.variability_threshold(varfeaturedir,
                                             varthreshinfof,
                                             lcformat='fakelc',
                                             min_stetj_stdev=stetson_stdev_min,
                                             min_inveta_stdev=inveta_stdev_min,
                                             verbose=False)

    # now get the true positives, false positives, true negatives, false
    # negatives, and calculate recall, precision, Matthews corr. coeff.
    actualvars = objectids[varflags]
    actualnotvars = objectids[~varflags]

    # this is the output dict
    recdict = {
        'simbasedir':simbasedir,
        'timecols':timecols,
        'magcols':magcols,
        'errcols':errcols,
        'stetj_min_stdev':stetson_stdev_min,
        'inveta_min_stdev':inveta_stdev_min,
        'actual_variables':actualvars,
        'actual_nonvariables':actualnotvars
    }

    for magcol in magcols:

        magbins = varthresh[magcol]['binned_sdssr_median']

        # stetson recovered variables
        stet_recoveredvars = varthresh[magcol][
            'objectids_stetsonj_thresh_all_magbins'
        ]
        stet_recoverednotvars = np.setdiff1d(objectids, stet_recoveredvars)

        stet_truepositives = np.intersect1d(stet_recoveredvars,
                                            actualvars)
        stet_falsepositives = np.intersect1d(stet_recoveredvars,
                                             actualnotvars)
        stet_truenegatives = np.intersect1d(stet_recoverednotvars,
                                            actualnotvars)
        stet_falsenegatives = np.intersect1d(stet_recoverednotvars,
                                             actualvars)

        # calculate stet recall, precision, and MCC
        stet_recall = recall(stet_truepositives.size,
                             stet_falsenegatives.size)

        stet_precision = precision(stet_truepositives.size,
                                   stet_falsepositives.size)

        stet_mcc = matthews_correl_coeff(stet_truepositives.size,
                                         stet_truenegatives.size,
                                         stet_falsepositives.size,
                                         stet_falsenegatives.size)


        # inveta recovered variables
        inveta_recoveredvars = varthresh[magcol][
            'objectids_inveta_thresh_all_magbins'
        ]
        inveta_recoverednotvars = np.setdiff1d(objectids, inveta_recoveredvars)

        inveta_truepositives = np.intersect1d(inveta_recoveredvars,
                                              actualvars)
        inveta_falsepositives = np.intersect1d(inveta_recoveredvars,
                                               actualnotvars)
        inveta_truenegatives = np.intersect1d(inveta_recoverednotvars,
                                              actualnotvars)
        inveta_falsenegatives = np.intersect1d(inveta_recoverednotvars,
                                               actualvars)

        # calculate inveta recall, precision, Matthews correl coeff
        inveta_recall = recall(inveta_truepositives.size,
                               inveta_falsenegatives.size)

        inveta_precision = precision(inveta_truepositives.size,
                                     inveta_falsepositives.size)

        inveta_mcc = matthews_correl_coeff(inveta_truepositives.size,
                                         inveta_truenegatives.size,
                                         inveta_falsepositives.size,
                                         inveta_falsenegatives.size)


        # calculate the stats for combined intersect(stet,inveta) variable flags
        intersect_recvars = np.intersect1d(
            varthresh[magcol]['objectids_stetsonj_thresh_all_magbins'],
            varthresh[magcol]['objectids_inveta_thresh_all_magbins']
        )
        intersect_recnonvars = np.setdiff1d(objectids, intersect_recvars)

        intersect_truepositives = np.intersect1d(intersect_recvars,
                                                 actualvars)
        intersect_falsepositives = np.intersect1d(intersect_recvars,
                                             actualnotvars)
        intersect_truenegatives = np.intersect1d(intersect_recnonvars,
                                            actualnotvars)
        intersect_falsenegatives = np.intersect1d(intersect_recnonvars,
                                             actualvars)

        # calculate intersection recall, precision, Matthews correl coeff
        intersect_recall = recall(intersect_truepositives.size,
                                  intersect_falsenegatives.size)

        intersect_precision = precision(intersect_truepositives.size,
                                        intersect_falsepositives.size)

        intersect_mcc = matthews_correl_coeff(intersect_truepositives.size,
                                              intersect_truenegatives.size,
                                              intersect_falsepositives.size,
                                              intersect_falsenegatives.size)

        # calculate the stats for combined union(stet,inveta) variable flags
        union_recvars = np.union1d(
            varthresh[magcol]['objectids_stetsonj_thresh_all_magbins'],
            varthresh[magcol]['objectids_inveta_thresh_all_magbins']
        )
        union_recnonvars = np.setdiff1d(objectids, union_recvars)

        union_truepositives = np.union1d(union_recvars,
                                         actualvars)
        union_falsepositives = np.union1d(union_recvars,
                                          actualnotvars)
        union_truenegatives = np.union1d(union_recnonvars,
                                         actualnotvars)
        union_falsenegatives = np.union1d(union_recnonvars,
                                          actualvars)

        # calculate union recall, precision, Matthews correl coeff
        union_recall = recall(union_truepositives.size,
                              union_falsenegatives.size)

        union_precision = precision(union_truepositives.size,
                                    union_falsepositives.size)

        union_mcc = matthews_correl_coeff(union_truepositives.size,
                                          union_truenegatives.size,
                                          union_falsepositives.size,
                                          union_falsenegatives.size)


        # calculate the items missed by one method but found by the other method
        stet_missed_inveta_found = np.setdiff1d(inveta_truepositives,
                                                stet_truepositives)
        inveta_missed_stet_found = np.setdiff1d(stet_truepositives,
                                                inveta_truepositives)


        if not statsonly:

            recdict[magcol] = {
                # stetson J alone
                'stet_recoveredvars':stet_recoveredvars,
                'stet_truepositives':stet_truepositives,
                'stet_falsepositives':stet_falsepositives,
                'stet_truenegatives':stet_truenegatives,
                'stet_falsenegatives':stet_falsenegatives,
                'stet_precision':stet_precision,
                'stet_recall':stet_recall,
                'stet_mcc':stet_mcc,
                # inveta alone
                'inveta_recoveredvars':inveta_recoveredvars,
                'inveta_truepositives':inveta_truepositives,
                'inveta_falsepositives':inveta_falsepositives,
                'inveta_truenegatives':inveta_truenegatives,
                'inveta_falsenegatives':inveta_falsenegatives,
                'inveta_precision':inveta_precision,
                'inveta_recall':inveta_recall,
                'inveta_mcc':inveta_mcc,
                # intersect of stetson J and inveta
                'intersect_recoveredvars':intersect_recvars,
                'intersect_truepositives':intersect_truepositives,
                'intersect_falsepositives':intersect_falsepositives,
                'intersect_truenegatives':intersect_truenegatives,
                'intersect_falsenegatives':intersect_falsenegatives,
                'intersect_precision':intersect_precision,
                'intersect_recall':intersect_recall,
                'intersect_mcc':intersect_mcc,
                # union of stetson J and inveta
                'union_recoveredvars':union_recvars,
                'union_truepositives':union_truepositives,
                'union_falsepositives':union_falsepositives,
                'union_truenegatives':union_truenegatives,
                'union_falsenegatives':union_falsenegatives,
                'union_precision':union_precision,
                'union_recall':union_recall,
                'union_mcc':union_mcc,
                # true positive variables missed by one method but picked up by
                # the other
                'stet_missed_inveta_found':stet_missed_inveta_found,
                'inveta_missed_stet_found':inveta_missed_stet_found,
                # the medians to plot these items vs mag
                'magbin_medians':varthresh[magcol]['binned_sdssr_median']
            }

        # if statsonly is set, then we only return the numbers but not the
        # arrays themselves
        else:

            recdict[magcol] = {
                # stetson J alone
                'stet_recoveredvars':stet_recoveredvars.size,
                'stet_truepositives':stet_truepositives.size,
                'stet_falsepositives':stet_falsepositives.size,
                'stet_truenegatives':stet_truenegatives.size,
                'stet_falsenegatives':stet_falsenegatives.size,
                'stet_precision':stet_precision,
                'stet_recall':stet_recall,
                'stet_mcc':stet_mcc,
                # inveta alone
                'inveta_recoveredvars':inveta_recoveredvars.size,
                'inveta_truepositives':inveta_truepositives.size,
                'inveta_falsepositives':inveta_falsepositives.size,
                'inveta_truenegatives':inveta_truenegatives.size,
                'inveta_falsenegatives':inveta_falsenegatives.size,
                'inveta_precision':inveta_precision,
                'inveta_recall':inveta_recall,
                'inveta_mcc':inveta_mcc,
                # intersect of stetson J and inveta
                'intersect_recoveredvars':intersect_recvars.size,
                'intersect_truepositives':intersect_truepositives.size,
                'intersect_falsepositives':intersect_falsepositives.size,
                'intersect_truenegatives':intersect_truenegatives.size,
                'intersect_falsenegatives':intersect_falsenegatives.size,
                'intersect_precision':intersect_precision,
                'intersect_recall':intersect_recall,
                'intersect_mcc':intersect_mcc,
                # union of stetson J and inveta
                'union_recoveredvars':union_recvars.size,
                'union_truepositives':union_truepositives.size,
                'union_falsepositives':union_falsepositives.size,
                'union_truenegatives':union_truenegatives.size,
                'union_falsenegatives':union_falsenegatives.size,
                'union_precision':union_precision,
                'union_recall':union_recall,
                'union_mcc':union_mcc,
                # true positive variables missed by one method but picked up by
                # the other
                'stet_missed_inveta_found':stet_missed_inveta_found.size,
                'inveta_missed_stet_found':inveta_missed_stet_found.size,
                # the medians to plot these items vs mag
                'magbin_medians':varthresh[magcol]['binned_sdssr_median']
            }

    return recdict



def varind_gridsearch_worker(task):
    '''
    This is a parallel grid seach worker for the function below.

    '''

    simbasedir, gridpoint = task

    try:
        res = get_overall_recovered_variables(simbasedir,
                                      stetson_stdev_min=gridpoint[0],
                                      inveta_stdev_min=gridpoint[1])
        return res
    except:
        LOGEXCEPTION('failed to get info for %s' % gridpoint)
        return None



def variable_index_gridsearch(simbasedir,
                              stetson_stdev_range=[1.0,20.0],
                              inveta_stdev_range=[1.0,20.0],
                              ngridpoints=50,
                              ngridworkers=None):
    '''This runs variability selection on all of the light curves in simbasedir.

    Reads the fakelcs-info.pkl in simbasedir to get:

    - the variable objects, their types, periods, epochs, and params
    - the nonvariable objects

    For each magbin, this does a grid search using the stetson and inveta ranges
    and tries to optimize the Matthews Correlation Coefficient (best value is
    +1.0), indicating the best possible separation of variables
    vs. nonvariables. The thresholds on these two variable indexes that produce
    the largest coeff for the collection of fake LCs will probably be the ones
    that work best for actual variable classification on the real LCs.

    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    For each grid-point, calculates the true positives, false positives, true
    negatives, false negatives. Then gets the precision and recall, confusion
    matrix, and the ROC curve for variable vs. nonvariable.


    Once we've identified the best thresholds to use, we can then calculate
    variable object numbers:

    - as a function of magnitude
    - as a function of period
    - as a function of number of detections
    - as a function of amplitude of variability

    Makes nice plots and writes everything back to
    simbasedir/fakevar-recovery.pkl.

    '''

    # make the output directory where all the pkls from the variability
    # threshold runs will go
    outdir = os.path.join(simbasedir,'recvar-threshold-pkls')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # get the info from the simbasedir
    with open(os.path.join(simbasedir, 'fakelcs-info.pkl'),'rb') as infd:
        siminfo = pickle.load(infd)

    # get the column defs for the fakelcs
    timecols = siminfo['timecols']
    magcols = siminfo['magcols']
    errcols = siminfo['errcols']

    stetson_grid = np.linspace(stetson_stdev_range[0],
                               stetson_stdev_range[1],
                               num=ngridpoints)
    inveta_grid = np.linspace(inveta_stdev_range[0],
                              inveta_stdev_range[1],
                              num=ngridpoints)

    # generate the grid
    stet_inveta_grid = []
    for stet in stetson_grid:
        for inveta in inveta_grid:
            grid_point = [stet, inveta]
            stet_inveta_grid.append(grid_point)

    # the output dict
    grid_results =  {'stetson_grid':stetson_grid,
                     'inveta_grid':inveta_grid,
                     'stet_inveta_grid':stet_inveta_grid,
                     'timecols':timecols,
                     'magcols':magcols,
                     'errcols':errcols,
                     'simbasedir':os.path.abspath(simbasedir)}

    # launch parallel workers
    LOGINFO('running grid-search for stetson J-inveta...')
    pool = mp.Pool(ngridworkers)
    tasks = [(simbasedir, gp) for gp in stet_inveta_grid]
    gridresults = pool.map(varind_gridsearch_worker, tasks)
    pool.close()
    pool.join()

    grid_results['recovery'] = gridresults

    LOGINFO('done.')
    with open(os.path.join(simbasedir,'fakevar-recovery.pkl'),'wb') as outfd:
        pickle.dump(grid_results,outfd,pickle.HIGHEST_PROTOCOL)

    return grid_results



def plot_varind_gridsearch_results(gridresults):
    '''
    This plots the results from variable_index_gridsearch above.

    If outprefix is not None, it'll be added into the output filename.

    '''

    # get the values
    gx, gy = np.meshgrid(gridresults['stetson_grid'],
                         gridresults['inveta_grid'])

    plotres = {'simbasedir':gridresults['simbasedir']}

    recgrid = gridresults['recovery']

    for magcol in gridresults['magcols']:

        intersect_mcc = np.array([x[magcol]['intersect_mcc']
                                  for x in recgrid])
        intersect_precision = np.array(
            [x[magcol]['intersect_precision']
             for x in recgrid]
        )
        intersect_recall = np.array(
            [x[magcol]['intersect_recall']
             for x in recgrid]
        )

        stet_mcc = np.array(
            [x[magcol]['stet_mcc']
             for x in recgrid]
        )[::gridresults['stetson_grid'].size]
        stet_precision = np.array(
            [x[magcol]['stet_precision']
             for x in recgrid]
        )[::gridresults['stetson_grid'].size]
        stet_recall = np.array(
            [x[magcol]['stet_recall']
             for x in recgrid]
        )[::gridresults['stetson_grid'].size]
        stet_missed_inveta_found = np.array(
            [x[magcol]['stet_missed_inveta_found']
             for x in recgrid]
        )[::gridresults['stetson_grid'].size]

        inveta_mcc = np.array(
            [x[magcol]['inveta_mcc']
             for x in recgrid]
        )[:gridresults['inveta_grid'].size]
        inveta_precision = np.array(
            [x[magcol]['inveta_precision']
             for x in recgrid]
        )[:gridresults['inveta_grid'].size]
        inveta_recall = np.array(
            [x[magcol]['inveta_recall']
             for x in recgrid]
        )[:gridresults['inveta_grid'].size]
        inveta_missed_stet_found = np.array(
            [x[magcol]['inveta_missed_stet_found']
             for x in recgrid]
        )[:gridresults['inveta_grid'].size]

        fig = plt.figure(figsize=(6.4*4,4.8*3))

        # FIRST ROW: intersect 2D plot

        intersect_mcc_gz = intersect_mcc.reshape(gx.shape).T
        intersect_precision_gz = intersect_precision.reshape(gx.shape).T
        intersect_recall_gz = intersect_recall.reshape(gx.shape).T

        plt.subplot(3,4,1)
        # make the mcc grid plot
        plt.pcolormesh(gx, gy, intersect_mcc_gz,
                       cmap='RdBu',
                       norm=mpc.LogNorm(vmin=intersect_mcc_gz.min(),
                                        vmax=intersect_mcc_gz.max()))
        plt.colorbar()
        plt.xlabel('stetson J stdev multiplier threshold')
        plt.ylabel('inveta multiplier threshold')
        plt.title('MCC for intersect(stetJ,inveta)')

        # make the precision grid plot
        plt.subplot(3,4,2)
        plt.pcolormesh(gx, gy, intersect_precision_gz,
                       cmap='RdBu',
                       norm=mpc.LogNorm(vmin=intersect_precision_gz.min(),
                                        vmax=intersect_precision_gz.max()))
        plt.colorbar()
        plt.xlabel('stetson J stdev multiplier threshold')
        plt.ylabel('inveta multiplier threshold')
        plt.title('precision for intersect(stetJ,inveta)')

        # make the recall grid plot
        plt.subplot(3,4,3)
        plt.pcolormesh(gx, gy, intersect_recall_gz,
                       cmap='RdBu',
                       norm=mpc.LogNorm(vmin=intersect_recall_gz.min(),
                                        vmax=intersect_recall_gz.max()))
        plt.colorbar()
        plt.xlabel('stetson J stdev multiplier threshold')
        plt.ylabel('inveta multiplier threshold')
        plt.title('recall for intersect(stetJ,inveta)')

        # SECOND ROW: Stetson J plot
        plt.subplot(3,4,5)
        plt.plot(gridresults['stetson_grid'],
                 stet_mcc)
        plt.xlabel('stetson J stdev multiplier threshold')
        plt.ylabel('MCC')
        plt.title('MCC for stetson J')

        plt.subplot(3,4,6)
        plt.plot(gridresults['stetson_grid'],
                 stet_precision)
        plt.xlabel('stetson J stdev multiplier threshold')
        plt.ylabel('precision')
        plt.title('precision for stetson J')

        plt.subplot(3,4,7)
        plt.plot(gridresults['stetson_grid'],
                 stet_recall)
        plt.xlabel('stetson J stdev multiplier threshold')
        plt.ylabel('recall')
        plt.title('recall for stetson J')

        plt.subplot(3,4,8)
        plt.plot(gridresults['stetson_grid'],
                 stet_missed_inveta_found)
        plt.xlabel('stetson J stdev multiplier threshold')
        plt.ylabel('# objects stetson missed but inveta found')
        plt.title('stetson J missed, inveta found')


        # THIRD ROW: inveta plot
        plt.subplot(3,4,9)
        plt.plot(gridresults['inveta_grid'],
                 inveta_mcc)
        plt.xlabel('inveta stdev multiplier threshold')
        plt.ylabel('MCC')
        plt.title('MCC for inveta')

        plt.subplot(3,4,10)
        plt.plot(gridresults['inveta_grid'],
                 inveta_precision)
        plt.xlabel('inveta stdev multiplier threshold')
        plt.ylabel('precision')
        plt.title('precision for inveta')

        plt.subplot(3,4,11)
        plt.plot(gridresults['inveta_grid'],
                 inveta_recall)
        plt.xlabel('inveta stdev multiplier threshold')
        plt.ylabel('recall')
        plt.title('recall for inveta')

        plt.subplot(3,4,12)
        plt.plot(gridresults['inveta_grid'],
                 inveta_missed_stet_found)
        plt.xlabel('inveta stdev multiplier threshold')
        plt.ylabel('# objects inveta missed but stetson found')
        plt.title('inveta missed, stetson J found')

        plt.subplots_adjust(hspace=0.25,wspace=0.25)
        plt.savefig(os.path.join(gridresults['simbasedir'],
                                 '%s-var-recoverygrid.png' % magcol),
                    dpi=100,bbox_inches='tight')
        plt.close('all')


        # get the best values of MCC, recall, precision and their associated
        # stet, inveta
        stet_mcc_maxind = np.where(stet_mcc == np.max(stet_mcc))
        stet_precision_maxind = np.where(
            stet_precision == np.max(stet_precision)
        )
        stet_recall_maxind = np.where(stet_recall == np.max(stet_recall))

        best_stet_mcc = stet_mcc[stet_mcc_maxind]
        best_stet_precision = stet_mcc[stet_precision_maxind]
        best_stet_recall = stet_mcc[stet_recall_maxind]

        stet_with_best_mcc = gridresults['stetson_grid'][stet_mcc_maxind]
        stet_with_best_precision = gridresults['stetson_grid'][
            stet_precision_maxind
        ]
        stet_with_best_recall = gridresults['stetson_grid'][stet_recall_maxind]


        inveta_mcc_maxind = np.where(inveta_mcc == np.max(inveta_mcc))
        inveta_precision_maxind = np.where(
            inveta_precision == np.max(inveta_precision)
        )
        inveta_recall_maxind = np.where(inveta_recall == np.max(inveta_recall))

        best_inveta_mcc = inveta_mcc[inveta_mcc_maxind]
        best_inveta_precision = inveta_mcc[inveta_precision_maxind]
        best_inveta_recall = inveta_mcc[inveta_recall_maxind]

        inveta_with_best_mcc = gridresults['inveta_grid'][inveta_mcc_maxind]
        inveta_with_best_precision = gridresults['inveta_grid'][
            inveta_precision_maxind
        ]
        inveta_with_best_recall = gridresults['inveta_grid'][
            inveta_recall_maxind
        ]

        plotres[magcol] = {
            # stetson
            'stet_grid':gridresults['stetson_grid'],
            'stet_mcc':stet_mcc,
            'stet_precision':stet_precision,
            'stet_recall':stet_recall,
            'stet_missed_inveta_found':stet_missed_inveta_found,
            'best_stet_mcc':best_stet_mcc,
            'stet_with_best_mcc':stet_with_best_mcc,
            'best_stet_precision':best_stet_precision,
            'stet_with_best_precision':stet_with_best_precision,
            'best_stet_recall':best_stet_recall,
            'stet_with_best_recall':stet_with_best_recall,
            # inveta
            'inveta_grid':gridresults['inveta_grid'],
            'inveta_mcc':inveta_mcc,
            'inveta_precision':inveta_precision,
            'inveta_recall':inveta_recall,
            'inveta_missed_stet_found':inveta_missed_stet_found,
            'best_inveta_mcc':best_inveta_mcc,
            'inveta_with_best_mcc':inveta_with_best_mcc,
            'best_inveta_precision':best_inveta_precision,
            'inveta_with_best_precision':inveta_with_best_precision,
            'best_inveta_recall':best_inveta_recall,
            'inveta_with_best_recall':inveta_with_best_recall,
            # plot info
            'recoveryplot':os.path.join(gridresults['simbasedir'],
                                        '%s-var-recoverygrid.png' % magcol)
        }


    return plotres



#######################################
## VARIABILITY RECOVERY (PER MAGBIN) ##
#######################################

def get_recovered_variables_for_magbin(simbasedir,
                                       magbinmedian,
                                       stetson_stdev_min=2.0,
                                       inveta_stdev_min=2.0,
                                       statsonly=True):
    '''This runs variability selection for the given magbinmedian.

    magbinmedian is an item from the fakelcs-info.pkl's
    fakelcinfo['magrms'][magcol] list for each magcol and designates which
    magbin to get the recovery stats for.

    To generate a full recovery matrix, run this function for each magbin over
    the specified stetson_stdev_min and inveta_stdev_min grid.

    '''


    # get the info from the simbasedir
    with open(os.path.join(simbasedir, 'fakelcs-info.pkl'),'rb') as infd:
        siminfo = pickle.load(infd)

    lcfpaths = siminfo['lcfpath']
    objectids = siminfo['objectid']
    varflags = siminfo['isvariable']
    sdssr = siminfo['sdssr']
    ndet = siminfo['ndet']

    # get the column defs for the fakelcs
    timecols = siminfo['timecols']
    magcols = siminfo['magcols']
    errcols = siminfo['errcols']

    # get the actual variables and non-variables
    actualvars = objectids[varflags]
    actualnotvars = objectids[~varflags]

    # register the fakelc pklc as a custom lcproc format
    # now we should be able to use all lcproc functions correctly
    if 'fakelc' not in lcproc.LCFORM:

        lcproc.register_custom_lcformat(
            'fakelc',
            '*-fakelc.pkl',
            lcproc.read_pklc,
            timecols,
            magcols,
            errcols,
            magsarefluxes=False,
            specialnormfunc=None
        )

    # make the output directory if it doesn't exit
    outdir = os.path.join(simbasedir, 'recvar-threshold-pkls')
    if not os.path.exists(outdir):
        os.mkdir(outdir)


    # run the variability search
    varfeaturedir = os.path.join(simbasedir, 'varfeatures')
    varthreshinfof = os.path.join(
        outdir,
        'varthresh-magbinmed%.2f-stet%.2f-inveta%.2f.pkl' % (magbinmedian,
                                                             stetson_stdev_min,
                                                             inveta_stdev_min)
    )
    varthresh = lcproc.variability_threshold(varfeaturedir,
                                             varthreshinfof,
                                             lcformat='fakelc',
                                             min_stetj_stdev=stetson_stdev_min,
                                             min_inveta_stdev=inveta_stdev_min,
                                             verbose=False)

    # get the magbins from the varthresh info
    magbins = varthresh['magbins']

    # get the magbininds
    magbininds = np.digitize(sdssr, magbins)

    # bin the objects according to these magbins
    binned_objectids = []
    binned_actualvars = []
    binned_actualnotvars = []

    # go through all the mag bins and bin up the objectids, actual variables,
    # and actual not-variables
    for mbinind, magi in zip(np.unique(magbininds),
                             range(len(magbins)-1)):

        thisbinind = np.where(magbininds == mbinind)

        thisbin_objectids = objectids[thisbinind]
        thisbin_varflags = varflags[thisbinind]

        thisbin_actualvars = thisbin_objectids[thisbin_varflags]
        thisbin_actualnotvars = thisbin_objectids[~thisbin_varflags]

        binned_objectids.append(thisbin_objectids)
        binned_actualvars.append(thisbin_actualvars)
        binned_actualnotvars.append(thisbin_actualnotvars)


    # this is the output dict
    recdict = {
        'simbasedir':simbasedir,
        'timecols':timecols,
        'magcols':magcols,
        'errcols':errcols,
        'stetj_min_stdev':stetson_stdev_min,
        'inveta_min_stdev':inveta_stdev_min,
        'magbinmedian':magbinmedian,
    }


    # now, for each magcol, find the magbin corresponding to magbinmedian, and
    # get its stats
    for magcol in magcols:

        # this is the index of the matching magnitude bin for the magbinmedian
        # provided
        magbinind = np.where(
            np.array(varthresh[magcol]['binned_sdssr_median']) == magbinmedian
        )

        magbinind = np.asscalar(magbinind[0])

        # get the objectids, actual vars and actual notvars in this magbin
        thisbin_objectids = binned_objectids[magbinind]
        thisbin_actualvars = binned_actualvars[magbinind]
        thisbin_actualnotvars = binned_actualnotvars[magbinind]

        # stetson recovered variables in this magbin
        stet_recoveredvars = varthresh[magcol][
            'binned_objectids_thresh_stetsonj'
        ][magbinind]

        # calculate TP, FP, TN, FN
        stet_recoverednotvars = np.setdiff1d(thisbin_objectids,
                                             stet_recoveredvars)

        stet_truepositives = np.intersect1d(stet_recoveredvars,
                                            thisbin_actualvars)
        stet_falsepositives = np.intersect1d(stet_recoveredvars,
                                             thisbin_actualnotvars)
        stet_truenegatives = np.intersect1d(stet_recoverednotvars,
                                            thisbin_actualnotvars)
        stet_falsenegatives = np.intersect1d(stet_recoverednotvars,
                                             thisbin_actualvars)

        # calculate stetson recall, precision, Matthews correl coeff
        stet_recall = recall(stet_truepositives.size,
                             stet_falsenegatives.size)

        stet_precision = precision(stet_truepositives.size,
                                   stet_falsepositives.size)

        stet_mcc = matthews_correl_coeff(stet_truepositives.size,
                                         stet_truenegatives.size,
                                         stet_falsepositives.size,
                                         stet_falsenegatives.size)


        # inveta recovered variables in this magbin
        inveta_recoveredvars = varthresh[magcol][
            'binned_objectids_thresh_inveta'
        ][magbinind]
        inveta_recoverednotvars = np.setdiff1d(thisbin_objectids,
                                               inveta_recoveredvars)

        inveta_truepositives = np.intersect1d(inveta_recoveredvars,
                                              thisbin_actualvars)
        inveta_falsepositives = np.intersect1d(inveta_recoveredvars,
                                               thisbin_actualnotvars)
        inveta_truenegatives = np.intersect1d(inveta_recoverednotvars,
                                              thisbin_actualnotvars)
        inveta_falsenegatives = np.intersect1d(inveta_recoverednotvars,
                                               thisbin_actualvars)

        # calculate inveta recall, precision, Matthews correl coeff
        inveta_recall = recall(inveta_truepositives.size,
                               inveta_falsenegatives.size)

        inveta_precision = precision(inveta_truepositives.size,
                                     inveta_falsepositives.size)

        inveta_mcc = matthews_correl_coeff(inveta_truepositives.size,
                                           inveta_truenegatives.size,
                                           inveta_falsepositives.size,
                                           inveta_falsenegatives.size)


        # calculate the stats for combined intersect(stet,inveta) variable flags
        intersect_recvars = np.intersect1d(stet_recoveredvars,
                                           inveta_recoveredvars)
        intersect_recnonvars = np.setdiff1d(thisbin_objectids,
                                            intersect_recvars)

        intersect_truepositives = np.intersect1d(intersect_recvars,
                                                 thisbin_actualvars)
        intersect_falsepositives = np.intersect1d(intersect_recvars,
                                                  thisbin_actualnotvars)
        intersect_truenegatives = np.intersect1d(intersect_recnonvars,
                                                 thisbin_actualnotvars)
        intersect_falsenegatives = np.intersect1d(intersect_recnonvars,
                                                  thisbin_actualvars)

        # calculate intersection recall, precision, Matthews correl coeff
        intersect_recall = recall(intersect_truepositives.size,
                                  intersect_falsenegatives.size)

        intersect_precision = precision(intersect_truepositives.size,
                                        intersect_falsepositives.size)

        intersect_mcc = matthews_correl_coeff(intersect_truepositives.size,
                                              intersect_truenegatives.size,
                                              intersect_falsepositives.size,
                                              intersect_falsenegatives.size)

        # calculate the stats for combined union(stet,inveta) variable flags
        union_recvars = np.union1d(stet_recoveredvars,
                                   inveta_recoveredvars)
        union_recnonvars = np.setdiff1d(thisbin_objectids, union_recvars)

        union_truepositives = np.union1d(union_recvars,
                                         thisbin_actualvars)
        union_falsepositives = np.union1d(union_recvars,
                                          thisbin_actualnotvars)
        union_truenegatives = np.union1d(union_recnonvars,
                                         thisbin_actualnotvars)
        union_falsenegatives = np.union1d(union_recnonvars,
                                          thisbin_actualvars)

        # calculate union recall, precision, Matthews correl coeff
        union_recall = recall(union_truepositives.size,
                              union_falsenegatives.size)

        union_precision = precision(union_truepositives.size,
                                    union_falsepositives.size)

        union_mcc = matthews_correl_coeff(union_truepositives.size,
                                          union_truenegatives.size,
                                          union_falsepositives.size,
                                          union_falsenegatives.size)


        # calculate the items missed by one method but found by the other method
        stet_missed_inveta_found = np.setdiff1d(inveta_truepositives,
                                                stet_truepositives)
        inveta_missed_stet_found = np.setdiff1d(stet_truepositives,
                                                inveta_truepositives)


        if not statsonly:

            recdict[magcol] = {
                # stetson J alone
                'stet_recoveredvars':stet_recoveredvars,
                'stet_truepositives':stet_truepositives,
                'stet_falsepositives':stet_falsepositives,
                'stet_truenegatives':stet_truenegatives,
                'stet_falsenegatives':stet_falsenegatives,
                'stet_precision':stet_precision,
                'stet_recall':stet_recall,
                'stet_mcc':stet_mcc,
                # inveta alone
                'inveta_recoveredvars':inveta_recoveredvars,
                'inveta_truepositives':inveta_truepositives,
                'inveta_falsepositives':inveta_falsepositives,
                'inveta_truenegatives':inveta_truenegatives,
                'inveta_falsenegatives':inveta_falsenegatives,
                'inveta_precision':inveta_precision,
                'inveta_recall':inveta_recall,
                'inveta_mcc':inveta_mcc,
                # intersect of stetson J and inveta
                'intersect_recoveredvars':intersect_recvars,
                'intersect_truepositives':intersect_truepositives,
                'intersect_falsepositives':intersect_falsepositives,
                'intersect_truenegatives':intersect_truenegatives,
                'intersect_falsenegatives':intersect_falsenegatives,
                'intersect_precision':intersect_precision,
                'intersect_recall':intersect_recall,
                'intersect_mcc':intersect_mcc,
                # union of stetson J and inveta
                'union_recoveredvars':union_recvars,
                'union_truepositives':union_truepositives,
                'union_falsepositives':union_falsepositives,
                'union_truenegatives':union_truenegatives,
                'union_falsenegatives':union_falsenegatives,
                'union_precision':union_precision,
                'union_recall':union_recall,
                'union_mcc':union_mcc,
                # true positive variables missed by one method but picked up by
                # the other
                'stet_missed_inveta_found':stet_missed_inveta_found,
                'inveta_missed_stet_found':inveta_missed_stet_found,
                # bin info
                'actual_variables':thisbin_actualvars,
                'actual_nonvariables':thisbin_actualnotvars,
                'all_objectids':thisbin_objectids,
                'magbinind':magbinind,

            }

        # if statsonly is set, then we only return the numbers but not the
        # arrays themselves
        else:

            recdict[magcol] = {
                # stetson J alone
                'stet_recoveredvars':stet_recoveredvars.size,
                'stet_truepositives':stet_truepositives.size,
                'stet_falsepositives':stet_falsepositives.size,
                'stet_truenegatives':stet_truenegatives.size,
                'stet_falsenegatives':stet_falsenegatives.size,
                'stet_precision':stet_precision,
                'stet_recall':stet_recall,
                'stet_mcc':stet_mcc,
                # inveta alone
                'inveta_recoveredvars':inveta_recoveredvars.size,
                'inveta_truepositives':inveta_truepositives.size,
                'inveta_falsepositives':inveta_falsepositives.size,
                'inveta_truenegatives':inveta_truenegatives.size,
                'inveta_falsenegatives':inveta_falsenegatives.size,
                'inveta_precision':inveta_precision,
                'inveta_recall':inveta_recall,
                'inveta_mcc':inveta_mcc,
                # intersect of stetson J and inveta
                'intersect_recoveredvars':intersect_recvars.size,
                'intersect_truepositives':intersect_truepositives.size,
                'intersect_falsepositives':intersect_falsepositives.size,
                'intersect_truenegatives':intersect_truenegatives.size,
                'intersect_falsenegatives':intersect_falsenegatives.size,
                'intersect_precision':intersect_precision,
                'intersect_recall':intersect_recall,
                'intersect_mcc':intersect_mcc,
                # union of stetson J and inveta
                'union_recoveredvars':union_recvars.size,
                'union_truepositives':union_truepositives.size,
                'union_falsepositives':union_falsepositives.size,
                'union_truenegatives':union_truenegatives.size,
                'union_falsenegatives':union_falsenegatives.size,
                'union_precision':union_precision,
                'union_recall':union_recall,
                'union_mcc':union_mcc,
                # true positive variables missed by one method but picked up by
                # the other
                'stet_missed_inveta_found':stet_missed_inveta_found.size,
                'inveta_missed_stet_found':inveta_missed_stet_found.size,
                # bin info
                'actual_variables':thisbin_actualvars.size,
                'actual_nonvariables':thisbin_actualnotvars.size,
                'all_objectids':thisbin_objectids.size,
                'magbinind':magbinind,
            }


    #
    # done with per magcol
    #

    return recdict



def magbin_varind_gridsearch_worker(task):
    '''
    This is a parallel grid search worker for the function below.

    '''

    simbasedir, gridpoint, magbinmedian = task

    try:
        res = get_recovered_variables_for_magbin(simbasedir,
                                                 magbinmedian,
                                                 stetson_stdev_min=gridpoint[0],
                                                 inveta_stdev_min=gridpoint[1],
                                                 statsonly=True)
        return res
    except:
        LOGEXCEPTION('failed to get info for %s' % gridpoint)
        return None



def variable_index_gridsearch_magbin(simbasedir,
                                         stetson_stdev_range=[1.0,20.0],
                                         inveta_stdev_range=[1.0,20.0],
                                         ngridpoints=50,
                                         ngridworkers=None):
    '''This runs a variable index grid search per magbin.

    Similar to variable_index_gridsearch above.

    Gets the magbin medians from the fakelcinfo.pkl's
    dict['magrms'][magcols[0]['binned_sdssr_median'] value.

    '''

    # make the output directory where all the pkls from the variability
    # threshold runs will go
    outdir = os.path.join(simbasedir,'recvar-threshold-pkls')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # get the info from the simbasedir
    with open(os.path.join(simbasedir, 'fakelcs-info.pkl'),'rb') as infd:
        siminfo = pickle.load(infd)

    # get the column defs for the fakelcs
    timecols = siminfo['timecols']
    magcols = siminfo['magcols']
    errcols = siminfo['errcols']

    # get the magbinmedians to use for the recovery processing
    magbinmedians = siminfo['magrms'][magcols[0]]['binned_sdssr_median']

    # generate the grids for stetson and inveta
    stetson_grid = np.linspace(stetson_stdev_range[0],
                               stetson_stdev_range[1],
                               num=ngridpoints)
    inveta_grid = np.linspace(inveta_stdev_range[0],
                              inveta_stdev_range[1],
                              num=ngridpoints)

    # generate the grid
    stet_inveta_grid = []
    for stet in stetson_grid:
        for inveta in inveta_grid:
            grid_point = [stet, inveta]
            stet_inveta_grid.append(grid_point)

    # the output dict
    grid_results =  {'stetson_grid':stetson_grid,
                     'inveta_grid':inveta_grid,
                     'stet_inveta_grid':stet_inveta_grid,
                     'magbinmedians':magbinmedians,
                     'timecols':timecols,
                     'magcols':magcols,
                     'errcols':errcols,
                     'simbasedir':os.path.abspath(simbasedir),
                     'recovery':[]}


    # set up the pool
    pool = mp.Pool(ngridworkers)

    # run the grid search per magbinmedian
    for magbinmedian in magbinmedians:

        LOGINFO('running stetson J-inveta grid-search '
                'for magbinmedian = %.3f...' % magbinmedian)

        tasks = [(simbasedir, gp, magbinmedian) for gp in stet_inveta_grid]
        thisbin_results = pool.map(magbin_varind_gridsearch_worker, tasks)
        grid_results['recovery'].append(thisbin_results)

    pool.close()
    pool.join()


    LOGINFO('done.')
    with open(os.path.join(simbasedir,
                           'fakevar-recovery-per-magbin.pkl'),'wb') as outfd:
        pickle.dump(grid_results,outfd,pickle.HIGHEST_PROTOCOL)

    return grid_results



def plot_varind_gridsearch_magbin_results(gridresults):
    '''
    This plots the gridsearch results from variable_index_gridsearch_magbin.

    '''

    # get the values
    gx, gy = np.meshgrid(gridresults['stetson_grid'],
                         gridresults['inveta_grid'])

    plotres = {'simbasedir':gridresults['simbasedir']}

    recgrid = gridresults['recovery']

    for magcol in gridresults['magcols']:

        plotres[magcol] = {}

        for magbinind, magbinmedian in enumerate(gridresults['magbinmedians']):

            LOGINFO('plotting results for %s: magbin: %.3f' %
                    (magcol, magbinmedian))

            intersect_mcc = np.array([x[magcol]['intersect_mcc']
                                      for x in recgrid[magbinind]])
            intersect_precision = np.array(
                [x[magcol]['intersect_precision']
                 for x in recgrid[magbinind]]
            )
            intersect_recall = np.array(
                [x[magcol]['intersect_recall']
                 for x in recgrid[magbinind]]
            )

            stet_mcc = np.array(
                [x[magcol]['stet_mcc']
                 for x in recgrid[magbinind]]
            )[::gridresults['stetson_grid'].size]
            stet_precision = np.array(
                [x[magcol]['stet_precision']
                 for x in recgrid[magbinind]]
            )[::gridresults['stetson_grid'].size]
            stet_recall = np.array(
                [x[magcol]['stet_recall']
                 for x in recgrid[magbinind]]
            )[::gridresults['stetson_grid'].size]
            stet_missed_inveta_found = np.array(
                [x[magcol]['stet_missed_inveta_found']
                 for x in recgrid[magbinind]]
            )[::gridresults['stetson_grid'].size]

            inveta_mcc = np.array(
                [x[magcol]['inveta_mcc']
                 for x in recgrid[magbinind]]
            )[:gridresults['inveta_grid'].size]
            inveta_precision = np.array(
                [x[magcol]['inveta_precision']
                 for x in recgrid[magbinind]]
            )[:gridresults['inveta_grid'].size]
            inveta_recall = np.array(
                [x[magcol]['inveta_recall']
                 for x in recgrid[magbinind]]
            )[:gridresults['inveta_grid'].size]
            inveta_missed_stet_found = np.array(
                [x[magcol]['inveta_missed_stet_found']
                 for x in recgrid[magbinind]]
            )[:gridresults['inveta_grid'].size]

            fig = plt.figure(figsize=(6.4*4,4.8*3))

            # FIRST ROW: intersect 2D plot

            intersect_mcc_gz = intersect_mcc.reshape(gx.shape).T
            intersect_precision_gz = intersect_precision.reshape(gx.shape).T
            intersect_recall_gz = intersect_recall.reshape(gx.shape).T

            # make the mcc grid plot
            plt.subplot(3,4,1)
            if np.any(np.isfinite(intersect_mcc_gz) & (intersect_mcc_gz > 0.0)):
                plt.pcolormesh(
                    gx, gy, intersect_mcc_gz,
                    cmap='RdBu',
                    norm=mpc.LogNorm(vmin=np.nanmin(intersect_mcc_gz),
                                     vmax=np.nanmax(intersect_mcc_gz))
                )
                plt.colorbar()
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('inveta multiplier threshold')
                plt.title('MCC for intersect(stetJ,inveta)')

            else:
                plt.text(0.5,0.5,
                         'intersect(stet,inveta) MCC values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            # make the precision grid plot
            plt.subplot(3,4,2)
            if np.any(np.isfinite(intersect_precision_gz) &
                      (intersect_precision_gz > 0.0)):
                plt.pcolormesh(
                    gx, gy, intersect_precision_gz,
                    cmap='RdBu',
                    norm=mpc.LogNorm(vmin=np.nanmin(intersect_precision_gz),
                                     vmax=np.nanax(intersect_precision_gz))
                )
                plt.colorbar()
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('inveta multiplier threshold')
                plt.title('precision for intersect(stetJ,inveta)')
            else:
                plt.text(0.5,0.5,
                         'intersect(stet,inveta) precision values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            # make the recall grid plot
            plt.subplot(3,4,3)
            if np.any(np.isfinite(intersect_recall_gz) &
                      (intersect_recall_gz > 0.0)):
                plt.pcolormesh(
                    gx, gy, intersect_recall_gz,
                    cmap='RdBu',
                    norm=mpc.LogNorm(vmin=np.nanmin(intersect_recall_gz),
                                     vmax=np.nanmax(intersect_recall_gz))
                )
                plt.colorbar()
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('inveta multiplier threshold')
                plt.title('recall for intersect(stetJ,inveta)')
            else:
                plt.text(0.5,0.5,
                         'intersect(stet,inveta) recall values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])


            # SECOND ROW: Stetson J plot
            plt.subplot(3,4,5)
            if np.any(np.isfinite(stet_mcc)):
                plt.plot(gridresults['stetson_grid'],
                         stet_mcc)
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('MCC')
                plt.title('MCC for stetson J')
            else:
                plt.text(0.5,0.5,
                         'stet MCC values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,4,6)
            if np.any(np.isfinite(stet_precision)):
                plt.plot(gridresults['stetson_grid'],
                         stet_precision)
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('precision')
                plt.title('precision for stetson J')
            else:
                plt.text(0.5,0.5,
                         'stet precision values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,4,7)
            if np.any(np.isfinite(stet_recall)):
                plt.plot(gridresults['stetson_grid'],
                         stet_recall)
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('recall')
                plt.title('recall for stetson J')
            else:
                plt.text(0.5,0.5,
                         'stet recall values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,4,8)
            if np.any(np.isfinite(stet_missed_inveta_found)):
                plt.plot(gridresults['stetson_grid'],
                         stet_missed_inveta_found)
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('# objects stetson missed but inveta found')
                plt.title('stetson J missed, inveta found')
            else:
                plt.text(0.5,0.5,
                         'stet-missed/inveta-found values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])


            # THIRD ROW: inveta plot

            plt.subplot(3,4,9)
            if np.any(np.isfinite(inveta_mcc)):
                plt.plot(gridresults['inveta_grid'],
                         inveta_mcc)
                plt.xlabel('inveta stdev multiplier threshold')
                plt.ylabel('MCC')
                plt.title('MCC for inveta')
            else:
                plt.text(0.5,0.5,
                         'inveta MCC values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,4,10)
            if np.any(np.isfinite(inveta_precision)):
                plt.plot(gridresults['inveta_grid'],
                         inveta_precision)
                plt.xlabel('inveta stdev multiplier threshold')
                plt.ylabel('precision')
                plt.title('precision for inveta')
            else:
                plt.text(0.5,0.5,
                         'inveta precision values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,4,11)
            if np.any(np.isfinite(inveta_recall)):
                plt.plot(gridresults['inveta_grid'],
                         inveta_recall)
                plt.xlabel('inveta stdev multiplier threshold')
                plt.ylabel('recall')
                plt.title('recall for inveta')
            else:
                plt.text(0.5,0.5,
                         'inveta recall values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,4,12)
            if np.any(np.isfinite(inveta_missed_stet_found)):
                plt.plot(gridresults['inveta_grid'],
                         inveta_missed_stet_found)
                plt.xlabel('inveta stdev multiplier threshold')
                plt.ylabel('# objects inveta missed but stetson found')
                plt.title('inveta missed, stetson J found')
            else:
                plt.text(0.5,0.5,
                         'inveta-missed-stet-found values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes)
                plt.xticks([])
                plt.yticks([])

            plt.subplots_adjust(hspace=0.25,wspace=0.25)

            plt.suptitle('magcol: %s, magbin: %.3f' % (magcol, magbinmedian))

            gridplotf = os.path.join(gridresults['simbasedir'],
                                     '%s-%.3f-var-recoverygrid-permagbin.png' %
                                     (magcol, magbinmedian))

            plt.savefig(gridplotf,dpi=100,bbox_inches='tight')
            plt.close('all')


            # get the best values of MCC, recall, precision and their associated
            # stet, inveta
            stet_mcc_maxind = np.where(stet_mcc == np.max(stet_mcc))
            stet_precision_maxind = np.where(
                stet_precision == np.max(stet_precision)
            )
            stet_recall_maxind = np.where(stet_recall == np.max(stet_recall))

            best_stet_mcc = stet_mcc[stet_mcc_maxind]
            best_stet_precision = stet_mcc[stet_precision_maxind]
            best_stet_recall = stet_mcc[stet_recall_maxind]

            stet_with_best_mcc = gridresults['stetson_grid'][stet_mcc_maxind]
            stet_with_best_precision = gridresults['stetson_grid'][
                stet_precision_maxind
            ]
            stet_with_best_recall = (
                gridresults['stetson_grid'][stet_recall_maxind]
            )


            inveta_mcc_maxind = np.where(inveta_mcc == np.max(inveta_mcc))
            inveta_precision_maxind = np.where(
                inveta_precision == np.max(inveta_precision)
            )
            inveta_recall_maxind = (
                np.where(inveta_recall == np.max(inveta_recall))
            )

            best_inveta_mcc = inveta_mcc[inveta_mcc_maxind]
            best_inveta_precision = inveta_mcc[inveta_precision_maxind]
            best_inveta_recall = inveta_mcc[inveta_recall_maxind]

            inveta_with_best_mcc = gridresults['inveta_grid'][inveta_mcc_maxind]
            inveta_with_best_precision = gridresults['inveta_grid'][
                inveta_precision_maxind
            ]
            inveta_with_best_recall = gridresults['inveta_grid'][
                inveta_recall_maxind
            ]

            plotres[magcol][magbinmedian] = {
                # stetson
                'stet_grid':gridresults['stetson_grid'],
                'stet_mcc':stet_mcc,
                'stet_precision':stet_precision,
                'stet_recall':stet_recall,
                'stet_missed_inveta_found':stet_missed_inveta_found,
                'best_stet_mcc':best_stet_mcc,
                'stet_with_best_mcc':stet_with_best_mcc,
                'best_stet_precision':best_stet_precision,
                'stet_with_best_precision':stet_with_best_precision,
                'best_stet_recall':best_stet_recall,
                'stet_with_best_recall':stet_with_best_recall,
                # inveta
                'inveta_grid':gridresults['inveta_grid'],
                'inveta_mcc':inveta_mcc,
                'inveta_precision':inveta_precision,
                'inveta_recall':inveta_recall,
                'inveta_missed_stet_found':inveta_missed_stet_found,
                'best_inveta_mcc':best_inveta_mcc,
                'inveta_with_best_mcc':inveta_with_best_mcc,
                'best_inveta_precision':best_inveta_precision,
                'inveta_with_best_precision':inveta_with_best_precision,
                'best_inveta_recall':best_inveta_recall,
                'inveta_with_best_recall':inveta_with_best_recall,
                # plot info
                'recoveryplot':gridplotf
            }

            # recommend inveta and stetson index for this magbin

            # if there are multiple stets, choose the smallest one
            if stet_with_best_mcc.size > 1:
                LOGINFO('smallest stetson J stdev multiplier with best '
                        'MCC for magbin: %.3f = %.3f' % (magbinmedian,
                                                         stet_with_best_mcc[0]))
            else:
                LOGINFO('stetson J stdev multiplier with best '
                        'MCC for magbin: %.3f = %.3f' % (magbinmedian,
                                                         stet_with_best_mcc[0]))

            # if there are multiple best invetas, choose the smallest one
            if inveta_with_best_mcc.size > 1:
                LOGINFO('smallest inveta stdev multiplier with best '
                        'MCC for magbin: %.3f = %.3f'
                        % (magbinmedian,
                           inveta_with_best_mcc[0]))
            else:
                LOGINFO('inveta stdev multiplier with best '
                        'MCC for magbin: %.3f = %.3f'
                        % (magbinmedian,
                           inveta_with_best_mcc[0]))



    # write the plotresults to a pickle
    plotrespicklef = os.path.join('simbasedir',
                                  'varindex-gridsearch-magbin-results.pkl')
    with open(plotrespicklef, 'wb') as outfd:
        pickle.dump(plotres, outfd, pickle.HIGHEST_PROTOCOL)

    return plotres
