#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''fakelcrecovery - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This is a companion module for fakelcgen.py. It runs LCs generated using
functions in that module through variable star detection and classification to
see how well they are recovered.

TODO: add in grid-search for IQR for all functions below in addition to
existing grid-searches for stetson and inveta.

'''
import os
import os.path
import pickle
import gzip
import glob

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
                    mindet=1000,
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
                                          mindet=mindet,
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



#######################################
## VARIABILITY RECOVERY (PER MAGBIN) ##
#######################################

def get_recovered_variables_for_magbin(simbasedir,
                                       magbinmedian,
                                       stetson_stdev_min=2.0,
                                       inveta_stdev_min=2.0,
                                       iqr_stdev_min=2.0,
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
                                             min_iqr_stdev=iqr_stdev_min,
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


        # iqr recovered variables in this magbin
        iqr_recoveredvars = varthresh[magcol][
            'binned_objectids_thresh_iqr'
        ][magbinind]
        iqr_recoverednotvars = np.setdiff1d(thisbin_objectids,
                                            iqr_recoveredvars)

        iqr_truepositives = np.intersect1d(iqr_recoveredvars,
                                           thisbin_actualvars)
        iqr_falsepositives = np.intersect1d(iqr_recoveredvars,
                                            thisbin_actualnotvars)
        iqr_truenegatives = np.intersect1d(iqr_recoverednotvars,
                                           thisbin_actualnotvars)
        iqr_falsenegatives = np.intersect1d(iqr_recoverednotvars,
                                            thisbin_actualvars)

        # calculate iqr recall, precision, Matthews correl coeff
        iqr_recall = recall(iqr_truepositives.size,
                            iqr_falsenegatives.size)

        iqr_precision = precision(iqr_truepositives.size,
                                  iqr_falsepositives.size)

        iqr_mcc = matthews_correl_coeff(iqr_truepositives.size,
                                        iqr_truenegatives.size,
                                        iqr_falsepositives.size,
                                        iqr_falsenegatives.size)


        # calculate the items missed by one method but found by the other
        # methods
        stet_missed_inveta_found = np.setdiff1d(inveta_truepositives,
                                                stet_truepositives)
        stet_missed_iqr_found = np.setdiff1d(iqr_truepositives,
                                             stet_truepositives)

        inveta_missed_stet_found = np.setdiff1d(stet_truepositives,
                                                inveta_truepositives)
        inveta_missed_iqr_found = np.setdiff1d(iqr_truepositives,
                                               inveta_truepositives)

        iqr_missed_stet_found = np.setdiff1d(stet_truepositives,
                                             iqr_truepositives)
        iqr_missed_inveta_found = np.setdiff1d(inveta_truepositives,
                                               iqr_truepositives)

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
                # iqr alone
                'iqr_recoveredvars':iqr_recoveredvars,
                'iqr_truepositives':iqr_truepositives,
                'iqr_falsepositives':iqr_falsepositives,
                'iqr_truenegatives':iqr_truenegatives,
                'iqr_falsenegatives':iqr_falsenegatives,
                'iqr_precision':iqr_precision,
                'iqr_recall':iqr_recall,
                'iqr_mcc':iqr_mcc,
                # true positive variables missed by one method but picked up by
                # the others
                'stet_missed_inveta_found':stet_missed_inveta_found,
                'stet_missed_iqr_found':stet_missed_iqr_found,
                'inveta_missed_stet_found':inveta_missed_stet_found,
                'inveta_missed_iqr_found':inveta_missed_iqr_found,
                'iqr_missed_stet_found':iqr_missed_stet_found,
                'iqr_missed_inveta_found':iqr_missed_inveta_found,
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
                # iqr alone
                'iqr_recoveredvars':iqr_recoveredvars.size,
                'iqr_truepositives':iqr_truepositives.size,
                'iqr_falsepositives':iqr_falsepositives.size,
                'iqr_truenegatives':iqr_truenegatives.size,
                'iqr_falsenegatives':iqr_falsenegatives.size,
                'iqr_precision':iqr_precision,
                'iqr_recall':iqr_recall,
                'iqr_mcc':iqr_mcc,
                # true positive variables missed by one method but picked up by
                # the others
                'stet_missed_inveta_found':stet_missed_inveta_found.size,
                'stet_missed_iqr_found':stet_missed_iqr_found.size,
                'inveta_missed_stet_found':inveta_missed_stet_found.size,
                'inveta_missed_iqr_found':inveta_missed_iqr_found.size,
                'iqr_missed_stet_found':iqr_missed_stet_found.size,
                'iqr_missed_inveta_found':iqr_missed_inveta_found.size,
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
                                                 iqr_stdev_min=gridpoint[2],
                                                 statsonly=True)
        return res
    except:
        LOGEXCEPTION('failed to get info for %s' % gridpoint)
        return None



def variable_index_gridsearch_magbin(simbasedir,
                                     stetson_stdev_range=[1.0,20.0],
                                     inveta_stdev_range=[1.0,20.0],
                                     iqr_stdev_range=[1.0,20.0],
                                     ngridpoints=32,
                                     ngridworkers=None):
    '''This runs a variable index grid search per magbin.

    Similar to variable_index_gridsearch above.

    Gets the magbin medians from the fakelcinfo.pkl's
    dict['magrms'][magcols[0]['binned_sdssr_median'] value.

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

    # get the magbinmedians to use for the recovery processing
    magbinmedians = siminfo['magrms'][magcols[0]]['binned_sdssr_median']

    # generate the grids for stetson and inveta
    stetson_grid = np.linspace(stetson_stdev_range[0],
                               stetson_stdev_range[1],
                               num=ngridpoints)
    inveta_grid = np.linspace(inveta_stdev_range[0],
                              inveta_stdev_range[1],
                              num=ngridpoints)
    iqr_grid = np.linspace(iqr_stdev_range[0],
                           iqr_stdev_range[1],
                           num=ngridpoints)

    # generate the grid
    stet_inveta_iqr_grid = []
    for stet in stetson_grid:
        for inveta in inveta_grid:
            for iqr in iqr_grid:
                grid_point = [stet, inveta, iqr]
                stet_inveta_iqr_grid.append(grid_point)

    # the output dict
    grid_results =  {'stetson_grid':stetson_grid,
                     'inveta_grid':inveta_grid,
                     'iqr_grid':iqr_grid,
                     'stet_inveta_iqr_grid':stet_inveta_iqr_grid,
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

        tasks = [(simbasedir, gp, magbinmedian) for gp in stet_inveta_iqr_grid]
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

    plotres = {'simbasedir':gridresults['simbasedir']}

    recgrid = gridresults['recovery']
    simbasedir = gridresults['simbasedir']

    for magcol in gridresults['magcols']:

        plotres[magcol] = {'best_stetsonj':[],
                           'best_inveta':[],
                           'best_iqr':[],
                           'magbinmedians':gridresults['magbinmedians']}

        # go through all the magbins
        for magbinind, magbinmedian in enumerate(gridresults['magbinmedians']):

            LOGINFO('plotting results for %s: magbin: %.3f' %
                    (magcol, magbinmedian))

            # FIXME: figure out the correct indexes for a 3D grid here

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
            stet_missed_iqr_found = np.array(
                [x[magcol]['stet_missed_iqr_found']
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
            inveta_missed_iqr_found = np.array(
                [x[magcol]['inveta_missed_iqr_found']
                 for x in recgrid[magbinind]]
            )[:gridresults['inveta_grid'].size]

            iqr_mcc = np.array(
                [x[magcol]['iqr_mcc']
                 for x in recgrid[magbinind]]
            )[:gridresults['iqr_grid'].size]
            iqr_precision = np.array(
                [x[magcol]['iqr_precision']
                 for x in recgrid[magbinind]]
            )[:gridresults['iqr_grid'].size]
            iqr_recall = np.array(
                [x[magcol]['iqr_recall']
                 for x in recgrid[magbinind]]
            )[:gridresults['iqr_grid'].size]
            iqr_missed_stet_found = np.array(
                [x[magcol]['iqr_missed_stet_found']
                 for x in recgrid[magbinind]]
            )[:gridresults['iqr_grid'].size]
            iqr_missed_inveta_found = np.array(
                [x[magcol]['iqr_missed_inveta_found']
                 for x in recgrid[magbinind]]
            )[:gridresults['iqr_grid'].size]


            fig = plt.figure(figsize=(6.4*5, 4.8*3))

            # FIRST ROW: intersect 2D plot

            intersect_mcc_gz = intersect_mcc.reshape(gx.shape).T
            intersect_precision_gz = intersect_precision.reshape(gx.shape).T
            intersect_recall_gz = intersect_recall.reshape(gx.shape).T

            # get rid of 0.0 values because they mess up logs
            intersect_mcc_gz[intersect_mcc_gz == 0.0] = 1.0e-3
            intersect_recall_gz[intersect_recall_gz == 0.0] = 1.0e-3
            intersect_precision_gz[intersect_precision_gz == 0.0] = 1.0e-3

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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                                     vmax=np.nanmax(intersect_precision_gz))
                )
                plt.colorbar()
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('inveta multiplier threshold')
                plt.title('precision for intersect(stetJ,inveta)')
            else:
                plt.text(0.5,0.5,
                         'intersect(stet,inveta) precision values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
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
                plotres[magcol]['best_stetsonj'].append(stet_with_best_mcc[0])
            elif stet_with_best_mcc.size > 0:
                plotres[magcol]['best_stetsonj'].append(stet_with_best_mcc[0])
            else:
                plotres[magcol]['best_stetsonj'].append(np.nan)

            # if there are multiple best invetas, choose the smallest one
            if inveta_with_best_mcc.size > 1:
                plotres[magcol]['best_inveta'].append(inveta_with_best_mcc[0])
            elif inveta_with_best_mcc.size > 0:
                plotres[magcol]['best_inveta'].append(inveta_with_best_mcc[0])
            else:
                plotres[magcol]['best_inveta'].append(np.nan)

    # write the plotresults to a pickle
    plotrespicklef = os.path.join(simbasedir,
                                  'varindex-gridsearch-magbin-results.pkl')
    with open(plotrespicklef, 'wb') as outfd:
        pickle.dump(plotres, outfd, pickle.HIGHEST_PROTOCOL)


    # recommend the values of stetson J and inveta to use
    for magcol in gridresults['magcols']:

        LOGINFO('best stdev multipliers for each %s magbin:' % magcol)
        LOGINFO('magbin    inveta    stetson J')

        for magbin, inveta, stet in zip(plotres[magcol]['magbinmedians'],
                                        plotres[magcol]['best_inveta'],
                                        plotres[magcol]['best_stetsonj']):
            LOGINFO('%.3f    %.3f    %.3f' % (magbin,inveta,stet))


    return plotres



################################
## PERIODIC VARIABLE RECOVERY ##
################################

PERIODIC_VARTYPES = ['EB','RRab','RRc','rotator',
                     'HADS','planet','LPV','cepheid']

ALIAS_TYPES = ['actual',
               'twice',
               'half',
               'ratio_over_1plus',
               'ratio_over_1minus',
               'ratio_over_1plus_twice',
               'ratio_over_1minus_twice',
               'ratio_over_1plus_thrice',
               'ratio_over_1minus_thrice',
               'ratio_over_minus1',
               'ratio_over_twice_minus1']


def run_periodfinding(simbasedir,
                      pfmethods=['gls','pdm','bls'],
                      pfkwargs=[{},{},{'startp':1.0,'maxtransitduration':0.3}],
                      getblssnr=False,
                      sigclip=5.0,
                      nperiodworkers=10,
                      ncontrolworkers=4,
                      liststartindex=None,
                      listmaxobjects=None):
    '''This runs periodfinding using several periodfinders on a collection of
    fakelcs.

    Use pfmethods to specify which periodfinders to run. These must be in
    lcproc.PFMETHODS.

    Use pfkwargs to provide optional kwargs to the periodfinders.

    If getblssnr is True, will run BLS SNR calculations for each object and
    magcol. This takes a while to run, so it's disabled (False) by default.

    sigclip sets the sigma-clip to use for the light curves before putting them
    through each of the periodfinders.

    nperiodworkers is the number of period-finder workers to launch.

    ncontrolworkers is the number of controlling processes to launch.

    liststartindex sets the index from where to start in the list of
    fakelcs. listmaxobjects sets the maximum number of objects in the fakelc
    list to run periodfinding for in this invocation. Together, these can be
    used to distribute processing over several independent machines if the
    number of light curves is very large.

    As a rough benchmark, 25000 fakelcs with up to 50000 points per lc take
    about 26 days in total to run on an invocation of this function using
    GLS+PDM+BLS and 10 periodworkers and 4 controlworkers (so all 40 'cores') on
    a 2 x Xeon E5-2660v3 machine.

    '''

    # get the info from the simbasedir
    with open(os.path.join(simbasedir, 'fakelcs-info.pkl'),'rb') as infd:
        siminfo = pickle.load(infd)

    lcfpaths = siminfo['lcfpath']
    pfdir = os.path.join(simbasedir,'periodfinding')

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

    if liststartindex:
        lcfpaths = lcfpaths[liststartindex:]

    if listmaxobjects:
        lcfpaths = lcfpaths[:listmaxobjects]

    pfinfo = lcproc.parallel_pf(lcfpaths,
                                pfdir,
                                lcformat='fakelc',
                                pfmethods=pfmethods,
                                pfkwargs=pfkwargs,
                                getblssnr=getblssnr,
                                sigclip=sigclip,
                                nperiodworkers=nperiodworkers,
                                ncontrolworkers=ncontrolworker)

    with open(os.path.join(simbasedir,
                           'fakelc-periodfinding.pkl'),'wb') as outfd:
        pickle.dump(varinfo, outfd, pickle.HIGHEST_PROTOCOL)

    return os.path.join(simbasedir,'fakelc-periodfinding.pkl')



def check_periodrec_alias(actualperiod, recoveredperiod, tolerance=1.0e-3):
    '''This determines what kind of aliasing (if any) exists between
    recoveredperiod and actualperiod.

    '''

    if not (np.isfinite(actualperiod) and np.isfinite(recoveredperiod)):

        LOGERROR("can't compare nan values for actual/recovered periods")
        return 'unknown'

    else:

        #################
        ## ALIAS TYPES ##
        #################

        # simple ratios
        twotimes_p = actualperiod*2.0
        half_p = actualperiod*0.5

        # first kind of alias
        alias_1a = actualperiod/(1.0+actualperiod)
        alias_1b = actualperiod/(1.0-actualperiod)

        # second kind of alias
        alias_2a = actualperiod/(1.0+2.0*actualperiod)
        alias_2b = actualperiod/(1.0-2.0*actualperiod)

        # third kind of alias
        alias_3a = actualperiod/(1.0+3.0*actualperiod)
        alias_3b = actualperiod/(1.0-3.0*actualperiod)

        # fourth kind of alias
        alias_4a = actualperiod/(actualperiod - 1.0)
        alias_4b = actualperiod/(2.0*actualperiod - 1.0)

        aliases = np.ravel(np.array([
            actualperiod,
            twotimes_p,
            half_p,
            alias_1a,
            alias_1b,
            alias_2a,
            alias_2b,
            alias_3a,
            alias_3b,
            alias_4a,
            alias_4b]
        ))
        alias_labels = np.array(ALIAS_TYPES)

        # check type of alias
        closest_alias = np.isclose(recoveredperiod, aliases, rtol=tolerance)

        if np.any(closest_alias):

            closest_alias_type = alias_labels[closest_alias]
            return ','.join(closest_alias_type.tolist())

        else:

            return 'other'



def periodicvar_recovery(fakepfpkl,
                         simbasedir,
                         period_tolerance=1.0e-3):

    '''Recovers the periodic variable status/info for the simulated pf pickle.

    fakepfpkl is a single periodfinding-<objectid>.pkl[.gz] file produced in the
    <simbasedir>/periodfinding subdirectory after run_periodfinding above is
    done.

    - uses simbasedir and the lcfbasename stored in fakepfpkl to figure out
      where the LC for this object is

    - gets the actual_varparams, actual_varperiod, actual_vartype,
      actual_varamplitude elements from the LC

    - figures out if the current objectid is a periodic variable (using
      actual_vartype)

    - if it is a periodic variable, gets the canonical period assigned to it

    - checks if the period was recovered in any of the five best periods
      reported by any of the periodfinders, checks if the period recovered was a
      harmonic of the period

    - returns the objectid, actual period and vartype, recovered period, and
      recovery status

    '''

    if fakepfpkl.endswith('.gz'):
        infd = gzip.open(fakepfpkl,'rb')
    else:
        infd = open(fakepfpkl,'rb')

    fakepf = pickle.load(infd)
    infd.close()

    # get info from the fakepf dict
    objectid, lcfbasename = fakepf['objectid'], fakepf['lcfbasename']
    lcfpath = os.path.join(simbasedir,'lightcurves',lcfbasename)

    # if the LC doesn't exist, bail out
    if not os.path.exists(lcfpath):
        LOGERROR('light curve for %s does not exist at: %s' % (objectid,
                                                               lcfpath))
        return None

    # now, open the fakelc
    fakelc = lcproc.read_pklc(lcfpath)

    # get the actual_varparams, actual_varperiod, actual_varamplitude
    actual_varparams, actual_varperiod, actual_varamplitude, actual_vartype = (
        fakelc['actual_varparams'],
        fakelc['actual_varperiod'],
        fakelc['actual_varamplitude'],
        fakelc['actual_vartype']
    )

    # get the moments too so we can track LC noise, etc.
    actual_moments = fakelc['moments']

    # get the magcols for this LC
    magcols = fakelc['magcols']

    # get the recovered info from each of the available methods
    pfres = {
        'objectid':objectid,
        'simbasedir':simbasedir,
        'magcols':magcols,
        'fakelc':os.path.abspath(lcfpath),
        'fakepf':os.path.abspath(fakepfpkl),
        'actual_vartype':actual_vartype,
        'actual_varperiod':actual_varperiod,
        'actual_varamplitude':actual_varamplitude,
        'actual_varparams':actual_varparams,
        'actual_moments':actual_moments,
        'recovery_periods':[],
        'recovery_pfmethods':[],
        'recovery_magcols':[],
        'recovery_status':[],
        'recovery_pdiff':[],
    }

    # populate the pfres dict with the periods, pfmethods, and magcols
    for magcol in magcols:

        for pfm in lcproc.PFMETHODS:

            if pfm in fakepf[magcol]:

                # only get the unique recovered periods by using
                # period_tolerance
                for rp in fakepf[magcol][pfm]['nbestperiods']:

                    if ((not np.any(np.isclose(
                            rp,
                            np.array(pfres['recovery_periods']),
                            rtol=period_tolerance
                    ))) and np.isfinite(rp)):
                        pfres['recovery_periods'].append(rp)
                        pfres['recovery_pfmethods'].append(pfm)
                        pfres['recovery_magcols'].append(magcol)

    # convert the recovery_* lists to arrays
    pfres['recovery_periods'] = np.array(pfres['recovery_periods'])
    pfres['recovery_pfmethods'] = np.array(pfres['recovery_pfmethods'])
    pfres['recovery_magcols'] = np.array(pfres['recovery_magcols'])

    #
    # now figure out recovery status
    #

    # if this is an actual periodic variable, characterize the recovery
    if (actual_vartype and
        actual_vartype in PERIODIC_VARTYPES and
        np.isfinite(actual_varperiod)):

        if pfres['recovery_periods'].size > 0:

            for ri in range(pfres['recovery_periods'].size):

                pfres['recovery_pdiff'].append(pfres['recovery_periods'][ri] -
                                               np.asscalar(actual_varperiod))

                # get the alias types
                pfres['recovery_status'].append(
                    check_periodrec_alias(actual_varperiod,
                                          pfres['recovery_periods'][ri],
                                          tolerance=period_tolerance)
                )

            # turn the recovery_pdiff/status lists into arrays
            pfres['recovery_status'] = np.array(pfres['recovery_status'])
            pfres['recovery_pdiff'] = np.array(pfres['recovery_pdiff'])

            # find the best recovered period and its status
            rec_absdiff = np.abs(pfres['recovery_pdiff'])
            best_recp_ind = rec_absdiff == rec_absdiff.min()

            pfres['best_recovered_period'] = (
                pfres['recovery_periods'][best_recp_ind]
            )
            pfres['best_recovered_pfmethod'] = (
                pfres['recovery_pfmethods'][best_recp_ind]
            )
            pfres['best_recovered_magcol'] = (
                pfres['recovery_magcols'][best_recp_ind]
            )
            pfres['best_recovered_status'] = (
                pfres['recovery_status'][best_recp_ind]
            )
            pfres['best_recovered_pdiff'] = (
                pfres['recovery_pdiff'][best_recp_ind]
            )

        else:

            LOGWARNING(
                'no finite periods recovered from period-finding for %s' %
                fakepfpkl
            )

            pfres['recovery_status'] = np.array(['no_finite_periods_recovered'])
            pfres['recovery_pdiff'] = np.array([np.nan])
            pfres['best_recovered_period'] = np.array([np.nan])
            pfres['best_recovered_pfmethod'] = np.array([],dtype=np.unicode_)
            pfres['best_recovered_magcol'] = np.array([],dtype=np.unicode_)
            pfres['best_recovered_status'] = np.array([],dtype=np.unicode_)
            pfres['best_recovered_pdiff'] = np.array([np.nan])

    # if this is not actually a variable, get the recovered period,
    # etc. anyway. this way, we can see what we need to look out for and avoid
    # when getting these values for actual objects
    else:

        pfres['recovery_status'] = np.array(
            ['not_variable']*pfres['recovery_periods'].size
        )
        pfres['recovery_pdiff'] = np.zeros(pfres['recovery_periods'].size)

        pfres['best_recovered_period'] = np.array([np.nan])
        pfres['best_recovered_pfmethod'] = np.array([],dtype=np.unicode_)
        pfres['best_recovered_magcol'] = np.array([],dtype=np.unicode_)
        pfres['best_recovered_status'] = np.array(['not_variable'])
        pfres['best_recovered_pdiff'] = np.array([np.nan])

    return pfres



def periodrec_worker(task):
    '''
    This is a parallel worker for the function below.

    '''

    pfpkl, simbasedir, period_tolerance = task

    try:
        return periodicvar_recovery(pfpkl,
                                    simbasedir,
                                    period_tolerance=period_tolerance)

    except Exception as e:
        LOGEXCEPTION('periodic var recovery failed for %s' % repr(task))
        return None



def parallel_periodicvar_recovery(simbasedir,
                                  period_tolerance=1.0e-3,
                                  liststartind=None,
                                  listmaxobjects=None,
                                  nworkers=None):
    '''
    This is a parallel driver for periodicvar_recovery.


    '''

    # figure out the periodfinding pickles directory
    pfpkldir = os.path.join(simbasedir,'periodfinding')

    if not os.path.exists(pfpkldir):
        LOGERROR('no "periodfinding" subdirectory in %s, can\'t continue' %
                 simbasedir)
        return None

    # find all the periodfinding pickles
    pfpkl_list = glob.glob(os.path.join(pfpkldir,'*periodfinding*pkl*'))

    if len(pfpkl_list) > 0:

        if liststartind:
            pfpkl_list = pfpkl_list[liststartind:]

        if listmaxobjects:
            pfpkl_list = pfpkl_list[:listmaxobjects]

        tasks = [(x, simbasedir, period_tolerance) for x in pfpkl_list]

        pool = mp.Pool(nworkers)
        results = pool.map(periodrec_worker, tasks)
        pool.close()
        pool.join()

        resdict = {x['objectid']:x for x in results if x is not None}

        actual_periodicvars = np.array(
            [x['objectid'] for x in results
             if (x is not None and x['actual_vartype'] in PERIODIC_VARTYPES)],
            dtype=np.unicode_
        )

        recovered_periodicvars = np.array(
            [x['objectid'] for x in results
             if (x is not None and 'actual' in x['best_recovered_status'])],
            dtype=np.unicode_
        )
        alias_twice_periodicvars = np.array(
            [x['objectid'] for x in results
             if (x is not None and 'twice' in x['best_recovered_status'])],
            dtype=np.unicode_
        )
        alias_half_periodicvars = np.array(
            [x['objectid'] for x in results
             if (x is not None and 'half' in x['best_recovered_status'])],
            dtype=np.unicode_
        )

        all_objectids = [x['objectid'] for x in results]

        outdict = {'simbasedir':os.path.abspath(simbasedir),
                   'objectids':all_objectids,
                   'period_tolerance':period_tolerance,
                   'actual_periodicvars':actual_periodicvars,
                   'recovered_periodicvars':recovered_periodicvars,
                   'alias_twice_periodicvars':alias_twice_periodicvars,
                   'alias_half_periodicvars':alias_half_periodicvars,
                   'details':resdict}

        outfile = os.path.join(simbasedir,'periodicvar-recovery.pkl')
        with open(outfile, 'wb') as outfd:
            pickle.dump(outdict, outfd, pickle.HIGHEST_PROTOCOL)

        return outdict

    else:

        LOGERROR(
            'no periodfinding result pickles found in %s, can\'t continue' %
            pfpkldir
        )
        return None



def plot_periodicvar_recovery_results(
        precvar_results,
        magbins=np.arange(8.0,16.25,0.25),
        periodbins=np.arange(0.0,500.0,0.25),
        amplitudebins=np.arange(0.0,2.0,0.05),
        ndetbins=np.arange(0.0,60000.0,1000.0),
        minbinsize=1,
        aliases_count_as_recovered=None,
):
    '''This plots the results of periodic var recovery.

    precvar_results is either a dict returned by parallel_periodicvar_recovery
    or the pickle created by that function.

    aliases_count_as recovered is used to set which kinds of aliases this
    function considers as 'recovered' objects. Normally, we require that
    recovered objects have a recovery status of 'actual' to indicate the actual
    period was recovered. To change this default behavior,
    aliases_count_as_recovered can be set to a list of alias status strings that
    should be considered as 'recovered' objects as well. Choose from the
    following alias types:

    'twice'                     recovered_p = 2.0*actual_p
    'half'                      recovered_p = 0.5*actual_p
    'ratio_over_1plus'          recovered_p = actual_p/(1.0+actual_p)
    'ratio_over_1minus'         recovered_p = actual_p/(1.0-actual_p)
    'ratio_over_1plus_twice'    recovered_p = actual_p/(1.0+2.0*actual_p)
    'ratio_over_1minus_twice'   recovered_p = actual_p/(1.0-2.0*actual_p)
    'ratio_over_1plus_thrice'   recovered_p = actual_p/(1.0+3.0*actual_p)
    'ratio_over_1minus_thrice'  recovered_p = actual_p/(1.0-3.0*actual_p)
    'ratio_over_minus1'         recovered_p = actual_p/(actual_p - 1.0)
    'ratio_over_twice_minus1'   recovered_p = actual_p/(2.0*actual_p - 1.0)

    or set aliases_count_as_recovered='all' to include all of the above in the
    'recovered' periodic var list.

    This function makes plots for periodicvar recovered fraction as a function
    of:

      - magbin
      - periodbin
      - amplitude of variability
      - ndet

    with plot lines broken down by:

      - magcol
      - periodfinder
      - vartype
      - alias type

    Recovery rates are calculated using the recovered periodic vars and the
    actual periodic vars in the simulation.

    '''

    # get the result pickle/dict
    if isinstance(precvar_results, str) and os.path.exists(precvar_results):

        with open(precvar_results,'rb') as infd:
            precvar = pickle.load(infd)

    elif isinstance(precvar_results, dict):

        precvar = precvar_results

    else:
        LOGERROR('could not understand the input '
                 'periodic var recovery dict/pickle')
        return None

    # get the simbasedir and open the fakelc-info.pkl. we'll need the magbins
    # definition from here.
    simbasedir = precvar['simbasedir']

    lcinfof = os.path.join(simbasedir,'fakelcs-info.pkl')

    if not os.path.exists(lcinfof):
        LOGERROR('fakelcs-info.pkl does not exist in %s, can\'t continue' %
                 simbasedir)
        return None

    with open(lcinfof,'rb') as infd:
        lcinfo = pickle.load(infd)

    # get the magcols, vartypes, sdssr, isvariable flags
    magcols = lcinfo['magcols']
    objectid = lcinfo['objectid']
    ndet = lcinfo['ndet']
    sdssr = lcinfo['sdssr']

    # get the actual periodic vars
    actual_periodicvars = precvar['actual_periodicvars']

    # generate lists of objects binned by magbins and periodbins
    LOGINFO('getting sdssr and ndet for actual periodic vars...')

    # get the sdssr and ndet for all periodic vars
    periodicvar_sdssr = []
    periodicvar_ndet = []
    periodicvar_objectids = []

    for pobj in actual_periodicvars:

        pobjind = objectid == pobj
        periodicvar_objectids.append(pobj)
        periodicvar_sdssr.append(sdssr[pobjind])
        periodicvar_ndet.append(ndet[pobjind])


    periodicvar_sdssr = np.array(periodicvar_sdssr)
    periodicvar_objectids = np.array(periodicvar_objectids)
    periodicvar_ndet = np.array(periodicvar_ndet)

    LOGINFO('getting periods, vartypes, '
            'amplitudes, ndet for actual periodic vars...')

    # get the periods, vartypes, amplitudes for the actual periodic vars
    periodicvar_periods = [
        np.asscalar(precvar['details'][x]['actual_varperiod'])
        for x in periodicvar_objectids
    ]
    periodicvar_amplitudes = [
        np.asscalar(precvar['details'][x]['actual_varamplitude'])
        for x in periodicvar_objectids
    ]
    periodicvar_vartypes = [
        precvar['details'][x]['actual_vartype'] for x in periodicvar_objectids
    ]

    #
    # do the binning
    #

    # bin by mag
    LOGINFO('binning actual periodic vars by magnitude...')

    magbinned_sdssr = []
    magbinned_periodicvars = []
    magbininds = np.digitize(np.ravel(periodicvar_sdssr), magbins)

    for mbinind, magi in zip(np.unique(magbininds),
                             range(len(magbins)-1)):

        thisbin_periodicvars = periodicvar_objectids[magbininds == mbinind]

        if (thisbin_periodicvars.size > (minbinsize-1)):

            magbinned_sdssr.append((magbins[magi] + magbins[magi+1])/2.0)
            magbinned_periodicvars.append(thisbin_periodicvars)


    # bin by period
    LOGINFO('binning actual periodic vars by period...')

    periodbinned_periods = []
    periodbinned_periodicvars = []
    periodbininds = np.digitize(np.ravel(periodicvar_periods), periodbins)

    for pbinind, peri in zip(np.unique(periodbininds),
                             range(len(periodbins)-1)):

        thisbin_periodicvars = periodicvar_objectids[periodbininds == pbinind]

        if (thisbin_periodicvars.size > (minbinsize-1)):

            periodbinned_periods.append((periodbins[peri] +
                                         periodbins[peri+1])/2.0)
            periodbinned_periodicvars.append(thisbin_periodicvars)


    # bin by amplitude of variability
    LOGINFO('binning actual periodic vars by variability amplitude...')

    amplitudebinned_amplitudes = []
    amplitudebinned_periodicvars = []
    amplitudebininds = np.digitize(np.ravel(np.abs(periodicvar_amplitudes)),
                                   amplitudebins)

    for abinind, ampi in zip(np.unique(amplitudebininds),
                             range(len(amplitudebins)-1)):

        thisbin_periodicvars = periodicvar_objectids[
            amplitudebininds == abinind
        ]

        if (thisbin_periodicvars.size > (minbinsize-1)):

            amplitudebinned_amplitudes.append(
                (amplitudebins[ampi] +
                 amplitudebins[ampi+1])/2.0
            )
            amplitudebinned_periodicvars.append(thisbin_periodicvars)


    # bin by ndet
    LOGINFO('binning actual periodic vars by ndet...')

    ndetbinned_ndets = []
    ndetbinned_periodicvars = []
    ndetbininds = np.digitize(np.ravel(periodicvar_ndet), ndetbins)

    for nbinind, ndeti in zip(np.unique(ndetbininds),
                             range(len(ndetbins)-1)):

        thisbin_periodicvars = periodicvar_objectids[ndetbininds == nbinind]

        if (thisbin_periodicvars.size > (minbinsize-1)):

            ndetbinned_ndets.append(
                (ndetbins[ndeti] +
                 ndetbins[ndeti+1])/2.0
            )
            ndetbinned_periodicvars.append(thisbin_periodicvars)


    # now figure out what 'recovered' means using the provided
    # aliases_count_as_recovered kwarg
    recovered_status = ['actual']

    if aliases_count_as_recovered and aliases_count_as_recovered != 'all':

        for atype in aliases_count_as_recovered:

            if atype in ALIAS_TYPES:
                recovered_status.append(atype)
            else:
                LOGWARNING('unknown alias type: %s, skipping' % atype)

    elif aliases_count_as_recovered and aliases_count_as_recovered == 'all':
        for atype in ALIAS_TYPES[1:]:
            recovered_status.append(atype)

    # find all the matching objects for these recovered statuses
    recovered_periodicvars = np.array(
            [precvar['details'][x]['objectid'] for x in precvar['details']
             if (precvar['details'][x] is not None and
                 precvar['details'][x]['best_recovered_status']
                 in recovered_status)],
            dtype=np.unicode_
        )

    LOGINFO('found %s objects with requested period recovery status: %s' %
            (recovered_periodicvars.size, ', '.join(recovered_status)))


    # get the objects recovered per bin and overall recovery fractions per bin
    magbinned_recovered_objects = [
        np.intersect1d(x,recovered_periodicvars)
        for x in magbinned_periodicvars
    ]
    magbinned_recfrac = [float(x.size/y.size) for x,y
                         in zip(magbinned_recovered_objects,
                                magbinned_periodicvars)]

    periodbinned_recovered_objects = [
        np.intersect1d(x,recovered_periodicvars)
        for x in periodbinned_periodicvars
    ]
    periodbinned_recfrac = [float(x.size/y.size) for x,y
                            in zip(periodbinned_recovered_objects,
                                   periodbinned_periodicvars)]

    amplitudebinned_recovered_objects = [
        np.intersect1d(x,recovered_periodicvars)
        for x in amplitudebinned_periodicvars
    ]
    amplitudebinned_recfrac = [float(x.size/y.size) for x,y
                               in zip(amplitudebinned_recovered_objects,
                                      amplitudebinned_periodicvars)]

    ndetbinned_recovered_objects = [
        np.intersect1d(x,recovered_periodicvars)
        for x in ndetbinned_periodicvars
    ]
    ndetbinned_recfrac = [float(x.size/y.size) for x,y
                          in zip(ndetbinned_recovered_objects,
                                 ndetbinned_periodicvars)]


    # this is the initial output dict
    outdict = {
        'simbasedir':simbasedir,
        'precvar_results':precvar,
        'magcols':magcols,
        'objectids':objectid,
        'ndet':ndet,
        'sdssr':sdssr,
        'actual_periodicvars':actual_periodicvars,
        'recovered_periodicvars':recovered_periodicvars,
        'recovery_definition':recovered_status,
        # mag binned actual periodicvars
        # note that only bins with nobjects > minbinsize are included
        'magbins':magbins,
        'magbinned_mags':magbinned_sdssr,
        'magbinned_periodicvars':magbinned_periodicvars,
        'magbinned_recoveredvars':magbinned_recovered_objects,
        'magbinned_recfrac':magbinned_recfrac,
        # period binned actual periodicvars
        # note that only bins with nobjects > minbinsize are included
        'periodbins':periodbins,
        'periodbinned_periods':periodbinned_periods,
        'periodbinned_periodicvars':periodbinned_periodicvars,
        'periodbinned_recoveredvars':periodbinned_recovered_objects,
        'periodbinned_recfrac':periodbinned_recfrac,
        # amplitude binned actual periodicvars
        # note that only bins with nobjects > minbinsize are included
        'amplitudebins':amplitudebins,
        'amplitudebinned_amplitudes':amplitudebinned_amplitudes,
        'amplitudebinned_periodicvars':amplitudebinned_periodicvars,
        'amplitudebinned_recoveredvars':amplitudebinned_recovered_objects,
        'amplitudebinned_recfrac':amplitudebinned_recfrac,
        # ndet binned actual periodicvars
        # note that only bins with nobjects > minbinsize are included
        'ndetbins':ndetbins,
        'ndetbinned_ndets':ndetbinned_ndets,
        'ndetbinned_periodicvars':ndetbinned_periodicvars,
        'ndetbinned_recoveredvars':ndetbinned_recovered_objects,
        'ndetbinned_recfrac':ndetbinned_recfrac,
    }

    #
    # finally, we do stuff for the plots!
    #

    #
    # by magbin
    #



    # 1. recovery-rate by magbin
    # 1a. plot of overall recovery rate per magbin
    # 1b. plot of recovery rate per magbin per magcol
    # 1c. plot of recovery rate per magbin per periodfinder
    # 1d. plot of recovery rate per magbin per variable type
    # 1e. plot of recovery rate per magbin per alias type






    # at the end, write the outdict to a pickle and return it
    outfile = os.path.join(simbasedir, 'periodicvar-recovery-plotresults.pkl')
    with open(outfile,'wb') as outfd:
        pickle.dump(outdict, outfd, pickle.HIGHEST_PROTOCOL)

    return outdict
