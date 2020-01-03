#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# recovery - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
# License: MIT. See the LICENSE file for more details.

'''This is a companion module for fakelcs/generation.py. It runs LCs generated
using functions in that module through variable star detection and
classification to see how well they are recovered.

'''

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


#############
## IMPORTS ##
#############

import os
import os.path
import pickle
import gzip
import glob

import multiprocessing as mp

from math import sqrt as msqrt

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem


def _dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)


import numpy as np
import numpy.random as npr
# seed the numpy random generator
npr.seed(0xdecaff)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt


###################
## LOCAL IMPORTS ##
###################

from .. import lcproc
from ..lcproc import lcvfeatures, varthreshold, periodsearch


#######################
## LC FORMATS SET UP ##
#######################

def read_fakelc(fakelcfile):
    '''
    This just reads a pickled fake LC.

    Parameters
    ----------

    fakelcfile : str
        The fake LC file to read.

    Returns
    -------

    dict
        This returns an lcdict.

    '''

    try:
        with open(fakelcfile,'rb') as infd:
            lcdict = pickle.load(infd)
    except UnicodeDecodeError:
        with open(fakelcfile,'rb') as infd:
            lcdict = pickle.load(infd, encoding='latin1')

    return lcdict


#######################
## UTILITY FUNCTIONS ##
#######################

def get_varfeatures(simbasedir,
                    mindet=1000,
                    nworkers=None):
    '''This runs `lcproc.lcvfeatures.parallel_varfeatures` on fake LCs in
    `simbasedir`.

    Parameters
    ----------

    simbasedir : str
        The directory containing the fake LCs to process.

    mindet : int
        The minimum number of detections needed to accept an LC and process it.

    nworkers : int or None
        The number of parallel workers to use when extracting variability
        features from the input light curves.

    Returns
    -------

    str
        The path to the `varfeatures` pickle created after running the
        `lcproc.lcvfeatures.parallel_varfeatures` function.

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

    # get the column defs for the fakelcs
    timecols = siminfo['timecols']
    magcols = siminfo['magcols']
    errcols = siminfo['errcols']

    # register the fakelc pklc as a custom lcproc format
    # now we should be able to use all lcproc functions correctly
    fakelc_formatkey = 'fake-%s' % siminfo['lcformat']
    lcproc.register_lcformat(
        fakelc_formatkey,
        '*-fakelc.pkl',
        timecols,
        magcols,
        errcols,
        'astrobase.lcproc',
        '_read_pklc',
        magsarefluxes=siminfo['magsarefluxes']
    )

    # now we can use lcproc.parallel_varfeatures directly
    varinfo = lcvfeatures.parallel_varfeatures(lcfpaths,
                                               varfeaturedir,
                                               lcformat=fakelc_formatkey,
                                               mindet=mindet,
                                               nworkers=nworkers)

    with open(os.path.join(simbasedir,'fakelc-varfeatures.pkl'),'wb') as outfd:
        pickle.dump(varinfo, outfd, pickle.HIGHEST_PROTOCOL)

    return os.path.join(simbasedir,'fakelc-varfeatures.pkl')


def precision(ntp, nfp):
    '''
    This calculates precision.

    https://en.wikipedia.org/wiki/Precision_and_recall

    Parameters
    ----------

    ntp : int
        The number of true positives.

    nfp : int
        The number of false positives.

    Returns
    -------

    float
        The precision calculated using `ntp/(ntp + nfp)`.

    '''

    if (ntp+nfp) > 0:
        return ntp/(ntp+nfp)
    else:
        return np.nan


def recall(ntp, nfn):
    '''
    This calculates recall.

    https://en.wikipedia.org/wiki/Precision_and_recall

    Parameters
    ----------

    ntp : int
        The number of true positives.

    nfn : int
        The number of false negatives.

    Returns
    -------

    float
        The precision calculated using `ntp/(ntp + nfn)`.

    '''

    if (ntp+nfn) > 0:
        return ntp/(ntp+nfn)
    else:
        return np.nan


def matthews_correl_coeff(ntp, ntn, nfp, nfn):
    '''
    This calculates the Matthews correlation coefficent.

    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    Parameters
    ----------

    ntp : int
        The number of true positives.

    ntn : int
        The number of true negatives

    nfp : int
        The number of false positives.

    nfn : int
        The number of false negatives.

    Returns
    -------

    float
        The Matthews correlation coefficient.

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

    To generate a full recovery matrix over all magnitude bins, run this
    function for each magbin over the specified stetson_stdev_min and
    inveta_stdev_min grid.

    Parameters
    ----------

    simbasedir : str
        The input directory of fake LCs.

    magbinmedian : float
        The magbin to run the variable recovery for. This is an item from the
        dict from `simbasedir/fakelcs-info.pkl: `fakelcinfo['magrms'][magcol]`
        list for each magcol and designates which magbin to get the recovery
        stats for.

    stetson_stdev_min : float
        The minimum sigma above the trend in the Stetson J variability index
        distribution for this magbin to use to consider objects as variable.

    inveta_stdev_min : float
        The minimum sigma above the trend in the 1/eta variability index
        distribution for this magbin to use to consider objects as variable.

    iqr_stdev_min : float
        The minimum sigma above the trend in the IQR variability index
        distribution for this magbin to use to consider objects as variable.

    statsonly : bool
        If this is True, only the final stats will be returned. If False, the
        full arrays used to generate the stats will also be returned.

    Returns
    -------

    dict
        The returned dict contains statistics for this magbin and if requested,
        the full arrays used to calculate the statistics.

    '''

    # get the info from the simbasedir
    with open(os.path.join(simbasedir, 'fakelcs-info.pkl'),'rb') as infd:
        siminfo = pickle.load(infd)

    objectids = siminfo['objectid']
    varflags = siminfo['isvariable']
    sdssr = siminfo['sdssr']

    # get the column defs for the fakelcs
    timecols = siminfo['timecols']
    magcols = siminfo['magcols']
    errcols = siminfo['errcols']

    # register the fakelc pklc as a custom lcproc format
    # now we should be able to use all lcproc functions correctly
    fakelc_formatkey = 'fake-%s' % siminfo['lcformat']
    lcproc.register_lcformat(
        fakelc_formatkey,
        '*-fakelc.pkl',
        timecols,
        magcols,
        errcols,
        'astrobase.lcproc',
        '_read_pklc',
        magsarefluxes=siminfo['magsarefluxes']
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
    varthresh = varthreshold.variability_threshold(
        varfeaturedir,
        varthreshinfof,
        lcformat=fakelc_formatkey,
        min_stetj_stdev=stetson_stdev_min,
        min_inveta_stdev=inveta_stdev_min,
        min_iqr_stdev=iqr_stdev_min,
        verbose=False
    )

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
    for mbinind, _magi in zip(np.unique(magbininds),
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
        'magsarefluxes':siminfo['magsarefluxes'],
        'stetj_min_stdev':stetson_stdev_min,
        'inveta_min_stdev':inveta_stdev_min,
        'iqr_min_stdev':iqr_stdev_min,
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
    except Exception:
        LOGEXCEPTION('failed to get info for %s' % gridpoint)
        return None


def variable_index_gridsearch_magbin(simbasedir,
                                     stetson_stdev_range=(1.0,20.0),
                                     inveta_stdev_range=(1.0,20.0),
                                     iqr_stdev_range=(1.0,20.0),
                                     ngridpoints=32,
                                     ngridworkers=None):
    '''This runs a variable index grid search per magbin.

    For each magbin, this does a grid search using the stetson and inveta ranges
    provided and tries to optimize the Matthews Correlation Coefficient (best
    value is +1.0), indicating the best possible separation of variables
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


    Writes everything back to `simbasedir/fakevar-recovery.pkl`. Use the
    plotting function below to make plots for the results.

    Parameters
    ----------

    simbasedir : str
        The directory where the fake LCs are located.

    stetson_stdev_range : sequence of 2 floats
        The min and max values of the Stetson J variability index to generate a
        grid over these to test for the values of this index that produce the
        'best' recovery rate for the injected variable stars.

    inveta_stdev_range : sequence of 2 floats
        The min and max values of the 1/eta variability index to generate a
        grid over these to test for the values of this index that produce the
        'best' recovery rate for the injected variable stars.

    iqr_stdev_range : sequence of 2 floats
        The min and max values of the IQR variability index to generate a
        grid over these to test for the values of this index that produce the
        'best' recovery rate for the injected variable stars.

    ngridpoints : int
        The number of grid points for each variability index grid. Remember that
        this function will be searching in 3D and will require lots of time to
        run if ngridpoints is too large.

        For the default number of grid points and 25000 simulated light curves,
        this takes about 3 days to run on a 40 (effective) core machine with 2 x
        Xeon E5-2650v3 CPUs.

    ngridworkers : int or None
        The number of parallel grid search workers that will be launched.

    Returns
    -------

    dict
        The returned dict contains a list of recovery stats for each magbin and
        each grid point in the variability index grids that were used. This dict
        can be passed to the plotting function below to plot the results.

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
    grid_results = {'stetson_grid':stetson_grid,
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


def plot_varind_gridsearch_magbin_results(gridsearch_results):
    '''This plots the gridsearch results from `variable_index_gridsearch_magbin`.

    Parameters
    ----------

    gridsearch_results : dict
        This is the dict produced by `variable_index_gridsearch_magbin` above.

    Returns
    -------

    dict
        The returned dict contains filenames of the recovery rate plots made for
        each variability index. These include plots of the precision, recall,
        and Matthews Correlation Coefficient over each magbin and a heatmap of
        these values over the grid points of the variability index stdev values
        arrays used.

    '''

    # get the result pickle/dict
    if (isinstance(gridsearch_results, str) and
        os.path.exists(gridsearch_results)):

        with open(gridsearch_results,'rb') as infd:
            gridresults = pickle.load(infd)

    elif isinstance(gridsearch_results, dict):

        gridresults = gridsearch_results

    else:
        LOGERROR('could not understand the input '
                 'variable index grid-search result dict/pickle')
        return None

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

            stet_mcc = np.array(
                [x[magcol]['stet_mcc']
                 for x in recgrid[magbinind]]
            )[::(gridresults['inveta_grid'].size *
                 gridresults['stetson_grid'].size)]
            stet_precision = np.array(
                [x[magcol]['stet_precision']
                 for x in recgrid[magbinind]]
            )[::(gridresults['inveta_grid'].size *
                 gridresults['stetson_grid'].size)]
            stet_recall = np.array(
                [x[magcol]['stet_recall']
                 for x in recgrid[magbinind]]
            )[::(gridresults['inveta_grid'].size *
                 gridresults['stetson_grid'].size)]
            stet_missed_inveta_found = np.array(
                [x[magcol]['stet_missed_inveta_found']
                 for x in recgrid[magbinind]]
            )[::(gridresults['inveta_grid'].size *
                 gridresults['stetson_grid'].size)]
            stet_missed_iqr_found = np.array(
                [x[magcol]['stet_missed_iqr_found']
                 for x in recgrid[magbinind]]
            )[::(gridresults['inveta_grid'].size *
                 gridresults['stetson_grid'].size)]

            inveta_mcc = np.array(
                [x[magcol]['inveta_mcc']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    ::gridresults['inveta_grid'].size
                ]
            inveta_precision = np.array(
                [x[magcol]['inveta_precision']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    ::gridresults['inveta_grid'].size
                ]
            inveta_recall = np.array(
                [x[magcol]['inveta_recall']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    ::gridresults['inveta_grid'].size
                ]
            inveta_missed_stet_found = np.array(
                [x[magcol]['inveta_missed_stet_found']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    ::gridresults['inveta_grid'].size
                ]
            inveta_missed_iqr_found = np.array(
                [x[magcol]['inveta_missed_iqr_found']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    ::gridresults['inveta_grid'].size
                ]

            iqr_mcc = np.array(
                [x[magcol]['iqr_mcc']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    :gridresults['inveta_grid'].size
                ]
            iqr_precision = np.array(
                [x[magcol]['iqr_precision']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    :gridresults['inveta_grid'].size
                ]
            iqr_recall = np.array(
                [x[magcol]['iqr_recall']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    :gridresults['inveta_grid'].size
                ]
            iqr_missed_stet_found = np.array(
                [x[magcol]['iqr_missed_stet_found']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    :gridresults['inveta_grid'].size
                ]
            iqr_missed_inveta_found = np.array(
                [x[magcol]['iqr_missed_inveta_found']
                 for x in recgrid[magbinind]]
            )[:(gridresults['iqr_grid'].size *
                gridresults['stetson_grid'].size)][
                    :gridresults['inveta_grid'].size
                ]

            plt.figure(figsize=(6.4*5, 4.8*3))

            # FIRST ROW: stetson J plot

            plt.subplot(3,5,1)
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

            plt.subplot(3,5,2)
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

            plt.subplot(3,5,3)
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

            plt.subplot(3,5,4)
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

            plt.subplot(3,5,5)
            if np.any(np.isfinite(stet_missed_iqr_found)):
                plt.plot(gridresults['stetson_grid'],
                         stet_missed_iqr_found)
                plt.xlabel('stetson J stdev multiplier threshold')
                plt.ylabel('# objects stetson missed but IQR found')
                plt.title('stetson J missed, IQR found')
            else:
                plt.text(0.5,0.5,
                         'stet-missed/IQR-found values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.xticks([])
                plt.yticks([])

            # SECOND ROW: inveta plots

            plt.subplot(3,5,6)
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

            plt.subplot(3,5,7)
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

            plt.subplot(3,5,8)
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

            plt.subplot(3,5,9)
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

            plt.subplot(3,5,10)
            if np.any(np.isfinite(inveta_missed_iqr_found)):
                plt.plot(gridresults['inveta_grid'],
                         inveta_missed_iqr_found)
                plt.xlabel('inveta stdev multiplier threshold')
                plt.ylabel('# objects inveta missed but IQR found')
                plt.title('inveta missed, IQR found')
            else:
                plt.text(0.5,0.5,
                         'inveta-missed-iqr-found values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.xticks([])
                plt.yticks([])

            # THIRD ROW: inveta plots

            plt.subplot(3,5,11)
            if np.any(np.isfinite(iqr_mcc)):
                plt.plot(gridresults['iqr_grid'],
                         iqr_mcc)
                plt.xlabel('IQR stdev multiplier threshold')
                plt.ylabel('MCC')
                plt.title('MCC for IQR')
            else:
                plt.text(0.5,0.5,
                         'IQR MCC values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,5,12)
            if np.any(np.isfinite(iqr_precision)):
                plt.plot(gridresults['iqr_grid'],
                         iqr_precision)
                plt.xlabel('IQR stdev multiplier threshold')
                plt.ylabel('precision')
                plt.title('precision for IQR')
            else:
                plt.text(0.5,0.5,
                         'IQR precision values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,5,13)
            if np.any(np.isfinite(iqr_recall)):
                plt.plot(gridresults['iqr_grid'],
                         iqr_recall)
                plt.xlabel('IQR stdev multiplier threshold')
                plt.ylabel('recall')
                plt.title('recall for IQR')
            else:
                plt.text(0.5,0.5,
                         'IQR recall values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,5,14)
            if np.any(np.isfinite(iqr_missed_stet_found)):
                plt.plot(gridresults['iqr_grid'],
                         iqr_missed_stet_found)
                plt.xlabel('IQR stdev multiplier threshold')
                plt.ylabel('# objects IQR missed but stetson found')
                plt.title('IQR missed, stetson J found')
            else:
                plt.text(0.5,0.5,
                         'iqr-missed-stet-found values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.xticks([])
                plt.yticks([])

            plt.subplot(3,5,15)
            if np.any(np.isfinite(iqr_missed_inveta_found)):
                plt.plot(gridresults['iqr_grid'],
                         iqr_missed_inveta_found)
                plt.xlabel('IQR stdev multiplier threshold')
                plt.ylabel('# objects IQR missed but inveta found')
                plt.title('IQR missed, inveta found')
            else:
                plt.text(0.5,0.5,
                         'iqr-missed-inveta-found values are all nan '
                         'for this magbin',
                         transform=plt.gca().transAxes,
                         horizontalalignment='center',
                         verticalalignment='center')
                plt.xticks([])
                plt.yticks([])

            plt.subplots_adjust(hspace=0.25,wspace=0.25)

            plt.suptitle('magcol: %s, magbin: %.3f' % (magcol, magbinmedian))

            plotdir = os.path.join(gridresults['simbasedir'],
                                   'varindex-gridsearch-plots')
            if not os.path.exists(plotdir):
                os.mkdir(plotdir)

            gridplotf = os.path.join(
                plotdir,
                '%s-magbin-%.3f-var-recoverygrid-permagbin.png' %
                (magcol, magbinmedian)
            )

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

            iqr_mcc_maxind = np.where(iqr_mcc == np.max(iqr_mcc))
            iqr_precision_maxind = np.where(
                iqr_precision == np.max(iqr_precision)
            )
            iqr_recall_maxind = (
                np.where(iqr_recall == np.max(iqr_recall))
            )

            best_iqr_mcc = iqr_mcc[iqr_mcc_maxind]
            best_iqr_precision = iqr_mcc[iqr_precision_maxind]
            best_iqr_recall = iqr_mcc[iqr_recall_maxind]

            iqr_with_best_mcc = gridresults['iqr_grid'][iqr_mcc_maxind]
            iqr_with_best_precision = gridresults['iqr_grid'][
                iqr_precision_maxind
            ]
            iqr_with_best_recall = gridresults['iqr_grid'][
                iqr_recall_maxind
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
                # iqr
                'iqr_grid':gridresults['iqr_grid'],
                'iqr_mcc':iqr_mcc,
                'iqr_precision':iqr_precision,
                'iqr_recall':iqr_recall,
                'iqr_missed_stet_found':iqr_missed_stet_found,
                'best_iqr_mcc':best_iqr_mcc,
                'iqr_with_best_mcc':iqr_with_best_mcc,
                'best_iqr_precision':best_iqr_precision,
                'iqr_with_best_precision':iqr_with_best_precision,
                'best_iqr_recall':best_iqr_recall,
                'iqr_with_best_recall':iqr_with_best_recall,
                # plot info
                'recoveryplot':gridplotf
            }

            # recommend inveta, stetson index, and iqr for this magbin

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

            # if there are multiple best iqrs, choose the smallest one
            if iqr_with_best_mcc.size > 1:
                plotres[magcol]['best_iqr'].append(iqr_with_best_mcc[0])
            elif iqr_with_best_mcc.size > 0:
                plotres[magcol]['best_iqr'].append(iqr_with_best_mcc[0])
            else:
                plotres[magcol]['best_iqr'].append(np.nan)

    # write the plotresults to a pickle
    plotrespicklef = os.path.join(simbasedir,
                                  'varindex-gridsearch-magbin-results.pkl')
    with open(plotrespicklef, 'wb') as outfd:
        pickle.dump(plotres, outfd, pickle.HIGHEST_PROTOCOL)

    # recommend the values of stetson J and inveta to use
    for magcol in gridresults['magcols']:

        LOGINFO('best stdev multipliers for each %s magbin:' % magcol)
        LOGINFO('magbin    inveta    stetson J    IQR')

        for magbin, inveta, stet, iqr in zip(
                plotres[magcol]['magbinmedians'],
                plotres[magcol]['best_inveta'],
                plotres[magcol]['best_stetsonj'],
                plotres[magcol]['best_iqr']):
            LOGINFO('%.3f    %.3f    %.3f    %.3f' % (magbin,
                                                      inveta,
                                                      stet,
                                                      iqr))
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
                      pfmethods=('gls','pdm','bls'),
                      pfkwargs=({},{},{'startp':1.0,'maxtransitduration':0.3}),
                      getblssnr=False,
                      sigclip=5.0,
                      nperiodworkers=10,
                      ncontrolworkers=4,
                      liststartindex=None,
                      listmaxobjects=None):
    '''This runs periodfinding using several period-finders on a collection of
    fake LCs.

    As a rough benchmark, 25000 fake LCs with 10000--50000 points per LC take
    about 26 days in total to run on an invocation of this function using
    GLS+PDM+BLS and 10 periodworkers and 4 controlworkers (so all 40 'cores') on
    a 2 x Xeon E5-2660v3 machine.

    Parameters
    ----------

    pfmethods : sequence of str
        This is used to specify which periodfinders to run. These must be in the
        `lcproc.periodsearch.PFMETHODS` dict.

    pfkwargs : sequence of dict
        This is used to provide optional kwargs to the period-finders.

    getblssnr : bool
        If this is True, will run BLS SNR calculations for each object and
        magcol. This takes a while to run, so it's disabled (False) by default.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    nperiodworkers : int
        This is the number of parallel period-finding worker processes to use.

    ncontrolworkers : int
        This is the number of parallel period-finding control workers to
        use. Each control worker will launch `nperiodworkers` worker processes.

    liststartindex : int
        The starting index of processing. This refers to the filename list
        generated by running `glob.glob` on the fake LCs in `simbasedir`.

    maxobjects : int
        The maximum number of objects to process in this run. Use this with
        `liststartindex` to effectively distribute working on a large list of
        input light curves over several sessions or machines.

    Returns
    -------

    str
        The path to the output summary pickle produced by
        `lcproc.periodsearch.parallel_pf`

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
    fakelc_formatkey = 'fake-%s' % siminfo['lcformat']
    lcproc.register_lcformat(
        fakelc_formatkey,
        '*-fakelc.pkl',
        timecols,
        magcols,
        errcols,
        'astrobase.lcproc',
        '_read_pklc',
        magsarefluxes=siminfo['magsarefluxes']
    )

    if liststartindex:
        lcfpaths = lcfpaths[liststartindex:]

    if listmaxobjects:
        lcfpaths = lcfpaths[:listmaxobjects]

    pfinfo = periodsearch.parallel_pf(lcfpaths,
                                      pfdir,
                                      lcformat=fakelc_formatkey,
                                      pfmethods=pfmethods,
                                      pfkwargs=pfkwargs,
                                      getblssnr=getblssnr,
                                      sigclip=sigclip,
                                      nperiodworkers=nperiodworkers,
                                      ncontrolworkers=ncontrolworkers)

    with open(os.path.join(simbasedir,
                           'fakelc-periodsearch.pkl'),'wb') as outfd:
        pickle.dump(pfinfo, outfd, pickle.HIGHEST_PROTOCOL)

    return os.path.join(simbasedir,'fakelc-periodsearch.pkl')


def check_periodrec_alias(actualperiod,
                          recoveredperiod,
                          tolerance=1.0e-3):
    '''This determines what kind of aliasing (if any) exists between
    `recoveredperiod` and `actualperiod`.

    Parameters
    ----------

    actualperiod : float
        The actual period of the object.

    recoveredperiod : float
        The recovered period of the object.

    tolerance : float
        The absolute difference required between the input periods to mark the
        recovered period as close to the actual period.

    Returns
    -------

    str
        The type of alias determined for the input combination of periods. This
        will be CSV string with values taken from the following list, based on
        the types of alias found::

            ['actual',
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
        closest_alias = np.isclose(recoveredperiod, aliases, atol=tolerance)

        if np.any(closest_alias):

            closest_alias_type = alias_labels[closest_alias]
            return ','.join(closest_alias_type.tolist())

        else:

            return 'other'


def periodicvar_recovery(fakepfpkl,
                         simbasedir,
                         period_tolerance=1.0e-3):

    '''Recovers the periodic variable status/info for the simulated PF result.

    - Uses simbasedir and the lcfbasename stored in fakepfpkl to figure out
      where the LC for this object is.
    - Gets the actual_varparams, actual_varperiod, actual_vartype,
      actual_varamplitude elements from the LC.
    - Figures out if the current objectid is a periodic variable (using
      actual_vartype).
    - If it is a periodic variable, gets the canonical period assigned to it.
    - Checks if the period was recovered in any of the five best periods
      reported by any of the period-finders, checks if the period recovered was
      a harmonic of the period.
    - Returns the objectid, actual period and vartype, recovered period, and
      recovery status.


    Parameters
    ----------

    fakepfpkl : str
        This is a periodfinding-<objectid>.pkl[.gz] file produced in the
        `simbasedir/periodfinding` subdirectory after `run_periodfinding` above
        is done.

    simbasedir : str
        The base directory where all of the fake LCs and period-finding results
        are.

    period_tolerance : float
        The maximum difference that this function will consider between an
        actual period (or its aliases) and a recovered period to consider it as
        as a 'recovered' period.

    Returns
    -------

    dict
        Returns a dict of period-recovery results.

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
    fakelc = lcproc._read_pklc(lcfpath)

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
        'recovery_lspvals':[],
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
                for rpi, rp in enumerate(
                        fakepf[magcol][pfm]['nbestperiods']
                ):

                    if ((not np.any(np.isclose(
                            rp,
                            np.array(pfres['recovery_periods']),
                            rtol=period_tolerance
                    ))) and np.isfinite(rp)):

                        # populate the recovery periods, pfmethods, and magcols
                        pfres['recovery_periods'].append(rp)
                        pfres['recovery_pfmethods'].append(pfm)
                        pfres['recovery_magcols'].append(magcol)

                        # normalize the periodogram peak value to between
                        # 0 and 1 so we can put in the results of multiple
                        # periodfinders on one scale
                        if pfm == 'pdm':

                            this_lspval = (
                                np.max(fakepf[magcol][pfm]['lspvals']) -
                                fakepf[magcol][pfm]['nbestlspvals'][rpi]
                            )

                        else:

                            this_lspval = (
                                fakepf[magcol][pfm]['nbestlspvals'][rpi] /
                                np.max(fakepf[magcol][pfm]['lspvals'])
                            )

                        # add the normalized lspval to the outdict for
                        # this object as well. later, we'll use this to
                        # construct a periodogram for objects that were actually
                        # not variables
                        pfres['recovery_lspvals'].append(this_lspval)

    # convert the recovery_* lists to arrays
    pfres['recovery_periods'] = np.array(pfres['recovery_periods'])
    pfres['recovery_lspvals'] = np.array(pfres['recovery_lspvals'])
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
    '''This is a parallel worker for running period-recovery.

    Parameters
    ----------

    task : tuple
        This is used to pass args to the `periodicvar_recovery` function::

            task[0] = period-finding result pickle to work on
            task[1] = simbasedir
            task[2] = period_tolerance

    Returns
    -------

    dict
        This is the dict produced by the `periodicvar_recovery` function for the
        input period-finding result pickle.

    '''

    pfpkl, simbasedir, period_tolerance = task

    try:
        return periodicvar_recovery(pfpkl,
                                    simbasedir,
                                    period_tolerance=period_tolerance)

    except Exception:
        LOGEXCEPTION('periodic var recovery failed for %s' % repr(task))
        return None


def parallel_periodicvar_recovery(simbasedir,
                                  period_tolerance=1.0e-3,
                                  liststartind=None,
                                  listmaxobjects=None,
                                  nworkers=None):
    '''This is a parallel driver for `periodicvar_recovery`.

    Parameters
    ----------

    simbasedir : str
        The base directory where all of the fake LCs and period-finding results
        are.

    period_tolerance : float
        The maximum difference that this function will consider between an
        actual period (or its aliases) and a recovered period to consider it as
        as a 'recovered' period.

    liststartindex : int
        The starting index of processing. This refers to the filename list
        generated by running `glob.glob` on the period-finding result pickles in
        `simbasedir/periodfinding`.

    listmaxobjects : int
        The maximum number of objects to process in this run. Use this with
        `liststartindex` to effectively distribute working on a large list of
        input period-finding result pickles over several sessions or machines.

    nperiodworkers : int
        This is the number of parallel period-finding worker processes to use.

    Returns
    -------

    str
        Returns the filename of the pickle produced containing all of the period
        recovery results.

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


PERIODREC_DEFAULT_MAGBINS = np.arange(8.0,16.25,0.25)
PERIODREC_DEFAULT_PERIODBINS = np.arange(0.0,500.0,0.5)
PERIODREC_DEFAULT_AMPBINS = np.arange(0.0,2.0,0.05)
PERIODREC_DEFAULT_NDETBINS = np.arange(0.0,60000.0,1000.0)


def plot_periodicvar_recovery_results(
        precvar_results,
        aliases_count_as_recovered=None,
        magbins=None,
        periodbins=None,
        amplitudebins=None,
        ndetbins=None,
        minbinsize=1,
        plotfile_ext='png',
):
    '''This plots the results of periodic var recovery.

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
    - recovery status

    The kwargs `magbins`, `periodbins`, `amplitudebins`, and `ndetbins` can be
    used to set the bin lists as needed. The kwarg `minbinsize` controls how
    many elements per bin are required to accept a bin in processing its
    recovery characteristics for mags, periods, amplitudes, and ndets.

    Parameters
    ----------

    precvar_results : dict or str
        This is either a dict returned by parallel_periodicvar_recovery or the
        pickle created by that function.

    aliases_count_as_recovered : list of str or 'all'
        This is used to set which kinds of aliases this function considers as
        'recovered' objects. Normally, we require that recovered objects have a
        recovery status of 'actual' to indicate the actual period was
        recovered. To change this default behavior, aliases_count_as_recovered
        can be set to a list of alias status strings that should be considered
        as 'recovered' objects as well. Choose from the following alias types::

          'twice'                    recovered_p = 2.0*actual_p
          'half'                     recovered_p = 0.5*actual_p
          'ratio_over_1plus'         recovered_p = actual_p/(1.0+actual_p)
          'ratio_over_1minus'        recovered_p = actual_p/(1.0-actual_p)
          'ratio_over_1plus_twice'   recovered_p = actual_p/(1.0+2.0*actual_p)
          'ratio_over_1minus_twice'  recovered_p = actual_p/(1.0-2.0*actual_p)
          'ratio_over_1plus_thrice'  recovered_p = actual_p/(1.0+3.0*actual_p)
          'ratio_over_1minus_thrice' recovered_p = actual_p/(1.0-3.0*actual_p)
          'ratio_over_minus1'        recovered_p = actual_p/(actual_p - 1.0)
          'ratio_over_twice_minus1'  recovered_p = actual_p/(2.0*actual_p - 1.0)

        or set `aliases_count_as_recovered='all'` to include all of the above in
        the 'recovered' periodic var list.

    magbins : np.array
        The magnitude bins to plot the recovery rate results over. If None, the
        default mag bins will be used: `np.arange(8.0,16.25,0.25)`.

    periodbins : np.array
        The period bins to plot the recovery rate results over. If None, the
        default period bins will be used: `np.arange(0.0,500.0,0.5)`.

    amplitudebins : np.array
        The variability amplitude bins to plot the recovery rate results
        over. If None, the default amplitude bins will be used:
        `np.arange(0.0,2.0,0.05)`.

    ndetbins : np.array
        The ndet bins to plot the recovery rate results over. If None, the
        default ndet bins will be used: `np.arange(0.0,60000.0,1000.0)`.

    minbinsize : int
        The minimum number of objects per bin required to plot a bin and its
        recovery fraction on the plot.

    plotfile_ext : {'png','pdf'}
        Sets the plot output files' extension.

    Returns
    -------

    dict
        A dict containing recovery fraction statistics and the paths to each of
        the plots made.

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

    #
    # do the binning
    #

    # bin by mag
    LOGINFO('binning actual periodic vars by magnitude...')

    magbinned_sdssr = []
    magbinned_periodicvars = []

    if not magbins:
        magbins = PERIODREC_DEFAULT_MAGBINS
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

    if not periodbins:
        periodbins = PERIODREC_DEFAULT_PERIODBINS
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

    if not amplitudebins:
        amplitudebins = PERIODREC_DEFAULT_AMPBINS
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

    if not ndetbins:
        ndetbins = PERIODREC_DEFAULT_NDETBINS
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

    if isinstance(aliases_count_as_recovered, list):

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

    LOGINFO('recovered %s/%s periodic variables (frac: %.3f) with '
            'period recovery status: %s' %
            (recovered_periodicvars.size,
             actual_periodicvars.size,
             float(recovered_periodicvars.size/actual_periodicvars.size),
             ', '.join(recovered_status)))

    # get the objects recovered per bin and overall recovery fractions per bin
    magbinned_recovered_objects = [
        np.intersect1d(x,recovered_periodicvars)
        for x in magbinned_periodicvars
    ]
    magbinned_recfrac = np.array([float(x.size/y.size) for x,y
                                  in zip(magbinned_recovered_objects,
                                         magbinned_periodicvars)])

    periodbinned_recovered_objects = [
        np.intersect1d(x,recovered_periodicvars)
        for x in periodbinned_periodicvars
    ]
    periodbinned_recfrac = np.array([float(x.size/y.size) for x,y
                                     in zip(periodbinned_recovered_objects,
                                            periodbinned_periodicvars)])

    amplitudebinned_recovered_objects = [
        np.intersect1d(x,recovered_periodicvars)
        for x in amplitudebinned_periodicvars
    ]
    amplitudebinned_recfrac = np.array(
        [float(x.size/y.size) for x,y
         in zip(amplitudebinned_recovered_objects,
                amplitudebinned_periodicvars)]
    )

    ndetbinned_recovered_objects = [
        np.intersect1d(x,recovered_periodicvars)
        for x in ndetbinned_periodicvars
    ]
    ndetbinned_recfrac = np.array([float(x.size/y.size) for x,y
                                   in zip(ndetbinned_recovered_objects,
                                          ndetbinned_periodicvars)])

    # convert the bin medians to arrays
    magbinned_sdssr = np.array(magbinned_sdssr)
    periodbinned_periods = np.array(periodbinned_periods)
    amplitudebinned_amplitudes = np.array(amplitudebinned_amplitudes)
    ndetbinned_ndets = np.array(ndetbinned_ndets)

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

    # figure out which pfmethods were used
    all_pfmethods = np.unique(
        np.concatenate(
            [np.unique(precvar['details'][x]['recovery_pfmethods'])
             for x in precvar['details']]
        )
    )

    # figure out all vartypes
    all_vartypes = np.unique(
        [(precvar['details'][x]['actual_vartype'])
         for x in precvar['details'] if
         (precvar['details'][x]['actual_vartype'] is not None)]
    )

    # figure out all alias types
    all_aliastypes = recovered_status

    # add these to the outdict
    outdict['aliastypes'] = all_aliastypes
    outdict['pfmethods'] = all_pfmethods
    outdict['vartypes'] = all_vartypes

    # these are recfracs per-magcol, -vartype, -periodfinder, -aliastype
    # binned appropriately by mags, periods, amplitudes, and ndet
    # all of these have the shape as the magcols, aliastypes, pfmethods, and
    # vartypes lists above.

    magbinned_per_magcol_recfracs = []
    magbinned_per_vartype_recfracs = []
    magbinned_per_pfmethod_recfracs = []
    magbinned_per_aliastype_recfracs = []

    periodbinned_per_magcol_recfracs = []
    periodbinned_per_vartype_recfracs = []
    periodbinned_per_pfmethod_recfracs = []
    periodbinned_per_aliastype_recfracs = []

    amplitudebinned_per_magcol_recfracs = []
    amplitudebinned_per_vartype_recfracs = []
    amplitudebinned_per_pfmethod_recfracs = []
    amplitudebinned_per_aliastype_recfracs = []

    ndetbinned_per_magcol_recfracs = []
    ndetbinned_per_vartype_recfracs = []
    ndetbinned_per_pfmethod_recfracs = []
    ndetbinned_per_aliastype_recfracs = []

    #
    # finally, we do stuff for the plots!
    #
    recplotdir = os.path.join(simbasedir, 'periodic-variable-recovery-plots')
    if not os.path.exists(recplotdir):
        os.mkdir(recplotdir)

    # 1. recovery-rate by magbin

    # 1a. plot of overall recovery rate per magbin
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    plt.plot(magbinned_sdssr, magbinned_recfrac,marker='.',ms=0.0)
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('overall recovery fraction by periodic var magnitudes')
    plt.ylim((0,1))
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-magnitudes-overall.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 1b. plot of recovery rate per magbin per magcol
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    for magcol in magcols:

        thismagcol_recfracs = []

        for magbin_pv, magbin_rv in zip(magbinned_periodicvars,
                                        magbinned_recovered_objects):

            thisbin_thismagcol_recvars = [
                x for x in magbin_rv
                if (precvar['details'][x]['best_recovered_magcol'] == magcol)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thismagcol_recvars).size /
                magbin_pv.size
            )
            thismagcol_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(magbinned_sdssr,
                 np.array(thismagcol_recfracs),
                 marker='.',
                 label='magcol: %s' % magcol,
                 ms=0.0)

        # add this to the outdict array
        magbinned_per_magcol_recfracs.append(np.array(thismagcol_recfracs))

    # finish up the plot
    plt.plot(magbinned_sdssr, magbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per magcol recovery fraction by periodic var magnitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-magnitudes-magcols.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 1c. plot of recovery rate per magbin per periodfinder
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out which pfmethods were used
    all_pfmethods = np.unique(
        np.concatenate(
            [np.unique(precvar['details'][x]['recovery_pfmethods'])
             for x in precvar['details']]
        )
    )

    for pfm in all_pfmethods:

        thispf_recfracs = []

        for magbin_pv, magbin_rv in zip(magbinned_periodicvars,
                                        magbinned_recovered_objects):

            thisbin_thispf_recvars = [
                x for x in magbin_rv
                if (precvar['details'][x]['best_recovered_pfmethod'] == pfm)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thispf_recvars).size /
                magbin_pv.size
            )
            thispf_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(magbinned_sdssr,
                 np.array(thispf_recfracs),
                 marker='.',
                 label='%s' % pfm.upper(),
                 ms=0.0)

        # add this to the outdict array
        magbinned_per_pfmethod_recfracs.append(np.array(thispf_recfracs))

    # finish up the plot
    plt.plot(magbinned_sdssr, magbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per period-finder recovery fraction by periodic var magnitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-magnitudes-pfmethod.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 1d. plot of recovery rate per magbin per variable type
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out all vartypes
    all_vartypes = np.unique(
        [(precvar['details'][x]['actual_vartype'])
         for x in precvar['details'] if
         (precvar['details'][x]['actual_vartype'] is not None)]
    )

    for vt in all_vartypes:

        thisvt_recfracs = []

        for magbin_pv, magbin_rv in zip(magbinned_periodicvars,
                                        magbinned_recovered_objects):

            thisbin_thisvt_recvars = [
                x for x in magbin_rv
                if (precvar['details'][x]['actual_vartype'] == vt)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thisvt_recvars).size /
                magbin_pv.size
            )
            thisvt_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(magbinned_sdssr,
                 np.array(thisvt_recfracs),
                 marker='.',
                 label='%s' % vt,
                 ms=0.0)

        # add this to the outdict array
        magbinned_per_vartype_recfracs.append(np.array(thisvt_recfracs))

    # finish up the plot
    plt.plot(magbinned_sdssr, magbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per vartype recovery fraction by periodic var magnitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-magnitudes-vartype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 1e. plot of recovery rate per magbin per alias type
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out all alias types
    all_aliastypes = recovered_status

    for at in all_aliastypes:

        thisat_recfracs = []

        for magbin_pv, magbin_rv in zip(magbinned_periodicvars,
                                        magbinned_recovered_objects):

            thisbin_thisat_recvars = [
                x for x in magbin_rv
                if (precvar['details'][x]['best_recovered_status'][0] == at)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thisat_recvars).size /
                magbin_pv.size
            )
            thisat_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(magbinned_sdssr,
                 np.array(thisat_recfracs),
                 marker='.',
                 label='%s' % at,
                 ms=0.0)

        # add this to the outdict array
        magbinned_per_aliastype_recfracs.append(np.array(thisat_recfracs))

    # finish up the plot
    plt.plot(magbinned_sdssr, magbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per alias-type recovery fraction by periodic var magnitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-magnitudes-aliastype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 2. recovery-rate by periodbin

    # 2a. plot of overall recovery rate per periodbin
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    plt.plot(periodbinned_periods, periodbinned_recfrac,
             marker='.',ms=0.0)
    plt.xlabel('periodic variable period [days]')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('overall recovery fraction by periodic var periods')
    plt.ylim((0,1))
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-periods-overall.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 2b. plot of recovery rate per periodbin per magcol
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    for magcol in magcols:

        thismagcol_recfracs = []

        for periodbin_pv, periodbin_rv in zip(periodbinned_periodicvars,
                                              periodbinned_recovered_objects):

            thisbin_thismagcol_recvars = [
                x for x in periodbin_rv
                if (precvar['details'][x]['best_recovered_magcol'] == magcol)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thismagcol_recvars).size /
                periodbin_pv.size
            )
            thismagcol_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(periodbinned_periods,
                 np.array(thismagcol_recfracs),
                 marker='.',
                 label='magcol: %s' % magcol,
                 ms=0.0)

        # add this to the outdict array
        periodbinned_per_magcol_recfracs.append(np.array(thismagcol_recfracs))

    # finish up the plot
    plt.plot(periodbinned_periods, periodbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per magcol recovery fraction by periodic var periods')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-periods-magcols.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 2c. plot of recovery rate per periodbin per periodfinder
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out which pfmethods were used
    all_pfmethods = np.unique(
        np.concatenate(
            [np.unique(precvar['details'][x]['recovery_pfmethods'])
             for x in precvar['details']]
        )
    )

    for pfm in all_pfmethods:

        thispf_recfracs = []

        for periodbin_pv, periodbin_rv in zip(periodbinned_periodicvars,
                                              periodbinned_recovered_objects):

            thisbin_thispf_recvars = [
                x for x in periodbin_rv
                if (precvar['details'][x]['best_recovered_pfmethod'] == pfm)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thispf_recvars).size /
                periodbin_pv.size
            )
            thispf_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(periodbinned_periods,
                 np.array(thispf_recfracs),
                 marker='.',
                 label='%s' % pfm.upper(),
                 ms=0.0)

        # add this to the outdict array
        periodbinned_per_pfmethod_recfracs.append(np.array(thispf_recfracs))

    # finish up the plot
    plt.plot(periodbinned_periods, periodbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per period-finder recovery fraction by periodic var periods')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-periods-pfmethod.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 2d. plot of recovery rate per periodbin per variable type
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out all vartypes
    all_vartypes = np.unique(
        [(precvar['details'][x]['actual_vartype'])
         for x in precvar['details'] if
         (precvar['details'][x]['actual_vartype'] is not None)]
    )

    for vt in all_vartypes:

        thisvt_recfracs = []

        for periodbin_pv, periodbin_rv in zip(periodbinned_periodicvars,
                                              periodbinned_recovered_objects):

            thisbin_thisvt_recvars = [
                x for x in periodbin_rv
                if (precvar['details'][x]['actual_vartype'] == vt)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thisvt_recvars).size /
                periodbin_pv.size
            )
            thisvt_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(periodbinned_periods,
                 np.array(thisvt_recfracs),
                 marker='.',
                 label='%s' % vt,
                 ms=0.0)

        # add this to the outdict array
        periodbinned_per_vartype_recfracs.append(np.array(thisvt_recfracs))

    # finish up the plot
    plt.plot(periodbinned_periods, periodbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per vartype recovery fraction by periodic var magnitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-periods-vartype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 2e. plot of recovery rate per periodbin per alias type
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out all vartypes
    all_aliastypes = recovered_status

    for at in all_aliastypes:

        thisat_recfracs = []

        for periodbin_pv, periodbin_rv in zip(
                periodbinned_periodicvars,
                periodbinned_recovered_objects
        ):

            thisbin_thisat_recvars = [
                x for x in periodbin_rv
                if (precvar['details'][x]['best_recovered_status'][0] == at)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thisat_recvars).size /
                periodbin_pv.size
            )
            thisat_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(periodbinned_periods,
                 np.array(thisat_recfracs),
                 marker='.',
                 label='%s' % at,
                 ms=0.0)

        # add this to the outdict array
        periodbinned_per_aliastype_recfracs.append(np.array(thisat_recfracs))

    # finish up the plot
    plt.plot(periodbinned_periods, periodbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per alias-type recovery fraction by periodic var magnitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-periods-aliastype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 3. recovery-rate by amplitude bin

    # 3a. plot of overall recovery rate per amplitude bin
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    plt.plot(amplitudebinned_amplitudes, amplitudebinned_recfrac,
             marker='.',ms=0.0)
    plt.xlabel('periodic variable amplitude [mag]')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('overall recovery fraction by periodic var amplitudes')
    plt.ylim((0,1))
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-amplitudes-overall.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 3b. plot of recovery rate per amplitude bin per magcol
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    for magcol in magcols:

        thismagcol_recfracs = []

        for amplitudebin_pv, amplitudebin_rv in zip(
                amplitudebinned_periodicvars,
                amplitudebinned_recovered_objects
        ):

            thisbin_thismagcol_recvars = [
                x for x in amplitudebin_rv
                if (precvar['details'][x]['best_recovered_magcol'] == magcol)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thismagcol_recvars).size /
                amplitudebin_pv.size
            )
            thismagcol_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(amplitudebinned_amplitudes,
                 np.array(thismagcol_recfracs),
                 marker='.',
                 label='magcol: %s' % magcol,
                 ms=0.0)

        # add this to the outdict array
        amplitudebinned_per_magcol_recfracs.append(
            np.array(thismagcol_recfracs)
        )

    # finish up the plot
    plt.plot(amplitudebinned_amplitudes, amplitudebinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per magcol recovery fraction by periodic var amplitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-amplitudes-magcols.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 3c. plot of recovery rate per amplitude bin per periodfinder
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out which pfmethods were used
    all_pfmethods = np.unique(
        np.concatenate(
            [np.unique(precvar['details'][x]['recovery_pfmethods'])
             for x in precvar['details']]
        )
    )

    for pfm in all_pfmethods:

        thispf_recfracs = []

        for amplitudebin_pv, amplitudebin_rv in zip(
                amplitudebinned_periodicvars,
                amplitudebinned_recovered_objects
        ):

            thisbin_thispf_recvars = [
                x for x in amplitudebin_rv
                if (precvar['details'][x]['best_recovered_pfmethod'] == pfm)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thispf_recvars).size /
                amplitudebin_pv.size
            )
            thispf_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(amplitudebinned_amplitudes,
                 np.array(thispf_recfracs),
                 marker='.',
                 label='%s' % pfm.upper(),
                 ms=0.0)

        # add this to the outdict array
        amplitudebinned_per_pfmethod_recfracs.append(
            np.array(thispf_recfracs)
        )

    # finish up the plot
    plt.plot(amplitudebinned_amplitudes, amplitudebinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per period-finder recovery fraction by periodic var amplitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-amplitudes-pfmethod.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 3d. plot of recovery rate per amplitude bin per variable type
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out all vartypes
    all_vartypes = np.unique(
        [(precvar['details'][x]['actual_vartype'])
         for x in precvar['details'] if
         (precvar['details'][x]['actual_vartype'] is not None)]
    )

    for vt in all_vartypes:

        thisvt_recfracs = []

        for amplitudebin_pv, amplitudebin_rv in zip(
                amplitudebinned_periodicvars,
                amplitudebinned_recovered_objects
        ):

            thisbin_thisvt_recvars = [
                x for x in amplitudebin_rv
                if (precvar['details'][x]['actual_vartype'] == vt)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thisvt_recvars).size /
                amplitudebin_pv.size
            )
            thisvt_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(amplitudebinned_amplitudes,
                 np.array(thisvt_recfracs),
                 marker='.',
                 label='%s' % vt,
                 ms=0.0)

        # add this to the outdict array
        amplitudebinned_per_vartype_recfracs.append(
            np.array(thisvt_recfracs)
        )

    # finish up the plot
    plt.plot(amplitudebinned_amplitudes, amplitudebinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per vartype recovery fraction by periodic var amplitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-amplitudes-vartype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 3e. plot of recovery rate per amplitude bin per alias type
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out all vartypes
    all_aliastypes = recovered_status

    for at in all_aliastypes:

        thisat_recfracs = []

        for amplitudebin_pv, amplitudebin_rv in zip(
                amplitudebinned_periodicvars,
                amplitudebinned_recovered_objects
        ):

            thisbin_thisat_recvars = [
                x for x in amplitudebin_rv
                if (precvar['details'][x]['best_recovered_status'][0] == at)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thisat_recvars).size /
                amplitudebin_pv.size
            )
            thisat_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(amplitudebinned_amplitudes,
                 np.array(thisat_recfracs),
                 marker='.',
                 label='%s' % at,
                 ms=0.0)

        # add this to the outdict array
        amplitudebinned_per_aliastype_recfracs.append(
            np.array(thisat_recfracs)
        )

    # finish up the plot
    plt.plot(amplitudebinned_amplitudes, amplitudebinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per alias-type recovery fraction by periodic var amplitudes')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-amplitudes-aliastype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 4. recovery-rate by ndet bin

    # 4a. plot of overall recovery rate per ndet bin
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    plt.plot(ndetbinned_ndets, ndetbinned_recfrac,
             marker='.',ms=0.0)
    plt.xlabel('periodic variable light curve points')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('overall recovery fraction by periodic var ndet')
    plt.ylim((0,1))
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-ndet-overall.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 4b. plot of recovery rate per ndet bin per magcol
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    for magcol in magcols:

        thismagcol_recfracs = []

        for ndetbin_pv, ndetbin_rv in zip(ndetbinned_periodicvars,
                                          ndetbinned_recovered_objects):

            thisbin_thismagcol_recvars = [
                x for x in ndetbin_rv
                if (precvar['details'][x]['best_recovered_magcol'] == magcol)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thismagcol_recvars).size /
                ndetbin_pv.size
            )
            thismagcol_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(ndetbinned_ndets,
                 np.array(thismagcol_recfracs),
                 marker='.',
                 label='magcol: %s' % magcol,
                 ms=0.0)

        # add this to the outdict array
        ndetbinned_per_magcol_recfracs.append(
            np.array(thismagcol_recfracs)
        )

    # finish up the plot
    plt.plot(ndetbinned_ndets, ndetbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per magcol recovery fraction by periodic var ndets')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-ndet-magcols.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 4c. plot of recovery rate per ndet bin per periodfinder
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out which pfmethods were used
    all_pfmethods = np.unique(
        np.concatenate(
            [np.unique(precvar['details'][x]['recovery_pfmethods'])
             for x in precvar['details']]
        )
    )

    for pfm in all_pfmethods:

        thispf_recfracs = []

        for ndetbin_pv, ndetbin_rv in zip(ndetbinned_periodicvars,
                                          ndetbinned_recovered_objects):

            thisbin_thispf_recvars = [
                x for x in ndetbin_rv
                if (precvar['details'][x]['best_recovered_pfmethod'] == pfm)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thispf_recvars).size /
                ndetbin_pv.size
            )
            thispf_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(ndetbinned_ndets,
                 np.array(thispf_recfracs),
                 marker='.',
                 label='%s' % pfm.upper(),
                 ms=0.0)

        # add this to the outdict array
        ndetbinned_per_pfmethod_recfracs.append(
            np.array(thispf_recfracs)
        )

    # finish up the plot
    plt.plot(ndetbinned_ndets, ndetbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per period-finder recovery fraction by periodic var ndets')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-ndet-pfmethod.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 4d. plot of recovery rate per ndet bin per variable type
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out all vartypes
    all_vartypes = np.unique(
        [(precvar['details'][x]['actual_vartype'])
         for x in precvar['details'] if
         (precvar['details'][x]['actual_vartype'] in PERIODIC_VARTYPES)]
    )

    for vt in all_vartypes:

        thisvt_recfracs = []

        for ndetbin_pv, ndetbin_rv in zip(ndetbinned_periodicvars,
                                          ndetbinned_recovered_objects):

            thisbin_thisvt_recvars = [
                x for x in ndetbin_rv
                if (precvar['details'][x]['actual_vartype'] == vt)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thisvt_recvars).size /
                ndetbin_pv.size
            )
            thisvt_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(ndetbinned_ndets,
                 np.array(thisvt_recfracs),
                 marker='.',
                 label='%s' % vt,
                 ms=0.0)

        # add this to the outdict array
        ndetbinned_per_vartype_recfracs.append(
            np.array(thisvt_recfracs)
        )

    # finish up the plot
    plt.plot(ndetbinned_ndets, ndetbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per vartype recovery fraction by periodic var ndets')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-ndet-vartype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 4e. plot of recovery rate per ndet bin per alias type
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    # figure out all vartypes
    all_aliastypes = recovered_status

    for at in all_aliastypes:

        thisat_recfracs = []

        for ndetbin_pv, ndetbin_rv in zip(ndetbinned_periodicvars,
                                          ndetbinned_recovered_objects):

            thisbin_thisat_recvars = [
                x for x in ndetbin_rv
                if (precvar['details'][x]['best_recovered_status'][0] == at)
            ]
            thisbin_thismagcol_recfrac = (
                np.array(thisbin_thisat_recvars).size /
                ndetbin_pv.size
            )
            thisat_recfracs.append(thisbin_thismagcol_recfrac)

        # now that we have per magcol recfracs, plot them
        plt.plot(ndetbinned_ndets,
                 np.array(thisat_recfracs),
                 marker='.',
                 label='%s' % at,
                 ms=0.0)

        # add this to the outdict array
        ndetbinned_per_aliastype_recfracs.append(
            np.array(thisat_recfracs)
        )

    # finish up the plot
    plt.plot(ndetbinned_ndets, ndetbinned_recfrac,
             marker='.',ms=0.0, label='overall', color='k')
    plt.xlabel(r'SDSS $r$ magnitude')
    plt.ylabel('recovered fraction of periodic variables')
    plt.title('per alias-type recovery fraction by periodic var ndets')
    plt.ylim((0,1))
    plt.legend(markerscale=10.0)
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-binned-ndet-aliastype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # update the lists in the outdict
    outdict['magbinned_per_magcol_recfracs'] = (
        magbinned_per_magcol_recfracs
    )
    outdict['magbinned_per_pfmethod_recfracs'] = (
        magbinned_per_pfmethod_recfracs
    )
    outdict['magbinned_per_vartype_recfracs'] = (
        magbinned_per_vartype_recfracs
    )
    outdict['magbinned_per_aliastype_recfracs'] = (
        magbinned_per_aliastype_recfracs
    )

    outdict['periodbinned_per_magcol_recfracs'] = (
        periodbinned_per_magcol_recfracs
    )
    outdict['periodbinned_per_pfmethod_recfracs'] = (
        periodbinned_per_pfmethod_recfracs
    )
    outdict['periodbinned_per_vartype_recfracs'] = (
        periodbinned_per_vartype_recfracs
    )
    outdict['periodbinned_per_aliastype_recfracs'] = (
        periodbinned_per_aliastype_recfracs
    )

    outdict['amplitudebinned_per_magcol_recfracs'] = (
        amplitudebinned_per_magcol_recfracs
    )
    outdict['amplitudebinned_per_pfmethod_recfracs'] = (
        amplitudebinned_per_pfmethod_recfracs
    )
    outdict['amplitudebinned_per_vartype_recfracs'] = (
        amplitudebinned_per_vartype_recfracs
    )
    outdict['amplitudebinned_per_aliastype_recfracs'] = (
        amplitudebinned_per_aliastype_recfracs
    )

    outdict['ndetbinned_per_magcol_recfracs'] = (
        ndetbinned_per_magcol_recfracs
    )
    outdict['ndetbinned_per_pfmethod_recfracs'] = (
        ndetbinned_per_pfmethod_recfracs
    )
    outdict['ndetbinned_per_vartype_recfracs'] = (
        ndetbinned_per_vartype_recfracs
    )
    outdict['ndetbinned_per_aliastype_recfracs'] = (
        ndetbinned_per_aliastype_recfracs
    )

    # get the overall recovered vars per pfmethod
    overall_recvars_per_pfmethod = []

    for pfm in all_pfmethods:

        thispfm_recvars = np.array([
            x for x in precvar['details'] if
            ((x in recovered_periodicvars) and
             (precvar['details'][x]['best_recovered_pfmethod'] == pfm))
        ])
        overall_recvars_per_pfmethod.append(thispfm_recvars)

    # get the overall recovered vars per vartype
    overall_recvars_per_vartype = []

    for vt in all_vartypes:

        thisvt_recvars = np.array([
            x for x in precvar['details'] if
            ((x in recovered_periodicvars) and
             (precvar['details'][x]['actual_vartype'] == vt))
        ])
        overall_recvars_per_vartype.append(thisvt_recvars)

    # get the overall recovered vars per magcol
    overall_recvars_per_magcol = []

    for mc in magcols:

        thismc_recvars = np.array([
            x for x in precvar['details'] if
            ((x in recovered_periodicvars) and
             (precvar['details'][x]['best_recovered_magcol'] == mc))
        ])
        overall_recvars_per_magcol.append(thismc_recvars)

    # get the overall recovered vars per aliastype
    overall_recvars_per_aliastype = []

    for at in all_aliastypes:

        thisat_recvars = np.array([
            x for x in precvar['details'] if
            ((x in recovered_periodicvars) and
             (precvar['details'][x]['best_recovered_status'] == at))
        ])
        overall_recvars_per_aliastype.append(thisat_recvars)

    # update the outdict with these
    outdict['overall_recfrac_per_pfmethod'] = np.array([
        x.size/actual_periodicvars.size for x in overall_recvars_per_pfmethod
    ])
    outdict['overall_recfrac_per_vartype'] = np.array([
        x.size/actual_periodicvars.size for x in overall_recvars_per_vartype
    ])
    outdict['overall_recfrac_per_magcol'] = np.array([
        x.size/actual_periodicvars.size for x in overall_recvars_per_magcol
    ])
    outdict['overall_recfrac_per_aliastype'] = np.array([
        x.size/actual_periodicvars.size for x in overall_recvars_per_aliastype
    ])

    # 5. bar plot of overall recovery rate per pfmethod
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    xt = np.arange(len(all_pfmethods))
    xl = all_pfmethods

    plt.barh(xt, outdict['overall_recfrac_per_pfmethod'], 0.50)
    plt.yticks(xt, xl)
    plt.xlabel('period-finding method')
    plt.ylabel('overall recovery rate')
    plt.title('overall recovery rate per period-finding method')
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-overall-pfmethod.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 6. bar plot of overall recovery rate per magcol
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    xt = np.arange(len(magcols))
    xl = magcols

    plt.barh(xt, outdict['overall_recfrac_per_magcol'], 0.50)
    plt.yticks(xt, xl)
    plt.xlabel('light curve magnitude column')
    plt.ylabel('overall recovery rate')
    plt.title('overall recovery rate per light curve magcol')
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-overall-magcol.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 7. bar plot of overall recovery rate per aliastype
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    xt = np.arange(len(all_aliastypes))
    xl = all_aliastypes

    plt.barh(xt, outdict['overall_recfrac_per_aliastype'], 0.50)
    plt.yticks(xt, xl)
    plt.xlabel('period recovery status')
    plt.ylabel('overall recovery rate')
    plt.title('overall recovery rate per period recovery status')
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-overall-aliastype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 8. bar plot of overall recovery rate per vartype
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))

    xt = np.arange(len(all_vartypes))
    xl = all_vartypes

    plt.barh(xt, outdict['overall_recfrac_per_vartype'], 0.50)
    plt.yticks(xt, xl)
    plt.xlabel('periodic variable type')
    plt.ylabel('overall recovery rate')
    plt.title('overall recovery rate per periodic variable type')
    plt.savefig(
        os.path.join(recplotdir,
                     'recfrac-overall-vartype.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 9. overall recovered period periodogram for objects that aren't actual
    # periodic variables. this effectively should give us the window function of
    # the observations

    notvariable_recovered_periods = np.concatenate([
        precvar['details'][x]['recovery_periods']
        for x in precvar['details'] if
        (precvar['details'][x]['actual_vartype'] is None)
    ])
    notvariable_recovered_lspvals = np.concatenate([
        precvar['details'][x]['recovery_lspvals']
        for x in precvar['details'] if
        (precvar['details'][x]['actual_vartype'] is None)
    ])

    sortind = np.argsort(notvariable_recovered_periods)
    notvariable_recovered_periods = notvariable_recovered_periods[sortind]
    notvariable_recovered_lspvals = notvariable_recovered_lspvals[sortind]

    outdict['notvariable_recovered_periods'] = notvariable_recovered_periods
    outdict['notvariable_recovered_lspvals'] = notvariable_recovered_lspvals

    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))
    plt.plot(notvariable_recovered_periods,
             notvariable_recovered_lspvals,
             ms=1.0,linestyle='none',marker='.')
    plt.xscale('log')
    plt.xlabel('recovered periods [days]')
    plt.ylabel('recovered normalized periodogram power')
    plt.title('periodogram for actual not-variable objects')
    plt.savefig(
        os.path.join(recplotdir,
                     'recovered-periodogram-nonvariables.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # 10. overall recovered period histogram for objects marked
    # not-variable. this gives us the most common periods
    fig = plt.figure(figsize=(6.4*1.5,4.8*1.5))
    plt.hist(notvariable_recovered_periods,bins=np.arange(0.02,300.0,1.0e-3),
             histtype='step')
    plt.xscale('log')
    plt.xlabel('recovered periods [days]')
    plt.ylabel('number of times periods recovered')
    plt.title('recovered period histogram for non-variable objects')
    plt.savefig(
        os.path.join(recplotdir,
                     'recovered-period-hist-nonvariables.%s' % plotfile_ext),
        dpi=100,
        bbox_inches='tight'
    )
    plt.close('all')

    # at the end, write the outdict to a pickle and return it
    outfile = os.path.join(simbasedir, 'periodicvar-recovery-plotresults.pkl')
    with open(outfile,'wb') as outfd:
        pickle.dump(outdict, outfd, pickle.HIGHEST_PROTOCOL)

    return outdict
