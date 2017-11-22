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



###########################
## VARIABILITY FUNCTIONS ##
###########################

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



def matthews_correl_coeff(ntp, ntn, nfp, nfn):
    '''
    This calculates the Matthews correlation coefficent.

    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    '''

    mcc_top = (ntp*ntn - nfp*nfn)
    mcc_bot = msqrt((ntp + nfp)*(ntp + nfn)*(ntn + nfp)*(ntn + nfn))

    return mcc_top/mcc_bot



def get_recovered_variables(simbasedir,
                            stetson_stdev_min=2.0,
                            inveta_stdev_min=2.0):
    '''This runs variability selection for LCs in simbasedir and gets recovery
    stats.

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

    # run the variability search using the results of get_varfeatures
    varfeaturedir = os.path.join(simbasedir,'varfeatures')
    varthreshinfof = os.path.join(
        simbasedir,
        'varthresh-stet%.2f-inveta%.2f.pkl' % (stetson_stdev_min,
                                               inveta_stdev_min)
    )
    varthresh = lcproc.variability_threshold(varfeaturedir,
                                             varthreshinfof,
                                             lcformat='fakelc',
                                             min_stetj_stdev=stetson_stdev_min,
                                             min_inveta_stdev=inveta_stdev_min)

    # now get the true positives, false positives, true negatives, false
    # negatives, and calculate recall, precision, Matthews corr. coeff.
    actualvars = objectids[varflags]
    actualnotvars = objectids[~varflags]

    # this is the output directory
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


    # FIXME: make this recovery fraction by magnitude bin!
    for magcol in magcols:

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

        # calculate stetson recall, precision, Matthews correl coeff
        stet_recall = stet_truepositives.size/(stet_truepositives.size +
                                               stet_falsenegatives.size)
        stet_precision = stet_truepositives.size/(stet_truepositives.size +
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
        inveta_recall = inveta_truepositives.size/(inveta_truepositives.size +
                                               inveta_falsenegatives.size)
        inveta_precision = inveta_truepositives.size/(inveta_truepositives.size +
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

        # calculate intersectson recall, precision, Matthews correl coeff
        intersect_recall = (
            intersect_truepositives.size/(intersect_truepositives.size +
                                          intersect_falsenegatives.size)
        )
        intersect_precision = (
            intersect_truepositives.size/(intersect_truepositives.size +
                                          intersect_falsepositives.size)
        )
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
        union_recall = (
            union_truepositives.size/(union_truepositives.size +
                                      union_falsenegatives.size)
        )
        union_precision = (
            union_truepositives.size/(union_truepositives.size +
                                      union_falsepositives.size)
        )
        union_mcc = matthews_correl_coeff(union_truepositives.size,
                                          union_truenegatives.size,
                                          union_falsepositives.size,
                                          union_falsenegatives.size)

        recdict[magcol] = {
            # stetson J alone
            'stet_recoveredvars':stet_recoveredvars,
            'stet_truepositives':stet_truepositives,
            'stet_falsepositives':stet_falsepositives,
            'stet_truenegatives':stet_truenegatives,
            'stet_falsenegative':stet_falsenegatives,
            'stet_precision':stet_precision,
            'stet_recall':stet_recall,
            'stet_mcc':stet_mcc,
            # inveta alone
            'inveta_recoveredvars':inveta_recoveredvars,
            'inveta_truepositives':inveta_truepositives,
            'inveta_falsepositives':inveta_falsepositives,
            'inveta_truenegatives':inveta_truenegatives,
            'inveta_falsenegative':inveta_falsenegatives,
            'inveta_precision':inveta_precision,
            'inveta_recall':inveta_recall,
            'inveta_mcc':inveta_mcc,
            # intersect of stetson J and inveta
            'intersect_recoveredvars':intersect_recvars,
            'intersect_truepositives':intersect_truepositives,
            'intersect_falsepositives':intersect_falsepositives,
            'intersect_truenegatives':intersect_truenegatives,
            'intersect_falsenegative':intersect_falsenegatives,
            'intersect_precision':intersect_precision,
            'intersect_recall':intersect_recall,
            'intersect_mcc':intersect_mcc,
            # union of stetson J and inveta
            'union_recoveredvars':union_recvars,
            'union_truepositives':union_truepositives,
            'union_falsepositives':union_falsepositives,
            'union_truenegatives':union_truenegatives,
            'union_falsenegative':union_falsenegatives,
            'union_precision':union_precision,
            'union_recall':union_recall,
            'union_mcc':union_mcc,
            'magbin_medians':varthresh[magcol]['binned_sdssr_median']
        }


    return recdict



def variable_index_gridsearch(simbasedir,
                              stetson_stdev_range=[1.0,20.0],
                              inveta_stdev_range=[1.0,20.0],
                              ngridpoints=10):
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

    stetson_grid = np.linspace(stetson_stdev_range[0],
                               stetson_stdev_range[1],
                               num=ngridpoints)
    inveta_grid = np.linspace(inveta_stdev_range[0],
                              inveta_stdev_range[1],
                              num=ngridpoints)


    # first, run just stet

    # then, run just inveta

    # finally, run a grid of (stet, inveta) and get results for union/intersect
