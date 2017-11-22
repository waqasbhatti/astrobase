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
    actualvars = objectids[isvariable]
    actualnotvars = objectids[~isvariable]

    # this is the output directory
    recdict = {'stetj_min_stdev':stetson_stdev_min,
               'inveta_min_stdev':inveta_stdev_min,
               'actual_variables':actualvars,
               'actual_nonvariables':actualnotvars,
               'magbin_medians':varthresh['binned_sdssr_median']}


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

        stet_tpfrac = stet_truepositives.size/actualvars.size
        stet_fpfrac = stet_falsepositives.size/actualnotvars.size
        stet_tnfrac = stet_truenegatives.size/actualnotvars.size
        stet_fnfrac = stet_falsenegatives.size/actualvars.size

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

        inveta_tpfrac = inveta_truepositives.size/actualvars.size
        inveta_fpfrac = inveta_falsepositives.size/actualnotvars.size
        inveta_tnfrac = inveta_truenegatives.size/actualnotvars.size
        inveta_fnfrac = inveta_falsenegatives.size/actualvars.size

        # calculate stetson recall, precision, Matthews correl coeff
        stet_recall = stet_truepositives.size/(stet_truepositives.size +
                                               stet_falsenegatives.size)
        stet_precision = stet_truepositives.size/(stet_truepositives.size +
                                                  stet_falsepositives.size)
        stet_mcc = matthews_correl_coeff(stet_truepositives.size,
                                         stet_truenegatives.size,
                                         stet_falsepositives.size,
                                         stet_falsenegatives.size)

        # calculate inveta recall, precision, Matthews correl coeff
        inveta_recall = inveta_truepositives.size/(inveta_truepositives.size +
                                               inveta_falsenegatives.size)
        inveta_precision = inveta_truepositives.size/(inveta_truepositives.size +
                                                      inveta_falsepositives.size)
        inveta_mcc = matthews_correl_coeff(inveta_truepositives.size,
                                         inveta_truenegatives.size,
                                         inveta_falsepositives.size,
                                         inveta_falsenegatives.size)


        recdict[magcol] = {
            'stet_recoveredvars':stet_recoveredvars,
            'stet_truepositives':stet_truepositives,
            'stet_falsepositives':stet_falsepositives,
            'stet_truenegatives':stet_truenegatives,
            'stet_falsenegative':stet_falsenegatives,
            'stet_tpfrac':stet_tpfrac,
            'stet_fpfrac':stet_fpfrac,
            'stet_tnfrac':stet_tnfrac,
            'stet_fnfrac':stet_fnfrac,
            'stet_recall':stet_recall,
            'stet_precision':stet_precision,
            'stet_mcc':stet_mcc,
            'inveta_recoveredvars':inveta_recoveredvars,
            'inveta_truepositives':inveta_truepositives,
            'inveta_falsepositives':inveta_falsepositives,
            'inveta_truenegatives':inveta_truenegatives,
            'inveta_falsenegative':inveta_falsenegatives,
            'inveta_tpfrac':inveta_tpfrac,
            'inveta_fpfrac':inveta_fpfrac,
            'inveta_tnfrac':inveta_tnfrac,
            'inveta_fnfrac':inveta_fnfrac,
            'inveta_recall':inveta_recall,
            'inveta_precision':inveta_precision,
            'inveta_mcc':inveta_mcc,
        }


    return recdict



def variable_index_gridsearch(simbasedir,
                              stetson_stdev_range=[1.0,20.0],
                              inveta_stdev_range=[1.0,20.0],
                              dgrid=0.5):
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
