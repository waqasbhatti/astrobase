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


# register the fakelc pklc as a custom lcproc format
# now we should be able to use all lcproc functions correctly
lcproc.register_custom_lcformat(
    'fakelc',
    '*-fakelc.pkl',
    read_pklc,
    None, # these are None because we make sure to get from fakelc directly
    None, # these are None because we make sure to get from fakelc directly
    None, # these are None because we make sure to get from fakelc directly
    magsarefluxes=False,
    specialnormfunc=None
)



###########################
## VARIABILITY FUNCTIONS ##
###########################


def variable_index_gridsearch(simbasedir,
                              stetson_stdev_range=[1.0,20.0],
                              inveta_stdev_range=[1.0,20.0]):
    '''This runs variability selection on all of the light curves in simbasedir.

    Reads the fakelcs-info.pkl in simbasedir to get:

    - the variable objects, their types, periods, epochs, and params
    - the nonvariable objects

    For each magbin, this does a grid search using the stetson and inveta ranges
    and tries to optimize the Matthews Correlation Coefficient, indicating the
    best possible separation of variables vs. nonvariables. The thresholds on
    these two variable indexes that produce the largest coeff for the collection
    of fake LCs will probably be the ones that work best for actual variable
    classification on the real LCs.

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
