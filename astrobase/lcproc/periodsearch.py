#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''periodsearch.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

This contains functions to run period-finding in a parallelized manner on large
collections of light curves.

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

try:
    import cPickle as pickle
except Exception as e:
    import pickle

import os
import os.path
import glob
import multiprocessing as mp

from tornado.escape import squeeze

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem
def dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)



###################
## LOCAL IMPORTS ##
###################

from astrobase import periodbase

from astrobase.lcproc import LCFORM



############
## CONFIG ##
############

NCPUS = mp.cpu_count()

# used to figure out which period finder to run given a list of methods
PFMETHODS = {'bls':periodbase.bls_parallel_pfind,
             'gls':periodbase.pgen_lsp,
             'aov':periodbase.aov_periodfind,
             'mav':periodbase.aovhm_periodfind,
             'pdm':periodbase.stellingwerf_pdm,
             'acf':periodbase.macf_period_find,
             'win':periodbase.specwindow_lsp}

PFMETHOD_NAMES = {
    'gls':'Generalized Lomb-Scargle periodogram',
    'pdm':'Stellingwerf phase-dispersion minimization',
    'aov':'Schwarzenberg-Czerny AoV',
    'mav':'Schwarzenberg-Czerny AoV multi-harmonic',
    'bls':'Box Least-squared Search',
    'acf':'McQuillan+ ACF Period Search',
    'win':'Timeseries Sampling Lomb-Scargle periodogram'
}
