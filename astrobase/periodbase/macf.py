#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''macf.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

This contains the ACF period-finding algorithm from McQuillian+ 2013a and
McQuillian+ 2014.

'''


from multiprocessing import Pool, cpu_count
import logging
from datetime import datetime
from traceback import format_exc

import numpy as np

# import these to avoid lookup overhead
from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, \
    arctan as nparctan, nanargmax as npnanargmax, nanargmin as npnanargmin, \
    empty as npempty, ceil as npceil, mean as npmean, \
    digitize as npdigitize, unique as npunique, \
    argmax as npargmax, argmin as npargmin

from scipy.signal import argrelmax, argrelmin
from astropy.convolution import convolve, Gaussian1DKernel


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.macf' % parent_name)

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

from ..lcmath import phase_magseries, sigclip_magseries, time_bin_magseries, \
    phase_bin_magseries, fill_magseries_gaps


############
## CONFIG ##
############

NCPUS = cpu_count()



######################
## HELPER FUNCTIONS ##
######################


def _smooth_acf(acf, windowfwhm=7):
    '''
    This returns a smoothed version of the ACF.

    Convolves the ACF with a Gaussian of given windowsize and windowfwhm

    '''

    convkernel = Gaussian1DKernel(windowfwhm)
    smoothed = convolve(acf, convkernel, boundary='extend')

    return smoothed



def _get_acf_peakheights(lags, acf, npeaks=10):
    '''This calculates the relative peak heights for first npeaks in ACF.

    Usually, the first peak or the second peak (if its peak height > first peak)
    corresponds to the correct period.

    '''

    maxinds = argrelmax(acf)
    mininds = argrelmin(acf)

    # TODO:
    # - go through each max
    # - find the two mininds on either side of each maxind
    # - calculate the relative peak height:
    #   hp = acf[maxind]/np.mean(acf[leftminind],acf[rightminind])
    # - calculate up to npeaks relative heights
    # - return them
