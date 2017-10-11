#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''saov.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Contains the Schwarzenberg-Cerny Analysis of Variance period-search algorithm
implementation for periodbase. This uses the multi-harmonic version presented in
Schwarzenberg-Cerny (1996).

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


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.smav' % parent_name)

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
    phase_bin_magseries

from . import get_frequency_grid


############
## CONFIG ##
############

NCPUS = cpu_count()


###################################################################
## MULTIHARMONIC ANALYSIS of VARIANCE (Schwarzenberg-Cerny 1996) ##
###################################################################


def harmonic_aov_theta(times, mags, errs, frequency, nharmonics,
                       binsize=0.05, minbin=9):
    '''
    This calculates the harmonic AoV theta for a frequency.

    Schwarzenberg-Cerny 1996 equation 11:

    theta_prefactor = (K - 2N - 1)/(2N)
    theta_top = sum(c_n*c_n) (from n=0 to n=2N)
    theta_bot = variance(timeseries) - sum(c_n*c_n) (from n=0 to n=2N)

    theta = theta_prefactor * (theta_top/theta_bot)

    N = number of harmonics (nharmonics)
    K = length of time series

    times, mags, errs should all be free of nans/infs and be normalized to zero.

    z_k = e^(i*w*t_k), where k is the time index, 1 -> N

    z^n = e^(i*n*w*t), where n is the harmonic index, 1 -> N


    recurrence relation for successive orders (SC96 eqn 6):

    phi_tilde_(n+1)(z) =
         z * phi_tilde_n - alpha_n * z^n * conjugate(phi_tilde_n(z))


    SC96 equation 2:

    scalar_product(phi, psi) =
           sum_k^K(weights_k * phi(z_k) * conjugate(psi(z_k)))


    '''
