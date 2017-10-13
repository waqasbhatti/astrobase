#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''saov.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Contains the Schwarzenberg-Czerny Analysis of Variance period-search algorithm
implementation for periodbase. This uses the multi-harmonic version presented in
Schwarzenberg-Czerny (1996).

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

from ..lcmath import phase_magseries_with_errs, sigclip_magseries, \
    time_bin_magseries, phase_bin_magseries

from . import get_frequency_grid


############
## CONFIG ##
############

NCPUS = cpu_count()


###################################################################
## MULTIHARMONIC ANALYSIS of VARIANCE (Schwarzenberg-Czerny 1996) ##
###################################################################


def harmonic_aov_theta(times, mags, errs, frequency,
                       nharmonics, magvariance):
    '''This calculates the harmonic AoV theta for a frequency.

    Schwarzenberg-Czerny 1996 equation 11:

    theta_prefactor = (K - 2N - 1)/(2N)
    theta_top = sum(c_n*c_n) (from n=0 to n=2N)
    theta_bot = variance(timeseries) - sum(c_n*c_n) (from n=0 to n=2N)

    theta = theta_prefactor * (theta_top/theta_bot)

    N = number of harmonics (nharmonics)
    K = length of time series (times.size)

    times, mags, errs should all be free of nans/infs and be normalized to zero.

    nharmonics is the number of harmonics to calculate up to. The recommended
    range is 4 to 8.

    magvariance is the (weighted by errors) variance of the magnitude time
    series.

    This is a mostly faithful translation of the inner loop in aovper.f90.

    See http://users.camk.edu.pl/alex/ and Schwarzenberg-Czerny (1996).

    http://iopscience.iop.org/article/10.1086/309985/meta

    '''

    period = 1.0/frequency

    ndet = times.size
    two_nharmonics = nharmonics + nharmonics

    # phase with test period
    phasedseries = phase_magseries_with_errs(
        times, mags, errs, period, times[0],
        sort=True, wrap=False
    )

    # get the phased quantities
    phase = phasedseries['phase']
    pmags = phasedseries['mags']
    perrs = phasedseries['errs']

    # this is sqrt(1.0/errs^2) -> the weights
    pweights = 1.0/perrs

    # multiply by 2.0*PI (for omega*time)
    phase = phase * 2.0 * MPI

    # this is the z complex vector
    z = np.cos(phase) + 1.0j*np.sin(phase)

    # multiply phase with N
    phase = nharmonics * phase

    # this is the psi complex vector
    psi = pmags * pweights * (np.cos(phase) + 1j*np.sin(phase))

    # this is the initial value of z^n
    zn = 1.0 + 0.0j

    # this is the initial value of phi
    phi = pweights + 0.0j

    # initialize theta to zero
    theta_aov = 0.0

    # go through all the harmonics now up to 2N
    for n in range(two_nharmonics):

        # this is <phi, phi>
        phi_dot_phi = np.sum(phi * phi.conjugate())

        # this is the alpha_n numerator
        alpha = np.sum(pweights * z * phi)

        # this is <phi, psi>
        phi_dot_psi = np.dot(phi, psi)

        # make sure phi_dot_phi is not zero
        phi_dot_phi = np.max([phi_dot_phi, 10.0e-9])

        # this is the expression for alpha_n
        alpha = alpha / phi_dot_phi

        # update theta_aov for this harmonic
        theta_aov = (theta_aov +
                     np.abs(phi_dot_psi) * np.abs(phi_dot_psi) / phi_dot_phi)

        # use the recurrence relation to find the next phi
        phi = phi * z - alpha * zn * phi.conjugate()

        # update z^n
        zn = zn * z


    # done with all harmonics, calculate the theta_aov for this freq
    # the max below makes sure that magvariance - theta_aov > zero
    theta_aov = ( (ndet - two_nharmonics - 1.0) * theta_aov /
                  (two_nharmonics * np.max([magvariance - theta_aov,
                                            1.0e-9])) )

    return theta_aov
