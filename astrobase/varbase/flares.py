#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''flares.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - May 2017
License: MIT. See the LICENSE file for the full text.

Contains functions to deal with finding stellar flares in time series.

'''


import logging
from datetime import datetime
from traceback import format_exc
from time import time as unixtime
import os.path
import os

import numpy as np

from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, all as npall, \
    correlate as npcorrelate

from scipy.signal import savgol_filter

###################
## LOCAL IMPORTS ##
###################



#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.varbase.flares' % parent_name)

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



###########################
## FLARE MODEL FUNCTIONS ##
###########################


def flare_model(times,
                mags,
                errs,
                amplitude,
                flare_peak_time,
                rise_gaussian_stdev,
                decay_time_constant,
                magsarefluxes=False):
    '''This is a flare model function, similar to Kowalski+ 2011.

    From the paper by Pitkin+ 2014:

    http://adsabs.harvard.edu/abs/2014MNRAS.445.2268P

    times: a numpy array of times

    mags: a numpy array of magnitudes or fluxes. the flare will simply be added
    to mags at the appropriate times

    errs: a numpy array of measurement errors for each mag/flux measurement

    amplitude: the maximum flare amplitude in mags or flux

    flare_peak_time: time at which the flare maximum happens

    rise_gaussian_stdev: the stdev of the gaussian describing the rise of the
                         flare

    decay_time_constant: the time constant of the exponential fall of the flare

    If magsarefluxes = True: everything will be assumed to be in fluxes.

    '''

    # set up the model
    if magsarefluxes:

        # before peak gaussian rise...
        mags[times < flare_peak_time] = (
            mags[times < flare_peak_time] +
            amplitude * np.exp(
                -((times[times < flare_peak_time] -
                   flare_peak_time) *
                  (times[times < flare_peak_time] -
                   flare_peak_time)) /
                (2.0*rise_gaussian_stdev*rise_gaussian_stdev)
                )
        )

        # after peak exponential decay...
        mags[times > flare_peak_time] = (
            mags[times > flare_peak_time] +
            amplitude * np.exp(
                -((times[times > flare_peak_time] -
                   flare_peak_time)) /
                (decay_time_constant)
                )
        )

    else:

        # before peak gaussian rise...
        mags[times < flare_peak_time] = (
            mags[times < flare_peak_time] -
            amplitude * np.exp(
                -((times[times < flare_peak_time] -
                   flare_peak_time) *
                  (times[times < flare_peak_time] -
                   flare_peak_time)) /
                (2.0*rise_gaussian_stdev*rise_gaussian_stdev)
                )
        )

        # after peak exponential decay...
        mags[times > flare_peak_time] = (
            mags[times > flare_peak_time] -
            amplitude * np.exp(
                -((times[times > flare_peak_time] -
                   flare_peak_time)) /
                (decay_time_constant)
                )
        )


    return {'times':times,
            'mags':mags,
            'errs':errs,
            'amplitude':amplitude,
            'flare_peak_time':flare_peak_time,
            'rise_gaussian_stdev':rise_gaussian_stdev,
            'decay_time_constant':decay_time_constant}



###################
## FLARE FINDERS ##
###################

def simple_flare_find(times, mags, errs,
                      smoothbinsize=97,
                      flareminsigma=4.0,
                      flaremaxcadencediff=1,
                      flaremincadencepoints=3,
                      magsarefluxes=False,
                      savgolpolyorder=2,
                      **savgolkwargs):
    '''This finds flares in  time series using the method in Walkowicz+ 2011.

    Returns number of flares found, and their time indices.

    Args
    ----

    times, mags, errs are numpy arrays for the time series.

    Kwargs
    ------

    smoothbinsize: the number of consecutive light curve points to smooth over
    in the time series using a Savitsky-Golay filter. The smoothed light curve
    is then subtracted from the actual light curve to remove trends that
    potentially last smoothbinsize light curve points. The default value is
    chosen as ~6.5 hours (97 x 4 minute cadence for HATNet/HATSouth).

    flareminsigma: the minimum sigma above the median light curve level to
    designate points as belonging to possible flares

    flaremaxcadencediff: the maximum number of light curve points apart each
    possible flare event measurement is allowed to be. If this is 1, then we'll
    look for consecutive measurements.

    flaremincadencepoints: the minimum number of light curve points (each
    flaremaxcadencediff points apart) required that are at least flareminsigma
    above the median light curve level to call an event a flare.

    magsarefluxes: if True, indicates that mags is actually an array of fluxes.

    savgolpolyorder: the polynomial order of the function used by the
    Savitsky-Golay filter.

    Any remaining keyword arguments are passed directly to the savgol_filter
    function from scipy.

    '''

    # if no errs are given, assume 0.1% errors
    if errs is None:
        errs = 0.001*mags

    # get rid of nans first
    finiteind = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes = times[finiteind]
    fmags = mags[finiteind]
    ferrs = errs[finiteind]

    # now get the smoothed mag series using the filter
    # kwargs are provided to the savgol_filter function
    smoothed = savgol_filter(fmags,
                             smoothbinsize,
                             savgolpolyorder,
                             **savgolkwargs)
    subtracted = fmags - smoothed

    # calculate some stats
    # the series_median is ~zero after subtraction
    series_mad = npmedian(npabs(subtracted))
    series_stdev = 1.483*series_mad

    # find extreme positive deviations
    if magsarefluxes:
        extind = npwhere(subtracted > (minflaresigma*series_stdev))
    else:
        extind = npwhere(subtracted < (-minflaresigma*series_stdev))

    # see if there are any extrema
    if extind and extind[0]:

        extrema_indices = extind[0]
        flaregroups = []

        # find the deviations within the requested flaremaxcadencediff
        for ind, extrema_index in enumerate(extrema_indices):

            stuff_to_do()






############################
## FLARE CHARACTERIZATION ##
############################
