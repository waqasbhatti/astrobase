#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flares.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - May 2017
# License: MIT. See the LICENSE file for the full text.

'''
Contains functions to deal with finding stellar flares in time series.

FIXME: finish this module.

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

import numpy as np
from scipy.signal import savgol_filter

from astrobase.lcmodels import flares


###########################
## FLARE MODEL FUNCTIONS ##
###########################

def add_flare_model(flareparams,
                    times,
                    mags,
                    errs):
    '''This adds a flare model function to the input magnitude/flux time-series.

    Parameters
    ----------

    flareparams : list of float
        This defines the flare model::

            [amplitude,
             flare_peak_time,
             rise_gaussian_stdev,
             decay_time_constant]

        where:

        `amplitude`: the maximum flare amplitude in mags or flux. If flux, then
        amplitude should be positive. If mags, amplitude should be negative.

        `flare_peak_time`: time at which the flare maximum happens.

        `rise_gaussian_stdev`: the stdev of the gaussian describing the rise of
        the flare.

        `decay_time_constant`: the time constant of the exponential fall of the
        flare.

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the model will be generated. The times will be used to generate
        model mags.

    magsarefluxes : bool
        Sets the correct direction of the flare amplitude (+ve) for fluxes if
        True and for mags (-ve) if False.

    Returns
    -------

    dict
        A dict of the form below is returned::

        {'times': the original times array
         'mags': the original mags + the flare model mags evaluated at times,
         'errs': the original errs array,
         'flareparams': the input list of flare params}

    '''

    modelmags, ftimes, fmags, ferrs = flares.flare_model(
        flareparams,
        times,
        mags,
        errs
    )

    return {'times':times,
            'mags':mags + modelmags,
            'errs':errs,
            'flareparams':flareparams}


###################
## FLARE FINDERS ##
###################

def simple_flare_find(times, mags, errs,
                      smoothbinsize=97,
                      flare_minsigma=4.0,
                      flare_maxcadencediff=1,
                      flare_mincadencepoints=3,
                      magsarefluxes=False,
                      savgol_polyorder=2,
                      **savgol_kwargs):
    '''This finds flares in time series using the method in Walkowicz+ 2011.

    FIXME: finish this.

    Parameters
    ----------

    times,mags,errs : np.array
        The input time-series to find flares in.

    smoothbinsize : int
        The number of consecutive light curve points to smooth over in the time
        series using a Savitsky-Golay filter. The smoothed light curve is then
        subtracted from the actual light curve to remove trends that potentially
        last `smoothbinsize` light curve points. The default value is chosen as
        ~6.5 hours (97 x 4 minute cadence for HATNet/HATSouth).

    flare_minsigma : float
        The minimum sigma above the median LC level to designate points as
        belonging to possible flares.

    flare_maxcadencediff : int
        The maximum number of light curve points apart each possible flare event
        measurement is allowed to be. If this is 1, then we'll look for
        consecutive measurements.

    flare_mincadencepoints : int
        The minimum number of light curve points (each `flare_maxcadencediff`
        points apart) required that are at least `flare_minsigma` above the
        median light curve level to call an event a flare.

    magsarefluxes: bool
        If True, indicates that mags is actually an array of fluxes.

    savgol_polyorder: int
        The polynomial order of the function used by the Savitsky-Golay filter.

    savgol_kwargs : extra kwargs
        Any remaining keyword arguments are passed directly to the
        `savgol_filter` function from `scipy.signal`.

    Returns
    -------

    (nflares, flare_indices) : tuple
        Returns the total number of flares found and their time-indices (start,
        end) as tuples.

    '''

    # if no errs are given, assume 0.1% errors
    if errs is None:
        errs = 0.001*mags

    # get rid of nans first
    finiteind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes = times[finiteind]
    fmags = mags[finiteind]
    ferrs = errs[finiteind]

    # now get the smoothed mag series using the filter
    # kwargs are provided to the savgol_filter function
    smoothed = savgol_filter(fmags,
                             smoothbinsize,
                             savgol_polyorder,
                             **savgol_kwargs)
    subtracted = fmags - smoothed

    # calculate some stats
    # the series_median is ~zero after subtraction
    series_mad = np.median(np.abs(subtracted))
    series_stdev = 1.483*series_mad

    # find extreme positive deviations
    if magsarefluxes:
        extind = np.where(subtracted > (flare_minsigma*series_stdev))
    else:
        extind = np.where(subtracted < (-flare_minsigma*series_stdev))

    # see if there are any extrema
    if extind and extind[0]:

        extrema_indices = extind[0]
        flaregroups = []

        # find the deviations within the requested flaremaxcadencediff
        for ind, extrema_index in enumerate(extrema_indices):
            # FIXME: finish this
            pass


############################
## FLARE CHARACTERIZATION ##
############################
