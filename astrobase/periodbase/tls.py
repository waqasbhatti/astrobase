#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tls.py - Luke Bouma (luke@astro.princeton.edu) - Apr 2019

"""
Contains the Hippke & Heller (2019) transit-least-squared period-search
algorithm implementation for periodbase. This depends on the external package
written by Hippke & Heller, https://github.com/hippke/tls.
"""

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

from multiprocessing import cpu_count

from numpy import (
    nan as npnan, array as nparray,
    isfinite as npisfinite, argmax as npargmax,
    argsort as npargsort
)

try:
    from transitleastsquares import transitleastsquares
except Exception as e:
    errmsg = (
        'tried importing transitleastsquares and failed.\n'
        'are you sure you have installed it correctly?\n'
        'see https://transitleastsquares.readthedocs.io/en/latest/Installation.html'
    )
    raise ImportError(errmsg)


###################
## LOCAL IMPORTS ##
###################

from ..lcmath import sigclip_magseries


############
## CONFIG ##
############

NCPUS = cpu_count()


#######################
## UTILITY FUNCTIONS ##
#######################

def tls_parallel_pfind(times, mags, errs,
                       magsarefluxes=None,
                       startp=0.1,  # search from 0.1 d to...
                       endp=None,   # determine automatically from times
                       tlsoversample=5,
                       tlsmintransits=3,
                       tls_transit_template='default',
                       tls_rstar_min=0.13,
                       tls_rstar_max=3.5,
                       tls_mstar_min=0.1,
                       tls_mstar_max=2.0,
                       periodepsilon=0.1,
                       nbestpeaks=5,
                       sigclip=10.0,
                       verbose=True,
                       nworkers=None):
    """
    Wrapper to Hippke & Heller (2019)'s "transit least squares", which is BLS,
    but with a slightly better template (and niceties in the implementation).

    A few comments:

    * The time series must be in units of days.

    * The frequency sampling Hippke & Heller (2019) advocate for is cubic in
    frequencies, instead of linear. Ofir (2014) found that the
    linear-in-frequency sampling (which is correct for sinusoidal signal
    detection) isn't optimal for a Keplerian box signal. He gave an equation
    for "optimal" sampling. `tlsoversample` is the factor by which to
    oversample over that. The grid can be imported independently via::

        from transitleastsquares import period_grid

    The spacing equations are given here:
    https://transitleastsquares.readthedocs.io/en/latest/Python%20interface.html#period-grid

    * The boundaries of the period search are by default 0.1 day to 99% the
    baseline of times.

    Parameters
    ----------

    times,mags,errs : np.array
        The magnitude/flux time-series to search for transits.

    magsarefluxes : bool
        `transitleastsquares` requires fluxes. Therefore if magsarefluxes is
        set to false, the passed mags are converted to fluxes. All output
        dictionary vectors include fluxes, not mags.

    startp,endp : float
        The minimum and maximum periods to consider for the transit search.

    tlsoversample : int
        Factor by which to oversample the frequency grid.

    tlsmintransits : int
        Sets the `min_n_transits` kwarg for the `BoxLeastSquares.autoperiod()`
        function.

    tls_transit_template: str
        `default`, `grazing`, or `box`.

    tls_R_star_min,tls_R_star_max,tls_M_star_min,tls_M_star_max : float
        The range of stellar values used to create the frequency grid.

    periodepsilon : float
        The fractional difference between successive values of 'best' periods
        when sorting by periodogram power to consider them as separate periods
        (as opposed to part of the same periodogram peak). This is used to avoid
        broad peaks in the periodogram and make sure the 'best' periods returned
        are all actually independent.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

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

    verbose : bool
        Kept for consistency with `periodbase` functions.

    nworkers : int or None
        The number of parallel workers to launch for period-search. If None,
        nworkers = NCPUS.

    Returns
    -------

    dict
        This function returns a dict, referred to as an `lspinfo` dict in other
        astrobase functions that operate on periodogram results. The format is
        similar to the other astrobase period-finders -- it contains the
        nbestpeaks, which is the most important thing. (But isn't entirely
        standardized.)

        Crucially, it also contains "tlsresult", which is a dictionary with
        transitleastsquares spectra (used to get the SDE as defined in the TLS
        paper), statistics, transit period, mid-time, duration, depth, SNR, and
        the "odd_even_mismatch" statistic. The full key list is::

            dict_keys(['SDE', 'SDE_raw', 'chi2_min', 'chi2red_min', 'period',
            'period_uncertainty', 'T0', 'duration', 'depth', 'depth_mean',
            'depth_mean_even', 'depth_mean_odd', 'transit_depths',
            'transit_depths_uncertainties', 'rp_rs', 'snr', 'snr_per_transit',
            'snr_pink_per_transit', 'odd_even_mismatch', 'transit_times',
            'per_transit_count', 'transit_count', 'distinct_transit_count',
            'empty_transit_count', 'FAP', 'in_transit_count',
            'after_transit_count', 'before_transit_count', 'periods',
            'power', 'power_raw', 'SR', 'chi2',
            'chi2red', 'model_lightcurve_time', 'model_lightcurve_model',
            'model_folded_phase', 'folded_y', 'folded_dy', 'folded_phase',
            'model_folded_model'])

        The descriptions are here:

        https://transitleastsquares.readthedocs.io/en/latest/Python%20interface.html#return-values

        The remaining resultdict is::

            resultdict = {
                'tlsresult':tlsresult,
                'bestperiod': the best period value in the periodogram,
                'bestlspval': the peak associated with the best period,
                'nbestpeaks': the input value of nbestpeaks,
                'nbestlspvals': nbestpeaks-size list of best period peak values,
                'nbestperiods': nbestpeaks-size list of best periods,
                'lspvals': the full array of periodogram powers,
                'periods': the full array of periods considered,
                'tlsresult': Astropy tls result object (BoxLeastSquaresResult),
                'tlsmodel': Astropy tls BoxLeastSquares object used for work,
                'method':'tls' -> the name of the period-finder method,
                'kwargs':{ dict of all of the input kwargs for record-keeping}
            }
    """

    if not magsarefluxes:

        LOGWARNING('transitleastsquares requires relative flux...')
        LOGWARNING('converting input mags to relative flux...')
        LOGWARNING('and forcing magsarefluxes=True...')

        mag_0, f_0 = 12, 1e4
        flux = f_0 * 10**( -0.4 * (mags - mag_0) )
        flux /= np.nanmedian(flux)
        mags = flux
        magsarefluxes = True

    if nworkers is None:
        nworkers = NCPUS

    if errs is None:
        # uniform weights
        errs = np.ones_like(flux)*1e-4

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # make sure there are enough points to calculate a spectrum
    if not (len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9):
        LOGERROR('no good detections for these times and mags, skipping...')
        resultdict = {
            'tlsresult':npnan,
            'bestperiod':npnan,
            'bestlspval':npnan,
            'nbestpeaks':nbestpeaks,
            'nbestinds':None,
            'nbestlspvals':None,
            'nbestperiods':None,
            'lspvals':None,
            'periods':None,
            'method':'tls',
            'kwargs':{'startp':startp,
                      'endp':endp,
                      'tlsoversample':tlsoversample,
                      'tlsntransits':tlsmintransits,
                      'periodepsilon':periodepsilon,
                      'nbestpeaks':nbestpeaks,
                      'sigclip':sigclip,
                      'magsarefluxes':magsarefluxes}
        }
        return resultdict

    if endp is None:
        # out to 99% of the baseline. (for two transits).
        endp = 0.99*(np.nanmax(stimes) - np.nanmin(stimes))

    # run periodogram
    model = transitleastsquares(stimes, smags, serrs)
    tlsresult = model.power(use_threads=nworkers, show_progress_bar=False,
                            R_star_min=tls_rstar_min,
                            R_star_max=tls_rstar_max,
                            M_star_min=tls_mstar_min,
                            M_star_max=tls_mstar_max,
                            period_min=startp, period_max=endp,
                            n_transits_min=tlsmintransits,
                            transit_template=tls_transit_template,
                            oversampling_factor=tlsoversample)

    # get the peak values
    lsp = nparray(tlsresult.power)
    periods = nparray(tlsresult.periods)

    # find the nbestpeaks for the periodogram: 1. sort the lsp array by highest
    # value first 2. go down the values until we find five values that are
    # separated by at least periodepsilon in period make sure to get only the
    # finite peaks in the periodogram this is needed because tls may produce
    # infs for some peaks
    finitepeakind = npisfinite(lsp)
    finlsp = lsp[finitepeakind]
    finperiods = periods[finitepeakind]

    # make sure that finlsp has finite values before we work on it
    try:

        bestperiodind = npargmax(finlsp)

    except ValueError:

        LOGERROR('no finite periodogram values '
                 'for this mag series, skipping...')
        resultdict = {
            'tlsresult':npnan,
            'bestperiod':npnan,
            'bestlspval':npnan,
            'nbestpeaks':nbestpeaks,
            'nbestinds':None,
            'nbestlspvals':None,
            'nbestperiods':None,
            'lspvals':None,
            'periods':None,
            'method':'tls',
            'kwargs':{'startp':startp,
                      'endp':endp,
                      'tlsoversample':tlsoversample,
                      'tlsntransits':tlsmintransits,
                      'periodepsilon':periodepsilon,
                      'nbestpeaks':nbestpeaks,
                      'sigclip':sigclip,
                      'magsarefluxes':magsarefluxes}
        }
        return resultdict

    sortedlspind = npargsort(finlsp)[::-1]
    sortedlspperiods = finperiods[sortedlspind]
    sortedlspvals = finlsp[sortedlspind]

    # now get the nbestpeaks
    nbestperiods, nbestlspvals, nbestinds, peakcount = (
        [finperiods[bestperiodind]],
        [finlsp[bestperiodind]],
        [bestperiodind],
        1
    )
    prevperiod = sortedlspperiods[0]

    # find the best nbestpeaks in the lsp and their periods
    for period, lspval, ind in zip(sortedlspperiods,
                                   sortedlspvals,
                                   sortedlspind):

        if peakcount == nbestpeaks:
            break
        perioddiff = abs(period - prevperiod)
        bestperiodsdiff = [abs(period - x) for x in nbestperiods]

        # this ensures that this period is different from the last
        # period and from all the other existing best periods by
        # periodepsilon to make sure we jump to an entire different
        # peak in the periodogram
        if (perioddiff > (periodepsilon*prevperiod) and
            all(x > (periodepsilon*period)
                for x in bestperiodsdiff)):
            nbestperiods.append(period)
            nbestlspvals.append(lspval)
            nbestinds.append(ind)
            peakcount = peakcount + 1

        prevperiod = period

    # generate the return dict
    resultdict = {
        'tlsresult':tlsresult,
        'bestperiod':finperiods[bestperiodind],
        'bestlspval':finlsp[bestperiodind],
        'nbestpeaks':nbestpeaks,
        'nbestinds':nbestinds,
        'nbestlspvals':nbestlspvals,
        'nbestperiods':nbestperiods,
        'lspvals':lsp,
        'periods':periods,
        'method':'tls',
        'kwargs':{'startp':startp,
                  'endp':endp,
                  'tlsoversample':tlsoversample,
                  'tlsntransits':tlsmintransits,
                  'periodepsilon':periodepsilon,
                  'nbestpeaks':nbestpeaks,
                  'sigclip':sigclip,
                  'magsarefluxes':magsarefluxes}
    }

    return resultdict
