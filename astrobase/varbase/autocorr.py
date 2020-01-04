#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# autocorr.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''
Calculates the autocorrelation for magnitude time series.

'''

from numpy import (
    sum as npsum, arange as nparange, correlate as npcorrelate,
    max as npmax, median as npmedian, array as nparray, abs as npabs
)
from ..lcmath import fill_magseries_gaps


#####################
## AUTOCORRELATION ##
#####################

def _autocorr_func1(mags, lag, maglen, magmed, magstd):
    '''Calculates the autocorr of mag series for specific lag.

    This version of the function is taken from: Kim et al. (`2011
    <https://dx.doi.org/10.1088/0004-637X/735/2/68>`_)

    Parameters
    ----------

    mags : np.array
        This is the magnitudes array. MUST NOT have any nans.

    lag : float
        The specific lag value to calculate the auto-correlation for. This MUST
        be less than total number of observations in `mags`.

    maglen : int
        The number of elements in the `mags` array.

    magmed : float
        The median of the `mags` array.

    magstd : float
        The standard deviation of the `mags` array.

    Returns
    -------

    float
        The auto-correlation at this specific `lag` value.

    '''

    lagindex = nparange(1,maglen-lag)
    products = (mags[lagindex] - magmed) * (mags[lagindex+lag] - magmed)
    acorr = (1.0/((maglen - lag)*magstd)) * npsum(products)

    return acorr


def _autocorr_func2(mags, lag, maglen, magmed, magstd):
    '''
    This is an alternative function to calculate the autocorrelation.

    This version is from (first definition):

    https://en.wikipedia.org/wiki/Correlogram#Estimation_of_autocorrelations

    Parameters
    ----------

    mags : np.array
        This is the magnitudes array. MUST NOT have any nans.

    lag : float
        The specific lag value to calculate the auto-correlation for. This MUST
        be less than total number of observations in `mags`.

    maglen : int
        The number of elements in the `mags` array.

    magmed : float
        The median of the `mags` array.

    magstd : float
        The standard deviation of the `mags` array.

    Returns
    -------

    float
        The auto-correlation at this specific `lag` value.

    '''

    lagindex = nparange(0,maglen-lag)
    products = (mags[lagindex] - magmed) * (mags[lagindex+lag] - magmed)

    autocovarfunc = npsum(products)/lagindex.size
    varfunc = npsum(
        (mags[lagindex] - magmed)*(mags[lagindex] - magmed)
    )/mags.size

    acorr = autocovarfunc/varfunc

    return acorr


def _autocorr_func3(mags, lag, maglen, magmed, magstd):
    '''
    This is yet another alternative to calculate the autocorrelation.

    Taken from: `Bayesian Methods for Hackers by Cameron Pilon <http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/Chapter3.ipynb#Autocorrelation>`_

    (This should be the fastest method to calculate ACFs.)

    Parameters
    ----------

    mags : np.array
        This is the magnitudes array. MUST NOT have any nans.

    lag : float
        The specific lag value to calculate the auto-correlation for. This MUST
        be less than total number of observations in `mags`.

    maglen : int
        The number of elements in the `mags` array.

    magmed : float
        The median of the `mags` array.

    magstd : float
        The standard deviation of the `mags` array.

    Returns
    -------

    float
        The auto-correlation at this specific `lag` value.

    '''

    # from http://tinyurl.com/afz57c4
    result = npcorrelate(mags, mags, mode='full')
    result = result / npmax(result)

    return result[int(result.size / 2):]


def autocorr_magseries(times, mags, errs,
                       maxlags=1000,
                       func=_autocorr_func3,
                       fillgaps=0.0,
                       filterwindow=11,
                       forcetimebin=None,
                       sigclip=3.0,
                       magsarefluxes=False,
                       verbose=True):
    '''This calculates the ACF of a light curve.

    This will pre-process the light curve to fill in all the gaps and normalize
    everything to zero. If `fillgaps = 'noiselevel'`, fills the gaps with the
    noise level obtained via the procedure above. If `fillgaps = 'nan'`, fills
    the gaps with `np.nan`.

    Parameters
    ----------

    times,mags,errs : np.array
        The measurement time-series and associated errors.

    maxlags : int
        The maximum number of lags to calculate.

    func : Python function
        This is a function to calculate the lags.

    fillgaps : 'noiselevel' or float
        This sets what to use to fill in gaps in the time series. If this is
        'noiselevel', will smooth the light curve using a point window size of
        `filterwindow` (this should be an odd integer), subtract the smoothed LC
        from the actual LC and estimate the RMS. This RMS will be used to fill
        in the gaps. Other useful values here are 0.0, and npnan.

    filterwindow : int
        The light curve's smoothing filter window size to use if
        `fillgaps='noiselevel`'.

    forcetimebin : None or float
        This is used to force a particular cadence in the light curve other than
        the automatically determined cadence. This effectively rebins the light
        curve to this cadence. This should be in the same time units as `times`.

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

    magsarefluxes : bool
        If your input measurements in `mags` are actually fluxes instead of
        mags, set this is True.

    verbose : bool
        If True, will indicate progress and report errors.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'itimes': the interpolated time values after gap-filling,
             'imags': the interpolated mag/flux values after gap-filling,
             'ierrs': the interpolated mag/flux values after gap-filling,
             'cadence': the cadence of the output mag/flux time-series,
             'minitime': the minimum value of the interpolated times array,
             'lags': the lags used to calculate the auto-correlation function,
             'acf': the value of the ACF at each lag used}

    '''

    # get the gap-filled timeseries
    interpolated = fill_magseries_gaps(times, mags, errs,
                                       fillgaps=fillgaps,
                                       forcetimebin=forcetimebin,
                                       sigclip=sigclip,
                                       magsarefluxes=magsarefluxes,
                                       filterwindow=filterwindow,
                                       verbose=verbose)

    if not interpolated:
        print('failed to interpolate light curve to minimum cadence!')
        return None

    itimes, imags = interpolated['itimes'], interpolated['imags'],

    # calculate the lags up to maxlags
    if maxlags:
        lags = nparange(0, maxlags)
    else:
        lags = nparange(itimes.size)

    series_stdev = 1.483*npmedian(npabs(imags))

    if func != _autocorr_func3:

        # get the autocorrelation as a function of the lag of the mag series
        autocorr = nparray([func(imags, x, imags.size, 0.0, series_stdev)
                            for x in lags])

    # this doesn't need a lags array
    else:

        autocorr = _autocorr_func3(imags, lags[0], imags.size,
                                   0.0, series_stdev)
        # return only the maximum number of lags
        if maxlags is not None:
            autocorr = autocorr[:maxlags]

    interpolated.update({'minitime':itimes.min(),
                         'lags':lags,
                         'acf':autocorr})

    return interpolated
