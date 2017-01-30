#!/usr/bin/env python

'''autocorr.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Calculates the autocorrelation for magnitude time series.

'''


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


#####################
## AUTOCORRELATION ##
#####################


def _autocorr_func1(mags, lag, maglen, magmed, magstd):
    '''Calculates the autocorr of mag series for specific lag.

    mags MUST be an array with no nans.

    lag is the current lag to calculate the autocorr for. MUST be less than the
    total number of observations in mags (maglen).

    maglen, magmed, magstd are provided by auto_correlation below.

    This version of the function taken from:

    doi:10.1088/0004-637X/735/2/68 (Kim et al. 2011)

    '''

    lagindex = nparange(1,maglen-lag)
    products = (mags[lagindex] - magmed) * (mags[lagindex+lag] - magmed)
    acorr = (1.0/((maglen - lag)*magstd)) * npsum(products)

    return acorr



def _autocorr_func2(mags, lag, maglen, magmed, magstd):
    '''
    This is an alternative function to calculate the autocorrelation.

    mags MUST be an array with no nans.

    lag is the current lag to calculate the autocorr for. MUST be less than the
    total number of observations in mags (maglen).

    maglen, magmed, magstd are provided by auto_correlation below.

    This version is from (first definition):

    https://en.wikipedia.org/wiki/Correlogram#Estimation_of_autocorrelations

    '''

    lagindex = nparange(1,maglen-lag)
    products = (mags[lagindex] - magmed) * (mags[lagindex+lag] - magmed)

    autocovarfunc = npsum(products)/maglen
    varfunc = npsum((mags[lagindex] - magmed)*(mags[lagindex] - magmed))/maglen

    acorr = autocovarfunc/varfunc

    return acorr


def _autocorr_func3(mags, lag, maglen, magmed, magstd):
    '''
    This is yet another alternative to calculate the autocorrelation.

    Stolen from:

    http://nbviewer.jupyter.org/github/CamDavidsonPilon/
    Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/
    blob/master/Chapter3_MCMC/Chapter3.ipynb#Autocorrelation

    '''

    # from http://tinyurl.com/afz57c4
    result = npcorrelate(mags, mags, mode='full')
    result = result / npmax(result)
    return result[result.size / 2:]



def auto_correlation(mags, func=_autocorr_func1):
    '''Returns the auto correlation of the mag series as a function of the lag.

    Returns the lags array and the calculated associated autocorrelations.

    '''

    # remove all nans
    finiteind = npisfinite(mags)
    fmags = mags[finiteind]

    # calculate the median, MAD, and stdev
    series_median = npmedian(fmags)
    series_mad = npmedian(npabs(fmags - series_median))
    series_stdev = series_mad * 1.483

    lags = nparange(1,len(fmags))

    # get the autocorrelation as a function of the lag of the mag series
    autocorr = [func(fmags, x, len(fmags),
                     series_median, series_stdev) for x in lags]
    return lags, nparray(autocorr)
