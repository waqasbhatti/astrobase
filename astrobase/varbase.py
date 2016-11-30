#!/usr/bin/env python

'''varbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2016

Contains various useful tools for variability analysis. This is the base module,
with no dependencies on HAT specific light curve tools.

'''


import logging
from datetime import datetime
from traceback import format_exc
from time import time as unixtime

import os.path

try:
    import cPickle as pickle
except:
    import pickle

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

from scipy.stats import skew as spskew, kurtosis as spkurtosis
from scipy.optimize import leastsq as spleastsq, minimize as spminimize
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter

import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt



###################
## LOCAL IMPORTS ##
###################

from .periodbase import pgen_lsp


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.varbase' % parent_name)

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


##########################################
## BASE VARIABILITY FEATURE COMPUTATION ##
##########################################

def stetson_jindex(mags, errs):
    '''
    This calculates the Stetson index for the magseries, based on consecutive
    pairs of observations. Based on Nicole Loncke's work for her Planets and
    Life certificate at Princeton.

    if weights is None, all measurements will have equal weight.

    if weights is not None, then it's a ndarray of weights ranging in the
    interval [0,1] for each measurement in mags.

    '''

    # remove nans first
    finiteind = npisfinite(mags) & npisfinite(errs)
    fmags, ferrs = mags[finiteind], errs[finiteind]
    ndet = len(fmags)

    if ndet >= 10:

        # get the median and ndet
        medmag = npmedian(fmags)

        # get the stetson index elements
        delta_prefactor = (ndet/(ndet - 1))
        sigma_i = delta_prefactor*(fmags - medmag)/ferrs
        sigma_j = nproll(sigma_i,1) # Nicole's clever trick to advance indices
                                    # by 1 and do x_i*x_(i+1)
        products = (sigma_i*sigma_j)[1:] # ignore first elem since it's
                                         # actually x_0*x_n

        stetsonj = (
            npsum(npsign(products) * npsqrt(npabs(products)))
        ) / ndet

        return stetsonj

    else:

        LOGERROR('not enough detections in this magseries '
                 'to calculate stetson J index')
        return npnan


def stetson_kindex(mags, errs):
    '''
    This calculates the Stetson K index (robust measure of the kurtosis).

    '''

    # use a fill in value for the errors if they're none
    if errs is None:
        errs = npfull_like(mags, 0.005)

    # remove nans first
    finiteind = npisfinite(mags) & npisfinite(errs)
    fmags, ferrs = mags[finiteind], errs[finiteind]

    ndet = len(fmags)

    if ndet >= 10:

        # get the median and ndet
        medmag = npmedian(fmags)

        # get the stetson index elements
        delta_prefactor = (ndet/(ndet - 1))
        sigma_i = delta_prefactor*(fmags - medmag)/ferrs

        stetsonk = (
            npsum(npabs(sigma_i))/(npsqrt(npsum(sigma_i*sigma_i))) *
            (ndet**(-0.5))
        )

        return stetsonk

    else:

        LOGERROR('not enough detections in this magseries '
                 'to calculate stetson K index')
        return npnan



def nonperiodic_lightcurve_features(times, mags, errs):
    '''This calculates the following nonperiodic features of the light curve,
    listed in Richards, et al. 2011):

    amplitude
    beyond1std
    flux_percentile_ratio_mid20
    flux_percentile_ratio_mid35
    flux_percentile_ratio_mid50
    flux_percentile_ratio_mid65
    flux_percentile_ratio_mid80
    linear_trend
    max_slope
    median_absolute_deviation
    median_buffer_range_percentage
    pair_slope_trend
    percent_amplitude
    percent_difference_flux_percentile
    skew
    stdev
    timelength
    mintime
    maxtime

    '''

    # remove nans first
    finiteind = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[finiteind], mags[finiteind], errs[finiteind]
    ndet = len(fmags)

    if ndet >= 10:

        # get the length in time
        mintime, maxtime = npmin(times), npmax(times)
        timelength = maxtime - mintime

        # get the amplitude
        series_amplitude = 0.5*(npmax(fmags) - npmin(fmags))

        # now calculate the various things we need
        series_median = npmedian(fmags)
        series_wmean = (
            npsum(fmags*(1.0/(ferrs*ferrs)))/npsum(1.0/(ferrs*ferrs))
        )
        series_mad = npmedian(npabs(fmags - series_median))
        series_stdev = 1.483*series_mad
        series_skew = spskew(fmags)
        series_kurtosis = spkurtosis(fmags)

        # get the beyond1std fraction
        series_above1std = len(fmags[fmags > (fmags + series_stdev)])
        series_below1std = len(fmags[fmags < (fmags - series_stdev)])
        series_beyond1std = int((series_above1std + series_below1std)/float(ndet))

        # get the fluxes
        series_fluxes = 10.0**(-0.4*fmags)
        series_flux_median = npmedian(series_fluxes)

        # get the percent_amplitude for the fluxes
        series_flux_percent_amplitude = (
            npmax(npabs(series_fluxes))/series_flux_median
        )

        # get the flux percentiles
        series_flux_percentiles = nppercentile(
            series_fluxes,
            [5.0,10,17.5,25,32.5,40,60,67.5,75,82.5,90,95]
        )
        series_frat_595 = (
            series_flux_percentiles[-1] - series_flux_percentiles[0]
        )
        series_frat_1090 = (
            series_flux_percentiles[-2] - series_flux_percentiles[1]
        )
        series_frat_175825 = (
            series_flux_percentiles[-3] - series_flux_percentiles[2]
        )
        series_frat_2575 = (
            series_flux_percentiles[-4] - series_flux_percentiles[3]
        )
        series_frat_325675 = (
            series_flux_percentiles[-5] - series_flux_percentiles[4]
        )
        series_frat_4060 = (
            series_flux_percentiles[-6] - series_flux_percentiles[5]
        )

        # calculate the flux percentile ratios
        series_flux_percentile_ratio_mid20 = series_frat_4060/series_frat_595
        series_flux_percentile_ratio_mid35 = series_frat_325675/series_frat_595
        series_flux_percentile_ratio_mid50 = series_frat_2575/series_frat_595
        series_flux_percentile_ratio_mid65 = series_frat_175825/series_frat_595
        series_flux_percentile_ratio_mid80 = series_frat_1090/series_frat_595

        # calculate the ratio of F595/median flux
        series_percent_difference_flux_percentile = (
            series_frat_595/series_flux_median
        )
        series_percentile_magdiff = -2.5*nplog10(
            series_percent_difference_flux_percentile
        )

        # calculate the linear fit to the entire mag series
        fitcoeffs = nppolyfit(ftimes, fmags, 1, w=1.0/(ferrs*ferrs))
        series_linear_slope = fitcoeffs[1]

        # roll fmags by 1
        rolled_fmags = nproll(fmags,1)

        # calculate the point to point measures
        p2p_abs_magdiffs = npabs((rolled_fmags - fmags)[1:])
        p2p_squared_magdiffs = ((rolled_fmags - fmags)[1:])**2.0

        p2p_scatter_over_mad = npmedian(p2p_abs_magdiffs)/series_mad
        p2p_sqrdiff_over_var = npsum(p2p_squared_magdiffs)/npvar(fmags)

        # calculate the magnitude ratio (from the WISE paper)
        series_magratio = (
            (npmax(fmags) - series_median) / (npmax(fmags) - npmin(fmags) )
        )

        # calculate the abrupt-robust skew
        # (med(m) - med(m[0:p])) + (med(m) - med(m[p:1])), p chosen as 0.03
        # WTF is p supposed to be; stride?
        # FIXME: actually implement this later

        # this is the dictionary returned containing all the measures
        measures = {
            'mintime':mintime,
            'maxtime':maxtime,
            'timelength':timelength,
            'ndet':ndet,
            'median':series_median,
            'wmean':series_wmean,
            'mad':series_mad,
            'stdev':series_stdev,
            'amplitude':series_amplitude,
            'skew':series_skew,
            'kurtosis':series_kurtosis,
            'beyond1std':series_beyond1std,
            'flux_median':series_flux_median,
            'flux_percent_amplitude':series_flux_percent_amplitude,
            'flux_percentiles':series_flux_percentiles,
            'flux_percentile_ratio_mid20':series_flux_percentile_ratio_mid20,
            'flux_percentile_ratio_mid35':series_flux_percentile_ratio_mid35,
            'flux_percentile_ratio_mid50':series_flux_percentile_ratio_mid50,
            'flux_percentile_ratio_mid65':series_flux_percentile_ratio_mid65,
            'flux_percentile_ratio_mid80':series_flux_percentile_ratio_mid80,
            'percent_difference_flux_percentile':series_percentile_magdiff,
            'linear_fit_slope':series_linear_slope,
            'p2p_scatter_over_mad':p2p_scatter_over_mad,
            'p2p_sqrdiff_over_var':p2p_sqrdiff_over_var,
            'magnitude_ratio':series_magratio,
        }

        return measures

    else:

        LOGERROR('not enough detections in this magseries '
                 'to calculate non-periodic features')
        return None



def qso_variability_metrics(times, mags, errs, weights=None):
    '''
    This calculates the QSO variability and non-quasar variability metric.

    From Butler and Bloom (2011).

    FIXME: implement this

    '''

    # remove nans first
    finiteind = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[finiteind], mags[finiteind], errs[finiteind]
    ndet = len(fmags)

    if ndet >= 10:

        # get the amplitude
        amplitude = 0.5*(npmax(mags) - npmin(mags))


    else:

        LOGERROR('not enough detections in this magseries '
                 'to calculate QSO variability metrics')
        return None


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


#####################################################
## FOURIER FITTING TO PHASED MAGNITUDE TIME SERIES ##
#####################################################

def _fourier_func(fourierparams, phase, mags):
    '''
    This returns a summed Fourier series generated using fourierparams.

    fourierparams is a sequence like so:

    [ampl_1, ampl_2, ampl_3, ..., ampl_X, pha_1, pha_2, pha_3, ..., pha_X]

    where X is the Fourier order.

    mags and phase MUST have no nans.

    '''

    # figure out the order from the length of the Fourier param list
    order = len(fourierparams)/2

    # get the amplitude and phase coefficients
    f_amp = fourierparams[:order]
    f_pha = fourierparams[order:]

    # calculate all the individual terms of the series
    f_orders = [f_amp[x]*npcos(2.0*MPI*x*phase + f_pha[x])
                for x in range(order)]

    # this is the zeroth order coefficient - a constant equal to median mag
    total_f = npmedian(mags)

    # sum the series
    for fo in f_orders:
        total_f += fo

    return total_f



def _fourier_chisq(fourierparams,
                   phase,
                   mags,
                   errs):
    '''
    This is the chisq objective function to be minimized by scipy.minimize.

    The parameters are the same as _fourier_func above.

    '''

    f = _fourier_func(fourierparams, phase, mags)
    chisq = npsum(((mags - f)*(mags - f))/(errs*errs))

    return chisq



def _fourier_residual(fourierparams,
                      phase,
                      mags):
    '''
    This is the residual objective function to be minimized by scipy.leastsq.

    The parameters are the same as _fourier_func above.

    '''

    f = _fourier_func(fourierparams, phase, mags)
    residual = mags - f

    return residual



def fourier_fit_magseries(times, mags, errs, period,
                          initfourierparams=[0.6,0.2,0.2,0.2,0.2,0.2,0.2,0.2,
                                             0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                          sigclip=3.0,
                          plotfit=False,
                          ignoreinitfail=True):
    '''This fits a Fourier series to a magnitude time series.

    This uses an 8th-order Fourier series by default. This is good for light
    curves with many thousands of observations (HAT light curves have ~10k
    observations). Lower the order accordingly if you have less observations in
    your light curves to avoid over-fitting.

    Returns the Fourier fit parameters, the minimum chisq and reduced
    chisq. Makes a plot for the fit to the mag series if plotfit is a string
    containing a filename to write the plot to.

    This folds the time series using the given period and at the first
    observation. Can optionally sigma-clip observations.

    if ignoreinitfail is True, ignores the initial failure to find a set of
    optimized Fourier parameters and proceeds to do a least-squares fit anyway.

    '''

    # get rid of nans first
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next
    if sigclip:

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs

    # phase the mag series using the given period and faintest mag time
    # mintime = stimes[npwhere(smags == npmax(smags))]

    # phase the mag series using the given period and epoch = min(stimes)
    mintime = npmin(stimes)

    # calculate the unsorted phase, then sort it
    iphase = (stimes - mintime)/period - npfloor((stimes - mintime)/period)
    phasesortind = npargsort(iphase)

    # these are the final quantities to use for the Fourier fits
    phase = iphase[phasesortind]
    pmags = smags[phasesortind]
    perrs = serrs[phasesortind]

    # get the times sorted in phase order (useful to get the fit mag minimum
    # with respect to phase -- the light curve minimum)
    ptimes = stimes[phasesortind]

    fourierorder = len(initfourierparams)/2

    LOGINFO('fitting Fourier series of order %s to '
            'mag series with %s observations, '
            'using period %.6f, folded at %.6f' % (fourierorder,
                                                   len(phase),
                                                   period,
                                                   mintime))

    # initial minimize call to find global minimum in chi-sq
    initialfit = spminimize(_fourier_chisq,
                            initfourierparams,
                            method='BFGS',
                            args=(phase, pmags, perrs))

    # make sure this initial fit succeeds before proceeding
    if initialfit.success or ignoreinitfail:

        LOGINFO('initial fit done, refining...')

        leastsqparams = initialfit.x

        leastsqfit = spleastsq(_fourier_residual,
                               leastsqparams,
                               args=(phase, pmags))

        # if the fit succeeded, then we can return the final parameters
        if leastsqfit[-1] in (1,2,3,4):

            finalparams = leastsqfit[0]

            # calculate the chisq and reduced chisq
            fitmags = _fourier_func(finalparams, phase, pmags)

            fitchisq = npsum(
                ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
            )

            fitredchisq = fitchisq/(len(pmags) - len(finalparams) - 1)

            LOGINFO(
                'final fit done. chisq = %.5f, reduced chisq = %.5f' %
                (fitchisq,fitredchisq)
            )

            # figure out the time of light curve minimum (i.e. the fit epoch)
            # this is when the fit mag is maximum (i.e. the faintest)
            fitmagminind = npwhere(fitmags == npmax(fitmags))
            magseriesepoch = ptimes[fitmagminind]

            # assemble the returndict
            returndict =  {'finalparams':finalparams,
                           'initialfit':initialfit,
                           'leastsqfit':leastsqfit,
                           'fitchisq':fitchisq,
                           'fitredchisq':fitredchisq,
                           'fitplotfile':None,
                           'phase':phase,
                           'mags':pmags,
                           'errs':perrs,
                           'fitmags':fitmags,
                           'fitepoch':magseriesepoch}

            # make the fit plot if required
            if plotfit and isinstance(plotfit, str):

                plt.figure(figsize=(8,6))
                plt.axvline(0.5,color='g',linestyle='--')
                plt.errorbar(phase,pmags,fmt='bo',yerr=perrs,
                             markersize=2.0,capsize=0)
                plt.plot(phase,fitmags, 'r-',linewidth=2.0)
                ymin, ymax = plt.ylim()
                plt.ylim(ymax,ymin)
                plt.gca().set_xticks(
                    [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
                )
                plt.xlabel('phase')
                plt.ylabel('magnitude')
                plt.title('period: %.6f, folded at %.6f, fit epoch: %.6f' %
                          (period, mintime, magseriesepoch))
                plt.savefig(plotfit)
                plt.close()

                returndict['fitplotfile'] = plotfit

            return returndict


    # if the fit didn't succeed, we can't proceed
    else:

        LOGERROR('initial Fourier fit did not succeed, '
                 'reason: %s, returning scipy OptimizeResult'
                 % initialfit.message)

        return {'finalparams':None,
                'initialfit':initialfit,
                'leastsqfit':None,
                'fitchisq':None,
                'fitredchisq':None,
                'fitplotfile':None,
                'phase':phase,
                'mags':pmags,
                'errs':perrs,
                'fitmags':None,
                'fitepoch':None}


################################################
## REMOVING SIGNALS FROM MAGNITUDE TIMESERIES ##
################################################

def whiten_magseries(times, mags, errs,
                     whitenperiod,
                     whitenparams,
                     sigclip=3.0,
                     plotfit=None,
                     rescaletomedian=True):
    '''Removes a periodic signal generated using whitenparams from
    the input magnitude time series.

    whitenparams is like so:

    [ampl_1, ampl_2, ampl_3, ..., ampl_X, pha_1, pha_2, pha_3, ..., pha_X]

    where X is the Fourier order.

    if rescaletomedian is True, then we add back the constant median term of the
    magnitudes to the final whitened mag series.

    '''

    # get rid of nans first
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next
    if sigclip:

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs

    # phase the mag series using the given period and faintest mag time
    # mintime = stimes[npwhere(smags == npmax(smags))]

    # phase the mag series using the given period and epoch = min(stimes)
    mintime = npmin(stimes)

    # calculate the unsorted phase, then sort it
    iphase = (
        (stimes - mintime)/whitenperiod -
        npfloor((stimes - mintime)/whitenperiod)
    )
    phasesortind = npargsort(iphase)

    # these are the final quantities to use for the Fourier fits
    phase = iphase[phasesortind]
    pmags = smags[phasesortind]
    perrs = serrs[phasesortind]

    # get the times sorted in phase order (useful to get the fit mag minimum
    # with respect to phase -- the light curve minimum)
    ptimes = stimes[phasesortind]
    fourierorder = len(whitenparams)/2

    # now subtract the harmonic series from the phased LC
    # these are still in phase order
    wmags = pmags - _fourier_func(whitenparams, phase, pmags)

    # resort everything by time order
    wtimeorder = npargsort(ptimes)
    wtimes = ptimes[wtimeorder]
    wphase = phase[wtimeorder]
    wmags = wmags[wtimeorder]
    werrs = perrs[wtimeorder]

    if rescaletomedian:
        wmags = wmags + median_mag

    # prepare the returndict
    returndict = {'wtimes':wtimes, # these are in phase order
                  'wphase':wphase,
                  'wmags':wmags,
                  'werrs':werrs,
                  'whitenparams':whitenparams,
                  'whitenperiod':whitenperiod}


    # make the fit plot if required
    if plotfit and isinstance(plotfit, str):

        plt.figure(figsize=(8,16))

        plt.subplot(211)
        plt.errorbar(stimes,smags,fmt='bo',yerr=serrs,
                     markersize=2.0,capsize=0)
        ymin, ymax = plt.ylim()
        plt.ylim(ymax,ymin)
        plt.xlabel('JD')
        plt.ylabel('magnitude')
        plt.title('LC before whitening')

        plt.subplot(212)
        plt.errorbar(wtimes,wmags,fmt='bo',yerr=werrs,
                     markersize=2.0,capsize=0)
        ymin, ymax = plt.ylim()
        plt.ylim(ymax,ymin)
        plt.xlabel('JD')
        plt.ylabel('magnitude')
        plt.title('LC after whitening with period: %.6f' % whitenperiod)

        plt.savefig(plotfit)
        plt.close()

        returndict['fitplotfile'] = plotfit


    return returndict



def lsp_whiten(times, mags, errs, startp, endp,
               sigclip=30.0,
               stepsize=1.0e-4,
               initfparams=[0.6,0.2,0.2,0.1,0.1,0.1], # 3rd order series
               nbestpeaks=5,
               nworkers=4,
               plotfits=None):
    '''Iterative whitening using the LSP.

    This finds the best period, fits a fourier series with the best period, then
    whitens the time series with the best period, and repeats until nbestpeaks
    are done.

    '''

    # get rid of nans first
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next
    if sigclip:

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs

    # now start the cycle by doing an LSP on the initial timeseries
    lsp = pgen_lsp(stimes, smags, serrs, startp, endp,
                   sigclip=sigclip,
                   stepsize=stepsize,
                   nworkers=nworkers)
    wperiod = lsp['bestperiod']
    fseries = fourier_fit_magseries(stimes, smags, serrs, wperiod,
                                    initfourierparams=initfparams,
                                    sigclip=sigclip)
    ffitparams = fseries['finalparams']

    # this is the initial whitened series using the initial fourier fit and
    # initial found best period
    wseries = whiten_magseries(stimes,
                               smags,
                               serrs,
                               wperiod,
                               ffitparams,
                               sigclip=sigclip)

    LOGINFO('round %s: period = %.6f' % (1, wperiod))
    bestperiods = [wperiod]

    if plotfits and isinstance(plotfits, str):

        plt.figure(figsize=(8,6*nbestpeaks))

        nplots = nbestpeaks + 1

        plt.subplot(nplots,1,1)
        plt.errorbar(stimes,smags,fmt='bo',yerr=serrs,
                     markersize=2.0,capsize=0)
        ymin, ymax = plt.ylim()
        plt.ylim(ymax,ymin)
        plt.xlabel('JD')
        plt.ylabel('magnitude')
        plt.title('LC before whitening')

        plt.subplot(nplots,1,2)
        plt.errorbar(wseries['wtimes'],wseries['wmags'],
                     fmt='bo',yerr=wseries['werrs'],
                     markersize=2.0,capsize=0)
        ymin, ymax = plt.ylim()
        plt.ylim(ymax,ymin)
        plt.xlabel('JD')
        plt.ylabel('magnitude')
        plt.title('LC after whitening with period: %.6f' % wperiod)

    # now go through the rest of the cycles
    for fitind in range(nbestpeaks-1):

        wtimes, wmags, werrs = (wseries['wtimes'],
                                wseries['wmags'],
                                wseries['werrs'])

        wlsp = pgen_lsp(wtimes, wmags, werrs, startp, endp,
                        sigclip=sigclip,
                        stepsize=stepsize,
                        nworkers=nworkers)
        wperiod = wlsp['bestperiod']
        wfseries = fourier_fit_magseries(wtimes, wmags, werrs, wperiod,
                                         initfourierparams=initfparams,
                                         sigclip=sigclip)
        wffitparams = wfseries['finalparams']
        wseries = whiten_magseries(wtimes, wmags, werrs, wperiod, wffitparams,
                                   sigclip=sigclip)

        LOGINFO('round %s: period = %.6f' % (fitind+2, wperiod))
        bestperiods.append(wperiod)

        if plotfits and isinstance(plotfits, str):

            plt.subplot(nplots,1,fitind+3)
            plt.errorbar(wtimes,wmags,fmt='bo',yerr=werrs,
                         markersize=2.0,capsize=0)
            ymin, ymax = plt.ylim()
            plt.ylim(ymax,ymin)
            plt.xlabel('JD')
            plt.ylabel('magnitude')
            plt.title('LC after whitening with period: %.6f' % wperiod)


    if plotfits and isinstance(plotfits, str):

        plt.subplots_adjust(hspace=0.3,top=0.9)
        plt.savefig(plotfits,bbox_inches='tight')
        plt.close()
        return bestperiods, os.path.abspath(plotfits)

    else:

        return bestperiods


def mask_signal(times, mags, errs,
                signalperiod,
                signalepoch,
                maskphases=[0,0,0.5,1.0],
                maskphaselength=0.1,
                sigclip=30.0):
    '''This removes repeating signals in the magnitude time series.

    Useful for masking transit signals in light curves to search for other
    variability.

    '''

    # get rid of nans first
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next
    if sigclip:

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs


    # now phase the light curve using the period and epoch provided
    phases = (
        (stimes - signalepoch)/signalperiod -
        npfloor((stimes - signalepoch)/signalperiod)
    )

    # mask the requested phases using the mask length (in phase units)
    # this gets all the masks into one array
    masks = nparray([(npabs(phases - x) > maskphaselength)
                     for x in maskphases])
    # this flattens the masks to a single array for all combinations
    masks = npall(masks,axis=0)

    # apply the mask to the times, mags, and errs
    mphases = phases[masks]
    mtimes = stimes[masks]
    mmags = smags[masks]
    merrs = serrs[masks]

    return {'mphases':mphases,
            'mtimes':mtimes,
            'mmags':mmags,
            'merrs':merrs}


#################################################################
## SPLINE FITTING TO PHASED AND UNPHASED MAGNITUDE TIME SERIES ##
#################################################################

def spline_fit_magseries(times, mags, errs, period,
                         knotfraction=0.01,
                         maxknots=100,
                         sigclip=30.0,
                         plotfit=False,
                         ignoreinitfail=False):

    '''This fits a univariate cubic spline to the phased light curve.

    This fit may be better than the Fourier fit for sharply variable objects,
    like EBs, so can be used to distinguish them from other types of variables.

    The knot fraction is the number of internal knots to use for the spline. A
    value of 0.01 (or 1%) of the total number of non-nan observations appears to
    work quite well, without over-fitting.

    Returns the chisq of the fit, as well as the reduced chisq. FIXME: check
    this equation below to see if it's right.

    reduced_chisq = fit_chisq/(len(pmags) - len(knots) - 1)

    '''

    if errs is None:
        errs = npfull_like(mags, 0.005)

    # get rid of nans first
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next
    if sigclip:

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs

    # phase the mag series using the given period and faintest mag time
    # mintime = stimes[npwhere(smags == npmax(smags))]

    # phase the mag series using the given period and epoch = min(stimes)
    mintime = npmin(stimes)

    # calculate the unsorted phase, then sort it
    iphase = (stimes - mintime)/period - npfloor((stimes - mintime)/period)
    phasesortind = npargsort(iphase)

    # these are the final quantities to use for the Fourier fits
    phase = iphase[phasesortind]
    pmags = smags[phasesortind]
    perrs = serrs[phasesortind]

    # get the times sorted in phase order (useful to get the fit mag minimum
    # with respect to phase -- the light curve minimum)
    ptimes = stimes[phasesortind]

    # now figure out the number of knots up to max knots (=100)
    nobs = len(phase)
    nknots = int(npfloor(knotfraction*nobs))
    nknots = maxknots if nknots > maxknots else nknots
    splineknots = nplinspace(phase[0]+0.01,phase[-1]-0.01,num=nknots)

    # generate and fit the spline
    spl = LSQUnivariateSpline(phase,pmags,t=splineknots,w=1.0/perrs)

    # calculate the spline fit to the actual phases, the chisq and red-chisq
    fitmags = spl(phase)

    fitchisq = npsum(
        ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
    )

    fitredchisq = fitchisq/(len(pmags) - nknots - 1)

    LOGINFO(
        'spline fit done. nknots = %s,  chisq = %.5f, reduced chisq = %.5f' %
        (nknots, fitchisq, fitredchisq)
    )

    # figure out the time of light curve minimum (i.e. the fit epoch)
    # this is when the fit mag is maximum (i.e. the faintest)
    fitmagminind = npwhere(fitmags == npmax(fitmags))
    magseriesepoch = ptimes[fitmagminind]

    # assemble the returndict
    returndict =  {'nknots':nknots,
                   'fitchisq':fitchisq,
                   'fitredchisq':fitredchisq,
                   'fitplotfile':None,
                   'phase':phase,
                   'mags':pmags,
                   'errs':perrs,
                   'fitmags':fitmags,
                   'fitepoch':magseriesepoch}

    # make the fit plot if required
    if plotfit and isinstance(plotfit, str):

        plt.figure(figsize=(8,6))
        plt.axvline(0.5,color='g',linestyle='--')
        plt.errorbar(phase,pmags,fmt='bo',yerr=perrs,
                     markersize=2.0,capsize=0)
        plt.plot(phase,fitmags, 'r-',linewidth=2.0)
        ymin, ymax = plt.ylim()
        plt.ylim(ymax,ymin)
        plt.gca().set_xticks(
            [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        )
        plt.xlabel('phase')
        plt.ylabel('magnitude')
        plt.title('period: %.6f, folded at %.6f, fit epoch: %.6f' %
                  (period, mintime, magseriesepoch))
        plt.savefig(plotfit)
        plt.close()

        returndict['fitplotfile'] = plotfit

    return returndict



###############################################################
## KEPLER COMBINED DIFFERENTIAL PHOTOMETRIC PRECISION (CDPP) ##
###############################################################

def gilliland_cdpp(times, mags, errs,
                   windowlength=97,
                   polyorder=2,
                   sigclip=5.0,
                   **kwargs):
    '''This calculates the CDPP of a timeseries using the method in the paper:

    Gilliland, R. L., Chaplin, W. J., Dunham, E. W., et al. 2011, ApJS, 197, 6
    http://adsabs.harvard.edu/abs/2011ApJS..197....6G

    The steps are:

    - pass the time-series through a Savitsky-Golay filter
      - we use scipy.signal.savgol_filter, **kwargs are passed to this
      - also see: http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
      - the windowlength is the number of LC points to use (Kepler uses 2 days)
      - the polyorder is a quadratic by default

    - subtract the smoothed time-series from the actual light curve

    - sigma clip the remaining LC

    - get the binned mag series by averaging over 6.5 hour bins, only retaining
      bins with at least 7 points

    - the standard deviation of the binned averages is the CDPP

    - multiply this by 1.168 to correct for over-subtraction of white-noise

    '''

    if errs is None:
        errs = npfull_like(mags, 0.005)

    # get rid of nans first
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    # now get the smoothed mag series using the filter
    smoothed = savgol_filter(fmags, windowlength, polyorder)
    subtracted = fmags - smoothed

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next
    if sigclip:

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs

    # bin over 6.5 hour bins


    # throw away anything with less than 7 LC points in a bin


    # stdev of bin mags x 1.168 -> CDPP




###################################
## PERIODIC VARIABILITY FEATURES ##
###################################

# features to calculate

# freq_amplitude_ratio_21 - amp ratio of the 2nd to 1st Fourier component
# freq_amplitude_ratio_31 - amp ratio of the 3rd to 1st Fourier component
# freq_model_max_delta_mags - absval of magdiff btw model phased LC maxima
#                             using period x 2
# freq_model_max_delta_mags - absval of magdiff btw model phased LC minima
#                             using period x 2
# freq_model_phi1_phi2 - ratio of the phase difference between the first minimum
#                        and the first maximum to the phase difference between
#                        first minimum and second maximum
# freq_n_alias - number of top period estimates that are consistent with a 1
#                day period
# freq_rrd - 1 if freq_frequency_ratio_21 or freq_frequency_ratio_31 are close
#            to 0.746 (characteristic of RRc? -- double mode RRLyr), 0 otherwise
# scatter_res_raw - MAD of the GLS phased LC residuals divided by MAD of the raw
#                   light curve (unphased)
# p2p_scatter_2praw - sum of the squared mag differences between pairs of
#                     successive observations in the phased LC using best period
#                     x 2 divided by that of the unphased light curve
# p2p_scatter_pfold_over_mad - MAD of successive absolute mag diffs of the
#                              phased LC using best period divided by the MAD of
#                              the unphased LC
# medperc90_2p_p - 90th percentile of the absolute residual values around the
#                  light curve phased with best period x 2 divided by the same
#                  quantity for the residuals using the phased light curve with
#                  best period (to detect EBs)
# fold2P_slope_10percentile - 10th percentile of the slopes between adjacent
#                             mags after the light curve is folded on best
#                             period x 2
# fold2P_slope_90percentile - 90th percentile of the slopes between adjacent
#                             mags after the light curve is folded on best
#                             period x 2
# splchisq_over_fourierchisq - the ratio of the chi-squared value of a cubic
#                              spline fit to the phased LC with the best period
#                              to the chi-squared value of an 8th order Fourier
#                              fit to the phased LC with the best period -- this
#                              might be a good way to figure out if something
#                              looks like a detached EB vs a sinusoidal LC.

# in addition, we fit a Fourier series to the light curve using the best period
# and extract the amplitudes and phases up to the 8th order to fit the LC. the
# various ratios of the amplitudes A_ij and the differences in the phases phi_ij
# are also used as periodic variabilty features
