#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''features.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Calculates light curve features for variability classification.

'''

import logging
from datetime import datetime
from traceback import format_exc
from time import time as unixtime

from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, mean as npmean, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, all as npall, \
    correlate as npcorrelate, nonzero as npnonzero, diff as npdiff, exp as npexp

from scipy.stats import skew as spskew, kurtosis as spkurtosis
from scipy.signal import savgol_filter


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.varbase.features' % parent_name)

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

from ..lcmath import sigclip_magseries, \
    time_bin_magseries, time_bin_magseries_with_errs


##########################################
## BASE VARIABILITY FEATURE COMPUTATION ##
##########################################

def stetson_jindex(ftimes, fmags, ferrs, weightbytimediff=False):
    '''This calculates the Stetson index for the magseries, based on consecutive
    pairs of observations. Based on Nicole Loncke's work for her Planets and
    Life certificate at Princeton.

    This requires finite times, mags, and errs.

    If weightbytimediff is True, the Stetson index for any pair of mags will be
    reweighted by the difference in times between them using the scheme in
    Fruth+ 2012 and Zhange+ 2003 (as seen in Sokolovsky+ 2017).

    w_i = exp(- (t_i+1 - t_i)/ delta_t )

    '''

    ndet = len(fmags)

    if ndet > 9:

        # get the median and ndet
        medmag = npmedian(fmags)

        # get the stetson index elements
        delta_prefactor = (ndet/(ndet - 1))
        sigma_i = delta_prefactor*(fmags - medmag)/ferrs
        sigma_j = nproll(sigma_i,1) # Nicole's clever trick to advance indices
                                    # by 1 and do x_i*x_(i+1)

        if weightbytimediff:

            time_i = ftimes
            time_j = nproll(ftimes,1)
            difft = npdiff(ftimes)
            deltat = npmedian(difft)

            weights_i = npexp(- difft/deltat )
            products = (weights_i*sigma_i[1:]*sigma_j[1:])
        else:
            # ignore first elem since it's actually x_0*x_n
            products = (sigma_i*sigma_j)[1:]

        stetsonj = (
            npsum(npsign(products) * npsqrt(npabs(products)))
        ) / ndet

        return stetsonj

    else:

        LOGERROR('not enough detections in this magseries '
                 'to calculate stetson J index')
        return npnan



def stetson_kindex(fmags, ferrs):
    '''
    This calculates the Stetson K index (robust measure of the kurtosis).

    Requires finite mags and errs.

    '''

    # use a fill in value for the errors if they're none
    if ferrs is None:
        ferrs = npfull_like(mags, 0.005)

    ndet = len(fmags)

    if ndet > 9:

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



def lightcurve_moments(ftimes, fmags, ferrs):
    '''This calculates the weighted mean, stdev, median, MAD, percentiles, skew,
    kurtosis, fraction of LC beyond 1-stdev, and IQR.

    '''

    ndet = len(fmags)

    if ndet > 9:

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
        series_above1std = len(fmags[fmags > (series_median + series_stdev)])
        series_below1std = len(fmags[fmags < (series_median - series_stdev)])

        # this is the fraction beyond 1 stdev
        series_beyond1std = (series_above1std + series_below1std)/float(ndet)

        # get the magnitude percentiles
        series_mag_percentiles = nppercentile(
            fmags,
            [5.0,10,17.5,25,32.5,40,60,67.5,75,82.5,90,95]
        )

        return {
            'median':series_median,
            'wmean':series_wmean,
            'mad':series_mad,
            'stdev':series_stdev,
            'skew':series_skew,
            'kurtosis':series_kurtosis,
            'beyond1std':series_beyond1std,
            'mag_percentiles':series_mag_percentiles,
            'mag_iqr': series_mag_percentiles[8] - series_mag_percentiles[3],
        }


    else:
        LOGERROR('not enough detections in this magseries '
                 'to calculate light curve moments')
        return None



def lightcurve_flux_measures(ftimes, fmags, ferrs, magsarefluxes=False):
    '''
    This calculates percentiles of the flux.

    '''

    ndet = len(fmags)

    if ndet > 9:

        # get the fluxes
        if magsarefluxes:
            series_fluxes = fmags
        else:
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

        return {
            'flux_median':series_flux_median,
            'flux_percent_amplitude':series_flux_percent_amplitude,
            'flux_percentiles':series_flux_percentiles,
            'flux_percentile_ratio_mid20':series_flux_percentile_ratio_mid20,
            'flux_percentile_ratio_mid35':series_flux_percentile_ratio_mid35,
            'flux_percentile_ratio_mid50':series_flux_percentile_ratio_mid50,
            'flux_percentile_ratio_mid65':series_flux_percentile_ratio_mid65,
            'flux_percentile_ratio_mid80':series_flux_percentile_ratio_mid80,
            'percent_difference_flux_percentile':series_percentile_magdiff,
        }


    else:

        LOGERROR('not enough detections in this magseries '
                 'to calculate flux measures')
        return None



def lightcurve_ptp_measures(ftimes, fmags, ferrs):
    '''
    This calculates various point-to-point measures (eta in Kim+ 2014).

    '''

    ndet = len(fmags)

    if ndet > 9:

        timediffs = npdiff(ftimes)

        # get rid of stuff with time diff = 0.0
        nzind = npnonzero(timediffs)
        ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

        # recalculate ndet and diffs
        ndet = ftimes.size
        timediffs = npdiff(ftimes)

        # calculate the point to point measures
        p2p_abs_magdiffs = npabs(npdiff(fmags))
        p2p_squared_magdiffs = npdiff(fmags)*npdiff(fmags)

        robstd = npmedian(npabs(fmags - npmedian(fmags)))*1.483
        robvar = robstd*robstd

        # these are eta from the Kim+ 2014 paper - ratio of point-to-point
        # difference to the variance of the entire series

        # this is the robust version
        eta_robust = npmedian(p2p_abs_magdiffs)/robvar
        eta_robust = eta_robust/(ndet - 1.0)

        # this is the usual version
        eta_normal = npsum(p2p_squared_magdiffs)/npvar(fmags)
        eta_normal = eta_normal/(ndet - 1.0)


        timeweights = 1.0/(timediffs*timediffs)

        # this is eta_e modified for uneven sampling from the Kim+ 2014 paper
        eta_uneven_normal = (
            (npsum(timeweights*p2p_squared_magdiffs) /
             (npvar(fmags) * npsum(timeweights)) ) *
            npmean(timeweights) *
            (ftimes.max() - ftimes.min())*(ftimes.max() - ftimes.min())
        )

        # this is robust eta_e modified for uneven sampling from the Kim+ 2014
        # paper
        eta_uneven_robust = (
            (npsum(timeweights*p2p_abs_magdiffs) /
             (robvar * npsum(timeweights)) ) *
            npmedian(timeweights) *
            (ftimes[-1] - ftimes[0])*(ftimes[-1] - ftimes[0])
        )

        return {
            'eta_normal':eta_normal,
            'eta_robust':eta_robust,
            'eta_uneven_normal':eta_uneven_normal,
            'eta_uneven_robust':eta_uneven_robust
        }

    else:

        return None



def nonperiodic_lightcurve_features(times, mags, errs, magsarefluxes=False):
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

    # remove zero errors
    nzind = npnonzero(ferrs)
    ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

    ndet = len(fmags)

    if ndet > 9:

        # calculate the moments
        moments = lightcurve_moments(ftimes, fmags, ferrs)

        # calculate the flux measures
        fluxmeasures = lightcurve_flux_measures(ftimes, fmags, ferrs,
                                                magsarefluxes=magsarefluxes)

        # calculate the point-to-point measures
        ptpmeasures = lightcurve_ptp_measures(ftimes, fmags, ferrs)

        # get the length in time
        mintime, maxtime = npmin(ftimes), npmax(ftimes)
        timelength = maxtime - mintime

        # get the amplitude
        series_amplitude = 0.5*(npmax(fmags) - npmin(fmags))

        # calculate the linear fit to the entire mag series
        fitcoeffs = nppolyfit(ftimes, fmags, 1, w=1.0/(ferrs*ferrs))
        series_linear_slope = fitcoeffs[1]

        # roll fmags by 1
        rolled_fmags = nproll(fmags,1)

        # calculate the magnitude ratio (from the WISE paper)
        series_magratio = (
            (npmax(fmags) - moments['median']) / (npmax(fmags) - npmin(fmags) )
        )

        # this is the dictionary returned containing all the measures
        measures = {
            'ndet':fmags.size,
            'mintime':mintime,
            'maxtime':maxtime,
            'timelength':timelength,
            'amplitude':series_amplitude,
            'ndetobslength_ratio':ndet/timelength,
            'ndet':ndet,
            'linear_fit_slope':series_linear_slope,
            'magnitude_ratio':series_magratio,
        }
        if moments:
            measures.update(moments)
        if ptpmeasures:
            measures.update(ptpmeasures)
        if fluxmeasures:
            measures.update(fluxmeasures)

        return measures

    else:

        LOGERROR('not enough detections in this magseries '
                 'to calculate non-periodic features')
        return None



###############################################################
## KEPLER COMBINED DIFFERENTIAL PHOTOMETRIC PRECISION (CDPP) ##
###############################################################

def gilliland_cdpp(times, mags, errs,
                   windowlength=97,
                   polyorder=2,
                   binsize=23400, # in seconds: 6.5 hours for classic CDPP
                   sigclip=5.0,
                   magsarefluxes=False,
                   **kwargs):
    '''This calculates the CDPP of a timeseries using the method in the paper:

    Gilliland, R. L., Chaplin, W. J., Dunham, E. W., et al. 2011, ApJS, 197, 6
    http://adsabs.harvard.edu/abs/2011ApJS..197....6G

    use magsarefluxes to indicate if the values in the mags array are actually
    flux values.

    The steps are:

    - pass the time-series through a Savitsky-Golay filter

      - we use scipy.signal.savgol_filter, **kwargs are passed to this

      - also see: http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay

      - the windowlength is the number of LC points to use (Kepler uses 2 days =
        (1440 minutes/day / 30 minutes/LC point) x 2 days = 96 -> 97 LC points)

      - the polyorder is a quadratic by default

    - subtract the smoothed time-series from the actual light curve

    - sigma clip the remaining LC

    - get the binned mag series by averaging over 6.5 hour bins, only retaining
      bins with at least 7 points

    - the standard deviation of the binned averages is the CDPP

    - multiply this by 1.168 to correct for over-subtraction of white-noise

    '''

    # if no errs are given, assume 0.1% errors
    if errs is None:
        errs = 0.001*mags

    # get rid of nans first
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    if ftimes.size < (3*windowlength):
        LOGERROR('not enough LC points to calculate CDPP')
        return npnan

    # now get the smoothed mag series using the filter
    # kwargs are provided to the savgol_filter function
    smoothed = savgol_filter(fmags, windowlength, polyorder, **kwargs)
    subtracted = fmags - smoothed

    # sigclip the subtracted light curve
    stimes, smags, serrs = sigclip_magseries(ftimes, fmags, ferrs,
                                             magsarefluxes=magsarefluxes)

    # bin over 6.5 hour bins and throw away all bins with less than 7 elements
    binned = time_bin_magseries_with_errs(stimes, smags, serrs,
                                          binsize=binsize,
                                          minbinelems=7)

    btimes, bmags, berrs = (binned['binnedtimes'],
                            binned['binnedmags'],
                            binned['binnederrs'])

    # stdev of bin mags x 1.168 -> CDPP
    cdpp = npstd(bmags) * 1.168

    return cdpp



#####################
## ROLLUP FUNCTION ##
#####################


def all_nonperiodic_features(times, mags, errs,
                             magsarefluxes=False,
                             stetson_weightbytimediff=True):
    '''
    This rolls up the functions above and returns a single dict.

    NOTE: this doesn't calculate the CDPP; that's a separate function.

    '''

    # remove nans first
    finiteind = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[finiteind], mags[finiteind], errs[finiteind]

    # remove zero errors
    nzind = npnonzero(ferrs)
    ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

    xfeatures = nonperiodic_lightcurve_features(times, mags, errs,
                                                magsarefluxes=magsarefluxes)
    stetj = stetson_jindex(ftimes, fmags, ferrs,
                           weightbytimediff=stetson_weightbytimediff)
    stetk = stetson_kindex(fmags, ferrs)

    xfeatures.update({'stetsonj':stetj,
                      'stetsonk':stetk})

    return xfeatures



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
