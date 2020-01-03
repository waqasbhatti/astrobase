#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lcmath.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2015

'''
Contains various useful tools for calculating various things related to
lightcurves (like phasing, sigma-clipping, finding and filling gaps, etc.)

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
from numpy import (
    isfinite as npisfinite, median as npmedian, mean as npmean,
    abs as npabs, std as npstddev
)

from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
import scipy.stats


############################
## NORMALIZING MAG SERIES ##
############################

def find_lc_timegroups(lctimes, mingap=4.0):
    '''Finds gaps in the provided time-series and indexes them into groups.

    This finds the gaps in the provided `lctimes` array, so we can figure out
    which times are for consecutive observations and which represent gaps
    between seasons or observing eras.

    Parameters
    ----------

    lctimes : array-like
        This contains the times to analyze for gaps; assumed to be some form of
        Julian date.

    mingap : float
        This defines how much the difference between consecutive measurements is
        allowed to be to consider them as parts of different timegroups. By
        default it is set to 4.0 days.

    Returns
    -------

    tuple
        A tuple of the form: `(ngroups, [slice(start_ind_1, end_ind_1), ...])`
        is returned.  This contains the number of groups as the first element,
        and a list of Python `slice` objects for each time-group found. These
        can be used directly to index into the array of times to quickly get
        measurements associated with each group.

    '''

    lc_time_diffs = np.diff(lctimes)
    group_start_indices = np.where(lc_time_diffs > mingap)[0]

    if len(group_start_indices) > 0:

        group_indices = []

        for i, gindex in enumerate(group_start_indices):

            if i == 0:
                group_indices.append(slice(0,gindex+1))
            else:
                group_indices.append(slice(group_start_indices[i-1]+1,gindex+1))

        # at the end, add the slice for the last group to the end of the times
        # array
        group_indices.append(slice(group_start_indices[-1]+1,len(lctimes)))

    # if there's no large gap in the LC, then there's only one group to worry
    # about
    else:
        group_indices = [slice(0,len(lctimes))]

    return len(group_indices), group_indices


def normalize_magseries(times,
                        mags,
                        mingap=4.0,
                        normto='globalmedian',
                        magsarefluxes=False,
                        debugmode=False):
    '''This normalizes the magnitude time-series to a specified value.

    This is used to normalize time series measurements that may have large time
    gaps and vertical offsets in mag/flux measurement between these
    'timegroups', either due to instrument changes or different filters.

    NOTE: this works in-place! The mags array will be replaced with normalized
    mags when this function finishes.

    Parameters
    ----------

    times,mags : array-like
        The times (assumed to be some form of JD) and mags (or flux)
        measurements to be normalized.

    mingap : float
        This defines how much the difference between consecutive measurements is
        allowed to be to consider them as parts of different timegroups. By
        default it is set to 4.0 days.

    normto : {'globalmedian', 'zero'} or a float
        Specifies the normalization type::

          'globalmedian' -> norms each mag to the global median of the LC column
          'zero'         -> norms each mag to zero
          a float        -> norms each mag to this specified float value.

    magsarefluxes : bool
        Indicates if the input `mags` array is actually an array of flux
        measurements instead of magnitude measurements. If this is set to True,
        then:

        - if `normto` is 'zero', then the median flux is divided from each
          observation's flux value to yield normalized fluxes with 1.0 as the
          global median.

        - if `normto` is 'globalmedian', then the global median flux value
          across the entire time series is multiplied with each measurement.

        - if `norm` is set to a `float`, then this number is multiplied with the
          flux value for each measurement.

    debugmode : bool
        If this is True, will print out verbose info on each timegroup found.

    Returns
    -------

    times,normalized_mags : np.arrays
        Normalized magnitude values after normalization. If normalization fails
        for some reason, `times` and `normalized_mags` will both be None.

    '''

    ngroups, timegroups = find_lc_timegroups(times,
                                             mingap=mingap)

    # find all the non-nan indices
    finite_ind = np.isfinite(mags)

    if any(finite_ind):

        # find the global median
        global_mag_median = np.median(mags[finite_ind])

        # go through the groups and normalize them to the median for
        # each group
        for tgind, tg in enumerate(timegroups):

            finite_ind = np.isfinite(mags[tg])

            # find this timegroup's median mag and normalize the mags in
            # it to this median
            group_median = np.median((mags[tg])[finite_ind])

            if magsarefluxes:
                mags[tg] = mags[tg]/group_median
            else:
                mags[tg] = mags[tg] - group_median

            if debugmode:
                LOGDEBUG('group %s: elems %s, '
                         'finite elems %s, median mag %s' %
                         (tgind,
                          len(mags[tg]),
                          len(finite_ind),
                          group_median))

        # now that everything is normalized to 0.0, add the global median
        # offset back to all the mags and write the result back to the dict
        if isinstance(normto, str) and normto == 'globalmedian':

            if magsarefluxes:
                mags = mags * global_mag_median
            else:
                mags = mags + global_mag_median

        # if the normto is a float, add everything to that float and return
        elif isinstance(normto, float):

            if magsarefluxes:
                mags = mags * normto
            else:
                mags = mags + normto

        # anything else just returns the normalized mags as usual
        return times, mags

    else:
        LOGERROR('measurements are all nan!')
        return None, None


####################
## SIGMA-CLIPPING ##
####################

def sigclip_magseries(times, mags, errs,
                      sigclip=None,
                      iterative=False,
                      niterations=None,
                      meanormedian='median',
                      magsarefluxes=False):
    '''Sigma-clips a magnitude or flux time-series.

    Selects the finite times, magnitudes (or fluxes), and errors from the passed
    values, and apply symmetric or asymmetric sigma clipping to them.

    Parameters
    ----------

    times,mags,errs : np.array
        The magnitude or flux time-series arrays to sigma-clip. This doesn't
        assume all values are finite or if they're positive/negative. All of
        these arrays will have their non-finite elements removed, and then will
        be sigma-clipped based on the arguments to this function.

        `errs` is optional. Set it to None if you don't have values for these. A
        'faked' `errs` array will be generated if necessary, which can be
        ignored in the output as well.

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

    iterative : bool
        If this is set to True, will perform iterative sigma-clipping. If
        `niterations` is not set and this is True, sigma-clipping is iterated
        until no more points are removed.

    niterations : int
        The maximum number of iterations to perform for sigma-clipping. If None,
        the `iterative` arg takes precedence, and `iterative=True` will
        sigma-clip until no more points are removed.  If `niterations` is not
        None and `iterative` is False, `niterations` takes precedence and
        iteration will occur for the specified number of iterations.

    meanormedian : {'mean', 'median'}
        Use 'mean' for sigma-clipping based on the mean value, or 'median' for
        sigma-clipping based on the median value.  Default is 'median'.

    magsareflux : bool
        True if your "mags" are in fact fluxes, i.e. if "fainter" corresponds to
        `mags` getting smaller.

    Returns
    -------

    (stimes, smags, serrs) : tuple
        The sigma-clipped and nan-stripped time-series.

    '''

    returnerrs = True

    # fake the errors if they don't exist
    # this is inconsequential to sigma-clipping
    # we don't return these dummy values if the input errs are None
    if errs is None:
        # assume 0.1% errors if not given
        # this should work for mags and fluxes
        errs = 0.001*mags
        returnerrs = False

    # filter the input times, mags, errs; do sigclipping and normalization
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[find], mags[find], errs[find]

    # get the center value and stdev
    if meanormedian == 'median':  # stddev = 1.483 x MAD

        center_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - center_mag))) * 1.483

    elif meanormedian == 'mean':

        center_mag = npmean(fmags)
        stddev_mag = npstddev(fmags)

    else:
        LOGWARNING("unrecognized meanormedian value given to "
                   "sigclip_magseries: %s, defaulting to 'median'" %
                   meanormedian)
        meanormedian = 'median'
        center_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - center_mag))) * 1.483

    # sigclip next for a single sigclip value
    if sigclip and isinstance(sigclip, (float, int)):

        if not iterative and niterations is None:

            sigind = (npabs(fmags - center_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

        else:

            #
            # iterative version adapted from scipy.stats.sigmaclip
            #

            # First, if niterations is not set, iterate until covergence
            if niterations is None:

                delta = 1

                this_times = ftimes
                this_mags = fmags
                this_errs = ferrs

                while delta:

                    if meanormedian == 'mean':
                        this_center = npmean(this_mags)
                        this_stdev = npstddev(this_mags)
                    elif meanormedian == 'median':
                        this_center = npmedian(this_mags)
                        this_stdev = (
                            npmedian(npabs(this_mags - this_center))
                        ) * 1.483
                    this_size = this_mags.size

                    # apply the sigclip
                    tsi = (
                        (npabs(this_mags - this_center)) <
                        (sigclip * this_stdev)
                    )

                    # update the arrays
                    this_times = this_times[tsi]
                    this_mags = this_mags[tsi]
                    this_errs = this_errs[tsi]

                    # update delta and go to the top of the loop
                    delta = this_size - this_mags.size

            else:  # If iterating only a certain number of times

                this_times = ftimes
                this_mags = fmags
                this_errs = ferrs

                iter_num = 0
                delta = 1
                while iter_num < niterations and delta:

                    if meanormedian == 'mean':

                        this_center = npmean(this_mags)
                        this_stdev = npstddev(this_mags)

                    elif meanormedian == 'median':

                        this_center = npmedian(this_mags)
                        this_stdev = (npmedian(npabs(this_mags -
                                                     this_center))) * 1.483
                    this_size = this_mags.size

                    # apply the sigclip
                    tsi = (
                        (npabs(this_mags - this_center)) <
                        (sigclip * this_stdev)
                    )

                    # update the arrays
                    this_times = this_times[tsi]
                    this_mags = this_mags[tsi]
                    this_errs = this_errs[tsi]

                    # update the number of iterations and delta and
                    # go to the top of the loop
                    delta = this_size - this_mags.size
                    iter_num += 1

            # final sigclipped versions
            stimes, smags, serrs = this_times, this_mags, this_errs

    # this handles sigclipping for asymmetric +ve and -ve clip values
    elif sigclip and isinstance(sigclip, (list,tuple)) and len(sigclip) == 2:

        # sigclip is passed as [dimmingclip, brighteningclip]
        dimmingclip = sigclip[0]
        brighteningclip = sigclip[1]

        if not iterative and niterations is None:

            if magsarefluxes:
                nottoodimind = (
                    (fmags - center_mag) > (-dimmingclip*stddev_mag)
                )
                nottoobrightind = (
                    (fmags - center_mag) < (brighteningclip*stddev_mag)
                )
            else:
                nottoodimind = (
                    (fmags - center_mag) < (dimmingclip*stddev_mag)
                )
                nottoobrightind = (
                    (fmags - center_mag) > (-brighteningclip*stddev_mag)
                )

            sigind = nottoodimind & nottoobrightind

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

        else:

            #
            # iterative version adapted from scipy.stats.sigmaclip
            #
            if niterations is None:

                delta = 1

                this_times = ftimes
                this_mags = fmags
                this_errs = ferrs

                while delta:

                    if meanormedian == 'mean':

                        this_center = npmean(this_mags)
                        this_stdev = npstddev(this_mags)

                    elif meanormedian == 'median':
                        this_center = npmedian(this_mags)
                        this_stdev = (npmedian(npabs(this_mags -
                                                     this_center))) * 1.483
                    this_size = this_mags.size

                    if magsarefluxes:
                        nottoodimind = (
                            (this_mags - this_center) >
                            (-dimmingclip*this_stdev)
                        )
                        nottoobrightind = (
                            (this_mags - this_center) <
                            (brighteningclip*this_stdev)
                        )
                    else:
                        nottoodimind = (
                            (this_mags - this_center) <
                            (dimmingclip*this_stdev)
                        )
                        nottoobrightind = (
                            (this_mags - this_center) >
                            (-brighteningclip*this_stdev)
                        )

                    # apply the sigclip
                    tsi = nottoodimind & nottoobrightind

                    # update the arrays
                    this_times = this_times[tsi]
                    this_mags = this_mags[tsi]
                    this_errs = this_errs[tsi]

                    # update delta and go to top of the loop
                    delta = this_size - this_mags.size

            else:  # If iterating only a certain number of times
                this_times = ftimes
                this_mags = fmags
                this_errs = ferrs

                iter_num = 0
                delta = 1

                while iter_num < niterations and delta:

                    if meanormedian == 'mean':
                        this_center = npmean(this_mags)
                        this_stdev = npstddev(this_mags)
                    elif meanormedian == 'median':
                        this_center = npmedian(this_mags)
                        this_stdev = (npmedian(npabs(this_mags -
                                                     this_center))) * 1.483
                    this_size = this_mags.size

                    if magsarefluxes:
                        nottoodimind = (
                            (this_mags - this_center) >
                            (-dimmingclip*this_stdev)
                        )
                        nottoobrightind = (
                            (this_mags - this_center) <
                            (brighteningclip*this_stdev)
                        )
                    else:
                        nottoodimind = (
                            (this_mags - this_center) < (dimmingclip*this_stdev)
                        )
                        nottoobrightind = (
                            (this_mags - this_center) >
                            (-brighteningclip*this_stdev)
                        )

                    # apply the sigclip
                    tsi = nottoodimind & nottoobrightind

                    # update the arrays
                    this_times = this_times[tsi]
                    this_mags = this_mags[tsi]
                    this_errs = this_errs[tsi]

                    # update the number of iterations and delta
                    # and go to top of the loop
                    delta = this_size - this_mags.size
                    iter_num += 1

            # final sigclipped versions
            stimes, smags, serrs = this_times, this_mags, this_errs

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs

    if returnerrs:
        return stimes, smags, serrs
    else:
        return stimes, smags, None


def sigclip_magseries_with_extparams(times, mags, errs, extparams,
                                     sigclip=None,
                                     iterative=False,
                                     magsarefluxes=False):
    '''Sigma-clips a magnitude or flux time-series and associated measurement
    arrays.

    Selects the finite times, magnitudes (or fluxes), and errors from the passed
    values, and apply symmetric or asymmetric sigma clipping to them.  Uses the
    same array indices as these values to filter out the values of all arrays in
    the `extparams` list. This can be useful for simultaneously sigma-clipping a
    magnitude/flux time-series along with their associated values of external
    parameters, such as telescope hour angle, zenith distance, temperature, moon
    phase, etc.

    Parameters
    ----------

    times,mags,errs : np.array
        The magnitude or flux time-series arrays to sigma-clip. This doesn't
        assume all values are finite or if they're positive/negative. All of
        these arrays will have their non-finite elements removed, and then will
        be sigma-clipped based on the arguments to this function.

        `errs` is optional. Set it to None if you don't have values for these. A
        'faked' `errs` array will be generated if necessary, which can be
        ignored in the output as well.

    extparams : list of np.array
        This is a list of all external parameter arrays to simultaneously filter
        along with the magnitude/flux time-series. All of these arrays should
        have the same length as the `times`, `mags`, and `errs` arrays.

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

    iterative : bool
        If this is set to True, will perform iterative sigma-clipping. If
        `niterations` is not set and this is True, sigma-clipping is iterated
        until no more points are removed.

    magsareflux : bool
        True if your "mags" are in fact fluxes, i.e. if "fainter" corresponds to
        `mags` getting smaller.

    Returns
    -------

    (stimes, smags, serrs) : tuple
        The sigma-clipped and nan-stripped time-series in `stimes`, `smags`,
        `serrs` and the associated values of the `extparams` in `sextparams`.

    '''

    returnerrs = True

    # fake the errors if they don't exist
    # this is inconsequential to sigma-clipping
    # we don't return these dummy values if the input errs are None
    if errs is None:
        # assume 0.1% errors if not given
        # this should work for mags and fluxes
        errs = 0.001*mags
        returnerrs = False

    # filter the input times, mags, errs; do sigclipping and normalization
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[find], mags[find], errs[find]

    # apply the same indices to the external parameters
    for epi, eparr in enumerate(extparams):
        extparams[epi] = eparr[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next for a single sigclip value
    if sigclip and isinstance(sigclip, (float, int)):

        if not iterative:

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

            # apply the same indices to the external parameters
            for epi, eparr in enumerate(extparams):
                extparams[epi] = eparr[sigind]

        else:

            #
            # iterative version adapted from scipy.stats.sigmaclip
            #
            delta = 1

            this_times = ftimes
            this_mags = fmags
            this_errs = ferrs

            while delta:

                this_median = npmedian(this_mags)
                this_stdev = (npmedian(npabs(this_mags - this_median))) * 1.483
                this_size = this_mags.size

                # apply the sigclip
                tsi = (npabs(this_mags - this_median)) < (sigclip * this_stdev)

                # update the arrays
                this_times = this_times[tsi]
                this_mags = this_mags[tsi]
                this_errs = this_errs[tsi]

                # apply the same indices to the external parameters
                for epi, eparr in enumerate(extparams):
                    extparams[epi] = eparr[tsi]

                # update delta and go to the top of the loop
                delta = this_size - this_mags.size

            # final sigclipped versions
            stimes, smags, serrs = this_times, this_mags, this_errs

    # this handles sigclipping for asymmetric +ve and -ve clip values
    elif sigclip and isinstance(sigclip, (list, tuple)) and len(sigclip) == 2:

        # sigclip is passed as [dimmingclip, brighteningclip]
        dimmingclip = sigclip[0]
        brighteningclip = sigclip[1]

        if not iterative:

            if magsarefluxes:
                nottoodimind = (
                    (fmags - median_mag) > (-dimmingclip*stddev_mag)
                )
                nottoobrightind = (
                    (fmags - median_mag) < (brighteningclip*stddev_mag)
                )
            else:
                nottoodimind = (
                    (fmags - median_mag) < (dimmingclip*stddev_mag)
                )
                nottoobrightind = (
                    (fmags - median_mag) > (-brighteningclip*stddev_mag)
                )

            sigind = nottoodimind & nottoobrightind

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

            # apply the same indices to the external parameters
            for epi, eparr in enumerate(extparams):
                extparams[epi] = eparr[sigind]

        else:

            #
            # iterative version adapted from scipy.stats.sigmaclip
            #
            delta = 1

            this_times = ftimes
            this_mags = fmags
            this_errs = ferrs

            while delta:

                this_median = npmedian(this_mags)
                this_stdev = (npmedian(npabs(this_mags - this_median))) * 1.483
                this_size = this_mags.size

                if magsarefluxes:
                    nottoodimind = (
                        (this_mags - this_median) > (-dimmingclip*this_stdev)
                    )
                    nottoobrightind = (
                        (this_mags - this_median) < (brighteningclip*this_stdev)
                    )
                else:
                    nottoodimind = (
                        (this_mags - this_median) < (dimmingclip*this_stdev)
                    )
                    nottoobrightind = (
                        (this_mags - this_median) >
                        (-brighteningclip*this_stdev)
                    )

                # apply the sigclip
                tsi = nottoodimind & nottoobrightind

                # update the arrays
                this_times = this_times[tsi]
                this_mags = this_mags[tsi]
                this_errs = this_errs[tsi]

                # apply the same indices to the external parameters
                for epi, eparr in enumerate(extparams):
                    extparams[epi] = eparr[tsi]

                # update delta and go to top of the loop
                delta = this_size - this_mags.size

            # final sigclipped versions
            stimes, smags, serrs = this_times, this_mags, this_errs

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs

    if returnerrs:
        return stimes, smags, serrs, extparams
    else:
        return stimes, smags, None, extparams


#################
## PHASING LCS ##
#################

def phase_magseries(times, mags, period, epoch, wrap=True, sort=True):
    '''Phases a magnitude/flux time-series using a given period and epoch.

    The equation used is::

        phase = (times - epoch)/period - floor((times - epoch)/period)

    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.

    Parameters
    ----------

    times,mags : np.array
        The magnitude/flux time-series values to phase using the provided
        `period` and `epoch`. Non-fiinite values will be removed.

    period : float
        The period to use to phase the time-series.

    epoch : float
        The epoch to phase the time-series. This is usually the time-of-minimum
        or time-of-maximum of some periodic light curve
        phenomenon. Alternatively, one can use the minimum time value in
        `times`.

    wrap : bool
        If this is True, the returned phased time-series will be wrapped around
        phase 0.0, which is useful for plotting purposes. The arrays returned
        will have twice the number of input elements because of this wrapping.

    sort : bool
        If this is True, the returned phased time-series will be sorted in
        increasing phase order.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'phase': the phase values,
             'mags': the mags/flux values at each phase,
             'period': the input `period` used to phase the time-series,
             'epoch': the input `epoch` used to phase the time-series}

    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags) & np.isfinite(times)

    finite_times = times[finiteind]
    finite_mags = mags[finiteind]

    magseries_phase = (
        (finite_times - epoch)/period -
        np.floor(((finite_times - epoch)/period))
    )

    outdict = {'phase':magseries_phase,
               'mags':finite_mags,
               'period':period,
               'epoch':epoch}

    if sort:
        sortorder = np.argsort(outdict['phase'])
        outdict['phase'] = outdict['phase'][sortorder]
        outdict['mags'] = outdict['mags'][sortorder]

    if wrap:
        outdict['phase'] = np.concatenate((outdict['phase']-1.0,
                                           outdict['phase']))
        outdict['mags'] = np.concatenate((outdict['mags'],
                                          outdict['mags']))

    return outdict


def phase_magseries_with_errs(times, mags, errs, period, epoch,
                              wrap=True, sort=True):
    '''Phases a magnitude/flux time-series using a given period and epoch.

    The equation used is::

        phase = (times - epoch)/period - floor((times - epoch)/period)

    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.

    Parameters
    ----------

    times,mags,errs : np.array
        The magnitude/flux time-series values and associated measurement errors
        to phase using the provided `period` and `epoch`. Non-fiinite values
        will be removed.

    period : float
        The period to use to phase the time-series.

    epoch : float
        The epoch to phase the time-series. This is usually the time-of-minimum
        or time-of-maximum of some periodic light curve
        phenomenon. Alternatively, one can use the minimum time value in
        `times`.

    wrap : bool
        If this is True, the returned phased time-series will be wrapped around
        phase 0.0, which is useful for plotting purposes. The arrays returned
        will have twice the number of input elements because of this wrapping.

    sort : bool
        If this is True, the returned phased time-series will be sorted in
        increasing phase order.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'phase': the phase values,
             'mags': the mags/flux values at each phase,
             'errs': the err values at each phase,
             'period': the input `period` used to phase the time-series,
             'epoch': the input `epoch` used to phase the time-series}

    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]
    finite_errs = errs[finiteind]

    magseries_phase = (
        (finite_times - epoch)/period -
        np.floor(((finite_times - epoch)/period))
    )

    outdict = {'phase':magseries_phase,
               'mags':finite_mags,
               'errs':finite_errs,
               'period':period,
               'epoch':epoch}

    if sort:
        sortorder = np.argsort(outdict['phase'])
        outdict['phase'] = outdict['phase'][sortorder]
        outdict['mags'] = outdict['mags'][sortorder]
        outdict['errs'] = outdict['errs'][sortorder]

    if wrap:
        outdict['phase'] = np.concatenate((outdict['phase']-1.0,
                                           outdict['phase']))
        outdict['mags'] = np.concatenate((outdict['mags'],
                                          outdict['mags']))
        outdict['errs'] = np.concatenate((outdict['errs'],
                                          outdict['errs']))

    return outdict


#################
## BINNING LCs ##
#################

def time_bin_magseries(times, mags,
                       binsize=540.0,
                       minbinelems=7):
    '''Bins the given mag/flux time-series in time using the bin size given.

    Parameters
    ----------

    times,mags : np.array
        The magnitude/flux time-series to bin in time. Non-finite elements will
        be removed from these arrays. At least 10 elements in each array are
        required for this function to operate.

    binsize : float
        The bin size to use to group together measurements closer than this
        amount in time. This is in seconds.

    minbinelems : int
        The minimum number of elements required per bin to include it in the
        output.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'jdbin_indices': a list of the index arrays into the nan-filtered
                              input arrays per each bin,
             'jdbins': list of bin boundaries for each bin,
             'nbins': the number of bins generated,
             'binnedtimes': the time values associated with each time bin;
                            this is the median of the times in each bin,
             'binnedmags': the mag/flux values associated with each time bin;
                           this is the median of the mags/fluxes in each bin}

    '''

    # check if the input arrays are ok
    if not(times.shape and mags.shape and len(times) > 9 and len(mags) > 9):

        LOGERROR("input time/mag arrays don't have enough elements")
        return

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags) & np.isfinite(times)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]

    # convert binsize in seconds to JD units
    binsizejd = binsize/(86400.0)
    nbins = int(np.ceil((np.nanmax(finite_times) -
                         np.nanmin(finite_times))/binsizejd) + 1)

    minjd = np.nanmin(finite_times)
    jdbins = [(minjd + x*binsizejd) for x in range(nbins)]

    # make a KD-tree on the JDs so we can do fast distance calculations.  we
    # need to add a bogus y coord to make this a problem that KD-trees can
    # solve.
    time_coords = np.array([[x,1.0] for x in finite_times])
    jdtree = cKDTree(time_coords)
    binned_finite_timeseries_indices = []

    collected_binned_mags = {}

    for jd in jdbins:
        # find all bin indices close to within binsizejd of this point
        # using the cKDTree query. we use the p-norm = 1 (I think this
        # means straight-up pairwise distance? FIXME: check this)
        bin_indices = jdtree.query_ball_point(np.array([jd,1.0]),
                                              binsizejd/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_timeseries_indices and
            len(bin_indices) >= minbinelems):

            binned_finite_timeseries_indices.append(bin_indices)

    # convert to ndarrays
    binned_finite_timeseries_indices = [np.array(x) for x in
                                        binned_finite_timeseries_indices]

    collected_binned_mags['jdbins_indices'] = binned_finite_timeseries_indices
    collected_binned_mags['jdbins'] = jdbins
    collected_binned_mags['nbins'] = len(binned_finite_timeseries_indices)

    # collect the finite_times
    binned_jd = np.array([np.median(finite_times[x])
                          for x in binned_finite_timeseries_indices])
    collected_binned_mags['binnedtimes'] = binned_jd
    collected_binned_mags['binsize'] = binsize

    # median bin the magnitudes according to the calculated indices
    collected_binned_mags['binnedmags'] = (
        np.array([np.median(finite_mags[x])
                  for x in binned_finite_timeseries_indices])
    )

    return collected_binned_mags


def time_bin_magseries_with_errs(times, mags, errs,
                                 binsize=540.0,
                                 minbinelems=7):
    '''Bins the given mag/flux time-series in time using the bin size given.

    Parameters
    ----------

    times,mags,errs : np.array
        The magnitude/flux time-series and associated measurement errors to bin
        in time. Non-finite elements will be removed from these arrays. At least
        10 elements in each array are required for this function to operate.

    binsize : float
        The bin size to use to group together measurements closer than this
        amount in time. This is in seconds.

    minbinelems : int
        The minimum number of elements required per bin to include it in the
        output.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'jdbin_indices': a list of the index arrays into the nan-filtered
                              input arrays per each bin,
             'jdbins': list of bin boundaries for each bin,
             'nbins': the number of bins generated,
             'binnedtimes': the time values associated with each time bin;
                            this is the median of the times in each bin,
             'binnedmags': the mag/flux values associated with each time bin;
                           this is the median of the mags/fluxes in each bin,
             'binnederrs': the err values associated with each time bin;
                           this is the median of the errs in each bin}

    '''

    # check if the input arrays are ok
    if not(times.shape and mags.shape and errs.shape and
           len(times) > 9 and len(mags) > 9):

        LOGERROR("input time/mag/err arrays don't have enough elements")
        return

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags) & np.isfinite(times) & np.isfinite(errs)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]
    finite_errs = errs[finiteind]

    # convert binsize in seconds to JD units
    binsizejd = binsize/(86400.0)
    nbins = int(np.ceil((np.nanmax(finite_times) -
                         np.nanmin(finite_times))/binsizejd) + 1)

    minjd = np.nanmin(finite_times)
    jdbins = [(minjd + x*binsizejd) for x in range(nbins)]

    # make a KD-tree on the JDs so we can do fast distance calculations.  we
    # need to add a bogus y coord to make this a problem that KD-trees can
    # solve.
    time_coords = np.array([[x,1.0] for x in finite_times])
    jdtree = cKDTree(time_coords)
    binned_finite_timeseries_indices = []

    collected_binned_mags = {}

    for jd in jdbins:

        # find all bin indices close to within binsize of this point using the
        # cKDTree query. we use the p-norm = 1 for pairwise Euclidean distance.
        bin_indices = jdtree.query_ball_point(np.array([jd,1.0]),
                                              binsizejd/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_timeseries_indices and
            len(bin_indices) >= minbinelems):

            binned_finite_timeseries_indices.append(bin_indices)

    # convert to ndarrays
    binned_finite_timeseries_indices = [np.array(x) for x in
                                        binned_finite_timeseries_indices]

    collected_binned_mags['jdbins_indices'] = binned_finite_timeseries_indices
    collected_binned_mags['jdbins'] = np.array(jdbins)
    collected_binned_mags['nbins'] = len(binned_finite_timeseries_indices)

    # collect the finite_times
    binned_jd = np.array([np.median(finite_times[x])
                          for x in binned_finite_timeseries_indices])
    collected_binned_mags['binnedtimes'] = binned_jd
    collected_binned_mags['binsize'] = binsize

    # median bin the magnitudes according to the calculated indices
    collected_binned_mags['binnedmags'] = (
        np.array([np.median(finite_mags[x])
                  for x in binned_finite_timeseries_indices])
    )

    # FIXME: calculate the error in the median-binned magnitude correctly
    # for now, just take the median of the errors in this bin
    collected_binned_mags['binnederrs'] = (
        np.array([np.median(finite_errs[x])
                  for x in binned_finite_timeseries_indices])
    )

    return collected_binned_mags


def phase_bin_magseries(phases, mags,
                        binsize=0.005,
                        minbinelems=7):
    '''Bins a phased magnitude/flux time-series using the bin size provided.

    Parameters
    ----------

    phases,mags : np.array
        The phased magnitude/flux time-series to bin in phase. Non-finite
        elements will be removed from these arrays. At least 10 elements in each
        array are required for this function to operate.

    binsize : float
        The bin size to use to group together measurements closer than this
        amount in phase. This is in units of phase.

    minbinelems : int
        The minimum number of elements required per bin to include it in the
        output.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'phasebin_indices': a list of the index arrays into the
                                 nan-filtered input arrays per each bin,
             'phasebins': list of bin boundaries for each bin,
             'nbins': the number of bins generated,
             'binnedphases': the phase values associated with each phase bin;
                            this is the median of the phase value in each bin,
             'binnedmags': the mag/flux values associated with each phase bin;
                           this is the median of the mags/fluxes in each bin}

    '''

    # check if the input arrays are ok
    if not(phases.shape and mags.shape and len(phases) > 10 and len(mags) > 10):

        LOGERROR("input time/mag arrays don't have enough elements")
        return

    # find all the finite values of the magnitudes and phases
    finiteind = np.isfinite(mags) & np.isfinite(phases)
    finite_phases = phases[finiteind]
    finite_mags = mags[finiteind]

    nbins = int(np.ceil((np.nanmax(finite_phases) -
                         np.nanmin(finite_phases))/binsize) + 1)

    minphase = np.nanmin(finite_phases)
    phasebins = [(minphase + x*binsize) for x in range(nbins)]

    # make a KD-tree on the PHASEs so we can do fast distance calculations.  we
    # need to add a bogus y coord to make this a problem that KD-trees can
    # solve.
    time_coords = np.array([[x,1.0] for x in finite_phases])
    phasetree = cKDTree(time_coords)
    binned_finite_phaseseries_indices = []

    collected_binned_mags = {}

    for phase in phasebins:

        # find all bin indices close to within binsize of this point using the
        # cKDTree query. we use the p-norm = 1 for pairwise Euclidean distance.
        bin_indices = phasetree.query_ball_point(np.array([phase,1.0]),
                                                 binsize/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_phaseseries_indices and
            len(bin_indices) >= minbinelems):

            binned_finite_phaseseries_indices.append(bin_indices)

    # convert to ndarrays
    binned_finite_phaseseries_indices = [np.array(x) for x in
                                         binned_finite_phaseseries_indices]

    collected_binned_mags['phasebins_indices'] = (
        binned_finite_phaseseries_indices
    )
    collected_binned_mags['phasebins'] = phasebins
    collected_binned_mags['nbins'] = len(binned_finite_phaseseries_indices)

    # collect the finite_phases
    binned_phase = np.array([np.median(finite_phases[x])
                             for x in binned_finite_phaseseries_indices])
    collected_binned_mags['binnedphases'] = binned_phase
    collected_binned_mags['binsize'] = binsize

    # median bin the magnitudes according to the calculated indices
    collected_binned_mags['binnedmags'] = (
        np.array([np.median(finite_mags[x])
                  for x in binned_finite_phaseseries_indices])
    )

    return collected_binned_mags


def phase_bin_magseries_with_errs(phases, mags, errs,
                                  binsize=0.005,
                                  minbinelems=7):
    '''Bins a phased magnitude/flux time-series using the bin size provided.

    Parameters
    ----------

    phases,mags,errs : np.array
        The phased magnitude/flux time-series and associated errs to bin in
        phase. Non-finite elements will be removed from these arrays. At least
        10 elements in each array are required for this function to operate.

    binsize : float
        The bin size to use to group together measurements closer than this
        amount in phase. This is in units of phase.

    minbinelems : int
        The minimum number of elements required per bin to include it in the
        output.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'phasebin_indices': a list of the index arrays into the
                                 nan-filtered input arrays per each bin,
             'phasebins': list of bin boundaries for each bin,
             'nbins': the number of bins generated,
             'binnedphases': the phase values associated with each phase bin;
                            this is the median of the phase value in each bin,
             'binnedmags': the mag/flux values associated with each phase bin;
                           this is the median of the mags/fluxes in each bin,
             'binnederrs': the err values associated with each phase bin;
                           this is the median of the errs in each bin}

    '''

    # check if the input arrays are ok
    if not(phases.shape and mags.shape and len(phases) > 10 and len(mags) > 10):

        LOGERROR("input time/mag arrays don't have enough elements")
        return

    # find all the finite values of the magnitudes and phases
    finiteind = np.isfinite(mags) & np.isfinite(phases) & np.isfinite(errs)
    finite_phases = phases[finiteind]
    finite_mags = mags[finiteind]
    finite_errs = errs[finiteind]

    nbins = int(np.ceil((np.nanmax(finite_phases) -
                         np.nanmin(finite_phases))/binsize) + 1)

    minphase = np.nanmin(finite_phases)
    phasebins = [(minphase + x*binsize) for x in range(nbins)]

    # make a KD-tree on the PHASEs so we can do fast distance calculations.  we
    # need to add a bogus y coord to make this a problem that KD-trees can
    # solve.
    time_coords = np.array([[x,1.0] for x in finite_phases])
    phasetree = cKDTree(time_coords)
    binned_finite_phaseseries_indices = []

    collected_binned_mags = {}

    for phase in phasebins:

        # find all bin indices close to within binsize of this point using the
        # cKDTree query. we use the p-norm = 1 for pairwise Euclidean distance.
        bin_indices = phasetree.query_ball_point(np.array([phase,1.0]),
                                                 binsize/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_phaseseries_indices and
            len(bin_indices) >= minbinelems):

            binned_finite_phaseseries_indices.append(bin_indices)

    # convert to ndarrays
    binned_finite_phaseseries_indices = [np.array(x) for x in
                                         binned_finite_phaseseries_indices]

    collected_binned_mags['phasebins_indices'] = (
        binned_finite_phaseseries_indices
    )
    collected_binned_mags['phasebins'] = phasebins
    collected_binned_mags['nbins'] = len(binned_finite_phaseseries_indices)

    # collect the finite_phases
    binned_phase = np.array([np.median(finite_phases[x])
                             for x in binned_finite_phaseseries_indices])
    collected_binned_mags['binnedphases'] = binned_phase
    collected_binned_mags['binsize'] = binsize

    # median bin the magnitudes according to the calculated indices
    collected_binned_mags['binnedmags'] = (
        np.array([np.median(finite_mags[x])
                  for x in binned_finite_phaseseries_indices])
    )
    collected_binned_mags['binnederrs'] = (
        np.array([np.median(finite_errs[x])
                  for x in binned_finite_phaseseries_indices])
    )

    return collected_binned_mags


#############################
## FILLING TIMESERIES GAPS ##
#############################

def fill_magseries_gaps(times, mags, errs,
                        fillgaps=0.0,
                        sigclip=3.0,
                        magsarefluxes=False,
                        filterwindow=11,
                        forcetimebin=None,
                        verbose=True):
    '''This fills in gaps in a light curve.

    This is mainly intended for use in ACF period-finding, but maybe useful
    otherwise (i.e. when we figure out ARMA stuff for LCs). The main steps here
    are:

    - normalize the light curve to zero
    - remove giant outliers
    - interpolate gaps in the light curve
      (since ACF requires evenly spaced sampling)

    From McQuillan+ 2013a (https://doi.org/10.1093/mnras/stt536):

        "The ACF calculation requires the light curves to be regularly sampled
        and normalized to zero. We divided the flux in each quarter by its
        median and subtracted unity. Gaps in the light curve longer than the
        Kepler long cadence were filled using linear interpolation with added
        white Gaussian noise. This noise level was estimated using the variance
        of the residuals following subtraction of a smoothed version of the
        flux. To smooth the flux, we applied an iterative non-linear filter
        which consists of a median filter followed by a boxcar filter, both with
        11-point windows, with iterative 3σ clipping of outliers."

    Parameters
    ----------

    times,mags,errs : np.array
        The magnitude/flux time-series and associated measurement errors to
        operate on. Non-finite elements will be removed from these arrays. At
        least 10 elements in each array are required for this function to
        operate.

    fillgaps : {'noiselevel', 'nan'} or float
        If `fillgap='noiselevel'`, fills the gaps with the noise level obtained
        via the procedure above. If `fillgaps='nan'`, fills the gaps with
        `np.nan`. Otherwise, if `fillgaps` is a float, will use that value to
        fill the gaps. The default is to fill the gaps with 0.0 (as in
        McQuillan+ 2014) to "...prevent them contributing to the ACF".

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

    magsareflux : bool
        True if your "mags" are in fact fluxes, i.e. if "fainter" corresponds to
        `mags` getting smaller.

    filterwindow : int
        The number of time-series points to include in the Savitsky-Golay filter
        operation when smoothing the light curve. This should be an odd integer.

    forcetimebin : float or None
        If `forcetimebin` is a float, this value will be used to generate the
        interpolated time series, effectively binning the light curve to this
        cadence. If `forcetimebin` is None, the mode of the gaps (the forward
        difference between successive time values in `times`) in the provided
        light curve will be used as the effective cadence. NOTE: `forcetimebin`
        must be in the same units as `times`, e.g. if times are JD then
        `forcetimebin` must be in days as well

    verbose : bool
        If this is True, will indicate progress at various stages in the
        operation.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'itimes': the interpolated time values after gap-filling,
             'imags': the interpolated mag/flux values after gap-filling,
             'ierrs': the interpolated mag/flux values after gap-filling,
             'cadence': the cadence of the output mag/flux time-series}

    '''

    # remove nans
    finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes, fmags, ferrs = times[finind], mags[finind], errs[finind]

    # remove zero errs
    nzind = np.nonzero(ferrs)
    ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

    # sigma-clip
    stimes, smags, serrs = sigclip_magseries(ftimes, fmags, ferrs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # normalize to zero
    if magsarefluxes:
        smags = smags / np.median(smags) - 1.0
    else:
        smags = smags - np.median(smags)

    if isinstance(fillgaps, float):

        gapfiller = fillgaps

    elif isinstance(fillgaps, str) and fillgaps == 'noiselevel':

        # figure out the gaussian noise level by subtracting a Savitsky-Golay
        # filtered version of the light curve
        smoothed = smags - savgol_filter(smags, filterwindow, 2)
        noiselevel = 1.483 * np.median(np.abs(smoothed - np.median(smoothed)))
        gapfiller = noiselevel

    elif isinstance(fillgaps, str) and fillgaps == 'nan':

        gapfiller = np.nan

    # figure out the gap size and where to interpolate. we do this by figuring
    # out the most common gap (this should be the cadence). to do this, we need
    # to calculate the mode of the gap distribution.

    # get the gaps
    gaps = np.diff(stimes)

    # just use scipy.stats.mode instead of our hacked together nonsense earlier.
    gapmoderes = scipy.stats.mode(gaps)
    gapmode = gapmoderes[0].item()

    LOGINFO('auto-cadence for mag series: %.5f' % gapmode)

    # sort the gaps
    if forcetimebin:
        LOGWARNING('forcetimebin is set, forcing cadence to %.5f' %
                   forcetimebin)
        gapmode = forcetimebin

    if gapmode == 0.0:
        LOGERROR('the smallest cadence of this light curve appears to be 0.0, '
                 'the automatic cadence finder probably failed. '
                 'try setting forcetimebin?')
        return None

    starttime, endtime = np.min(stimes), np.max(stimes)
    ntimes = int(np.ceil((endtime - starttime)/gapmode) + 1)
    if verbose:
        LOGINFO('generating new time series with %s measurements' % ntimes)

    # first, generate the full time series
    interpolated_times = np.linspace(starttime, endtime, ntimes)
    interpolated_mags = np.full_like(interpolated_times, gapfiller)
    interpolated_errs = np.full_like(interpolated_times, gapfiller)

    for ind, itime in enumerate(interpolated_times[:-1]):

        nextitime = itime + gapmode
        # find the mags between this and the next time bin
        itimeind = np.where((stimes > itime) & (stimes < nextitime))

        # if there's more than one elem in this time bin, median them
        if itimeind[0].size > 1:

            interpolated_mags[ind] = np.median(smags[itimeind[0]])
            interpolated_errs[ind] = np.median(serrs[itimeind[0]])

        # otherwise, if there's only one elem in this time bin, take it
        elif itimeind[0].size == 1:

            interpolated_mags[ind] = smags[itimeind[0]]
            interpolated_errs[ind] = serrs[itimeind[0]]

    return {'itimes':interpolated_times,
            'imags':interpolated_mags,
            'ierrs':interpolated_errs,
            'cadence':gapmode}
