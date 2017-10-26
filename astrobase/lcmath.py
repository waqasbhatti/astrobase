#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
lcmath.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2015

Contains various useful tools for calculating various things related to
lightcurves (like phasing, sigma-clipping, etc.)

'''

import logging
import multiprocessing as mp
from datetime import datetime

import numpy as np
from numpy import isfinite as npisfinite, median as npmedian, abs as npabs

from scipy.spatial import cKDTree as kdtree
from scipy.signal import medfilt
from scipy.linalg import lstsq
from scipy.stats import sigmaclip as stats_sigmaclip
from scipy.optimize import curve_fit

import scipy.stats
import numpy.random as nprand

from scipy.signal import savgol_filter



#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.lcmath' % parent_name)

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


# DEBUG mode
DEBUG = False

############################
## NORMALIZING MAG SERIES ##
############################

def find_lc_timegroups(lctimes, mingap=4.0):
    '''
    This finds the gaps in the lightcurve, so we can figure out which times are
    for consecutive observations and which represent gaps between
    seasons.

    lctimes is assumed to be in some form of JD.

    min_gap defines how much the difference between consecutive measurements is
    allowed to be to consider them as parts of different timegroups. By default
    it is set to 4.0 days.

    Returns number of groups and Python slice objects for each group like so:

    (ngroups, [slice(start_ind_1, end_ind_1), ...])

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
    '''This normalizes the mag series to the value specified by normto.

    This is used to normalize time series measurements that may have large time
    gaps and vertical offsets in mag/flux measurement between these
    'timegroups', either due to instrument changes or different filters.

    NOTE: this works in-place! The mags array will be replaced with normalized
    mags when this function finishes.

    The normto kwarg is one of the following strings:

    'globalmedian' -> norms each mag to the global median of the LC column
    'zero'         -> norms each mag to zero

    or a float indicating the canonical magnitude/flux to normalize to.

    If magsarefluxes = True:

      If normto='zero', then the median flux is divided from each observation's
      flux value to yield normalized fluxes with 1.0 as the global median.

      If normto='globalmedian', then the global median flux value across the
      entire time series is multiplied with each measurement.

      If normto=<some float number>, then this number is multiplied with the
      flux value for each measurement.

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
                LOGDEBUG('%s group %s: elems %s, '
                         'finite elems %s, median mag %s' %
                         (col, tgind,
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
                      magsarefluxes=False):
    '''
    Select the finite times, magnitudes (or fluxes), and errors from the
    passed values, and apply symmetric or asymmetric sigma clipping to them.
    Returns sigma-clipped times, mags, and errs.

    Args:
        times (np.array): ...

        mags (np.array): numpy array to sigma-clip. Does not assume all values
        are finite. Does not assume anything about whether they're
        positive/negative.

        errs (np.array): ...

        iterative (bool): True if you want iterative sigma-clipping.

        magsarefluxes (bool): True if your "mags" are in fact fluxes, i.e. if
        "dimming" corresponds to your "mags" getting smaller.

        sigclip (float or list): If float, apply symmetric sigma clipping. If
        list, e.g., [10., 3.], will sigclip out greater than 10-sigma dimmings
        and greater than 3-sigma brightenings. Here the meaning of "dimming"
        and "brightening" is set by *physics* (not the magnitude system), which
        is why the `magsarefluxes` kwarg must be correctly set.

    Returns:
        stimes, smags, serrs: (sigmaclipped values of each).
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

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next for a single sigclip value
    if sigclip and isinstance(sigclip,float):

        if not iterative:

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

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

                # update delta and go to the top of the loop
                delta = this_size - this_mags.size

            # final sigclipped versions
            stimes, smags, serrs = this_times, this_mags, this_errs


    # this handles sigclipping for asymmetric +ve and -ve clip values
    elif sigclip and isinstance(sigclip,list) and len(sigclip) == 2:

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
                        (this_mags - this_median) > (-brighteningclip*this_stdev)
                    )

                # apply the sigclip
                tsi = nottoodimind & nottoobrightind

                # update the arrays
                this_times = this_times[tsi]
                this_mags = this_mags[tsi]
                this_errs = this_errs[tsi]

                # update delta and go to top of the loop
                delta = this_size - this_mags.size

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



#################
## PHASING LCS ##
#################

def phase_magseries(times, mags, period, epoch, wrap=True, sort=True):
    '''
    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.

    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags)
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
    '''
    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.

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
    '''This bins the given mag timeseries in time using the binsize given.

    binsize is in seconds.

    minbinelems is the minimum number of elements per bin.

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
    jdtree = kdtree(time_coords)
    binned_finite_timeseries_indices = []

    collected_binned_mags = {}

    for jd in jdbins:
        # find all bin indices close to within binsizejd of this point
        # using the kdtree query. we use the p-norm = 1 (I think this
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
    '''This bins the given mag timeseries in time using the binsize given.

    binsize is in seconds.

    minbinelems is the number of minimum elements in a bin.

    '''

    # check if the input arrays are ok
    if not(times.shape and mags.shape and errs.shape and
           len(times) > 9 and len(mags) > 9):

        LOGERROR("input time/mag arrays don't have enough elements")
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
    jdtree = kdtree(time_coords)
    binned_finite_timeseries_indices = []

    collected_binned_mags = {}

    for jd in jdbins:

        # find all bin indices close to within binsize of this point using the
        # kdtree query. we use the p-norm = 1 for pairwise Euclidean distance.
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
    '''
    This bins a magnitude timeseries in phase using the binsize (in phase)
    provided.

    minbinelems is the minimum number of elements in each bin.

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
    phasetree = kdtree(time_coords)
    binned_finite_phaseseries_indices = []

    collected_binned_mags = {}

    for phase in phasebins:

        # find all bin indices close to within binsize of this point using the
        # kdtree query. we use the p-norm = 1 for pairwise Euclidean distance.
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
    '''
    This bins a magnitude timeseries in phase using the binsize (in phase)
    provided.

    minbinelems is the minimum number of elements in each bin.

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
    phasetree = kdtree(time_coords)
    binned_finite_phaseseries_indices = []

    collected_binned_mags = {}

    for phase in phasebins:

        # find all bin indices close to within binsize of this point using the
        # kdtree query. we use the p-norm = 1 for pairwise Euclidean distance.
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



###################
## EPD FUNCTIONS ##
###################


def epd_diffmags(coeff, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag):
    '''
    This calculates the difference in mags after EPD coefficients are
    calculated.

    final EPD mags = median(magseries) + epd_diffmags()

    '''

    return -(coeff[0]*fsv**2. +
             coeff[1]*fsv +
             coeff[2]*fdv**2. +
             coeff[3]*fdv +
             coeff[4]*fkv**2. +
             coeff[5]*fkv +
             coeff[6] +
             coeff[7]*fsv*fdv +
             coeff[8]*fsv*fkv +
             coeff[9]*fdv*fkv +
             coeff[10]*np.sin(2*np.pi*xcc) +
             coeff[11]*np.cos(2*np.pi*xcc) +
             coeff[12]*np.sin(2*np.pi*ycc) +
             coeff[13]*np.cos(2*np.pi*ycc) +
             coeff[14]*np.sin(4*np.pi*xcc) +
             coeff[15]*np.cos(4*np.pi*xcc) +
             coeff[16]*np.sin(4*np.pi*ycc) +
             coeff[17]*np.cos(4*np.pi*ycc) +
             coeff[18]*bgv +
             coeff[19]*bge -
             mag)


def epd_magseries(mag, fsv, fdv, fkv, xcc, ycc, bgv, bge,
                  smooth=21, sigmaclip=3.0):
    '''
    Detrends a magnitude series given in mag using accompanying values of S in
    fsv, D in fdv, K in fkv, x coords in xcc, y coords in ycc, background in
    bgv, and background error in bge. smooth is used to set a smoothing
    parameter for the fit function.

    This returns EPD mag corrections. To convert RMx to EPx, do:

    EPx = RMx + correction

    '''

    # find all the finite values of the magnitude
    finiteind = np.isfinite(mag)

    # calculate median and stdev
    mag_median = np.median(mag[finiteind])
    mag_stdev = np.nanstd(mag)

    # if we're supposed to sigma clip, do so
    if sigmaclip:
        excludeind = abs(mag - mag_median) < sigmaclip*mag_stdev
        finalind = finiteind & excludeind
    else:
        finalind = finiteind

    final_mag = mag[finalind]
    final_len = len(final_mag)

    if DEBUG:
        print('final epd fit mag len = %s' % final_len)

    # smooth the signal
    smoothedmag = medfilt(final_mag, smooth)

    # make the linear equation matrix
    epdmatrix = np.c_[fsv[finalind]**2.0,
                      fsv[finalind],
                      fdv[finalind]**2.0,
                      fdv[finalind],
                      fkv[finalind]**2.0,
                      fkv[finalind],
                      np.ones(final_len),
                      fsv[finalind]*fdv[finalind],
                      fsv[finalind]*fkv[finalind],
                      fdv[finalind]*fkv[finalind],
                      np.sin(2*np.pi*xcc[finalind]),
                      np.cos(2*np.pi*xcc[finalind]),
                      np.sin(2*np.pi*ycc[finalind]),
                      np.cos(2*np.pi*ycc[finalind]),
                      np.sin(4*np.pi*xcc[finalind]),
                      np.cos(4*np.pi*xcc[finalind]),
                      np.sin(4*np.pi*ycc[finalind]),
                      np.cos(4*np.pi*ycc[finalind]),
                      bgv[finalind],
                      bge[finalind]]

    # solve the equation epdmatrix * x = smoothedmag
    # return the EPD differential mags if the solution succeeds
    try:

        coeffs, residuals, rank, singulars = lstsq(epdmatrix, smoothedmag)

        if DEBUG:
            print('coeffs = %s, residuals = %s' % (coeffs, residuals))

        return epd_diffmags(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag)

    # if the solution fails, return nothing
    except Exception as e:

        LOGEXCEPTION('%sZ: EPD solution did not converge! Error was: %s' %
                     (datetime.utcnow().isoformat(), e))
        return None



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

    "The ACF calculation requires the light curves to be regularly sampled and
    normalized to zero. We divided the flux in each quarter by its median and
    subtracted unity. Gaps in the light curve longer than the Kepler long
    cadence were filled using linear interpolation with added white Gaussian
    noise. This noise level was estimated using the variance of the residuals
    following subtraction of a smoothed version of the flux. To smooth the flux,
    we applied an iterative non-linear filter which consists of a median filter
    followed by a boxcar filter, both with 11-point windows, with iterative 3σ
    clipping of outliers."

    If fillgaps == 'noiselevel', fills the gaps with the noise level obtained
    via the procedure above. If fillgaps == 'nan', fills the gaps with
    np.nan. Otherwise, if fillgaps is a float, will use that value to fill the
    gaps. The default is to fill the gaps with 0.0 (as in McQuillan+ 2014) to
    "...prevent them contributing to the ACF".

    If forcetimebin is a float, this value will be used to generate the
    interpolated time series, effectively binning the light curve to this
    cadence.

    NOTE: forcetimebin must be in the same units as times; e.g. if times are JD
    then forcetimebin must be in days.

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
    gapmode = np.asscalar(gapmoderes[0])

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
