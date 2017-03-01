#!/usr/bin/env python

'''
lcmath.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2015

Contains various useful tools for calculating various things related to
lightcurves (like phasing, sigma-clipping, etc.)

'''

import logging
import multiprocessing as mp
import datetime

import numpy as np
from numpy import isfinite as npisfinite, median as npmedian, abs as npabs

from scipy.spatial import cKDTree as kdtree
from scipy.signal import medfilt
from scipy.linalg import lstsq
from scipy.stats import sigmaclip as stats_sigmaclip
from scipy.optimize import curve_fit

import scipy.stats
import numpy.random as nprand

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
                        debugmode=False):
    '''This normalizes the mag series to the value specified by normto.

    the normto kwarg is one of the following strings:

    'globalmedian' -> norms each mag to the global median of the LC column
    'zero'         -> norms each mag to zero

    or a float indicating the canonical magnitude to normalize to.

    NOTE: this works in-place! The mags array will be replaced with normalized
    mags when this function finishes.

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

            mags = mags + global_mag_median

        # if the normto is a float, add everything to that float and return
        elif isinstance(normto, float):

            mags = mags + normto

        # anything else just returns the normalized mags as usual
        return times, mags

    else:
        LOGWARNING('mags are all nan!')
        return None, None



####################
## SIGMA-CLIPPING ##
####################

def sigclip_magseries(times, mags, errs,
                      magsarefluxes=False,
                      sigclip=None):
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
        errs = np.full_like(mags, 0.005)
        returnerrs = False

    # filter the input times, mags, errs; do sigclipping and normalization
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[find], mags[find], errs[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next for a single sigclip value
    if sigclip and isinstance(sigclip,float):

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

    # this handles sigclipping for asymmetric +ve and -ve clip values
    elif sigclip and isinstance(sigclip,list) and len(sigclip) == 2:

        # sigclip is passed as [dimmingclip, brighteningclip]
        dimmingclip = sigclip[0]
        brighteningclip = sigclip[1]

        if magsarefluxes:
            nottoodimind = (fmags - median_mag) > (-dimmingclip*stddev_mag)
            nottoobrightind = (fmags - median_mag) < (brighteningclip*stddev_mag)
        else:
            nottoodimind = (fmags - median_mag) < (dimmingclip*stddev_mag)
            nottoobrightind = (fmags - median_mag) > (-brighteningclip*stddev_mag)

        sigind = nottoodimind & nottoobrightind

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

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

def time_bin_magseries(times, mags, binsize=540.0):
    '''
    This bins the given mag timeseries in time using the binsize given. binsize
    is in seconds.

    '''

    # check if the input arrays are ok
    if not(times.shape and mags.shape and len(times) > 10 and len(mags) > 10):

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
            len(bin_indices)) > 0:
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



def time_bin_magseries_with_errs(times, mags, errs, binsize=540.0):
    '''
    This bins the given mag timeseries in time using the binsize given. binsize
    is in seconds.

    '''

    # check if the input arrays are ok
    if not(times.shape and mags.shape and errs.shape and
           len(times) > 10 and len(mags) > 10):

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
        # find all bin indices close to within binsizejd of this point
        # using the kdtree query. we use the p-norm = 1 (I think this
        # means straight-up pairwise distance? FIXME: check this)
        bin_indices = jdtree.query_ball_point(np.array([jd,1.0]),
                                              binsizejd/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_timeseries_indices and
            len(bin_indices)) > 0:
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

    # FIXME: calculate the error in the median-binned magnitude correctly
    # for now, just take the median of the errors in this bin
    collected_binned_mags['binnederrs'] = (
        np.array([np.median(finite_errs[x])
                  for x in binned_finite_timeseries_indices])
        )


    return collected_binned_mags



def phase_bin_magseries(phases, mags, binsize=0.005):
    '''
    This bins a magnitude timeseries in phase using the binsize (in phase)
    provided.

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
        # find all bin indices close to within binsize of this point
        # using the kdtree query. we use the p-norm = 1 (I think this
        # means straight-up pairwise distance? FIXME: check this)
        bin_indices = phasetree.query_ball_point(np.array([phase,1.0]),
                                              binsize/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_phaseseries_indices and
            len(bin_indices)) > 0:
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



def phase_bin_magseries_with_errs(phases, mags, errs, binsize=0.005):
    '''
    This bins a magnitude timeseries in phase using the binsize (in phase)
    provided.

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
        # find all bin indices close to within binsize of this point
        # using the kdtree query. we use the p-norm = 1 (I think this
        # means straight-up pairwise distance? FIXME: check this)
        bin_indices = phasetree.query_ball_point(np.array([phase,1.0]),
                                              binsize/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_phaseseries_indices and
            len(bin_indices)) > 0:
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
