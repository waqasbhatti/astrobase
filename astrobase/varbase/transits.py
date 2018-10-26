#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''transits.py - Luke Bouma (luke@astro.princeton.edu) - Oct 2018
License: MIT - see the LICENSE file for the full text.

Contains tools for analyzing transits.
'''

#############
## LOGGING ##
#############

import logging
from datetime import datetime
from traceback import format_exc

# setup a logger
LOGGER = None
LOGMOD = __name__
DEBUG = False

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.%s' % (parent_name, LOGMOD))

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('[%s - DBUG] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('[%s - INFO] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('[%s - ERR!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('[%s - WRN!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '[%s - EXC!] %s\nexception was: %s' % (
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                message, format_exc()
            )
        )


#############
## IMPORTS ##
#############

import numpy as np
from astropy import units as u

from astrobase.periodbase import kbls
from astrobase.varbase import lcfit

##############################
## TRANSIT MODEL ASSESSMENT ##
##############################

def get_snr_of_dip(times,
                   mags,
                   modeltimes,
                   modelmags,
                   magsarefluxes=False,
                   verbose=True):
    '''
    Calculate the total SNR of a transit assuming gaussian uncertainties.
    `modelmags` gets interpolated onto the cadence of `mags`. The noise is
    calculated as the 1-sigma std deviation of the residual (see below).

    Following Carter et al. 2009,

        Q = sqrt( Γ T ) * δ / σ

    for Q the total SNR of the transit in the r->0 limit, where
    r = Rp/Rstar,
    T = transit duration,
    δ = transit depth,
    σ = RMS of the lightcurve in transit. For simplicity, we assume this is the
        same as the RMS of the residual=mags-modelmags.
    Γ = sampling rate

    Thus Γ * T is roughly the number of points obtained during transit.
    (This doesn't correctly account for the SNR during ingress/egress, but this
    is a second-order correction).

    Note this is the same total SNR as described by e.g., Kovacs et al. 2002,
    their Equation 11.

    Args:

        mags (np.ndarray): data fluxes (magnitudes not yet implemented).

        modelmags (np.ndarray): model fluxes. Assumed to be a BLS model, or a
        trapezoidal model, or a Mandel-Agol model.

    Kwargs:

        magsarefluxes (bool): currently forced to be true.

    Returns:

        snr, transit depth, and noise of residual lightcurve (tuple)
    '''

    if magsarefluxes:
        if not np.nanmedian(modelmags) == 1:
            raise AssertionError('snr calculation assumes modelmags are '
                                 'median-normalized')
    else:
        raise NotImplementedError(
            'need to implement a method for identifying in-transit points when'
            'mags are mags, and not fluxes'
        )

    # calculate transit depth from whatever model magnitudes are passed.
    transitdepth = np.abs(np.max(modelmags) - np.min(modelmags))

    # generally, mags (data) and modelmags are at different cadence.
    # interpolate modelmags onto the cadence of mags.
    if not len(mags) == len(modelmags):
        from scipy.interpolate import interp1d

        fn = interp1d(modeltimes, modelmags, kind='cubic', bounds_error=True,
                      fill_value=np.nan)

        modelmags = fn(times)

        if verbose:
            LOGINFO('interpolated model timeseries onto the data timeseries')

    subtractedmags = mags - modelmags

    subtractedrms = np.std(subtractedmags)

    def _get_npoints_in_transit(modelmags):
        # assumes median-normalized fluxes are input
        return len(modelmags[(modelmags != 1)])

    npoints_in_transit = _get_npoints_in_transit(modelmags)

    snr = np.sqrt(npoints_in_transit) * transitdepth/subtractedrms

    if verbose:

        LOGINFO('\npoints in transit: {:d}'.format(npoints_in_transit ) +
                '\ndepth: {:.2e}'.format(transitdepth) +
                '\nrms in residual: {:.2e}'.format(subtractedrms) +
                '\n\t SNR: {:.2e}'.format(snr))

    return snr, transitdepth, subtractedrms



def estimate_achievable_tmid_precision(snr, t_ingress_min=10,
                                       t_duration_hr=2.14):
    '''
    Using Carter et al. 2009's estimate, calculate the theoretical optimal
    precision on mid-transit time measurement possible given a transit of a
    particular SNR.

    sigma_tc = Q^{-1} * T * sqrt(θ/2)

    Q = SNR of the transit.
    T = transit duration, which is 2.14 hours from discovery paper.
    θ = τ/T = ratio of ingress to total duration
            ~= (few minutes [guess]) / 2.14 hours

    args:

        snr (float): measured signal-to-noise of transit, e.g., from
        `periodbase.get_snr_of_dip`

    kwargs:

        t_ingress_min (float): ingress duration in minutes, t_I to t_II in Winn
        (2010) nomenclature.

        t_duration_hr (float): total transit duration in hours, t_I to t_IV.
    '''

    t_ingress = t_ingress_min*u.minute
    t_duration = t_duration_hr*u.hour

    theta = t_ingress/t_duration

    sigma_tc = (1/snr * t_duration * np.sqrt(theta/2))

    LOGINFO('assuming t_ingress = {:.1f}'.format(t_ingress))
    LOGINFO('assuming t_duration = {:.1f}'.format(t_duration))
    LOGINFO('measured SNR={:.2f}\n\t'.format(snr) +
            '-->theoretical sigma_tc = {:.2e} = {:.2e} = {:.2e}'.format(
                sigma_tc.to(u.minute), sigma_tc.to(u.hour), sigma_tc.to(u.day)))

    return sigma_tc.to(u.day).value



def get_transit_times(
        blsd,
        time,
        extra_maskfrac,
        trapd=None,
        nperiodint=1000
):
    '''Given a BLS period, epoch, and transit ingress/egress points
    (usually from kbls.bls_stats_singleperiod), return the times within
    transit durations + extra_maskfrac of each transit.

    Optionally, use the (more accurate) trapezoidal fit period and epoch, if
    it's passed.  Useful for inspecting individual transits, and masking them
    out if desired.

    args:

        blsd (dict): dictionary returned by kbls.bls_stats_singleperiod

        time (np.ndarray): vector of times

        extra_maskfrac (float): separation from in-transit points you desire, in
        units of the transit duration. extra_maskfrac = 0 if you just want
        points inside transit.  (see below).

    kwargs:

        trapd (dict): dictionary returned by lcfit.traptransit_fit_magseries

        nperiodint (int): how many periods back/forward to try and identify
        transits from the epoch reported in blsd or trapd.

    returns:

        tmids_obsd, t_starts, t_ends (tuple of np.ndarrays):

            tmids_obsd (np.ndarray): best guess of transit midtimes in
            lightcurve. Has length number of transits in lightcurve.

            t_starts (np.ndarray): t_Is - extra_maskfrac*tdur, for t_Is transit
            first contact point.

            t_ends (np.ndarray): t_Is + extra_maskfrac*tdur, for t_Is transit
            first contact point.

    '''

    if trapd:
        period = trapd['fitinfo']['finalparams'][0]
        t0 = trapd['fitinfo']['fitepoch']
        transitduration_phase = trapd['fitinfo']['finalparams'][3]
        tdur = period * transitduration_phase
    else:
        period = blsd['period']
        t0 = blsd['epoch']
        tdur = (
            period *
            (blsd['transegressbin']-blsd['transingressbin'])/blsd['nphasebins']
        )
        if not blsd['transegressbin'] > blsd['transingressbin']:

            raise NotImplementedError(
                'careful of the width. '
                'this edge case must be dealt with separately.'
            )

    tmids = [t0 + ix*period for ix in range(-nperiodint,nperiodint)]

    sel = (tmids > np.nanmin(time)) & (tmids < np.nanmax(time))
    tmids_obsd = np.array(tmids)[sel]

    t_Is = tmids_obsd - tdur/2
    t_IVs = tmids_obsd + tdur/2

    # focus on the times around transit
    t_starts = t_Is - extra_maskfrac * tdur
    t_ends = t_IVs + extra_maskfrac * tdur

    return tmids_obsd, t_starts, t_ends



def given_lc_get_transit_tmids_tstarts_tends(
        time,
        flux,
        err_flux,
        blsfit_savpath=None,
        trapfit_savpath=None,
        magsarefluxes=True,
        nworkers=1,
        sigclip=None,
        extra_maskfrac=0.03
):
    '''
    args:
        should be obvious

    kwargs:

        blsfit_savpath (str): path to plot the fit BLS model

        trapfit_savpath (str): path to plot the fit trapezoidal transit model

        sigclip (None/int/list): if list, will be asymmetric

        extra_maskfrac (float): t_starts = t_Is - N*tdur, t_ends = t_IVs +
        N*tdur. Thus setting N=0.03 masks slightly more than the guessed
        transit duration.

    returns:

        tmids_obsd, t_starts, t_ends (tuple of np.ndarrays): see
        get_transit_times docstring.
    '''

    # first, run BLS to get an initial epoch and period.
    endp = 1.05*(np.nanmax(time) - np.nanmin(time))/2

    blsdict = kbls.bls_parallel_pfind(time, flux, err_flux,
                                      magsarefluxes=magsarefluxes, startp=0.1,
                                      endp=endp, maxtransitduration=0.3,
                                      nworkers=nworkers, sigclip=sigclip)

    blsd = kbls.bls_stats_singleperiod(time, flux, err_flux,
                                       blsdict['bestperiod'],
                                       magsarefluxes=True, sigclip=sigclip,
                                       perioddeltapercent=5)
    #  plot the BLS model.
    if blsfit_savpath:
        lcfit._make_fit_plot(blsd['phases'], blsd['phasedmags'], None,
                             blsd['blsmodel'], blsd['period'], blsd['epoch'],
                             blsd['epoch'], blsfit_savpath,
                             magsarefluxes=magsarefluxes)

    ingduration_guess = blsd['transitduration'] * 0.2  # a guesstimate.
    transitparams = [
        blsd['period'], blsd['epoch'], blsd['transitdepth'],
        blsd['transitduration'], ingduration_guess
    ]

    # fit a trapezoidal transit model; plot the resulting phased LC.
    if trapfit_savpath:
        trapd = lcfit.traptransit_fit_magseries(time, flux, err_flux,
                                                transitparams,
                                                magsarefluxes=magsarefluxes,
                                                sigclip=sigclip,
                                                plotfit=trapfit_savpath)

    # use the trapezoidal model's epoch as the guess to identify (roughly) in
    # and out of transit points
    tmids, t_starts, t_ends = get_transit_times(blsd,
                                                time,
                                                extra_maskfrac,
                                                trapd=trapd)

    return tmids, t_starts, t_ends



def _in_out_transit_plot(time, flux, intransit, ootransit, savpath):

    import matplotlib.pyplot as plt

    f, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))

    ax.scatter(
        time[ootransit],
        flux[ootransit],
        c='k',
        s=1.5,
        rasterized=True,
        linewidths=0
    )
    ax.scatter(
        time[intransit],
        flux[intransit],
        c='r',
        s=1.5,
        rasterized=True,
        linewidths=0
    )

    ax.set_ylabel('relative flux')
    ax.set_xlabel('time [days]')
    f.tight_layout(h_pad=0, w_pad=0)
    f.savefig(savpath, dpi=400, bbox_inches='tight')



def given_lc_get_out_of_transit_points(
    time, flux, err_flux, blsfit_savpath=None, trapfit_savpath=None,
    in_out_transit_savpath=None, magsarefluxes=True, nworkers=1, sigclip=None,
    extra_maskfrac=0.03
):
    '''
    relevant during iterative masking of transits for multiple planet system
    search.

    returns:

        tuple of np.ndarrays:
        time[out_of_transit], flux[out_of_transit], err_flux[out_of_transit]
    '''

    tmids_obsd, t_starts, t_ends = (
        given_lc_get_transit_tmids_tstarts_tends(
            time, flux, err_flux, blsfit_savpath=blsfit_savpath,
            trapfit_savpath=trapfit_savpath, magsarefluxes=magsarefluxes,
            nworkers=nworkers, sigclip=sigclip, extra_maskfrac=extra_maskfrac
        )
    )

    in_transit = np.zeros_like(time).astype(bool)

    for t_start, t_end in zip(t_starts, t_ends):

        this_transit = ( (time > t_start) & (time < t_end) )

        in_transit |= this_transit

    out_of_transit = ~in_transit

    if in_out_transit_savpath:
        _in_out_transit_plot(time, flux, in_transit, out_of_transit,
                             in_out_transit_savpath)

    return time[out_of_transit], flux[out_of_transit], err_flux[out_of_transit]
