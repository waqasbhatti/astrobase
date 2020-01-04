#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# transits.py - Luke Bouma (luke@astro.princeton.edu) - Oct 2018
# License: MIT - see the LICENSE file for the full text.

'''
Contains tools for analyzing transits.

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
from astropy import units as u

from astrobase.periodbase import kbls

from ..lcfit.transits import traptransit_fit_magseries
from ..lcfit.utils import make_fit_plot


#######################
## UTILITY FUNCTIONS ##
#######################

def transit_duration_range(period,
                           min_radius_hint,
                           max_radius_hint):
    '''This figures out the minimum and max transit duration (q) given a period
    and min/max stellar radius hints.

    One can get stellar radii from various places:

    - GAIA distances and luminosities
    - the TESS input catalog
    - isochrone fits

    The equation used is::

        q ~ 0.076 x R**(2/3) x P**(-2/3)

        P = period in days
        R = stellar radius in solar radii

    Parameters
    ----------

    period : float
        The orbital period of the transiting planet.

    min_radius_hint,max_radius_hint : float
        The minimum and maximum radii of the star the planet is orbiting around.

    Returns
    -------

    (min_transit_duration, max_transit_duration) : tuple
        The returned tuple contains the minimum and maximum transit durations
        allowed for the orbital geometry of this planetary system. These can be
        used with the BLS period-search functions in
        :py:mod:`astrobase.periodbase.kbls` or
        :py:mod:`astrobase.periodbase.abls` to refine the period-search to only
        physically possible transit durations.

    '''

    return (
        0.076 * (min_radius_hint**(2./3.)) * (period**(-2./3.)),
        0.076 * (max_radius_hint**(2./3.)) * (period**(-2./3.))
    )


##############################
## TRANSIT MODEL ASSESSMENT ##
##############################

def get_snr_of_dip(times,
                   mags,
                   modeltimes,
                   modelmags,
                   atol_normalization=1e-8,
                   indsforrms=None,
                   magsarefluxes=False,
                   verbose=True,
                   transitdepth=None,
                   npoints_in_transit=None):
    '''Calculate the total SNR of a transit assuming gaussian uncertainties.

    `modelmags` gets interpolated onto the cadence of `mags`. The noise is
    calculated as the 1-sigma std deviation of the residual (see below).

    Following Carter et al. 2009::

        Q = sqrt( Γ T ) * δ / σ

    for Q the total SNR of the transit in the r->0 limit, where::

        r = Rp/Rstar,
        T = transit duration,
        δ = transit depth,
        σ = RMS of the lightcurve in transit.
        Γ = sampling rate

    Thus Γ * T is roughly the number of points obtained during transit.
    (This doesn't correctly account for the SNR during ingress/egress, but this
    is a second-order correction).

    Note this is the same total SNR as described by e.g., Kovacs et al. 2002,
    their Equation 11.

    NOTE: this only works with fluxes at the moment.

    Parameters
    ----------

    times,mags : np.array
        The input flux time-series to process.

    modeltimes,modelmags : np.array
        A transiting planet model, either from BLS, a trapezoid model, or a
        Mandel-Agol model.

    atol_normalization : float
        The absolute tolerance to which the median of the passed model fluxes
        must be equal to 1.

    indsforrms : np.array
        A array of bools of `len(mags)` used to select points for the RMS
        measurement. If not passed, the RMS of the entire passed timeseries is
        used as an approximation. Genearlly, it's best to use out of transit
        points, so the RMS measurement is not model-dependent.

    magsarefluxes : bool
        Currently forced to be True because this function only works with
        fluxes.

    verbose : bool
        If True, indicates progress and warns about problems.

    transitdepth : float or None
        If the transit depth is known, pass it in here. Otherwise, it is
        calculated assuming OOT flux is 1.

    npoints_in_transits : int or None
        If the number of points in transit is known, pass it in here. Otherwise,
        the function will guess at this value.

    Returns
    -------

    (snr, transit_depth, noise) : tuple
        The returned tuple contains the calculated SNR, transit depth, and noise
        of the residual lightcurve calculated using the relation described
        above.

    '''

    if magsarefluxes:
        if not np.isclose(np.nanmedian(modelmags), 1, atol=atol_normalization):
            raise AssertionError('snr calculation assumes modelmags are '
                                 'median-normalized')
    else:
        raise NotImplementedError(
            'need to implement a method for identifying in-transit points when'
            'mags are mags, and not fluxes'
        )

    if not transitdepth:
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

    if isinstance(indsforrms, np.ndarray):
        subtractedrms = np.std(subtractedmags[indsforrms])
        if verbose:
            LOGINFO('using selected points to measure RMS')
    else:
        subtractedrms = np.std(subtractedmags)
        if verbose:
            LOGINFO('using all points to measure RMS')

    def _get_npoints_in_transit(modelmags):
        # assumes median-normalized fluxes are input
        if np.nanmedian(modelmags) == 1:
            return len(modelmags[(modelmags != 1)])
        else:
            raise NotImplementedError

    if not npoints_in_transit:
        npoints_in_transit = _get_npoints_in_transit(modelmags)

    snr = np.sqrt(npoints_in_transit) * transitdepth/subtractedrms

    if verbose:

        LOGINFO('\npoints in transit: {:d}'.format(npoints_in_transit) +
                '\ndepth: {:.2e}'.format(transitdepth) +
                '\nrms in residual: {:.2e}'.format(subtractedrms) +
                '\n\t SNR: {:.2e}'.format(snr))

    return snr, transitdepth, subtractedrms


def estimate_achievable_tmid_precision(snr, t_ingress_min=10,
                                       t_duration_hr=2.14):
    '''Using Carter et al. 2009's estimate, calculate the theoretical optimal
    precision on mid-transit time measurement possible given a transit of a
    particular SNR.

    The relation used is::

        sigma_tc = Q^{-1} * T * sqrt(θ/2)

        Q = SNR of the transit.
        T = transit duration, which is 2.14 hours from discovery paper.
        θ = τ/T = ratio of ingress to total duration
                ~= (few minutes [guess]) / 2.14 hours

    Parameters
    ----------

    snr : float
        The measured signal-to-noise of the transit, e,g. from
        :py:func:`astrobase.periodbase.kbls.bls_stats_singleperiod` or from
        running the `.compute_stats()` method on an Astropy BoxLeastSquares
        object.

    t_ingress_min : float
        The ingress duration in minutes. This is t_I to t_II in Winn (2010)
        nomenclature.

    t_duration_hr : float
        The transit duration in hours. This is t_I to t_IV in Winn (2010)
        nomenclature.

    Returns
    -------

    float
        Returns the precision achievable for transit-center time as calculated
        from the relation above. This is in days.

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
    '''Given a BLS period, epoch, and transit ingress/egress points (usually
    from :py:func:`astrobase.periodbase.kbls.bls_stats_singleperiod`), return
    the times within transit durations + `extra_maskfrac` of each transit.

    Optionally, can use the (more accurate) trapezoidal fit period and epoch, if
    it's passed.  Useful for inspecting individual transits, and masking them
    out if desired.

    Parameters
    ----------

    blsd : dict
        This is the dict returned by
        :py:func:`astrobase.periodbase.kbls.bls_stats_singleperiod`.

    time : np.array
        The times from the time-series of transit observations used to calculate
        the initial period.

    extra_maskfrac : float
        This is the separation from in-transit points you desire, in units of
        the transit duration. `extra_maskfrac = 0` if you just want points
        inside transit (see below).

    trapd : dict
        This is a dict returned by
        :py:func:`astrobase.lcfit.transits.traptransit_fit_magseries` containing
        the trapezoid transit model.

    nperiodint : int
        This indicates how many periods backwards/forwards to try and identify
        transits from the epochs reported in `blsd` or `trapd`.

    Returns
    -------

    (tmids_obsd, t_starts, t_ends) : tuple of np.array
        The returned items are::

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
    '''Gets the transit start, middle, and end times for transits in a given
    time-series of observations.

    Parameters
    ----------

    time,flux,err_flux : np.array
        The input flux time-series measurements and their associated measurement
        errors

    blsfit_savpath : str or None
        If provided as a str, indicates the path of the fit plot to make for a
        simple BLS model fit to the transit using the obtained period and epoch.

    trapfit_savpath : str or None
        If provided as a str, indicates the path of the fit plot to make for a
        trapezoidal transit model fit to the transit using the obtained period
        and epoch.

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
        This is by default True for this function, since it works on fluxes only
        at the moment.

    nworkers : int
        The number of parallel BLS period-finder workers to use.

    extra_maskfrac : float
        This is the separation (N) from in-transit points you desire, in units
        of the transit duration. `extra_maskfrac = 0` if you just want points
        inside transit, otherwise::

            t_starts = t_Is - N*tdur, t_ends = t_IVs + N*tdur

        Thus setting N=0.03 masks slightly more than the guessed transit
        duration.

    Returns
    -------

    (tmids_obsd, t_starts, t_ends) : tuple
        The returned items are::

            tmids_obsd (np.ndarray): best guess of transit midtimes in
            lightcurve. Has length number of transits in lightcurve.

            t_starts (np.ndarray): t_Is - extra_maskfrac*tdur, for t_Is transit
            first contact point.

            t_ends (np.ndarray): t_Is + extra_maskfrac*tdur, for t_Is transit
            first contact point.

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
        make_fit_plot(blsd['phases'], blsd['phasedmags'], None,
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
        trapd = traptransit_fit_magseries(time, flux, err_flux,
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
        time, flux, err_flux,
        blsfit_savpath=None,
        trapfit_savpath=None,
        in_out_transit_savpath=None,
        sigclip=None,
        magsarefluxes=True,
        nworkers=1,
        extra_maskfrac=0.03
):
    '''This gets the out-of-transit light curve points.

    Relevant during iterative masking of transits for multiple planet system
    search.

    Parameters
    ----------

    time,flux,err_flux : np.array
        The input flux time-series measurements and their associated measurement
        errors

    blsfit_savpath : str or None
        If provided as a str, indicates the path of the fit plot to make for a
        simple BLS model fit to the transit using the obtained period and epoch.

    trapfit_savpath : str or None
        If provided as a str, indicates the path of the fit plot to make for a
        trapezoidal transit model fit to the transit using the obtained period
        and epoch.

    in_out_transit_savpath : str or None
        If provided as a str, indicates the path of the plot file that will be
        made for a plot showing the in-transit points and out-of-transit points
        tagged separately.

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
        This is by default True for this function, since it works on fluxes only
        at the moment.

    nworkers : int
        The number of parallel BLS period-finder workers to use.

    extra_maskfrac : float
        This is the separation (N) from in-transit points you desire, in units
        of the transit duration. `extra_maskfrac = 0` if you just want points
        inside transit, otherwise::

            t_starts = t_Is - N*tdur, t_ends = t_IVs + N*tdur

        Thus setting N=0.03 masks slightly more than the guessed transit
        duration.

    Returns
    -------

    (times_oot, fluxes_oot, errs_oot) : tuple of np.array
        The `times`, `flux`, `err_flux` values from the input at the time values
        out-of-transit are returned.

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
