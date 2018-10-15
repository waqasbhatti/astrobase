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