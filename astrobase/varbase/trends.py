#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# trends.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2018

'''
Contains light curve trend-removal tools, such as external parameter
decorrelation (EPD) and smoothing.

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

from functools import partial

import numpy as np
import numpy.random as npr
RANDSEED = 0xdecaff
npr.seed(RANDSEED)

from numpy import median as npmedian, abs as npabs, pi as pi_value
from numpy.linalg import lstsq

from scipy.optimize import leastsq, least_squares
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage.filters import median_filter
from astropy.convolution import convolve, Gaussian1DKernel

# for random forest EPD
from sklearn.ensemble import RandomForestRegressor

from ..lcmath import sigclip_magseries_with_extparams


#########################
## SMOOTHING FUNCTIONS ##
#########################

def smooth_magseries_ndimage_medfilt(mags, windowsize):
    '''This smooths the magseries with a median filter that reflects the array
    at the boundary.

    See https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html for
    details.

    Parameters
    ----------

    mags : np.array
        The input mags/flux time-series to smooth.

    windowsize : int
        This is a odd integer containing the smoothing window size.

    Returns
    -------

    np.array
        The smoothed mag/flux time-series array.

    '''
    return median_filter(mags, size=windowsize, mode='reflect')


def smooth_magseries_signal_medfilt(mags, windowsize):
    '''This smooths the magseries with a simple median filter.

    This function pads with zeros near the boundary, see:

    https://stackoverflow.com/questions/24585706/scipy-medfilt-wrong-result

    Typically this is bad.

    Parameters
    ----------

    mags : np.array
        The input mags/flux time-series to smooth.

    windowsize : int
        This is a odd integer containing the smoothing window size.

    Returns
    -------

    np.array
        The smoothed mag/flux time-series array.

    '''

    return medfilt(mags, windowsize)


def smooth_magseries_gaussfilt(mags, windowsize, windowfwhm=7):
    '''This smooths the magseries with a Gaussian kernel.

    Parameters
    ----------

    mags : np.array
        The input mags/flux time-series to smooth.

    windowsize : int
        This is a odd integer containing the smoothing window size.

    windowfwhm : int
        This is an odd integer containing the FWHM of the applied Gaussian
        window function.

    Returns
    -------

    np.array
        The smoothed mag/flux time-series array.

    '''

    convkernel = Gaussian1DKernel(windowfwhm, x_size=windowsize)
    smoothed = convolve(mags, convkernel, boundary='extend')
    return smoothed


def smooth_magseries_savgol(mags, windowsize, polyorder=2):
    '''This smooths the magseries with a Savitsky-Golay filter.

    Parameters
    ----------

    mags : np.array
        The input mags/flux time-series to smooth.

    windowsize : int
        This is a odd integer containing the smoothing window size.

    polyorder : int
        This is an integer containing the polynomial degree order to use when
        generating the Savitsky-Golay filter.

    Returns
    -------

    np.array
        The smoothed mag/flux time-series array.

    '''

    smoothed = savgol_filter(mags, windowsize, polyorder)
    return smoothed


########################################
## OLD EPD FUNCTIONS (USED FOR HATPI) ##
########################################

def _old_epd_diffmags(coeff, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag):
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


def _old_epd_magseries(times, mags, errs,
                       fsv, fdv, fkv, xcc, ycc, bgv, bge,
                       epdsmooth_windowsize=21,
                       epdsmooth_sigclip=3.0,
                       epdsmooth_func=smooth_magseries_signal_medfilt,
                       epdsmooth_extraparams=None):
    '''
    Detrends a magnitude series given in mag using accompanying values of S in
    fsv, D in fdv, K in fkv, x coords in xcc, y coords in ycc, background in
    bgv, and background error in bge. smooth is used to set a smoothing
    parameter for the fit function. Does EPD voodoo.

    '''

    # find all the finite values of the magsnitude
    finiteind = np.isfinite(mags)

    # calculate median and stdev
    mags_median = np.median(mags[finiteind])
    mags_stdev = np.nanstd(mags)

    # if we're supposed to sigma clip, do so
    if epdsmooth_sigclip:
        excludeind = abs(mags - mags_median) < epdsmooth_sigclip*mags_stdev
        finalind = finiteind & excludeind
    else:
        finalind = finiteind

    final_mags = mags[finalind]
    final_len = len(final_mags)

    # smooth the signal
    if isinstance(epdsmooth_extraparams, dict):
        smoothedmags = epdsmooth_func(final_mags,
                                      epdsmooth_windowsize,
                                      **epdsmooth_extraparams)
    else:
        smoothedmags = epdsmooth_func(final_mags, epdsmooth_windowsize)

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

    # solve the matrix equation [epdmatrix] . [x] = [smoothedmags]
    # return the EPD differential magss if the solution succeeds
    try:

        coeffs, residuals, rank, singulars = lstsq(epdmatrix, smoothedmags,
                                                   rcond=None)

        if DEBUG:
            print('coeffs = %s, residuals = %s' % (coeffs, residuals))

        retdict = {'times':times,
                   'mags':(mags_median +
                           _old_epd_diffmags(coeffs, fsv, fdv,
                                             fkv, xcc, ycc, bgv, bge, mags)),
                   'errs':errs,
                   'fitcoeffs':coeffs,
                   'residuals':residuals}

        return retdict

    # if the solution fails, return nothing
    except Exception:

        LOGEXCEPTION('EPD solution did not converge')

        retdict = {'times':times,
                   'mags':np.full_like(mags, np.nan),
                   'errs':errs,
                   'fitcoeffs':coeffs,
                   'residuals':residuals}

        return retdict


###################################################
## HAT-SPECIFIC EXTERNAL PARAMETER DECORRELATION ##
###################################################

def _epd_function(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd):
    '''
    This is the EPD function to fit using a smoothed mag-series.

    '''

    return (coeffs[0]*fsv*fsv +
            coeffs[1]*fsv +
            coeffs[2]*fdv*fdv +
            coeffs[3]*fdv +
            coeffs[4]*fkv*fkv +
            coeffs[5]*fkv +
            coeffs[6] +
            coeffs[7]*fsv*fdv +
            coeffs[8]*fsv*fkv +
            coeffs[9]*fdv*fkv +
            coeffs[10]*np.sin(2*pi_value*xcc) +
            coeffs[11]*np.cos(2*pi_value*xcc) +
            coeffs[12]*np.sin(2*pi_value*ycc) +
            coeffs[13]*np.cos(2*pi_value*ycc) +
            coeffs[14]*np.sin(4*pi_value*xcc) +
            coeffs[15]*np.cos(4*pi_value*xcc) +
            coeffs[16]*np.sin(4*pi_value*ycc) +
            coeffs[17]*np.cos(4*pi_value*ycc) +
            coeffs[18]*bgv +
            coeffs[19]*bge +
            coeffs[20]*iha +
            coeffs[21]*izd)


def _epd_residual(coeffs, mags, fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd):
    '''
    This is the residual function to minimize using scipy.optimize.leastsq.

    '''

    f = _epd_function(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd)
    residual = mags - f
    return residual


def _epd_residual2(coeffs,
                   times, mags, errs,
                   fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd):
    '''This is the residual function to minimize using
    scipy.optimize.least_squares.

    This variant is for :py:func:`.epd_magseries_extparams`.

    '''

    f = _epd_function(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd)
    residual = mags - f
    return residual


def epd_magseries(times, mags, errs,
                  fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd,
                  magsarefluxes=False,
                  epdsmooth_sigclip=3.0,
                  epdsmooth_windowsize=21,
                  epdsmooth_func=smooth_magseries_savgol,
                  epdsmooth_extraparams=None):
    '''Detrends a magnitude series using External Parameter Decorrelation.

    Requires a set of external parameters similar to those present in HAT light
    curves. At the moment, the HAT light-curve-specific external parameters are:

    - S: the 'fsv' column in light curves,
    - D: the 'fdv' column in light curves,
    - K: the 'fkv' column in light curves,
    - x coords: the 'xcc' column in light curves,
    - y coords: the 'ycc' column in light curves,
    - background value: the 'bgv' column in light curves,
    - background error: the 'bge' column in light curves,
    - hour angle: the 'iha' column in light curves,
    - zenith distance: the 'izd' column in light curves

    S, D, and K are defined as follows:

    - S -> measure of PSF sharpness (~1/sigma^2 sosmaller S = wider PSF)
    - D -> measure of PSF ellipticity in xy direction
    - K -> measure of PSF ellipticity in cross direction

    S, D, K are related to the PSF's variance and covariance, see eqn 30-33 in
    A. Pal's thesis: https://arxiv.org/abs/0906.3486

    NOTE: The errs are completely ignored and returned unchanged (except for
    sigclip and finite filtering).

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to detrend.

    fsv : np.array
        Array containing the external parameter `S` of the same length as times.

    fdv : np.array
        Array containing the external parameter `D` of the same length as times.

    fkv : np.array
        Array containing the external parameter `K` of the same length as times.

    xcc : np.array
        Array containing the external parameter `x-coords` of the same length as
        times.

    ycc : np.array
        Array containing the external parameter `y-coords` of the same length as
        times.

    bgv : np.array
        Array containing the external parameter `background value` of the same
        length as times.

    bge : np.array
        Array containing the external parameter `background error` of the same
        length as times.

    iha : np.array
        Array containing the external parameter `hour angle` of the same length
        as times.

    izd : np.array
        Array containing the external parameter `zenith distance` of the same
        length as times.

    magsarefluxes : bool
        Set this to True if `mags` actually contains fluxes.

    epdsmooth_sigclip : float or int or sequence of two floats/ints or None
        This specifies how to sigma-clip the input LC before fitting the EPD
        function to it.

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

    epdsmooth_windowsize : int
        This is the number of LC points to smooth over to generate a smoothed
        light curve that will be used to fit the EPD function.

    epdsmooth_func : Python function
        This sets the smoothing filter function to use. A Savitsky-Golay filter
        is used to smooth the light curve by default. The functions that can be
        used with this kwarg are listed in `varbase.trends`. If you want to use
        your own function, it MUST have the following signature::

                def smoothfunc(mags_array, window_size, **extraparams)

        and return a numpy array of the same size as `mags_array` with the
        smoothed time-series. Any extra params can be provided using the
        `extraparams` dict.

    epdsmooth_extraparams : dict
        This is a dict of any extra filter params to supply to the smoothing
        function.

    Returns
    -------

    dict
        Returns a dict of the following form::

            {'times':the input times after non-finite elems removed,
             'mags':the EPD detrended mag values (the EPD mags),
             'errs':the errs after non-finite elems removed,
             'fitcoeffs':EPD fit coefficient values,
             'fitinfo':the full tuple returned by scipy.leastsq,
             'fitmags':the EPD fit function evaluated at times,
             'mags_median': this is median of the EPD mags,
             'mags_mad': this is the MAD of EPD mags}

    '''

    finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes, fmags, ferrs = times[::][finind], mags[::][finind], errs[::][finind]
    ffsv, ffdv, ffkv, fxcc, fycc, fbgv, fbge, fiha, fizd = (
        fsv[::][finind],
        fdv[::][finind],
        fkv[::][finind],
        xcc[::][finind],
        ycc[::][finind],
        bgv[::][finind],
        bge[::][finind],
        iha[::][finind],
        izd[::][finind],
    )

    stimes, smags, serrs, separams = sigclip_magseries_with_extparams(
        times, mags, errs,
        [fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd],
        sigclip=epdsmooth_sigclip,
        magsarefluxes=magsarefluxes
    )
    sfsv, sfdv, sfkv, sxcc, sycc, sbgv, sbge, siha, sizd = separams

    # smooth the signal
    if isinstance(epdsmooth_extraparams, dict):
        smoothedmags = epdsmooth_func(smags,
                                      epdsmooth_windowsize,
                                      **epdsmooth_extraparams)
    else:
        smoothedmags = epdsmooth_func(smags, epdsmooth_windowsize)

    # initial fit coeffs
    initcoeffs = np.zeros(22)

    # fit the smoothed mags and find the EPD function coefficients
    leastsqfit = leastsq(_epd_residual,
                         initcoeffs,
                         args=(smoothedmags,
                               sfsv, sfdv, sfkv, sxcc,
                               sycc, sbgv, sbge, siha, sizd),
                         full_output=True)

    # if the fit succeeds, then get the EPD mags
    if leastsqfit[-1] in (1,2,3,4):

        fitcoeffs = leastsqfit[0]
        epdfit = _epd_function(fitcoeffs,
                               ffsv, ffdv, ffkv, fxcc, fycc,
                               fbgv, fbge, fiha, fizd)

        epdmags = npmedian(fmags) + fmags - epdfit

        retdict = {'times':ftimes,
                   'mags':epdmags,
                   'errs':ferrs,
                   'fitcoeffs':fitcoeffs,
                   'fitinfo':leastsqfit,
                   'fitmags':epdfit,
                   'mags_median':npmedian(epdmags),
                   'mags_mad':npmedian(npabs(epdmags - npmedian(epdmags)))}

        return retdict

    # if the solution fails, return nothing
    else:

        LOGERROR('EPD fit did not converge')
        return None


########################################
## EPD WITH ARBITRARY EXTERNAL PARAMS ##
########################################

def epd_magseries_extparams(
        times,
        mags,
        errs,
        externalparam_arrs,
        initial_coeff_guess,
        magsarefluxes=False,
        epdsmooth_sigclip=3.0,
        epdsmooth_windowsize=21,
        epdsmooth_func=smooth_magseries_savgol,
        epdsmooth_extraparams=None,
        objective_func=_epd_residual2,
        objective_kwargs=None,
        optimizer_func=least_squares,
        optimizer_kwargs=None,
):
    '''This does EPD on a mag-series with arbitrary external parameters.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to run EPD on.

    externalparam_arrs : list of np.arrays
        This is a list of ndarrays of external parameters to decorrelate
        against. These should all be the same size as `times`, `mags`, `errs`.

    initial_coeff_guess : np.array
        An array of initial fit coefficients to pass into the objective
        function.

    epdsmooth_sigclip : float or int or sequence of two floats/ints or None
        This specifies how to sigma-clip the input LC before smoothing it and
        fitting the EPD function to it. The actual LC will not be sigma-clipped.

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

    epdsmooth_windowsize : int
        This is the number of LC points to smooth over to generate a smoothed
        light curve that will be used to fit the EPD function.

    epdsmooth_func : Python function
        This sets the smoothing filter function to use. A Savitsky-Golay filter
        is used to smooth the light curve by default. The functions that can be
        used with this kwarg are listed in `varbase.trends`. If you want to use
        your own function, it MUST have the following signature::

                def smoothfunc(mags_array, window_size, **extraparams)

        and return a numpy array of the same size as `mags_array` with the
        smoothed time-series. Any extra params can be provided using the
        `extraparams` dict.

    epdsmooth_extraparams : dict
        This is a dict of any extra filter params to supply to the smoothing
        function.

    objective_func : Python function
        The function that calculates residuals between the model and the
        smoothed mag-series. This must have the following signature::

            def objective_func(fit_coeffs,
                               times,
                               mags,
                               errs,
                               *external_params,
                               **objective_kwargs)

        where `times`, `mags`, `errs` are arrays of the sigma-clipped and
        smoothed time-series, `fit_coeffs` is an array of EPD fit coefficients,
        `external_params` is a tuple of the passed in external parameter arrays,
        and `objective_kwargs` is a dict of any optional kwargs to pass into the
        objective function.

        This should return the value of the residual based on evaluating the
        model function (and any weights based on errs or times).

    objective_kwargs : dict or None
        A dict of kwargs to pass into the `objective_func` function.

    optimizer_func : Python function
        The function that minimizes the residual between the model and the
        smoothed mag-series using the `objective_func`. This should have a
        signature similar to one of the optimizer functions in `scipy.optimize
        <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_, i.e.::

            def optimizer_func(objective_func, initial_coeffs, args=(), ...)

        and return a `scipy.optimize.OptimizeResult
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_. We'll
        rely on the ``.success`` attribute to determine if the EPD fit was
        successful, and the ``.x`` attribute to get the values of the fit
        coefficients.

    optimizer_kwargs : dict or None
        A dict of kwargs to pass into the `optimizer_func` function.

    Returns
    -------

    dict
        Returns a dict of the following form::

            {'times':the input times after non-finite elems removed,
             'mags':the EPD detrended mag values (the EPD mags),
             'errs':the errs after non-finite elems removed,
             'fitcoeffs':EPD fit coefficient values,
             'fitinfo':the result returned by the optimizer function,
             'mags_median': this is the median of the EPD mags,
             'mags_mad': this is the MAD of EPD mags}

    '''

    # get finite times, mags, errs
    finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes, fmags, ferrs = times[::][finind], mags[::][finind], errs[::][finind]
    finalparam_arrs = []
    for ep in externalparam_arrs:
        finalparam_arrs.append(ep[::][finind])

    # sigclip the LC to pass into the smoothing for EPD fit
    stimes, smags, serrs, eparams = sigclip_magseries_with_extparams(
        times.copy(), mags.copy(), errs.copy(),
        [x.copy() for x in externalparam_arrs],
        sigclip=epdsmooth_sigclip,
        magsarefluxes=magsarefluxes
    )

    # smooth the signal before fitting the function to it
    if isinstance(epdsmooth_extraparams, dict):
        smoothedmags = epdsmooth_func(smags,
                                      epdsmooth_windowsize,
                                      **epdsmooth_extraparams)
    else:
        smoothedmags = epdsmooth_func(smags,
                                      epdsmooth_windowsize)

    # the initial coeffs are passed in here
    initial_coeffs = initial_coeff_guess

    # reform the objective function with any optional kwargs
    if objective_kwargs is not None:
        obj_func = partial(objective_func, **objective_kwargs)
    else:
        obj_func = objective_func

    # run the optimizer function by passing in the objective function, the
    # coeffs, and the smoothed mags and external params as part of the `args`
    # tuple
    if not optimizer_kwargs:
        optimizer_kwargs = {}

    fit_info = optimizer_func(
        obj_func,
        initial_coeffs,
        args=(stimes, smoothedmags, serrs, *eparams),
        **optimizer_kwargs
    )

    if fit_info.success:

        fit_coeffs = fit_info.x

        epd_mags = np.median(fmags) + obj_func(fit_coeffs,
                                               ftimes,
                                               fmags,
                                               ferrs,
                                               *finalparam_arrs)

        retdict = {'times':ftimes,
                   'mags':epd_mags,
                   'errs':ferrs,
                   'fitcoeffs':fit_coeffs,
                   'fitinfo':fit_info,
                   'mags_median':npmedian(epd_mags),
                   'mags_mad':npmedian(npabs(epd_mags - npmedian(epd_mags)))}

        return retdict

    # if the solution fails, return nothing
    else:

        LOGERROR('EPD fit did not converge')
        return None


#######################
## RANDOM FOREST EPD ##
#######################

def rfepd_magseries(times, mags, errs,
                    externalparam_arrs,
                    magsarefluxes=False,
                    epdsmooth=True,
                    epdsmooth_sigclip=3.0,
                    epdsmooth_windowsize=21,
                    epdsmooth_func=smooth_magseries_savgol,
                    epdsmooth_extraparams=None,
                    rf_subsample=1.0,
                    rf_ntrees=300,
                    rf_extraparams={'criterion':'mse',
                                    'oob_score':False,
                                    'n_jobs':-1}):
    '''This uses a `RandomForestRegressor` to de-correlate the given magseries.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to run EPD on.

    externalparam_arrs : list of np.arrays
        This is a list of ndarrays of external parameters to decorrelate
        against. These should all be the same size as `times`, `mags`, `errs`.

    epdsmooth : bool
        If True, sets the training LC for the RandomForestRegress to be a
        smoothed version of the sigma-clipped light curve provided in `times`,
        `mags`, `errs`.

    epdsmooth_sigclip : float or int or sequence of two floats/ints or None
        This specifies how to sigma-clip the input LC before smoothing it and
        fitting the EPD function to it. The actual LC will not be sigma-clipped.

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

    epdsmooth_windowsize : int
        This is the number of LC points to smooth over to generate a smoothed
        light curve that will be used to fit the EPD function.

    epdsmooth_func : Python function
        This sets the smoothing filter function to use. A Savitsky-Golay filter
        is used to smooth the light curve by default. The functions that can be
        used with this kwarg are listed in `varbase.trends`. If you want to use
        your own function, it MUST have the following signature::

                def smoothfunc(mags_array, window_size, **extraparams)

        and return a numpy array of the same size as `mags_array` with the
        smoothed time-series. Any extra params can be provided using the
        `extraparams` dict.

    epdsmooth_extraparams : dict
        This is a dict of any extra filter params to supply to the smoothing
        function.

    rf_subsample : float
        Defines the fraction of the size of the `mags` array to use for
        training the random forest regressor.

    rf_ntrees : int
        This is the number of trees to use for the `RandomForestRegressor`.

    rf_extraprams : dict
        This is a dict of any extra kwargs to provide to the
        `RandomForestRegressor` instance used.

    Returns
    -------

    dict
        Returns a dict with decorrelated mags and the usual info from the
        `RandomForestRegressor`: variable importances, etc.

    '''

    # get finite times, mags, errs
    finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes, fmags, ferrs = times[::][finind], mags[::][finind], errs[::][finind]
    finalparam_arrs = []
    for ep in externalparam_arrs:
        finalparam_arrs.append(ep[::][finind])

    stimes, smags, serrs, eparams = sigclip_magseries_with_extparams(
        times, mags, errs,
        externalparam_arrs,
        sigclip=epdsmooth_sigclip,
        magsarefluxes=magsarefluxes
    )

    # smoothing is optional for RFR because we train on a fraction of the mag
    # series and so should not require a smoothed input to fit a function to
    if epdsmooth:

        # smooth the signal
        if isinstance(epdsmooth_extraparams, dict):
            smoothedmags = epdsmooth_func(smags,
                                          epdsmooth_windowsize,
                                          **epdsmooth_extraparams)
        else:
            smoothedmags = epdsmooth_func(smags,
                                          epdsmooth_windowsize)

    else:

        smoothedmags = smags

    # set up the regressor
    if isinstance(rf_extraparams, dict):
        RFR = RandomForestRegressor(n_estimators=rf_ntrees,
                                    **rf_extraparams)
    else:
        RFR = RandomForestRegressor(n_estimators=rf_ntrees)

    # collect the features
    features = np.column_stack(eparams)

    # fit, then generate the predicted values, then get corrected values

    # we fit on a randomly selected subsample of all the mags
    if rf_subsample < 1.0:
        featureindices = np.arange(smoothedmags.size)

        # these are sorted because time-order should be important
        training_indices = np.sort(
            npr.choice(featureindices,
                       size=int(rf_subsample*smoothedmags.size),
                       replace=False)
        )
    else:
        training_indices = np.arange(smoothedmags.size)

    RFR.fit(features[training_indices,:], smoothedmags[training_indices])

    # predict on the full feature set
    flux_corrections = RFR.predict(np.column_stack(finalparam_arrs))
    corrected_fmags = npmedian(fmags) + fmags - flux_corrections

    retdict = {'times':ftimes,
               'mags':corrected_fmags,
               'errs':ferrs,
               'feature_importances':RFR.feature_importances_,
               'regressor':RFR,
               'mags_median':npmedian(corrected_fmags),
               'mags_mad':npmedian(npabs(corrected_fmags -
                                         npmedian(corrected_fmags)))}

    return retdict
