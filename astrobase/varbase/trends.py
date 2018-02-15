#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''trends.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2018

Contains light curve trend-removal tools, such as external parameter
decorrelation (EPD) and an implementation of the trend filtering algorithm
(TFA).

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
import numpy.random as npr
RANDSEED = 0xdecaff
npr.seed(RANDSEED)

from numpy import isfinite as npisfinite, median as npmedian, \
    abs as npabs, pi as MPI
from numpy.linalg import lstsq

from scipy.optimize import leastsq
from scipy.signal import medfilt, savgol_filter
from astropy.convolution import convolve, Gaussian1DKernel

# for random forest EPD
from sklearn.ensemble import RandomForestRegressor

from ..lcmath import sigclip_magseries_with_extparams

#########################
## SMOOTHING FUNCTIONS ##
#########################

def smooth_magseries_medfilt(mags, windowsize):
    '''
    This smooths the magseries with a median filter.

    '''

    return medfilt(mags, windowsize)



def smooth_magseries_gaussfilt(mags, windowsize, windowfwhm=7):
    '''
    This smooths the magseries with a Gaussian kernel.

    '''

    convkernel = Gaussian1DKernel(windowfwhm, x_size=windowsize)
    smoothed = convolve(mags, convkernel, boundary='extend')
    return smoothed



def smooth_magseries_savgol(mags, windowsize, polyorder=2):
    '''
    This smooths the magseries with a Savitsky-Golay filter.

    '''

    smoothed = savgol_filter(mags, windowsize, polyorder)
    return smoothed


########################################
## OLD EPD FUNCTIONS (USED FOR HATPI) ##
########################################

def old_epd_diffmags(coeff, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag):
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



def old_epd_magseries(times, mags, errs,
                      fsv, fdv, fkv, xcc, ycc, bgv, bge,
                      epdsmooth_windowsize=21,
                      epdsmooth_sigclip=3.0,
                      epdsmooth_func=smooth_magseries_medfilt,
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
                           old_epd_diffmags(coeffs, fsv, fdv,
                                            fkv, xcc, ycc, bgv, bge, mags)),
                   'errs':errs,
                   'fitcoeffs':coeffs,
                   'residuals':residuals}

        return retdict

    # if the solution fails, return nothing
    except Exception as e:

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
            coeffs[10]*np.sin(2*MPI*xcc) +
            coeffs[11]*np.cos(2*MPI*xcc) +
            coeffs[12]*np.sin(2*MPI*ycc) +
            coeffs[13]*np.cos(2*MPI*ycc) +
            coeffs[14]*np.sin(4*MPI*xcc) +
            coeffs[15]*np.cos(4*MPI*xcc) +
            coeffs[16]*np.sin(4*MPI*ycc) +
            coeffs[17]*np.cos(4*MPI*ycc) +
            coeffs[18]*bgv +
            coeffs[19]*bge +
            coeffs[20]*iha +
            coeffs[21]*izd
    )



def _epd_residual(coeffs, mags, fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd):
    '''
    This is the residual function to minimize using scipy.optimize.leastsq.

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

    The HAT light-curve-specific external parameters are:

    S: the 'fsv' column,
    D: the 'fdv' column,
    K: the 'fkv' column,
    x coords: the 'xcc' column,
    y coords: the 'ycc' column,
    background: the 'bgv' column,
    background error: the 'bge' column
    hour angle: the 'iha' column
    zenith distance: the 'izd' column

    epdsmooth_windowsize is the number of points to smooth over to generate a
    smoothed light curve to train the regressor against.

    epdsmooth_func sets the smoothing filter function to use. A Savitsky-Golay
    filter is used to smooth the light curve by default.

    epdsmooth_extraparams is a dict of any extra filter params to supply to the
    smoothing function.

    NOTE: The errs are completely ignored and returned unchanged (except for
    sigclip and finite filtering).

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
                   'fitmags':epdfit}

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
                    epdsmooth_windowsize=201,
                    epdsmooth_func=smooth_magseries_savgol,
                    epdsmooth_extraparams=None,
                    rf_subsample=1.0,
                    rf_ntrees=300,
                    rf_extraparams={'criterion':'mse',
                                    'oob_score':False,
                                    'n_jobs':-1}):
    '''This uses a RandomForestRegressor to de-correlate the given magseries.

    times, mags, errs are ndarrays of time and magnitude values to filter.

    externalparam_arrs is a list of ndarrays of external parameters to
    decorrelate against. These should all be the same size as times, mags, errs.

    epdsmooth = True sets the training light curve to be a smoothed version of
    the sigma-clipped light curve.

    epdsmooth_windowsize is the number of points to smooth over to generate a
    smoothed light curve to train the regressor against.

    epdsmooth_func sets the smoothing filter function to use. A Savitsky-Golay
    filter is used to smooth the light curve by default.

    epdsmooth_extraparams is a dict of any extra filter params to supply to the
    smoothing function.

    rf_subsample is the fraction of the size of the mags array to use for
    training the random forest regressor.

    rf_ntrees is the number of trees to use for the RandomForestRegressor.

    rf_extraparams is any extra params to provide to the RandomForestRegressor
    instance as a dict.

    Returns a dict with decorrelated mags and the usual info from the
    RandomForestRegressor: variable importances, etc.

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
               'regressor':RFR}

    return retdict



#####################################
## TREND FILTERING ALGORITHM (TFA) ##
#####################################
