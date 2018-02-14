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

from scipy.optimization import leastsq
from scipy.signal import medfilt, savgol_filter
from astropy.convolution import convolve, Gaussian1DKernel

# for random forest EPD
from sklearn.ensemble import RandomForestRegressor



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



###################################################
## HAT-SPECIFIC EXTERNAL PARAMETER DECORRELATION ##
###################################################

def _epd_function(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge):
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
            coeffs[19]*bge)



def _epd_residual(coeffs, mags, fsv, fdv, fkv, xcc, ycc, bgv, bge):
    '''
    This is the residual function to minimize using scipy.optimize.leastsq.

    '''

    f = _epd_function(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge)
    residual = mags - f
    return residual



def epd_magseries(times, mags, errs,
                  fsv, fdv, fkv, xcc, ycc, bgv, bge,
                  magsarefluxes=False,
                  sigclip=3.0,
                  epdsmooth_windowsize=27,
                  epdsmooth_func=_smooth_savgol,
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

    epdsmooth_windowsize is the number of points to smooth over to generate a
    smoothed light curve to train the regressor against.

    epdsmooth_func sets the smoothing filter function to use. A Savitsky-Golay
    filter is used to smooth the light curve by default.

    epdsmooth_extraparams is a dict of any extra filter params to supply to the
    smoothing function.

    NOTE: The errs are completely ignored and returned unchanged (except for
    sigclip and finite filtering).

    '''

    stimes, smags, serrs, separams = sigclip_magseries_with_extparams(
        times, mags, errs,
        [fsv, fdv, fkv, xcc, ycc, bgv, bge],
        sigclip=sigclip,
        magsarefluxes=magsarefluxes
    )
    sfsv, sfdv, sfkv, sxcc, sycc, sbgv, sbge = separams

    # smooth the signal
    if isinstance(epdsmooth_extraparams, dict):
        smoothedmags = epdsmooth_func(smags,
                                      epdsmooth_windowsize,
                                      **epdsmooth_extraparams)
    else:
        smoothedmags = epdsmooth_func(smags, epdsmooth_windowsize)

    # initial fit coeffs
    initcoeffs = npones(20)

    # fit the smoothed mags and find the EPD function coefficients
    leastsqfit = leastsq(_epd_residual,
                         initcoeffs,
                         args=(smoothedmags,
                               sfsv, sfdv, sfkv, sxcc, sycc, sbgv, sbge))

    # if the fit succeeds, then get the EPD mags
    if leastsqfit[-1] in (1,2,3,4):

        fitcoeffs = leastsqfit[0]
        epdfit = _epd_function(fitcoeffs,
                               sfsv, sfdv, sfkv, sxcc, sycc, sbgv, sbge)

        epdmags = npmedian(smags) + smags - epdfit

        retdict = {'times':stimes,
                   'mags':epdmags,
                   'errs':serrs,
                   'fitcoeffs':fitcoeffs,
                   'fitinfo':leastsqfit}

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
                    sigclip=3.0,
                    magsarefluxes=False,
                    epdsmooth=False,
                    epdsmooth_windowsize=27,
                    epdsmooth_func=_smooth_savgol,
                    epdsmooth_extraparams=None,
                    rf_subsample=0.5,
                    rf_ntrees=200,
                    rf_extraparams=None):
    '''This uses a RandomForestRegressor to trend-filter the given magseries.

    times and mags are ndarrays of time and magnitude values to filter.

    extparam_arrs is a list of ndarrays of external parameters to decorrelate
    against. These should all be the same size as times and mags.

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

    stimes, smags, serrs, eparams = sigclip_magseries_with_extparams(
        times, mags, errs,
        externalparam_arrs,
        sigclip=sigclip,
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
    featureindices = np.arange(smoothedmags.size)
    training_indices = npr.choice(featureindices,
                                  size=int(rf_subsample*smoothedmags.size),
                                  replace=False)

    RFR.fit(features[:,training_indices], smoothedmags[training_indices])

    # predict on the full feature set
    flux_corrections = RFR.predict(features)
    corrected_smags = npmedian(smags) + smags - flux_corrections

    retdict = {'times':stimes,
               'mags':corrected_smags,
               'errs':serrs,
               'feature_importances':RFR.feature_importances_,
               'regressor':RFR}

    return retdict



#####################################
## TREND FILTERING ALGORITHM (TFA) ##
#####################################
