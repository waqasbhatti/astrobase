#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
astrotess.py - Luke Bouma (luke@astro.princeton.edu) - 09/2018

Contains various tools for analyzing TESS light curves.
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


##################
## MAIN IMPORTS ##
##################

import numpy as np
from astropy.io import fits


########################################
## READING AMES LLC FITS LCS FOR TESS ##
########################################

def get_time_flux_errs_from_Ames_lightcurve(infile,
                                            lctype,
                                            cadence_min=2):
    '''
    MIT TOI alerts include Ames lightcurve files. This function gets the
    finite, nonzero times, fluxes, and errors with QUALITY == 0.

    NB. the PDCSAP lightcurve typically still need "pre-whitening" after this
    step.

    args:
        infile (str): path to *.fits.gz TOI alert file, from Ames pipeline.
        lctype (str): PDCSAP or SAP

    kwargs:
        cadence_min (float): expected frame cadence in units of minutes. Raises
        ValueError if you use the wrong cadence.

    returns:
        (tuple): times, normalized (to median) fluxes, flux errors.
    '''

    if lctype not in ('PDCSAP','SAP'):
        raise ValueError('unknown light curve type requested: %s' % lctype)

    hdulist = fits.open(infile)

    main_hdr = hdulist[0].header
    lc_hdr = hdulist[1].header
    lc = hdulist[1].data

    if (('Ames' not in main_hdr['ORIGIN']) or
        ('LIGHTCURVE' not in lc_hdr['EXTNAME'])):
        raise ValueError(
            'could not understand input LC format. '
            'Is it a TESS TOI LC file?'
        )

    time = lc['TIME']
    flux = lc['{:s}_FLUX'.format(lctype)]
    err_flux = lc['{:s}_FLUX_ERR'.format(lctype)]

    # REMOVE POINTS FLAGGED WITH:
    # attitude tweaks, safe mode, coarse/earth pointing, argabrithening events,
    # reaction wheel desaturation events, cosmic rays in optimal aperture
    # pixels, manual excludes, discontinuities, stray light from Earth or Moon
    # in camera FoV.
    # (Note: it's not clear to me what a lot of these mean. Also most of these
    # columns are probably not correctly propagated right now.)
    sel = (lc['QUALITY'] == 0)
    sel &= np.isfinite(time)
    sel &= np.isfinite(flux)
    sel &= np.isfinite(err_flux)
    sel &= ~np.isnan(time)
    sel &= ~np.isnan(flux)
    sel &= ~np.isnan(err_flux)
    sel &= (time != 0)
    sel &= (flux != 0)
    sel &= (err_flux != 0)

    time = time[sel]
    flux = flux[sel]
    err_flux = err_flux[sel]

    # ensure desired cadence
    lc_cadence_diff = np.abs(np.nanmedian(np.diff(time))*24*60 - cadence_min)

    if lc_cadence_diff > 1.0e-2:
        raise ValueError(
            'the light curve is not at the required cadence specified: %.2f' %
            cadence_min
        )

    fluxmedian = np.nanmedian(flux)
    flux /= fluxmedian
    err_flux /= fluxmedian

    return time, flux, err_flux
