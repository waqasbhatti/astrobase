#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
astrotess.py - Luke Bouma (luke@astro.princeton.edu) - 09/2018

Contains various tools for analyzing TESS light curves.
'''

import numpy as np
from astropy.io import fits

def get_time_flux_errs_from_Ames_lightcurve(infile, lctype, cadence_min=2):
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
        AssertionError is you use wrong cadence.

    returns:
        (tuple): times, fluxes (in electrons/sec), flux errors.
    '''
    assert lctype == 'PDCSAP' or lctype == 'SAP'

    hdulist = fits.open(infile)

    main_hdr = hdulist[0].header
    lc_hdr = hdulist[1].header
    lc = hdulist[1].data

    assert 'Ames' in main_hdr['ORIGIN']
    assert 'LIGHTCURVE' in lc_hdr['EXTNAME']

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

    # ensure desired minute cadence
    assert np.abs(np.nanmedian(np.diff(time))*24*60 - cadence_min) < 1e-2

    fluxmedian = np.nanmedian(flux)
    flux /= fluxmedian
    err_flux /= fluxmedian

    return time, flux, err_flux
