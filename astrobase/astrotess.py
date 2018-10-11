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

import pickle
import os.path
import gzip

import numpy as np
from astropy.io import fits as pyfits


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

    hdulist = pyfits.open(infile)

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



#########################################################
## LCDICT MAKING FUNCTIONS FOR TESS HLSP LC.FITS FILES ##
#########################################################

# these appear to be similar to Kepler LCs, so we'll copy over stuff from
# astrokep.py

# this is the list of keys to pull out of the top header of the FITS
LCTOPKEYS = [
    'DATE-OBS',
    'DATE-END',
    'PROCVER',
    'ORIGIN',
    'DATA_REL',
    'TIMVERSN',
    'OBJECT',
    'TICID',
    'SECTOR',
    'CAMERA',
    'CCD',
    'PXTABLE',
    'RADESYS',
    'RA_OBJ',
    'DEC_OBJ',
    'EQUINOX',
    'PMRA',
    'PMDEC',
    'PMTOTAL',
    'TESSMAG',
    'TEFF',
    'LOGG',
    'MH',
    'RADIUS',
    'TICVER',
    'CRMITEN',
    'CRBLKSZ',
    'CRSPOC',
]

# this is the list of keys to pull out of the light curve header
LCHEADERKEYS = [
    'EXPOSURE',
    'TIMEREF',
    'TASSIGN',
    'TIMESYS',
    'BJDREFI',
    'BJDREFF',
    'TELAPSE',
    'LIVETIME',
    'INT_TIME',
    'NUM_FRM',
    'TIMEDEL',
    'BACKAPP',
    'DEADAPP',
    'VIGNAPP',
    'GAINA',
    'GAINB',
    'GAINC',
    'GAIND',
    'READNOIA',
    'READNOIB',
    'READNOIC',
    'READNOID',
    'CDPP0_5',
    'CDPP1_0',
    'CDPP2_0',
    'PDCVAR',
    'PDCMETHD',
    'CROWDSAP',
    'FLFRCSAP',
    'NSPSDDET',
    'NSPSDCOR'
]

# this is the list of keys to pull out of the light curve FITS table
LCDATAKEYS = [
    'TIME',
    'TIMECORR',
    'CADENCENO',
    'QUALITY',
    'PSF_CENTR1','PSF_CENTR1_ERR','PSF_CENTR2','PSF_CENTR2_ERR',
    'MOM_CENTR1','MOM_CENTR1_ERR','MOM_CENTR2','MOM_CENTR2_ERR',
    'POS_CORR1','POS_CORR2'
]

# this is the list of columns to use for fluxes, backgrounds, errors
LCSAPKEYS = ['SAP_FLUX','SAP_FLUX_ERR','SAP_BKG','SAP_BKG_ERR']
LCPDCKEYS = ['PDCSAP_FLUX','PDCSAP_FLUX_ERR']

# this is the list of keys to pull out of the aperture part of the light curve
# we also pull out the whole pixel mask, which looks something like:
#
# array([[65, 69, 69, 69, 69, 69, 69, 69, 69, 65, 65],
#        [69, 69, 69, 69, 65, 65, 65, 65, 69, 69, 65],
#        [65, 65, 65, 65, 65, 65, 65, 65, 65, 69, 65],
#        [65, 65, 65, 65, 75, 75, 65, 65, 65, 69, 65],
#        [65, 65, 65, 75, 75, 75, 75, 65, 65, 65, 65],
#        [65, 65, 65, 75, 75, 75, 75, 65, 65, 69, 65],
#        [65, 65, 65, 65, 75, 75, 65, 65, 65, 69, 65],
#        [65, 65, 65, 65, 65, 65, 65, 65, 65, 69, 65],
#        [69, 69, 69, 65, 69, 65, 65, 65, 69, 69, 65],
#        [65, 69, 69, 69, 69, 69, 65, 69, 69, 65, 65],
#        [65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65]], dtype=int32)
#
# FIXME: figure out what these values mean (probably flux-collected = 75 /
# flux-available = 69 / flux-in-stamp = 65). we use CDELT1 and CDELT2 below to
# get the pixel scale in arcsec/px
LCAPERTUREKEYS = ['NPIXSAP','NPIXMISS',
                  'CDELT1','CDELT2']


def read_tess_fitslc(lcfits,
                     headerkeys=LCHEADERKEYS,
                     datakeys=LCDATAKEYS,
                     sapkeys=LCSAPKEYS,
                     pdckeys=LCPDCKEYS,
                     topkeys=LCTOPKEYS,
                     apkeys=LCAPERTUREKEYS,
                     normalize=False,
                     appendto=None):
    '''This extracts the light curve from a single TESS .lc.fits file.

    This works on the light curves available at MAST:

    Returns an lcdict.

    If normalize == True, then each component light curve's flux measurements
    will be normalized to 1.0 by dividing out the median flux for the component
    light curve.

    If appendto is an lcdict, will append measurements to that dict. This is
    used for consolidating light curves for the same object across different
    files (sectors/cameras/CCDs?). The appending does not care about the time
    order. To consolidate light curves in time order, use
    consolidate_tess_fitslc below.

    TODO: look at:

    https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf

    for details on the column descriptions and to fill in any other info we
    need.

    '''

    # read the fits file
    hdulist = pyfits.open(lcfits)
    lchdr, lcdata = hdulist[1].header, hdulist[1].data
    lctophdr, lcaperturehdr, lcaperturedata = (hdulist[0].header,
                                               hdulist[2].header,
                                               hdulist[2].data)
    hdulist.close()

    hdrinfo = {}

    # now get the values we want from the header
    for key in headerkeys:
        if key in lchdr and lchdr[key] is not None:
            hdrinfo[key.lower()] = lchdr[key]
        else:
            hdrinfo[key.lower()] = None

    # get the number of detections
    ndet = lchdr['NAXIS2']

    # get the info from the topheader
    for key in topkeys:
        if key in lctophdr and lctophdr[key] is not None:
            hdrinfo[key.lower()] = lctophdr[key]
        else:
            hdrinfo[key.lower()] = None

    # get the info from the lcaperturehdr
    for key in lcaperturehdr:
        if key in lcaperturehdr and lcaperturehdr[key] is not None:
            hdrinfo[key.lower()] = lcaperturehdr[key]
        else:
            hdrinfo[key.lower()] = None


    # if we're appending to another lcdict
    if appendto and isinstance(appendto, dict):

        lcdict = appendto

        # update lcinfo
        lcdict['lcinfo']['timesys'].append(hdrinfo['timesys'])
        lcdict['lcinfo']['bjdoffset'].append(
            hdrinfo['bjdrefi'] + hdrinfo['bjdreff']
        )
        lcdict['lcinfo']['lcaperture'].append(lcaperturedata)
        lcdict['lcinfo']['aperpix_used'].append(hdrinfo['npixsap'])
        lcdict['lcinfo']['aperpix_unused'].append(hdrinfo['npixmiss'])
        lcdict['lcinfo']['pixarcsec'].append(
            (np.abs(hdrinfo['cdelt1']) +
             np.abs(hdrinfo['cdelt2']))*3600.0/2.0
        )
        lcdict['lcinfo']['ndet'].append(ndet)
        lcdict['lcinfo']['exptime'].append(hdrinfo['exposure'])
        lcdict['lcinfo']['sector'].append(hdrinfo['sector'])
        lcdict['lcinfo']['camera'].append(hdrinfo['camera'])
        lcdict['lcinfo']['ccd'].append(hdrinfo['ccd'])

        lcdict['lcinfo']['date_obs_start'].append(hdrinfo['date-obs'])
        lcdict['lcinfo']['date_obs_end'].append(hdrinfo['date-end'])
        lcdict['lcinfo']['pixel_table_id'].append(hdrinfo['pxtable'])
        lcdict['lcinfo']['origin'].append(hdrinfo['origin'])
        lcdict['lcinfo']['datarelease'].append(hdrinfo['data_rel'])
        lcdict['lcinfo']['procversion'].append(hdrinfo['procver'])


        lcdict['lcinfo']['tic_version'].append(hdrinfo['ticver'])
        lcdict['lcinfo']['cr_mitigation'].append(hdrinfo['crmiten'])
        lcdict['lcinfo']['cr_blocksize'].append(hdrinfo['crblksz'])
        lcdict['lcinfo']['cr_spocclean'].append(hdrinfo['crspoc'])

        # update the varinfo for this light curve
        lcdict['varinfo']['cdpp0_5'].append(hdrinfo['cdpp0_5'])
        lcdict['varinfo']['cdpp1_0'].append(hdrinfo['cdpp1_0'])
        lcdict['varinfo']['cdpp2_0'].append(hdrinfo['cdpp2_0'])
        lcdict['varinfo']['pdcvar'].append(hdrinfo['pdcvar'])
        lcdict['varinfo']['pdcmethod'].append(hdrinfo['pdcmethd'])
        lcdict['varinfo']['target_flux_total_flux_ratio_in_aper'].append(
            hdrinfo['crowdsap']
        )
        lcdict['varinfo']['target_flux_fraction_in_aper'].append(
            hdrinfo['flfrcsap']
        )

        # update the light curve columns now
        for key in datakeys:

            if key.lower() in lcdict:
                lcdict[key.lower()] = (
                    np.concatenate((lcdict[key.lower()], lcdata[key]))
                )

        for key in sapkeys:

            if key.lower() in lcdict['sap']:

                sapflux_median = np.nanmedian(lcdata['SAP_FLUX'])
                sapbkg_median = np.nanmedian(lcdata['SAP_BKG'])

                # normalize the current flux measurements if needed
                if normalize and key == 'SAP_FLUX':
                    thislcdata = lcdata[key] / sapflux_median
                elif normalize and key == 'SAP_FLUX_ERR':
                    thislcdata = lcdata[key] / sapflux_median
                elif normalize and key == 'SAP_BKG':
                    thislcdata = lcdata[key] / sapbkg_median
                elif normalize and key == 'SAP_BKG_ERR':
                    thislcdata = lcdata[key] / sapbkg_median
                else:
                    thislcdata = lcdata[key]

                lcdict['sap'][key.lower()] = (
                    np.concatenate((lcdict['sap'][key.lower()], thislcdata))
                )

        for key in pdckeys:

            if key.lower() in lcdict['pdc']:

                pdcsap_flux_median = np.nanmedian(lcdata['PDCSAP_FLUX'])

                # normalize the current flux measurements if needed
                if normalize and key == 'PDCSAP_FLUX':
                    thislcdata = lcdata[key] / pdcsap_flux_median
                elif normalize and key == 'PDCSAP_FLUX_ERR':
                    thislcdata = lcdata[key] / pdcsap_flux_median
                else:
                    thislcdata = lcdata[key]

                lcdict['pdc'][key.lower()] = (
                    np.concatenate((lcdict['pdc'][key.lower()], thislcdata))
                )


        # append some of the light curve information into existing numpy arrays
        # so we can sort on them later
        lcdict['exptime'] = np.concatenate(
            (lcdict['exptime'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['exposure']))
        )
        lcdict['sector'] = np.concatenate(
            (lcdict['sector'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['sector']))
        )
        lcdict['camera'] = np.concatenate(
            (lcdict['camera'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['camera']))
        )
        lcdict['ccd'] = np.concatenate(
            (lcdict['ccd'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['ccd']))
        )

        lcdict['pixel_table_id'] = np.concatenate(
            (lcdict['pixel_table_id'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['pxtable']))
        )
        lcdict['origin'] = np.concatenate(
            (lcdict['origin'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['origin']))
        )
        lcdict['date_obs_start'] = np.concatenate(
            (lcdict['date_obs_start'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['date-obs']))
        )
        lcdict['date_obs_end'] = np.concatenate(
            (lcdict['date_obs_end'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['date-end']))
        )
        lcdict['procversion'] = np.concatenate(
            (lcdict['procversion'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['procver']))
        )
        lcdict['datarelease'] = np.concatenate(
            (lcdict['datarelease'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['data_rel']))
        )


    # otherwise, this is a new lcdict
    else:

        # form the lcdict
        # the metadata is one-elem arrays because we might add on to them later
        lcdict = {
            'objectid':hdrinfo['object'],
            'lcinfo':{
                'timesys':[hdrinfo['timesys']],
                'bjdoffset':[hdrinfo['bjdrefi'] + hdrinfo['bjdreff']],
                'exptime':[hdrinfo['exposure']],
                'lcaperture':[lcaperturedata],
                'aperpix_used':[hdrinfo['npixsap']],
                'aperpix_unused':[hdrinfo['npixmiss']],
                'pixarcsec':[(np.abs(hdrinfo['cdelt1']) +
                              np.abs(hdrinfo['cdelt2']))*3600.0/2.0],
                'ndet':[ndet],
                'origin':[hdrinfo['origin']],
                'procversion':[hdrinfo['procver']],
                'datarelease':[hdrinfo['data_rel']],
                'sector':[hdrinfo['sector']],
                'camera':[hdrinfo['camera']],
                'ccd':[hdrinfo['ccd']],
                'pixel_table_id':[hdrinfo['pxtable']],
                'date_obs_start':[hdrinfo['date-obs']],
                'date_obs_end':[hdrinfo['date-end']],
                'tic_version':[hdrinfo['ticver']],
                'cr_mitigation':[hdrinfo['crmiten']],
                'cr_blocksize':[hdrinfo['crblksz']],
                'cr_spocclean':[hdrinfo['crspoc']],
            },
            'objectinfo':{
                'objectid':hdrinfo['object'],  # repeated here for checkplot use
                'ticid':hdrinfo['ticid'],
                'tessmag':hdrinfo['tessmag'],
                'ra':hdrinfo['ra_obj'],
                'decl':hdrinfo['dec_obj'],
                'pmra':hdrinfo['pmra'],
                'pmdecl':hdrinfo['pmdec'],
                'pmtotal':hdrinfo['pmtotal'],
                'star_teff':hdrinfo['teff'],
                'star_logg':hdrinfo['logg'],
                'star_mh':hdrinfo['mh'],
                'star_radius':hdrinfo['radius'],
            },
            'varinfo':{
                'cdpp0_5':[hdrinfo['cdpp0_5']],
                'cdpp1_0':[hdrinfo['cdpp1_0']],
                'cdpp2_0':[hdrinfo['cdpp2_0']],
                'pdcvar':[hdrinfo['pdcvar']],
                'pdcmethod':[hdrinfo['pdcmethd']],
                'target_flux_total_flux_ratio_in_aper':[hdrinfo['crowdsap']],
                'target_flux_fraction_in_aper':[hdrinfo['flfrcsap']],
            },
            'sap':{},
            'pdc':{},
        }

        # get the LC columns
        for key in datakeys:
            lcdict[key.lower()] = lcdata[key]
        for key in sapkeys:
            lcdict['sap'][key.lower()] = lcdata[key]
        for key in pdckeys:
            lcdict['pdc'][key.lower()] = lcdata[key]

        # turn some of the light curve information into numpy arrays so we can
        # sort on them later
        lcdict['exptime'] = np.full_like(lcdict['time'],
                                         lcdict['lcinfo']['exptime'][0],
                                         dtype=np.float64)
        lcdict['sector'] = np.full_like(lcdict['time'],
                                        lcdict['lcinfo']['sector'][0],
                                        dtype=np.int64)
        lcdict['camera'] = np.full_like(lcdict['time'],
                                        lcdict['lcinfo']['camera'][0],
                                        dtype=np.int64)
        lcdict['ccd'] = np.full_like(lcdict['time'],
                                     lcdict['lcinfo']['ccd'][0],
                                     dtype=np.int64)

        lcdict['pixel_table_id'] = np.full_like(
            lcdict['time'],
            lcdict['lcinfo']['pixel_table_id'][0],
            dtype=np.int64,
        )
        lcdict['origin'] = np.full_like(
            lcdict['time'],
            lcdict['lcinfo']['origin'][0],
            dtype='U100',
        )
        lcdict['date_obs_start'] = np.full_like(
            lcdict['time'],
            lcdict['lcinfo']['date_obs_start'][0],
            dtype='U100',
        )
        lcdict['date_obs_end'] = np.full_like(
            lcdict['time'],
            lcdict['lcinfo']['date_obs_end'][0],
            dtype='U100',
        )
        lcdict['procversion'] = np.full_like(
            lcdict['time'],
            lcdict['lcinfo']['procversion'][0],
            dtype='U255',
        )
        lcdict['datarelease'] = np.full_like(
            lcdict['time'],
            lcdict['lcinfo']['datarelease'][0],
            dtype=np.int64,
        )


        # normalize the SAP and PDCSAP fluxes, errs, and backgrounds if needed
        if normalize:

            sapflux_median = np.nanmedian(lcdict['sap']['sap_flux'])
            sapbkg_median = np.nanmedian(lcdict['sap']['sap_bkg'])
            pdcsap_flux_median = np.nanmedian(lcdict['pdc']['pdcsap_flux'])

            lcdict['sap']['sap_flux'] = (
                lcdict['sap']['sap_flux'] /
                sapflux_median
            )
            lcdict['sap']['sap_flux_err'] = (
                lcdict['sap']['sap_flux_err'] /
                sapflux_median
            )

            lcdict['sap']['sap_bkg'] = (
                lcdict['sap']['sap_bkg'] /
                sapbkg_median
            )
            lcdict['sap']['sap_bkg_err'] = (
                lcdict['sap']['sap_bkg_err'] /
                sapbkg_median
            )

            lcdict['pdc']['pdcsap_flux'] = (
                lcdict['pdc']['pdcsap_flux'] /
                pdcsap_flux_median
            )
            lcdict['pdc']['pdcsap_flux_err'] = (
                lcdict['pdc']['pdcsap_flux_err'] /
                pdcsap_flux_median
            )

    ## END OF LIGHT CURVE CONSTRUCTION ##

    # update the lcdict columns with the actual columns
    lcdict['columns'] = (
        [x.lower() for x in datakeys] +
        ['sap.%s' % x.lower() for x in sapkeys] +
        ['pdc.%s' % x.lower() for x in pdckeys] +
        ['exptime','sector','camera','ccd', 'pixel_table_id',
         'origin', 'date_obs_start', 'date_obs_end',
         'procversion', 'datarelease']
    )

    # return the lcdict at the end
    return lcdict



##################
## INPUT/OUTPUT ##
##################

def tess_lcdict_to_pkl(lcdict,
                       outfile=None):
    '''This simply writes the lcdict to a pickle.

    '''

    if not outfile:
        outfile = '%s-tesslc.pkl' % lcdict['objectid'].replace(' ','')

    # we're using pickle.HIGHEST_PROTOCOL here, this will make Py3 pickles
    # unreadable for Python 2.7
    with open(outfile,'wb') as outfd:
        pickle.dump(lcdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    return os.path.abspath(outfile)



def read_tess_pklc(picklefile):
    '''This turns the pickled lightcurve back into an lcdict.

    '''

    if picklefile.endswith('.gz'):
        infd = gzip.open(picklefile, 'rb')
    else:
        infd = open(picklefile, 'rb')

    try:
        with infd:
            lcdict = pickle.load(infd)

    except UnicodeDecodeError:

        with open(picklefile,'rb') as infd:
            lcdict = pickle.load(infd, encoding='latin1')

        LOGWARNING('pickle %s was probably from Python 2 '
                   'and failed to load without using "latin1" encoding. '
                   'This is probably a numpy issue: '
                   'http://stackoverflow.com/q/11305790' % picklefile)

    return lcdict
