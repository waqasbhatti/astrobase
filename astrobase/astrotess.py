#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# astrotess.py - Luke Bouma (luke@astro.princeton.edu) - 09/2018

'''
Contains various tools for analyzing TESS light curves.

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


##################
## MAIN IMPORTS ##
##################

import pickle
import os.path
import gzip
import sys
import glob

import numpy as np
from astropy.io import fits as pyfits


#######################################
## UTILITY FUNCTIONS FOR FLUXES/MAGS ##
#######################################

def normalized_flux_to_mag(lcdict,
                           columns=('sap.sap_flux',
                                    'sap.sap_flux_err',
                                    'sap.sap_bkg',
                                    'sap.sap_bkg_err',
                                    'pdc.pdcsap_flux',
                                    'pdc.pdcsap_flux_err')):
    '''This converts the normalized fluxes in the TESS lcdicts to TESS mags.

    Uses the object's TESS mag stored in lcdict['objectinfo']['tessmag']::

        mag - object_tess_mag = -2.5 log (flux/median_flux)

    Parameters
    ----------

    lcdict : lcdict
        An `lcdict` produced by `read_tess_fitslc` or
        `consolidate_tess_fitslc`. This must have normalized fluxes in its
        measurement columns (use the `normalize` kwarg for these functions).

    columns : sequence of str
        The column keys of the normalized flux and background measurements in
        the `lcdict` to operate on and convert to magnitudes in TESS band (T).

    Returns
    -------

    lcdict
        The returned `lcdict` will contain extra columns corresponding to
        magnitudes for each input normalized flux/background column.

    '''

    tess_mag = lcdict['objectinfo']['tessmag']

    for key in columns:

        k1, k2 = key.split('.')

        if 'err' not in k2:

            lcdict[k1][k2.replace('flux','mag')] = (
                tess_mag - 2.5*np.log10(lcdict[k1][k2])
            )

        else:

            lcdict[k1][k2.replace('flux','mag')] = (
                - 2.5*np.log10(1.0 - lcdict[k1][k2])
            )

    return lcdict


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
                     appendto=None,
                     filterqualityflags=False,
                     nanfilter=None,
                     timestoignore=None):
    '''This extracts the light curve from a single TESS .lc.fits file.

    This works on the light curves available at MAST.

    TODO: look at:

    https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf

    for details on the column descriptions and to fill in any other info we
    need.

    Parameters
    ----------

    lcfits : str
        The filename of a MAST Kepler/K2 light curve FITS file.

    headerkeys : list
        A list of FITS header keys that will be extracted from the FITS light
        curve file. These describe the observations. The default value for this
        is given in `LCHEADERKEYS` above.

    datakeys : list
        A list of FITS column names that correspond to the auxiliary
        measurements in the light curve. The default is `LCDATAKEYS` above.

    sapkeys : list
        A list of FITS column names that correspond to the SAP flux
        measurements in the light curve. The default is `LCSAPKEYS` above.

    pdckeys : list
        A list of FITS column names that correspond to the PDC flux
        measurements in the light curve. The default is `LCPDCKEYS` above.

    topkeys : list
        A list of FITS header keys that describe the object in the light
        curve. The default is `LCTOPKEYS` above.

    apkeys : list
        A list of FITS header keys that describe the flux measurement apertures
        used by the TESS pipeline. The default is `LCAPERTUREKEYS` above.

    normalize : bool
        If True, then the light curve's SAP_FLUX and PDCSAP_FLUX measurements
        will be normalized to 1.0 by dividing out the median flux for the
        component light curve.

    appendto : lcdict or None
        If appendto is an `lcdict`, will append measurements of this `lcdict` to
        that `lcdict`. This is used for consolidating light curves for the same
        object across different files (sectors/cameras/CCDs?). The appending
        does not care about the time order. To consolidate light curves in time
        order, use `consolidate_tess_fitslc` below.

    filterqualityflags : bool
        If True, will remove any measurements that have non-zero quality flags
        present. This usually indicates an issue with the instrument or
        spacecraft.

    nanfilter : {'sap','pdc','sap,pdc'} or None
        Indicates the flux measurement type(s) to apply the filtering to.

    timestoignore : list of tuples or None
        This is of the form::

            [(time1_start, time1_end), (time2_start, time2_end), ...]

        and indicates the start and end times to mask out of the final
        lcdict. Use this to remove anything that wasn't caught by the quality
        flags.

    Returns
    -------

    lcdict
        Returns an `lcdict` (this is useable by most astrobase functions for LC
        processing).

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

                # normalize the current flux measurements if needed
                if normalize and key == 'SAP_FLUX':
                    thislcdata = lcdata[key] / sapflux_median
                elif normalize and key == 'SAP_FLUX_ERR':
                    thislcdata = lcdata[key] / sapflux_median
                elif normalize and key == 'SAP_BKG':
                    thislcdata = lcdata[key] / sapflux_median
                elif normalize and key == 'SAP_BKG_ERR':
                    thislcdata = lcdata[key] / sapflux_median
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
                          hdrinfo['exposure'],
                          dtype=np.float64))
        )
        lcdict['sector'] = np.concatenate(
            (lcdict['sector'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['sector'],
                          dtype=np.int64))
        )
        lcdict['camera'] = np.concatenate(
            (lcdict['camera'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['camera'],
                          dtype=np.int64))
        )
        lcdict['ccd'] = np.concatenate(
            (lcdict['ccd'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['ccd'],
                          dtype=np.int64))
        )

        lcdict['pixel_table_id'] = np.concatenate(
            (lcdict['pixel_table_id'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['pxtable'],
                          dtype=np.int64))
        )
        lcdict['origin'] = np.concatenate(
            (lcdict['origin'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['origin'],
                          dtype='U100'))
        )
        lcdict['date_obs_start'] = np.concatenate(
            (lcdict['date_obs_start'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['date-obs'],
                          dtype='U100'))
        )
        lcdict['date_obs_end'] = np.concatenate(
            (lcdict['date_obs_end'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['date-end'],
                          dtype='U100'))
        )
        lcdict['procversion'] = np.concatenate(
            (lcdict['procversion'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['procver'],
                          dtype='U255'))
        )
        lcdict['datarelease'] = np.concatenate(
            (lcdict['datarelease'],
             np.full_like(lcdata['TIME'],
                          hdrinfo['data_rel'],
                          dtype=np.int64))
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
                'objectid':hdrinfo['object'],
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
                'observatory':'TESS',
                'telescope':'TESS photometer',
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
                sapflux_median
            )
            lcdict['sap']['sap_bkg_err'] = (
                lcdict['sap']['sap_bkg_err'] /
                sapflux_median
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

    # update the ndet key in the objectinfo with the sum of all observations
    lcdict['objectinfo']['ndet'] = sum(lcdict['lcinfo']['ndet'])

    # filter the LC dict if requested
    if (filterqualityflags is not False or
        nanfilter is not None or
        timestoignore is not None):
        lcdict = filter_tess_lcdict(lcdict,
                                    filterqualityflags,
                                    nanfilter=nanfilter,
                                    timestoignore=timestoignore)

    # return the lcdict at the end
    return lcdict


def consolidate_tess_fitslc(lclist,
                            normalize=True,
                            filterqualityflags=False,
                            nanfilter=None,
                            timestoignore=None,
                            headerkeys=LCHEADERKEYS,
                            datakeys=LCDATAKEYS,
                            sapkeys=LCSAPKEYS,
                            pdckeys=LCPDCKEYS,
                            topkeys=LCTOPKEYS,
                            apkeys=LCAPERTUREKEYS):
    '''This consolidates a list of LCs for a single TIC object.

    NOTE: if light curve time arrays contain nans, these and their associated
    measurements will be sorted to the end of the final combined arrays.

    Parameters
    ----------

    lclist : list of str, or str
        `lclist` is either a list of actual light curve files or a string that
        is valid for glob.glob to search for and generate a light curve list
        based on the file glob. This is useful for consolidating LC FITS files
        across different TESS sectors for a single TIC ID using a glob like
        `*<TICID>*_lc.fits`.

    normalize : bool
        If True, then the light curve's SAP_FLUX and PDCSAP_FLUX measurements
        will be normalized to 1.0 by dividing out the median flux for the
        component light curve.

    filterqualityflags : bool
        If True, will remove any measurements that have non-zero quality flags
        present. This usually indicates an issue with the instrument or
        spacecraft.

    nanfilter : {'sap','pdc','sap,pdc'} or None
        Indicates the flux measurement type(s) to apply the filtering to.

    timestoignore : list of tuples or None
        This is of the form::

            [(time1_start, time1_end), (time2_start, time2_end), ...]

        and indicates the start and end times to mask out of the final
        lcdict. Use this to remove anything that wasn't caught by the quality
        flags.

    headerkeys : list
        A list of FITS header keys that will be extracted from the FITS light
        curve file. These describe the observations. The default value for this
        is given in `LCHEADERKEYS` above.

    datakeys : list
        A list of FITS column names that correspond to the auxiliary
        measurements in the light curve. The default is `LCDATAKEYS` above.

    sapkeys : list
        A list of FITS column names that correspond to the SAP flux
        measurements in the light curve. The default is `LCSAPKEYS` above.

    pdckeys : list
        A list of FITS column names that correspond to the PDC flux
        measurements in the light curve. The default is `LCPDCKEYS` above.

    topkeys : list
        A list of FITS header keys that describe the object in the light
        curve. The default is `LCTOPKEYS` above.

    apkeys : list
        A list of FITS header keys that describe the flux measurement apertures
        used by the TESS pipeline. The default is `LCAPERTUREKEYS` above.

    Returns
    -------

    lcdict
        Returns an `lcdict` (this is useable by most astrobase functions for LC
        processing).

    '''

    # if the lclist is a string, assume that we're passing in a fileglob
    if isinstance(lclist, str):

        matching = glob.glob(lclist,
                             recursive=True)
        LOGINFO('found %s LCs: %r' % (len(matching), matching))

        if len(matching) == 0:
            LOGERROR('could not find any TESS LC files matching glob: %s' %
                     lclist)
            return None

    # if the lclist is an actual list of LCs, then use it directly
    else:

        matching = lclist

    # get the first file
    consolidated = read_tess_fitslc(matching[0],
                                    normalize=normalize,
                                    headerkeys=LCHEADERKEYS,
                                    datakeys=LCDATAKEYS,
                                    sapkeys=LCSAPKEYS,
                                    pdckeys=LCPDCKEYS,
                                    topkeys=LCTOPKEYS,
                                    apkeys=LCAPERTUREKEYS)

    # get the rest of the files
    if len(matching) > 1:

        for lcf in matching[1:]:

            consolidated = read_tess_fitslc(lcf,
                                            appendto=consolidated,
                                            normalize=normalize,
                                            headerkeys=LCHEADERKEYS,
                                            datakeys=LCDATAKEYS,
                                            sapkeys=LCSAPKEYS,
                                            pdckeys=LCPDCKEYS,
                                            topkeys=LCTOPKEYS,
                                            apkeys=LCAPERTUREKEYS)

    # get the sort indices. we use time for the columns and sectors for the
    # bits in lcinfo and varinfo
    LOGINFO('sorting by time...')

    # NOTE: nans in time will be sorted to the end of the array
    finiteind = np.isfinite(consolidated['time'])
    if np.sum(finiteind) < consolidated['time'].size:
        LOGWARNING('some time values are nan! '
                   'measurements at these times will be '
                   'sorted to the end of the column arrays.')

    # get the time sort index
    column_sort_ind = np.argsort(consolidated['time'])

    # sort the columns by time
    for col in consolidated['columns']:
        if '.' in col:
            key, subkey = col.split('.')
            consolidated[key][subkey] = (
                consolidated[key][subkey][column_sort_ind]
            )
        else:
            consolidated[col] = consolidated[col][column_sort_ind]

    info_sort_ind = np.argsort(consolidated['lcinfo']['sector'])

    # sort the keys in lcinfo
    for key in consolidated['lcinfo']:
        consolidated['lcinfo'][key] = (
            np.array(consolidated['lcinfo'][key])[info_sort_ind].tolist()
        )

    # sort the keys in varinfo
    for key in consolidated['varinfo']:
        consolidated['varinfo'][key] = (
            np.array(consolidated['varinfo'][key])[info_sort_ind].tolist()
        )

    # filter the LC dict if requested
    # we do this at the end
    if (filterqualityflags is not False or
        nanfilter is not None or
        timestoignore is not None):
        consolidated = filter_tess_lcdict(consolidated,
                                          filterqualityflags,
                                          nanfilter=nanfilter,
                                          timestoignore=timestoignore)

    return consolidated


##################
## INPUT/OUTPUT ##
##################

def tess_lcdict_to_pkl(lcdict,
                       outfile=None):
    '''This writes the `lcdict` to a Python pickle.

    Parameters
    ----------

    lcdict : lcdict
        This is the input `lcdict` to write to a pickle.

    outfile : str or None
        If this is None, the object's Kepler ID/EPIC ID will determined from the
        `lcdict` and used to form the filename of the output pickle file. If
        this is a `str`, the provided filename will be used.

    Returns
    -------

    str
        The absolute path to the written pickle file.

    '''

    if not outfile:
        outfile = '%s-tesslc.pkl' % lcdict['objectid'].replace(' ','')

    # we're using pickle.HIGHEST_PROTOCOL here, this will make Py3 pickles
    # unreadable for Python 2.7
    with open(outfile,'wb') as outfd:
        pickle.dump(lcdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    return os.path.abspath(outfile)


def read_tess_pklc(picklefile):
    '''This turns the pickled lightcurve file back into an `lcdict`.

    Parameters
    ----------

    picklefile : str
        The path to a previously written Kepler LC picklefile generated by
        `tess_lcdict_to_pkl` above.

    Returns
    -------

    lcdict
        Returns an `lcdict` (this is useable by most astrobase functions for LC
        processing).

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


################################
## TESS LIGHTCURVE PROCESSING ##
################################

def filter_tess_lcdict(lcdict,
                       filterqualityflags=True,
                       nanfilter='sap,pdc,time',
                       timestoignore=None,
                       quiet=False):
    '''This filters the provided TESS `lcdict`, removing nans and bad
    observations.

    By default, this function removes points in the TESS LC that have ANY
    quality flags set.

    Parameters
    ----------

    lcdict : lcdict
        An `lcdict` produced by `consolidate_tess_fitslc` or
        `read_tess_fitslc`.

    filterflags : bool
        If True, will remove any measurements that have non-zero quality flags
        present. This usually indicates an issue with the instrument or
        spacecraft.

    nanfilter : {'sap','pdc','sap,pdc'}
        Indicates the flux measurement type(s) to apply the filtering to.

    timestoignore : list of tuples or None
        This is of the form::

            [(time1_start, time1_end), (time2_start, time2_end), ...]

        and indicates the start and end times to mask out of the final
        lcdict. Use this to remove anything that wasn't caught by the quality
        flags.

    Returns
    -------

    lcdict
        Returns an `lcdict` (this is useable by most astrobase functions for LC
        processing). The `lcdict` is filtered IN PLACE!

    '''

    cols = lcdict['columns']

    # filter all bad LC points as noted by quality flags
    if filterqualityflags:

        nbefore = lcdict['time'].size
        filterind = lcdict['quality'] == 0

        for col in cols:
            if '.' in col:
                key, subkey = col.split('.')
                lcdict[key][subkey] = lcdict[key][subkey][filterind]
            else:
                lcdict[col] = lcdict[col][filterind]

        nafter = lcdict['time'].size
        if not quiet:
            LOGINFO('applied quality flag filter, '
                    'ndet before = %s, ndet after = %s'
                    % (nbefore, nafter))

    if nanfilter and nanfilter == 'sap,pdc,time':
        notnanind = (
            np.isfinite(lcdict['sap']['sap_flux']) &
            np.isfinite(lcdict['sap']['sap_flux_err']) &
            np.isfinite(lcdict['pdc']['pdcsap_flux']) &
            np.isfinite(lcdict['pdc']['pdcsap_flux_err']) &
            np.isfinite(lcdict['time'])
        )
    elif nanfilter and nanfilter == 'sap,time':
        notnanind = (
            np.isfinite(lcdict['sap']['sap_flux']) &
            np.isfinite(lcdict['sap']['sap_flux_err']) &
            np.isfinite(lcdict['time'])
        )
    elif nanfilter and nanfilter == 'pdc,time':
        notnanind = (
            np.isfinite(lcdict['pdc']['pdcsap_flux']) &
            np.isfinite(lcdict['pdc']['pdcsap_flux_err']) &
            np.isfinite(lcdict['time'])
        )
    elif nanfilter is None:
        pass
    else:
        raise NotImplementedError

    # remove nans from all columns
    if nanfilter:

        nbefore = lcdict['time'].size
        for col in cols:
            if '.' in col:
                key, subkey = col.split('.')
                lcdict[key][subkey] = lcdict[key][subkey][notnanind]
            else:
                lcdict[col] = lcdict[col][notnanind]

        nafter = lcdict['time'].size

        if not quiet:
            LOGINFO('removed nans, ndet before = %s, ndet after = %s'
                    % (nbefore, nafter))

    # exclude all times in timestoignore
    if (timestoignore and
        isinstance(timestoignore, list) and
        len(timestoignore) > 0):

        exclind = np.full_like(lcdict['time'],True).astype(bool)
        nbefore = exclind.size

        # get all the masks
        for ignoretime in timestoignore:
            time0, time1 = ignoretime[0], ignoretime[1]
            thismask = ~((lcdict['time'] >= time0) & (lcdict['time'] <= time1))
            exclind = exclind & thismask

        # apply the masks
        for col in cols:
            if '.' in col:
                key, subkey = col.split('.')
                lcdict[key][subkey] = lcdict[key][subkey][exclind]
            else:
                lcdict[col] = lcdict[col][exclind]

        nafter = lcdict['time'].size
        if not quiet:
            LOGINFO('removed timestoignore, ndet before = %s, ndet after = %s'
                    % (nbefore, nafter))

    return lcdict
