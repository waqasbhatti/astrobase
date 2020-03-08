#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tesslightcurves.py - Luke Bouma (bouma.luke@gmail.com) - Nov 2019
# License: MIT - see the LICENSE file for the full text.

'''
Useful tools for acquiring TESS light-curves.  This module contains a number of
non-standard dependencies, including lightkurve, eleanor, and astroquery.

Light-curve retrieval: get light-curves from all sectors for a tic_id.
    get_two_minute_spoc_lightcurves
    get_hlsp_lightcurves
    get_eleanor_lightcurves

Visibility queries: check if an ra/dec was observed.
    is_two_minute_spoc_lightcurve_available
    get_tess_visibility_given_ticid
    get_tess_visibility_given_ticids

TODO:
    get_cpm_lightcurve
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

from astropy.coordinates import SkyCoord
from astrobase.services.identifiers import simbad_to_tic
from astrobase.services.mast import tic_objectsearch

# This module contains a number of non-standard dependencies, including
# lightkurve, astroquery, and eleanor.
#
# $ conda install -c conda-forge lightkurve
# $ conda install -c astropy astroquery
# $ pip install eleanor
#

try:
    from lightkurve.search import search_lightcurvefile
    lightkurve_dependency = True
except ImportError:
    lightkurve_dependency = False

try:
    from astroquery.mast import Tesscut
    from astroquery.mast import Observations
    astroquery_dependency = True
except ImportError:
    astroquery_dependency = False

try:
    import eleanor
    eleanor_dependency = True
except ImportError:
    eleanor_dependency = False

deps = {
    'lightkurve': lightkurve_dependency,
    'astroquery': astroquery_dependency,
    'eleanor': eleanor_dependency
}

for k,v in deps.items():
    if not v:
        wrn = (
            'Failed to import {:s} dependency. Trying anyway.'.
            format(k)
        )
        LOGWARNING(wrn)

from glob import glob
import os
import numpy as np, pandas as pd

##########
## WORK ##
##########

def get_two_minute_spoc_lightcurves(tic_id, download_dir=None):
    """
    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    Returns
    -------
    lcfiles : list or None
        List of light-curve file paths. None if none are found and downloaded.
    """

    if not isinstance(download_dir, str):
        errmsg = (
            'get_two_minute_spoc_lightcurves: failed to get valid download_dir'
        )
        LOGERROR(errmsg)
        return None

    search_str = 'TIC '+tic_id
    res = search_lightcurvefile(search_str, cadence='short', mission='TESS')

    if len(res.table)==0:
        errmsg = (
            'failed to get any SC data for TIC{}. need other LC source.'.
            format(tic_id)
        )
        LOGERROR(errmsg)
        return None

    available_sectors = list(res.table['sequence_number'])

    res.download_all(download_dir=download_dir)

    lcfiles = glob(os.path.join(download_dir, 'mastDownload', 'TESS',
                                '*{}*'.format(tic_id) ,
                                '*{}*.fits'.format(tic_id) ))

    return lcfiles


def get_hlsp_lightcurves(tic_id, hlsp_products=['CDIPS', 'TASOC'],
                         download_dir=None, verbose=True):
    """
    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    hlsp_products : list
        List of desired HLSP products to search. For instance, ["CDIPS"].

    download_dir : str
        Path of directory to which light-curve will be downloaded.

    Returns
    -------
    lcfiles : list or None
        List of light-curve file paths. None if none are found and downloaded.
    """

    lcfiles = []

    for hlsp in hlsp_products:

        obs_table = Observations.query_criteria(
            target_name=tic_id, provenance_name=hlsp
        )

        if verbose:
            LOGINFO('Found {} {} light-curves.'.format(len(obs_table), hlsp))

        # Get list of available products for this Observation.
        cdips_products = Observations.get_product_list(obs_table)

        # Download the products for this Observation.
        manifest = Observations.download_products(cdips_products,
                                                  download_dir=download_dir)
        if verbose:
            LOGINFO("Done")

        if len(manifest) >= 1:
            lcfiles.append(list(manifest['Local Path']))

    #
    # flatten lcfiles list
    #
    if len(lcfiles) >= 1:
        return_lcfiles = [item for sublist in lcfiles for item in sublist]
    else:
        return_lcfiles = None

    return return_lcfiles


def get_eleanor_lightcurves(tic_id, download_dir=None):
    """
    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    Returns
    -------
    lcfiles : list or None
        List of light-curve file paths. These are saved as CSV, rather than
        FITS, by this function.
    """

    stars = eleanor.multi_sectors(tic=np.int64(tic_id), sectors='all', tc=False)

    data = []

    for star in stars:

        d = eleanor.TargetData(star, height=15, width=15, bkg_size=31,
                               do_psf=False, do_pca=False)

        d.save(directory=download_dir)

    lcfiles = glob(os.path.join(
        download_dir, 'hlsp_eleanor_tess_ffi_tic{}*.fits'.format(tic_id)
    ))

    return lcfiles


def is_two_minute_spoc_lightcurve_available(tic_id):
    """
    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    Returns
    -------
    True if a 2 minute SPOC light-curve is available, else False.
    """

    search_str = 'TIC '+tic_id
    res = search_lightcurvefile(search_str, cadence='short', mission='TESS')

    if len(res.table)==0:
        return False
    else:
        return True


def get_tess_visibility_given_ticid(tic_id):
    """
    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.


    Returns
    -------
    sector_str, full_sector_str : tuple of strings
        For example, "[16, 17]" and "[tess-s0016-1-4, tess-s0017-2-3]". If
        empty, will return "[]" and "[]".
    """

    ticres = tic_objectsearch(ticid)
    with open(ticres['cachefname'], 'r') as json_file:
        data = json.load(json_file)

    ra = data['data'][0]['ra']
    dec = data['data'][0]['dec']

    coord = SkyCoord(ra, dec, unit="deg")
    sector_table = Tesscut.get_sectors(coord)

    sector_str = list(sector_table['sector'])
    full_sector_str = list(sector_table['sectorName'])

    return sector_str, full_sector_str


def get_tess_visibility_given_ticids(ticids):
    """
    Wrapper to get_tess_visibility_given_ticid for an iterable container of
    ticids.
    """

    sector_strs, full_sector_strs = [], []

    for ticid in ticids:
        sector_str, full_sector_str = (
            get_tess_visibility_given_ticid(ticid)
        )
        sector_strs.append(sector_str)
        full_sector_strs.append(full_sector_str)

    return sector_strs, full_sector_strs
