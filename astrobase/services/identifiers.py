#!/usr/bin/env python
# -*- coding: utf-8 -*-
# convert_identifiers.py - Luke Bouma (bouma.luke@gmail.com) - Oct 2019
# License: MIT - see the LICENSE file for the full text.

'''
Easy conversion between survey identifiers. Works best on bright and/or famous
objects, particularly when SIMBAD is involved.

given simbad name, attempt to get DR2 source_id

given DR2 source_id, attempt to get TIC ID

given simbad name, get TIC ID

given TIC ID, get simbad name
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

import pandas as pd, numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u, constants as const

from astrobase.services.simbad import tap_query as simbad_tap_query
from astrobase.services.gaia import objectid_search as gaia_objectid_search


try:
    from astroquery.mast import Catalogs
    from astroquery.simbad import Simbad
    astroquery_dependencies = True
except ImportError:
    astroquery_dependencies = False


def simbad2gaiadrtwo(simbad_name,
                     simbad_mirror='simbad', returnformat='csv',
                     forcefetch=False, cachedir='~/.astrobase/simbad-cache',
                     verbose=True, timeout=10.0, refresh=2.0, maxtimeout=90.0,
                     maxtries=1, complete_query_later=True):
    """
    Convenience function that, given a SIMBAD object name, returns string of
    the Gaia-DR2 identifier.

    simbad_name: string as you would search on SIMBAD.
    """

    assert isinstance(simbad_name, str)

    # TAP table list is here:
    # http://simbad.u-strasbg.fr/simbad/tap/tapsearch.html
    query = (
    "SELECT basic.OID, basic.RA, basic.DEC, ident.id, ident.oidref, ids.ids "
    "FROM basic "
    "LEFT OUTER JOIN ident ON ident.oidref = basic.oid "
    "LEFT OUTER JOIN ids ON ids.oidref = ident.oidref "
    "WHERE ident.id = '{simbad_name}'; "
    )

    formatted_query = query.format(simbad_name=simbad_name)

    # astroquery.simbad would have been fine here too. Sometimes pure astrobase
    # solutions are nice though ;-).
    r =  simbad_tap_query(formatted_query, simbad_mirror=simbad_mirror,
                          returnformat=returnformat, forcefetch=forcefetch,
                          cachedir=cachedir, verbose=verbose, timeout=timeout,
                          refresh=refresh, maxtimeout=maxtimeout,
                          maxtries=maxtries,
                          complete_query_later=complete_query_later)

    df = pd.read_csv(r['result'])

    if len(df) != 1:
        errmsg = (
            'Expected 1 result from name {}; got {} results.'.
            format(simbad_name, len(df))
        )
        raise ValueError(errmsg)

    if 'Gaia DR2' not in df['ids'].iloc[0]:
        errmsg = (
            'Failed to retrieve Gaia DR2 identifier for {}'.
            format(simbad_name)
        )
        raise NameError(errmsg)

    # simbad returns a "|"-separated list of cross-matched names
    names = df['ids'].iloc[0].split('|')

    gaia_name = [n for n in names if 'Gaia DR2' in n]

    gaia_id = gaia_name[0].split(' ')[-1]

    return gaia_id


def gaiadrtwo2tic(source_id, returnformat='csv', gaia_mirror='gaia',
                  forcefetch=False, cachedir='~/.astrobase/simbad-cache',
                  verbose=True, timeout=10.0, refresh=2.0, maxtimeout=90.0,
                  maxtries=1, complete_query_later=True):
    """
    First, gets RA/dec from Gaia DR2, given source_id. Then searches TICv8
    spatially, and returns matches with the correct DR2 source_id.

    Parameters
    ----------
    source_id: Gaia DR2 source identifier.

    Remainder are described in `astrobase.services.gaia.objectid_search`
    """

    if not astroquery_dependencies:
        raise ImportError(
            'This function depends on astroquery.'
        )

    r = gaia_objectid_search(source_id, gaia_mirror=gaia_mirror,
                             returnformat=returnformat, forcefetch=forcefetch,
                             cachedir=cachedir, verbose=verbose,
                             timeout=timeout, refresh=refresh,
                             maxtimeout=maxtimeout, maxtries=maxtries,
                             complete_query_later=complete_query_later)

    try:
        df = pd.read_csv(r['result'])
    except pd.errors.EmptyDataError:
        errmsg = (
            'Expected 1 Gaia result from source_id {}; got no results.'.
            format(source_id)
        )
        raise pd.errors.EmptyDataError(errmsg)

    ra, dec = df['ra'].iloc[0], df['dec'].iloc[0]

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    radius = 0.5*u.arcminute
    try:
        stars = Catalogs.query_region(
            "{} {}".format(float(coord.ra.value), float(coord.dec.value)),
            catalog="TIC", radius=radius
        )
    except requests.exceptions.ConnectionError:
        LOGWARNING('WRN! TIC query failed. trying again...')
        time.sleep(60)
        stars = Catalogs.query_region(
            "{} {}".format(float(coord.ra.value), float(coord.dec.value)),
            catalog="TIC", radius=radius
        )

    sel = ~stars['GAIA'].mask
    selstars = stars[sel]

    if len(selstars)>=1:

        # TICv8 was based on Gaia DR2: enforce that the TICv8 TICID match that's
        # returned has a Gaia source_id listed in the TIC that is the same as
        # the source_id passed to this function.
        if np.any(
            np.in1d(np.array(selstars['GAIA']).astype(np.int64),
                    np.array(np.int64(source_id)))
        ):

            ind = (
                int(np.where(
                        np.in1d(
                            np.array(selstars['GAIA']).astype(np.int64),
                            np.array(np.int64(source_id))))[0]
                )
            )

            mrow = selstars[ind]

    if len(mrow) == 0:
        errmsg = (
            'Failed to retrieve TIC identifier for Gaia DR2 {}'.
            format(source_id)
        )
        raise NameError(errmsg)

    ticid = mrow['ID']
    return ticid


def simbad2tic(simbad_name):

    source_id = simbad2gaiadrtwo(simbad_name)

    return gaiadrtwo2tic(source_id)
