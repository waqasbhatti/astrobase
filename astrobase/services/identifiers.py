#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# convert_identifiers.py - Luke Bouma (bouma.luke@gmail.com) - Oct 2019
# License: MIT - see the LICENSE file for the full text.

'''
Easy conversion between survey identifiers. Works best on bright and/or famous
objects, particularly when SIMBAD is involved.

``simbad_to_gaiadr2()``: given simbad name, attempt to get GAIA DR2 source_id

``gaiadr2_to_tic()``: given GAIA DR2 source_id, attempt to get TIC ID

``simbad_to_tic()``: given simbad name, get TIC ID

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

import json

import numpy as np
from astropy.table import Table

from astrobase.services.simbad import tap_query as simbad_tap_query
from astrobase.services.gaia import objectid_search as gaia_objectid_search
from astrobase.services.mast import tic_conesearch, tic_objectsearch


###############
## FUNCTIONS ##
###############

def simbad_to_gaiadr2(
        simbad_name,
        simbad_mirror='simbad',
        returnformat='csv',
        forcefetch=False,
        cachedir='~/.astrobase/simbad-cache',
        verbose=True,
        timeout=10.0,
        refresh=2.0,
        maxtimeout=90.0,
        maxtries=1,
        complete_query_later=True
):
    """
    Convenience function that, given a SIMBAD object name, returns string of
    the Gaia-DR2 identifier.

    Parameters
    ----------

    simbad_name : str
        The SIMBAD object name to search for.

    simbad_mirror : str
        This is the key used to select a SIMBAD mirror from the
        `SIMBAD_URLS` dict above. If set, the specified mirror will be used. If
        None, a random mirror chosen from that dict will be used.

    returnformat : {'csv','votable','json'}
        The returned file format to request from the GAIA catalog service.

    forcefetch : bool
        If this is True, the query will be retried even if cached results for
        it exist.

    cachedir : str
        This points to the directory where results will be downloaded.

    verbose : bool
        If True, will indicate progress and warn of any issues.

    timeout : float
        This sets the amount of time in seconds to wait for the service to
        respond to our initial request.

    refresh : float
        This sets the amount of time in seconds to wait before checking if the
        result file is available. If the results file isn't available after
        `refresh` seconds have elapsed, the function will wait for `refresh`
        seconds continuously, until `maxtimeout` is reached or the results file
        becomes available.

    maxtimeout : float
        The maximum amount of time in seconds to wait for a result to become
        available after submitting our query request.

    maxtries : int
        The maximum number of tries (across all mirrors tried) to make to either
        submit the request or download the results, before giving up.

    complete_query_later : bool
        If set to True, a submitted query that does not return a result before
        `maxtimeout` has passed will be cancelled but its input request
        parameters and the result URL provided by the service will be saved. If
        this function is then called later with these same input request
        parameters, it will check if the query finally finished and a result is
        available. If so, will download the results instead of submitting a new
        query. If it's not done yet, will start waiting for results again. To
        force launch a new query with the same request parameters, set the
        `forcefetch` kwarg to True.

    Returns
    -------

    gaiadr2_id : str
        Returns the GAIA DR2 ID as a string.

    """

    if not isinstance(simbad_name, str):
        LOGWARNING("The given simbad_name must be a string, "
                   "converting automatically...")
        use_simbad_name = str(simbad_name)
    else:
        use_simbad_name = simbad_name

    # TAP table list is here:
    # http://simbad.u-strasbg.fr/simbad/tap/tapsearch.html
    query = (
        "SELECT basic.OID, basic.RA, basic.DEC, "
        "ident.id, ident.oidref, ids.ids "
        "FROM basic "
        "LEFT OUTER JOIN ident ON ident.oidref = basic.oid "
        "LEFT OUTER JOIN ids ON ids.oidref = ident.oidref "
        "WHERE ident.id = '{use_simbad_name}'; "
    )

    formatted_query = query.format(use_simbad_name=use_simbad_name)

    # astroquery.simbad would have been fine here too. Sometimes pure astrobase
    # solutions are nice though ;-).
    r = simbad_tap_query(
        formatted_query,
        simbad_mirror=simbad_mirror,
        returnformat=returnformat,
        forcefetch=forcefetch,
        cachedir=cachedir,
        verbose=verbose,
        timeout=timeout,
        refresh=refresh,
        maxtimeout=maxtimeout,
        maxtries=maxtries,
        complete_query_later=complete_query_later
    )

    df = Table.read(r['result'],format='csv')

    if len(df) != 1:
        errmsg = (
            'Expected 1 result from name {}; got {} results.'.format(
                use_simbad_name, len(df)
            )
        )
        LOGERROR(errmsg)
        return None

    if 'Gaia DR2' not in df['ids'][0]:
        errmsg = (
            'Failed to retrieve Gaia DR2 identifier for {}'.format(
                use_simbad_name
            )
        )
        LOGERROR(errmsg)
        return None

    # simbad returns a "|"-separated list of cross-matched names
    names = df['ids'][0].split('|')

    gaia_name = [n for n in names if 'Gaia DR2' in n]

    gaia_id = gaia_name[0].split(' ')[-1]

    return gaia_id


def gaiadr2_to_tic(
        source_id,
        gaia_mirror='gaia',
        returnformat='csv',
        forcefetch=False,
        cachedir='~/.astrobase/simbad-cache',
        verbose=True,
        timeout=10.0,
        refresh=2.0,
        maxtimeout=90.0,
        maxtries=1,
        complete_query_later=True
):
    """
    First, gets RA/dec from Gaia DR2, given source_id. Then searches TICv8
    spatially, and returns matches with the correct DR2 source_id.

    Parameters
    ----------

    source_id : str
        The GAIA DR2 source identifier.

    gaia_mirror : {'gaia','heidelberg','vizier'} or None
        This is the key used to select a GAIA catalog mirror from the
        `GAIA_URLS` dict above. If set, the specified mirror will be used. If
        None, a random mirror chosen from that dict will be used.

    returnformat : {'csv','votable','json'}
        The returned file format to request from the GAIA catalog service.

    forcefetch : bool
        If this is True, the query will be retried even if cached results for
        it exist.

    cachedir : str
        This points to the directory where results will be downloaded.

    verbose : bool
        If True, will indicate progress and warn of any issues.

    timeout : float
        This sets the amount of time in seconds to wait for the service to
        respond to our initial request.

    refresh : float
        This sets the amount of time in seconds to wait before checking if the
        result file is available. If the results file isn't available after
        `refresh` seconds have elapsed, the function will wait for `refresh`
        seconds continuously, until `maxtimeout` is reached or the results file
        becomes available.

    maxtimeout : float
        The maximum amount of time in seconds to wait for a result to become
        available after submitting our query request.

    maxtries : int
        The maximum number of tries (across all mirrors tried) to make to either
        submit the request or download the results, before giving up.

    completequerylater : bool
        If set to True, a submitted query that does not return a result before
        `maxtimeout` has passed will be cancelled but its input request
        parameters and the result URL provided by the service will be saved. If
        this function is then called later with these same input request
        parameters, it will check if the query finally finished and a result is
        available. If so, will download the results instead of submitting a new
        query. If it's not done yet, will start waiting for results again. To
        force launch a new query with the same request parameters, set the
        `forcefetch` kwarg to True.

    Returns
    -------

    tic_id : str
        Returns the TIC ID of the object as a string.

    """

    r = gaia_objectid_search(source_id, gaia_mirror=gaia_mirror,
                             returnformat=returnformat, forcefetch=forcefetch,
                             cachedir=cachedir, verbose=verbose,
                             timeout=timeout, refresh=refresh,
                             maxtimeout=maxtimeout, maxtries=maxtries,
                             complete_query_later=complete_query_later)

    try:
        df = Table.read(r['result'], format='csv')

        if len(df) == 0 or len(df) > 1:
            errmsg = (
                'Expected 1 Gaia result from source_id {}; got {} results.'.
                format(source_id, len(df))
            )
            LOGERROR(errmsg)
            return None

    except Exception:
        LOGEXCEPTION("Could not fetch GAIA info for source_id = %s" % source_id)
        return None

    ra, dec = df['ra'][0], df['dec'][0]

    # use mast.tic_conesearch to find the closest match to the GAIA object
    tic_res = tic_conesearch(ra, dec, radius_arcmin=0.5,
                             timeout=timeout,refresh=refresh,
                             maxtimeout=maxtimeout,maxtries=maxtries)

    try:

        with open(tic_res['cachefname'],'r') as infd:
            tic_info = json.load(infd)

            if len(tic_info['data']) == 0:
                errmsg = (
                    'Expected 1 TIC result from source_id {}; got {} results.'.
                    format(source_id, len(tic_info['data']))
                )
                LOGERROR(errmsg)
                return None

    except Exception:

        LOGEXCEPTION("Could not fetch TIC info for source_id = %s" % source_id)
        return None

    #
    # now, select the appropriate row in the returned matches
    #
    gaia_ids = np.array([
        (int(x['GAIA']) if x['GAIA'] is not None else -1)
        for x in tic_info['data']
    ])
    tic_ids = np.array([
        x['ID'] for x in tic_info['data']
    ])

    matched_tic_id = tic_ids[gaia_ids == int(source_id)]
    if matched_tic_id.size > 0:
        return str(matched_tic_id.item())
    else:
        LOGERROR("Could not find TIC ID for "
                 "source ID: %s in TIC (version: %s)" %
                 (source_id, tic_info['data'][0]['version']))
        return None


def simbad_to_tic(simbad_name):
    """
    This goes from a SIMBAD name to a TIC name.

    Parameters
    ----------

    simbad_name : str
        The SIMBAD name of the object to look up the TIC ID for.

    Returns
    -------

    tic_id : str
        Returns the TIC ID of the object as a string.

    """

    source_id = simbad_to_gaiadr2(simbad_name)

    if source_id is not None:
        return gaiadr2_to_tic(source_id)
    else:
        LOGERROR("Could not find TIC ID for SIMBAD name: %s" % simbad_name)
        return None


def tic_to_gaiadr2(tic_id, raiseonfail=False):
    """
    This goes from a TIC name to a GAIA DR2 source_id.

    Parameters
    ----------

    tic_id : str
        The TIC ID to look for.

    Returns
    -------

    source_id : str
        Returns the GAIA DR2 ID of the object as a string.

    """

    tic_res = tic_objectsearch(tic_id, raiseonfail=raiseonfail)

    with open(tic_res['cachefname'],'r') as infd:
        tic_info = json.load(infd)

        if len(tic_info['data']) != 1:
            errmsg = (
                'Expected exactly 1 TIC result from tic_id {}; got {}.'.
                format(tic_id, len(tic_info['data']))
            )

            if raiseonfail:
                raise ValueError(errmsg)
            else:
                LOGEXCEPTION(errmsg)
                return None

    gaia_id = str(tic_info['data'][0]['GAIA'])

    return gaia_id
