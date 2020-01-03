#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coordutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 07/13
# License: MIT - see LICENSE for the full text.

'''
Contains various useful tools for coordinate conversion, etc.

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

from math import trunc, fabs, pi as pi_value

import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u

import scipy.spatial as sps


#######################
## ANGLE CONVERSIONS ##
#######################

def angle_wrap(angle, radians=False):
    '''Wraps the input angle to 360.0 degrees.

    Parameters
    ----------

    angle : float
        The angle to wrap around 360.0 deg.

    radians : bool
        If True, will assume that the input is in radians. The output will then
        also be in radians.

    Returns
    -------

    float
        Wrapped angle. If radians is True: input is assumed to be in radians,
        output is also in radians.

    '''

    if radians:
        wrapped = angle % (2.0*pi_value)
        if wrapped < 0.0:
            wrapped = 2.0*pi_value + wrapped

    else:

        wrapped = angle % 360.0
        if wrapped < 0.0:
            wrapped = 360.0 + wrapped

    return wrapped


def decimal_to_dms(decimal_value):
    '''Converts from decimal degrees (for declination coords) to DD:MM:SS.

    Parameters
    ----------

    decimal_value : float
        A decimal value to convert to degrees, minutes, seconds sexagesimal
        format.

    Returns
    -------

    tuple
        A four element tuple is returned: (sign, HH, MM, SS.ssss...)

    '''

    if decimal_value < 0:
        negative = True
        dec_val = fabs(decimal_value)
    else:
        negative = False
        dec_val = decimal_value

    degrees = trunc(dec_val)
    minutes_deg = dec_val - degrees

    minutes_mm = minutes_deg * 60.0
    minutes_out = trunc(minutes_mm)
    seconds = (minutes_mm - minutes_out)*60.0

    if negative:
        degrees = degrees
        return '-', degrees, minutes_out, seconds
    else:
        return '+', degrees, minutes_out, seconds


def decimal_to_hms(decimal_value):
    '''Converts from decimal degrees (for RA coords) to HH:MM:SS.

    Parameters
    ----------

    decimal_value : float
        A decimal value to convert to hours, minutes, seconds. Negative values
        will be wrapped around 360.0.

    Returns
    -------

    tuple
        A three element tuple is returned: (HH, MM, SS.ssss...)

    '''

    # wrap to 360.0
    if decimal_value < 0:
        dec_wrapped = 360.0 + decimal_value
    else:
        dec_wrapped = decimal_value

    # convert to decimal hours first
    dec_hours = dec_wrapped/15.0

    if dec_hours < 0:
        negative = True
        dec_val = fabs(dec_hours)
    else:
        negative = False
        dec_val = dec_hours

    hours = trunc(dec_val)
    minutes_hrs = dec_val - hours

    minutes_mm = minutes_hrs * 60.0
    minutes_out = trunc(minutes_mm)
    seconds = (minutes_mm - minutes_out)*60.0

    if negative:
        hours = -hours
        return hours, minutes_out, seconds
    else:
        return hours, minutes_out, seconds


def hms_str_to_tuple(hms_string):
    '''Converts a string of the form HH:MM:SS or HH MM SS to a tuple of the form
    (HH, MM, SS).

    Parameters
    ----------

    hms_string : str
        A RA coordinate string of the form 'HH:MM:SS.sss' or 'HH MM SS.sss'.

    Returns
    -------

    tuple
        A three element tuple is returned (HH, MM, SS.ssss...)

    '''

    if ':' in hms_string:
        separator = ':'
    else:
        separator = ' '

    hh, mm, ss = hms_string.split(separator)

    return int(hh), int(mm), float(ss)


def dms_str_to_tuple(dms_string):
    '''Converts a string of the form [+-]DD:MM:SS or [+-]DD MM SS to a tuple of
    the form (sign, DD, MM, SS).

    Parameters
    ----------

    dms_string : str
        A declination coordinate string of the form '[+-]DD:MM:SS.sss' or
        '[+-]DD MM SS.sss'. The sign in front of DD is optional. If it's not
        there, this function will assume that the coordinate string is a
        positive value.

    Returns
    -------

    tuple
        A four element tuple of the form: (sign, DD, MM, SS.ssss...).

    '''
    if ':' in dms_string:
        separator = ':'
    else:
        separator = ' '

    sign_dd, mm, ss = dms_string.split(separator)
    if sign_dd.startswith('+') or sign_dd.startswith('-'):
        sign, dd = sign_dd[0], sign_dd[1:]
    else:
        sign, dd = '+', sign_dd

    return sign, int(dd), int(mm), float(ss)


def hms_str_to_decimal(hms_string):
    '''Converts a HH:MM:SS string to decimal degrees.

    Parameters
    ----------

    hms_string : str
        A right ascension coordinate string of the form: 'HH:MM:SS.sss'
        or 'HH MM SS.sss'.

    Returns
    -------

    float
        The RA value in decimal degrees (wrapped around 360.0 deg if necessary.)

    '''
    return hms_to_decimal(*hms_str_to_tuple(hms_string))


def dms_str_to_decimal(dms_string):
    '''Converts a DD:MM:SS string to decimal degrees.

    Parameters
    ----------

    dms_string : str
        A declination coordinate string of the form: '[+-]DD:MM:SS.sss'
        or '[+-]DD MM SS.sss'.

    Returns
    -------

    float
        The declination value in decimal degrees.

    '''
    return dms_to_decimal(*dms_str_to_tuple(dms_string))


def hms_to_decimal(hours, minutes, seconds, returndeg=True):
    '''Converts from HH, MM, SS to a decimal value.

    Parameters
    ----------

    hours : int
        The HH part of a RA coordinate.

    minutes : int
        The MM part of a RA coordinate.

    seconds : float
        The SS.sss part of a RA coordinate.

    returndeg : bool
        If this is True, then will return decimal degrees as the output.
        If this is False, then will return decimal HOURS as the output.
        Decimal hours are sometimes used in FITS headers.

    Returns
    -------

    float
        The right ascension value in either decimal degrees or decimal hours
        depending on `returndeg`.

    '''

    if hours > 24:

        return None

    else:

        dec_hours = fabs(hours) + fabs(minutes)/60.0 + fabs(seconds)/3600.0

        if returndeg:

            dec_deg = dec_hours*15.0

            if dec_deg < 0:
                dec_deg = dec_deg + 360.0
            dec_deg = dec_deg % 360.0
            return dec_deg
        else:
            return dec_hours


def dms_to_decimal(sign, degrees, minutes, seconds):
    '''Converts from DD:MM:SS to a decimal value.

    Parameters
    ----------

    sign : {'+', '-', ''}
        The sign part of a Dec coordinate.

    degrees : int
        The DD part of a Dec coordinate.

    minutes : int
        The MM part of a Dec coordinate.

    seconds : float
        The SS.sss part of a Dec coordinate.

    Returns
    -------

    float
        The declination value in decimal degrees.

    '''

    dec_deg = fabs(degrees) + fabs(minutes)/60.0 + fabs(seconds)/3600.0

    if sign == '-':
        return -dec_deg
    else:
        return dec_deg


############################
## DISTANCE AND XMATCHING ##
############################

def great_circle_dist(ra1, dec1, ra2, dec2):
    '''Calculates the great circle angular distance between two coords.

    This calculates the great circle angular distance in arcseconds between two
    coordinates (ra1,dec1) and (ra2,dec2). This is basically a clone of GCIRC
    from the IDL Astrolib.

    Parameters
    ----------

    ra1,dec1 : float or array-like
        The first coordinate's right ascension and declination value(s) in
        decimal degrees.

    ra2,dec2 : float or array-like
        The second coordinate's right ascension and declination value(s) in
        decimal degrees.

    Returns
    -------

    float or array-like
        Great circle distance between the two coordinates in arseconds.

    Notes
    -----

    If (`ra1`, `dec1`) is scalar and (`ra2`, `dec2`) is scalar: the result is a
    float distance in arcseconds.

    If (`ra1`, `dec1`) is scalar and (`ra2`, `dec2`) is array-like: the result
    is an np.array with distance in arcseconds between (`ra1`, `dec1`) and each
    element of (`ra2`, `dec2`).

    If (`ra1`, `dec1`) is array-like and (`ra2`, `dec2`) is scalar: the result
    is an np.array with distance in arcseconds between (`ra2`, `dec2`) and each
    element of (`ra1`, `dec1`).

    If (`ra1`, `dec1`) and (`ra2`, `dec2`) are both array-like: the result is an
    np.array with the pair-wise distance in arcseconds between each element of
    the two coordinate lists. In this case, if the input array-likes are not the
    same length, then excess elements of the longer one will be ignored.

    '''

    # wrap RA if negative or larger than 360.0 deg
    in_ra1 = ra1 % 360.0
    in_ra1 = in_ra1 + 360.0*(in_ra1 < 0.0)
    in_ra2 = ra2 % 360.0
    in_ra2 = in_ra2 + 360.0*(in_ra1 < 0.0)

    # convert to radians
    ra1_rad, dec1_rad = np.deg2rad(in_ra1), np.deg2rad(dec1)
    ra2_rad, dec2_rad = np.deg2rad(in_ra2), np.deg2rad(dec2)

    del_dec2 = (dec2_rad - dec1_rad)/2.0
    del_ra2 = (ra2_rad - ra1_rad)/2.0
    sin_dist = np.sqrt(np.sin(del_dec2) * np.sin(del_dec2) +
                       np.cos(dec1_rad) * np.cos(dec2_rad) *
                       np.sin(del_ra2) * np.sin(del_ra2))

    dist_rad = 2.0 * np.arcsin(sin_dist)

    # return the distance in arcseconds
    return np.rad2deg(dist_rad)*3600.0


def xmatch_basic(ra1, dec1, ra2, dec2, match_radius=5.0):
    '''Finds the closest object in (`ra2`, `dec2`) to scalar coordinate pair
    (`ra1`, `dec1`) and returns the distance in arcseconds.

    This is a quick matcher that uses the `great_circle_dist` function to find
    the closest object in (`ra2`, `dec2`) within `match_radius` arcseconds to
    (`ra1`, `dec1`). (`ra1`, `dec1`) must be a scalar pair, while
    (`ra2`, `dec2`) must be array-likes of the same lengths.

    Parameters
    ----------

    ra1,dec1 : float
        Coordinate of the object to find matches to. In decimal degrees.

    ra2,dec2 : array-like
        The coordinates that will be searched for matches. In decimal degrees.

    match_radius : float
        The match radius in arcseconds to use for the match.

    Returns
    -------

    tuple
        A two element tuple like the following::

            (True -> no match found or False -> found a match,
             minimum distance between target and list in arcseconds)

    '''

    min_dist_arcsec = np.min(great_circle_dist(ra1,dec1,ra2,dec2))

    if (min_dist_arcsec < match_radius):
        return (True,min_dist_arcsec)
    else:
        return (False,min_dist_arcsec)


def xmatch_neighbors(ra1, dec1,
                     ra2, dec2,
                     match_radius=60.0,
                     includeself=False,
                     sortresults=True):
    '''Finds the closest objects in (`ra2`, `dec2`) to scalar coordinate pair
    (`ra1`, `dec1`) and returns the indices of the objects that match.

    This is a quick matcher that uses the `great_circle_dist` function to find
    the closest object in (`ra2`, `dec2`) within `match_radius` arcseconds to
    (`ra1`, `dec1`). (`ra1`, `dec1`) must be a scalar pair, while
    (`ra2`, `dec2`) must be array-likes of the same lengths.

    Parameters
    ----------

    ra1,dec1 : float
        Coordinate of the object to find matches to. In decimal degrees.

    ra2,dec2 : array-like
        The coordinates that will be searched for matches. In decimal degrees.

    match_radius : float
        The match radius in arcseconds to use for the match.

    includeself : bool
        If this is True, the object itself will be included in the match
        results.

    sortresults : bool
        If this is True, the match indices will be sorted by distance.

    Returns
    -------

    tuple
        A tuple like the following is returned::

            (True -> matches found or False -> no matches found,
             minimum distance between target and list,
             np.array of indices where list of coordinates is
             closer than `match_radius` arcseconds from the target,
             np.array of distances in arcseconds)

    '''

    dist = great_circle_dist(ra1,dec1,ra2,dec2)

    if includeself:
        match_dist_ind = np.where(dist < match_radius)

    else:
        # make sure we match only objects that are not the same as this object
        match_dist_ind = np.where((dist < match_radius) & (dist > 0.1))

    if len(match_dist_ind) > 0:
        match_dists = dist[match_dist_ind]
        dist_sort_ind = np.argsort(match_dists)

        if sortresults:
            match_dist_ind = (match_dist_ind[0])[dist_sort_ind]

        min_dist = np.min(match_dists)

        return (True,min_dist,match_dist_ind,match_dists[dist_sort_ind])

    else:
        return (False,)


######################
## KDTREE FUNCTIONS ##
######################

def make_kdtree(ra, decl):
    '''This makes a `scipy.spatial.CKDTree` on (`ra`, `decl`).

    Parameters
    ----------

    ra,decl : array-like
        The right ascension and declination coordinate pairs in decimal degrees.

    Returns
    -------

    `scipy.spatial.CKDTree`
        The cKDTRee object generated by this function is returned and can be
        used to run various spatial queries.

    '''

    # get the xyz unit vectors from ra,decl
    # since i had to remind myself:
    # https://en.wikipedia.org/wiki/Equatorial_coordinate_system
    cosdecl = np.cos(np.radians(decl))
    sindecl = np.sin(np.radians(decl))
    cosra = np.cos(np.radians(ra))
    sinra = np.sin(np.radians(ra))
    xyz = np.column_stack((cosra*cosdecl,sinra*cosdecl, sindecl))

    # generate the kdtree
    kdt = sps.cKDTree(xyz,copy_data=True)

    return kdt


def conesearch_kdtree(kdtree,
                      racenter,
                      declcenter,
                      searchradiusdeg,
                      conesearchworkers=1):
    '''This does a cone-search around (`racenter`, `declcenter`) in `kdtree`.

    Parameters
    ----------

    kdtree : scipy.spatial.CKDTree
        This is a kdtree object generated by the `make_kdtree` function.

    racenter,declcenter : float or array-like
        This is the center coordinate to run the cone-search around in decimal
        degrees. If this is an np.array, will search for all coordinate pairs in
        the array.

    searchradiusdeg : float
        The search radius to use for the cone-search in decimal degrees.

    conesearchworkers : int
        The number of parallel workers to launch for the cone-search.

    Returns
    -------

    list or np.array of lists
        If (`racenter`, `declcenter`) is a single coordinate, this will return a
        list of the indices of the matching objects in the kdtree. If
        (`racenter`, `declcenter`) are array-likes, this will return an object
        array containing lists of matching object indices for each coordinate
        searched.

    '''

    cosdecl = np.cos(np.radians(declcenter))
    sindecl = np.sin(np.radians(declcenter))
    cosra = np.cos(np.radians(racenter))
    sinra = np.sin(np.radians(racenter))

    # this is the search distance in xyz unit vectors
    xyzdist = 2.0 * np.sin(np.radians(searchradiusdeg)/2.0)

    # look up the coordinates
    kdtindices = kdtree.query_ball_point([cosra*cosdecl,
                                          sinra*cosdecl,
                                          sindecl],
                                         xyzdist,
                                         n_jobs=conesearchworkers)

    return kdtindices


def xmatch_kdtree(kdtree,
                  extra, extdecl,
                  xmatchdistdeg,
                  closestonly=True):
    '''This cross-matches between `kdtree` and (`extra`, `extdecl`) arrays.

    Returns the indices of the kdtree and the indices of extra, extdecl that
    xmatch successfully.

    Parameters
    ----------

    kdtree : scipy.spatial.CKDTree
        This is a kdtree object generated by the `make_kdtree` function.

    extra,extdecl : array-like
        These are np.arrays of 'external' coordinates in decimal degrees that
        will be cross-matched against the objects in `kdtree`.

    xmatchdistdeg : float
        The match radius to use for the cross-match in decimal degrees.

    closestonly : bool
        If closestonly is True, then this function returns only the closest
        matching indices in (extra, extdecl) for each object in kdtree if there
        are any matches. Otherwise, it returns a list of indices in (extra,
        extdecl) for all matches within xmatchdistdeg between kdtree and (extra,
        extdecl).

    Returns
    -------

    tuple of lists
        Returns a tuple of the form::

            (list of `kdtree` indices matching to external objects,
             list of all `extra`/`extdecl` indices that match to each
             element in `kdtree` within the specified cross-match distance)

    '''

    ext_cosdecl = np.cos(np.radians(extdecl))
    ext_sindecl = np.sin(np.radians(extdecl))
    ext_cosra = np.cos(np.radians(extra))
    ext_sinra = np.sin(np.radians(extra))

    ext_xyz = np.column_stack((ext_cosra*ext_cosdecl,
                               ext_sinra*ext_cosdecl,
                               ext_sindecl))
    ext_xyzdist = 2.0 * np.sin(np.radians(xmatchdistdeg)/2.0)

    # get our kdtree
    our_kdt = kdtree

    # get the external kdtree
    ext_kdt = sps.cKDTree(ext_xyz)

    # do a query_ball_tree
    extkd_matchinds = our_kdt.query_ball_tree(ext_kdt, ext_xyzdist)

    ext_matchinds = []
    kdt_matchinds = []

    for extind, mind in enumerate(extkd_matchinds):
        if len(mind) > 0:
            # our object indices
            kdt_matchinds.append(extind)

            # external object indices
            if closestonly:
                ext_matchinds.append(mind[0])
            else:
                ext_matchinds.append(mind)

    return kdt_matchinds, ext_matchinds


###################
## PROPER MOTION ##
###################

def total_proper_motion(pmra, pmdecl, decl):

    '''This calculates the total proper motion of an object.

    Parameters
    ----------

    pmra : float or array-like
        The proper motion(s) in right ascension, measured in mas/yr.

    pmdecl : float or array-like
        The proper motion(s) in declination, measured in mas/yr.

    decl : float or array-like
        The declination of the object(s) in decimal degrees.

    Returns
    -------

    float or array-like
        The total proper motion(s) of the object(s) in mas/yr.

    '''

    pm = np.sqrt( pmdecl*pmdecl + pmra*pmra*np.cos(np.radians(decl)) *
                  np.cos(np.radians(decl)) )

    return pm


def reduced_proper_motion(mag, propermotion):
    '''This calculates the reduced proper motion using the mag measurement
    provided.

    Parameters
    ----------

    mag : float or array-like
        The magnitude(s) to use to calculate the reduced proper motion(s).

    propermotion : float or array-like
        The total proper motion of the object(s). Use the `total_proper_motion`
        function to calculate this if you have `pmra`, `pmdecl`, and `decl`
        values. `propermotion` should be in mas/yr.

    Returns
    -------

    float or array-like
        The reduced proper motion for the object(s). This is effectively a
        measure of the absolute magnitude in the band provided.

    '''

    rpm = mag + 5.0*np.log10(propermotion/1000.0)
    return rpm


###########################
## COORDINATE CONVERSION ##
###########################

def equatorial_to_galactic(ra, decl, equinox='J2000'):
    '''This converts from equatorial coords to galactic coords.

    Parameters
    ----------

    ra : float or array-like
        Right ascension values(s) in decimal degrees.

    decl : float or array-like
        Declination value(s) in decimal degrees.

    equinox : str
        The equinox that the coordinates are measured at. This must be
        recognizable by Astropy's `SkyCoord` class.

    Returns
    -------

    tuple of (float, float) or tuple of (np.array, np.array)
        The galactic coordinates (l, b) for each element of the input
        (`ra`, `decl`).

    '''

    # convert the ra/decl to gl, gb
    radecl = SkyCoord(ra=ra*u.degree, dec=decl*u.degree, equinox=equinox)

    gl = radecl.galactic.l.degree
    gb = radecl.galactic.b.degree

    return gl, gb


def galactic_to_equatorial(gl, gb):
    '''This converts from galactic coords to equatorial coordinates.

    Parameters
    ----------

    gl : float or array-like
        Galactic longitude values(s) in decimal degrees.

    gb : float or array-like
        Galactic latitude value(s) in decimal degrees.

    Returns
    -------

    tuple of (float, float) or tuple of (np.array, np.array)
        The equatorial coordinates (RA, DEC) for each element of the input
        (`gl`, `gb`) in decimal degrees. These are reported in the ICRS frame.

    '''

    gal = SkyCoord(gl*u.degree, gl*u.degree, frame='galactic')

    transformed = gal.transform_to('icrs')

    return transformed.ra.degree, transformed.dec.degree


########################
## XI-ETA PROJECTIONS ##
########################

def xieta_from_radecl(inra, indecl,
                      incenterra, incenterdecl,
                      deg=True):
    '''This returns the image-plane projected xi-eta coords for inra, indecl.

    Parameters
    ----------

    inra,indecl : array-like
        The equatorial coordinates to get the xi, eta coordinates for in decimal
        degrees or radians.

    incenterra,incenterdecl : float
        The center coordinate values to use to calculate the plane-projected
        coordinates around.

    deg : bool
        If this is True, the input angles are assumed to be in degrees and the
        output is in degrees as well.

    Returns
    -------

    tuple of np.arrays
        This is the (`xi`, `eta`) coordinate pairs corresponding to the
        image-plane projected coordinates for each pair of input equatorial
        coordinates in (`inra`, `indecl`).

    '''

    if deg:

        ra = np.radians(inra)
        decl = np.radians(indecl)
        centerra = np.radians(incenterra)
        centerdecl = np.radians(incenterdecl)

    else:

        ra = inra
        decl = indecl
        centerra = incenterra
        centerdecl = incenterdecl

    cdecc = np.cos(centerdecl)
    sdecc = np.sin(centerdecl)
    crac = np.cos(centerra)
    srac = np.sin(centerra)

    uu = np.cos(decl)*np.cos(ra)
    vv = np.cos(decl)*np.sin(ra)
    ww = np.sin(decl)

    uun = uu*cdecc*crac + vv*cdecc*srac + ww*sdecc
    vvn = -uu*srac + vv*crac
    wwn = -uu*sdecc*crac - vv*sdecc*srac + ww*cdecc
    denom = vvn*vvn + wwn*wwn

    aunn = np.zeros_like(uun)
    aunn[uun >= 1.0] = 0.0
    aunn[uun < 1.0] = np.arccos(uun)

    xi, eta = np.zeros_like(aunn), np.zeros_like(aunn)

    xi[(aunn <= 0.0) | (denom <= 0.0)] = 0.0
    eta[(aunn <= 0.0) | (denom <= 0.0)] = 0.0

    sdenom = np.sqrt(denom)

    xi[(aunn > 0.0) | (denom > 0.0)] = aunn*vvn/sdenom
    eta[(aunn > 0.0) | (denom > 0.0)] = aunn*wwn/sdenom

    if deg:
        return np.degrees(xi), np.degrees(eta)
    else:
        return xi, eta
