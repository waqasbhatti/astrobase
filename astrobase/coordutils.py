#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
coordutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 07/13
License: MIT - see LICENSE for the full text.

Contains various useful tools for coordinate conversion, etc.

'''

import time
from math import trunc, radians, degrees, sin, cos, asin, atan2, fabs, pi as PI

import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u

import scipy.spatial as sps

#######################
## ANGLE CONVERSIONS ##
#######################

def angle_wrap(angle,radians=False):
    '''
    Wraps the input angle to 360.0 degrees.

    if radians is True: input is assumed to be in radians, output is also in
    radians

    '''

    if radians:
        wrapped = angle % (2.0*PI)
        if wrapped < 0.0:
            wrapped = 2.0*PI + wrapped

    else:

        wrapped = angle % 360.0
        if wrapped < 0.0:
            wrapped = 360.0 + wrapped

    return wrapped


def decimal_to_dms(decimal_value):
    '''
    This converts from decimal degrees to DD:MM:SS, returned as a tuple.

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
    '''
    This converts from decimal degrees to HH:MM:SS, returned as a
    tuple. Negative values of degrees are wrapped to 360.0.

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
    '''
    Converts a string of the form HH:MM:SS or HH MM SS to a tuple of the form
    (HH,MM,SS).

    '''

    if ':' in hms_string:
        separator = ':'
    else:
        separator = ' '

    hh, mm, ss = hms_string.split(separator)

    return int(hh), int(mm), float(ss)


def dms_str_to_tuple(dms_string):
    '''
    Converts a string of the form +/-DD:MM:SS or +/-DD MM SS to a tuple of the
    form (sign,DD,MM,SS).

    '''
    if ':' in dms_string:
        separator = ':'
    else:
        separator = ' '

    sign_dd, mm, ss = dms_string.split(separator)
    sign, dd = sign_dd[0], sign_dd[1:]

    return sign, int(dd), int(mm), float(ss)


def hms_str_to_decimal(hms_string):
    '''
    Converts a HH:MM:SS string to decimal degrees.

    '''
    return hms_to_decimal(*hms_str_to_tuple(hms_string))


def dms_str_to_decimal(dms_string):
    '''
    Converts a DD:MM:SS string to decimal degrees.

    '''
    return dms_to_decimal(*dms_str_to_tuple(dms_string))


def hms_to_decimal(hours, minutes, seconds, returndeg=True):
    '''
    Converts from HH:MM:SS to a decimal value.

    if returndeg is True: returns decimal degrees
    if returndeg is False: returns decimal hours

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
    '''
    Converts from DD:MM:SS to a decimal value. Returns decimal degrees.

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
    '''
    This calculates the great circle angular distance in arcseconds between two
    coordinates (ra1,dec1) and (ra2,dec2). This is basically a clone of GCIRC
    from the IDL Astrolib.

    PARAMETERS:

    ra1,dec1: first coordinate (decimal degrees) -- scalar or np.array
    ra2,dec2: second coordinate (decimal degrees) -- scalar or np.array

    RETURNS:

    great circle distance between the two coordinates in arseconds.

    if (ra1,dec1) scalar and (ra2,dec2) scalar: result is a scalar

    if (ra1,dec1) scalar and (ra2,dec2) np.array: result is np.array with
    distance between (ra1,dec1) and each element of (ra2,dec2)

    if (ra1,dec1) np.array and (ra2,dec2) scalar: result is np.array with
    distance between (ra2,dec2) and each element of (ra1,dec1)

    if (ra1,dec1) and (ra2,dec2) both np.arrays: result is np.array with
    pair-wise distance between each element of the two coordinate lists.

    If the input np.arrays are not the same length, then excess elements of the
    longer ones will be ignored.

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
    del_ra2 =  (ra2_rad - ra1_rad)/2.0
    sin_dist = np.sqrt(np.sin(del_dec2) * np.sin(del_dec2) + \
                           np.cos(dec1_rad) * np.cos(dec2_rad) * \
                           np.sin(del_ra2) * np.sin(del_ra2))

    dist_rad = 2.0 * np.arcsin(sin_dist)

    # return the distance in arcseconds
    return np.rad2deg(dist_rad)*3600.0


def xmatch_basic(ra1, dec1, ra2, dec2, match_radius=5.0):
    '''
    This is a quick matcher that uses great_circle_dist to find the closest
    object in (ra2,dec2) within match_radius to (ra1,dec1). (ra1,dec1) must be a
    scalar pair, while (ra2,dec2) must be np.arrays of the same lengths.

    PARAMETERS:
    ra1/dec1: coordinates of the target to match
    ra2/dec2: coordinate np.arrays of the list of coordinates to match to

    RETURNS:

    A tuple like the following:

    (True -> no match or False -> matched,
     minimum distance between target and list)

    '''

    min_dist_arcsec = np.min(great_circle_dist(ra1,dec1,ra2,dec2))

    if (min_dist_arcsec < match_radius):
        return (True,min_dist_arcsec)
    else:
        return (False,min_dist_arcsec)



def xmatch_neighbors(ra1, dec1, ra2, dec2, match_radius=60.0,
                     includeself=False,sortresults=True):
    '''
    This is a quick matcher that uses great_circle_dist to find the closest
    neighbors in (ra2,dec2) within match_radius to (ra1,dec1). (ra1,dec1) must
    be a scalar pair, while (ra2,dec2) must be np.arrays of the same lengths

    PARAMETERS:
    ra1/dec1: coordinates of the target to match

    ra2/dec2: coordinate np.arrays of the list of coordinates to match to

    includeself: if True, includes matches in list to self-coordinates

    sortresults: if True, returns match_index in order of increasing distance
    from target

    RETURNS:

    A tuple like the following:

    (True -> no match or False -> matched,
     minimum distance between target and list,
     np.array of indices where list of coordinates is closer than match_radius
     to the target)

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
    '''
    This makes a scipy.spatial.CKDTree on ra, decl.

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
                      racenter, declcenter,
                      searchradiusdeg,
                      conesearchworkers=1):
    '''
    This does a cone-search around (racenter, declcenter) in kdtree.

    Returns the indices of kdtree.data that match the specified criteria.

    '''

    cosdecl = np.cos(np.radians(declcenter))
    sindecl = np.sin(np.radians(declcenter))
    cosra = np.cos(np.radians(racenter))
    sinra = np.sin(np.radians(racenter))

    # this is the search distance in xyz unit vectors
    xyzdist = 2.0 * np.sin(np.radians(searchradius)/2.0)

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
    '''This cross-matches between kdtree and (extra, extdecl) arrays.

    Returns the indices of the kdtree and the indices of extra, extdecl that
    xmatch successfully.

    If closestonly is True, then this function returns only the closest matching
    indices in (extra, extdecl) for each object in kdtree if there are any
    matches. Otherwise, it returns a list of indices in (extra, extdecl) for all
    matches within xmatchdistdeg between kdtree and (extra, extdecl).

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
    extkd_matchinds = ext_kdt.query_ball_tree(our_kdt, ext_xyzdist)

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

    '''
    This calculates the total proper motion of an object.

    '''

    pm = np.sqrt( pmdecl*pmdecl + pmra*pmra*np.cos(np.radians(decl)) *
                  np.cos(np.radians(decl)) )
    return pm


def reduced_proper_motion(jmag, propermotion):
    '''
    This calculates the reduced proper motion using the J magnitude.

    This is an effective measure of the absolute magnitude in the J band.

    '''

    rpm = jmag + 5.0*np.log10(propermotion/1000.0)
    return rpm


###########################
## COORDINATE CONVERSION ##
###########################

def equatorial_to_galactic(ra, decl, equinox='J2000'):
    '''
    This simply converts from equatorial coords to galactic coords.

    '''

    # convert the ra/decl to gl, gb
    radecl = SkyCoord(ra=ra*u.degree, dec=decl*u.degree, equinox=equinox)

    gl = radecl.galactic.l.degree
    gb = radecl.galactic.b.degree

    return gl, gb


def galactic_to_equatorial(gl, gb):
    '''
    This converts from galactic coords to equatorial coordinates.

    '''

    gal = SkyCoord(gl*u.degree, gl*u.degree, frame='galactic')

    transformed = gal.transform_to('icrs')

    return transformed.ra.degree, transformed.dec.degree



########################
## XI-ETA PROJECTIONS ##
########################

def xieta_from_radecl(inra, indecl, incenterra, incenterdecl, deg=True):
    '''This returns the image-plane projected xi-eta coords for inra, indecl.

    If deg = True, the input angles are assumed to be in degrees and the output
    is in degrees as well. A center RA and DEC are required.

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
    denom = vvn*vnn + wwn*wwn

    aunn = np.zeros_like(uun)
    auun[uun >= 1.0] = 0.0
    auun[uun < 1.0] = np.acos(uun)

    xi, eta = np.zeros_like(auun), np.zeros_like(auun)

    xi[(auun <= 0.0) | (denom <= 0.0)] = 0.0
    eta[(auun <= 0.0) | (denom <= 0.0)] = 0.0

    sdenom = np.sqrt(denom)

    xi[(auun > 0.0) | (denom > 0.0)] = auun*vvn/sdenom
    eta[(auun > 0.0) | (denom > 0.0)] = auun*wwn/sdenom

    if deg:
        return np.degrees(xi), np.degrees(eta)
    else:
        return xi, eta
