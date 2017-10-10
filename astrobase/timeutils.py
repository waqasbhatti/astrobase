#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

'''
timeutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Sept 2013

Contains various useful tools for dealing with time in astronomical contexts.

'''

import logging
try:
    import ConfigParser
except:
    import configparser as ConfigParser

import time
import os.path
import os

import multiprocessing as mp

import numpy as np

# we need the astropy.time.Time to convert from UTC to TDB
# this should also add any leap seconds on top of the relevant corrections
import astropy.time as astime

# we need the jplephem package from Brandon Rhodes to import and use the JPL
# ephemerides
from jplephem.spk import SPK


#################
## JPL KERNELS ##
#################

modpath = os.path.abspath(os.path.dirname(__file__))
planetdatafile = os.path.join(modpath,'data/de430.bsp')

# we'll try to load the SPK kernel. if that fails, we'll download it direct from
# JPL so the source distribution is kept small

try:

    # load the JPL kernel
    jplkernel = SPK.open(planetdatafile)
    HAVEKERNEL = True

except Exception as e:

    # this function is used to check progress of the download
    def on_download_chunk(transferred,blocksize,totalsize):
        progress = transferred*blocksize/float(totalsize)*100.0
        print('{progress:.1f}%'.format(progress=progress),end='\r')

    # this is the URL to the SPK
    spkurl = (
        'http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp'
    )

    print('JPL kernel de430.bsp not found. Downloading from:\n\n%s\n' % spkurl)
    try:
        from urllib import urlretrieve
    except:
        from urllib.request import urlretrieve

    localf, headerr = urlretrieve(
        spkurl,planetdatafile,reporthook=on_download_chunk
    )
    if os.path.exists(localf):
        print('\nDone.')
        jplkernel = SPK.open(planetdatafile)
        HAVEKERNEL = True
    else:
        print('failed to download the JPL kernel!')
        HAVEKERNEL = False


##############
## LOGGING  ##
##############

# setup a logger
LOGGER = None
DEBUG = False

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.timeutils' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )



######################
## USEFUL CONSTANTS ##
######################

# physical constants
CLIGHT_KPS = 299792.458

# various JDs
JD1800 = 2378495.0
JD2000 = 2451545.0
JD2000INT = 2451545
JD2050 = 2469807.5

# conversion factors
MAS_P_YR_TO_RAD_P_DAY = 1.3273475e-11
ARCSEC_TO_RADIANS = 4.84813681109536e-6
KM_P_AU = 1.49597870691e8
SEC_P_DAY = 86400.0

# this is the Earth-Moon mass ratio
# needed for calculating the position of the earth's center wrt SSB.
# the JPL ephemerides provide the position of the Earth-moon barycenter with
# respect to the solar-system barycenter.
EMRAT = 81.30056941599857


#######################
## UTILITY FUNCTIONS ##
#######################

def precess_coordinates(ra, dec,
                        epoch_one, epoch_two,
                        jd=None,
                        mu_ra=0.0,
                        mu_dec=0.0,
                        outscalar=False):
    '''
    Precesses target coordinates ra, dec from epoch_one to epoch_two, given the
    jd of the observations, as well as the proper motion of the target mu_ra,
    mu_dec. Adapted from hatpipe/source/vartools/converttime.c [coordprecess].

    epoch_one, epoch_two = epochs (e.g. 1985.0, 2013.0, etc.)

    jd = Julian date (full JD, not reduced JD)

    ra = right ascension in decimal degrees
    dec = declination in decimal degrees

    mu_ra = proper motion in RA (mas/yr)
    mu_dec = proper motion in Dec (mas/yr)

    '''

    raproc, decproc = np.radians(ra), np.radians(dec)

    if ((mu_ra != 0.0) and (mu_dec != 0.0) and jd):

        jd_epoch_one = JD2000 + (epoch_one - epoch_two)*365.25
        raproc = (
            raproc +
            (jd - jd_epoch_one)*mu_ra*MAS_P_YR_TO_RAD_P_DAY/np.cos(decproc)
            )
        decproc = decproc + (jd - jd_epoch_one)*mu_dec*MAS_P_YR_TO_RAD_P_DAY

    ca = np.cos(raproc)
    cd = np.cos(decproc)
    sa = np.sin(raproc)
    sd = np.sin(decproc)

    if epoch_one != epoch_two:

        t1 = 1.0e-3 * (epoch_two - epoch_one)
        t2 = 1.0e-3 * (epoch_one - 2000.0)

        a = ( t1*ARCSEC_TO_RADIANS * (23062.181 + t2*(139.656 + 0.0139*t2) +
                                      t1*(30.188 - 0.344*t2+17.998*t1)) )
        b = t1*t1*ARCSEC_TO_RADIANS*(79.280 + 0.410*t2 + 0.205*t1) + a
        c = (
            ARCSEC_TO_RADIANS*t1*(20043.109 - t2*(85.33 + 0.217*t2) +
                                  t1*(-42.665 - 0.217*t2 - 41.833*t2))
            )
        sina, sinb, sinc = np.sin(a), np.sin(b), np.sin(c)
        cosa, cosb, cosc = np.cos(a), np.cos(b), np.cos(c)

        precmatrix = np.matrix([[cosa*cosb*cosc - sina*sinb,
                                 sina*cosb + cosa*sinb*cosc,
                                 cosa*sinc],
                                [-cosa*sinb - sina*cosb*cosc,
                                  cosa*cosb - sina*sinb*cosc,
                                  -sina*sinc],
                                [-cosb*sinc,
                                  -sinb*sinc,
                                  cosc]])

        precmatrix = precmatrix.transpose()

        x = (np.matrix([cd*ca, cd*sa, sd])).transpose()

        x2 = precmatrix * x

        outra = np.arctan2(x2[1],x2[0])
        outdec = np.arcsin(x2[2])


        outradeg = np.rad2deg(outra)
        outdecdeg = np.rad2deg(outdec)

        if outradeg < 0.0:
            outradeg = outradeg + 360.0

        if outscalar:
            return float(outradeg), float(outdecdeg)
        else:
            return outradeg, outdecdeg

    else:

        # if the epochs are the same and no proper motion, this will be the same
        # as the input values. if the epochs are the same, but there IS proper
        # motion (and a given JD), then these will be perturbed from the input
        # values of ra, dec by the appropriate amount of motion
        return np.degrees(raproc), np.degrees(decproc)



###########################
## JULIAN DATE FUNCTIONS ##
###########################

def unixtime_to_jd(unix_time):
    '''
    This converts UNIX time in seconds to a julian date in UTC (JD_UTC).

    '''

    # use astropy's time module
    jdutc = astime.Time(unix_time, format='unix', scale='utc')
    return jdutc.jd



def datetime_to_jd(dt):
    '''
    This converts a Python datetime object (naive, time in UT) to JD_UTC.

    '''


    jdutc = astime.Time(dt, format='datetime',scale='utc')
    return jdutc.jd


def jd_to_datetime(jd, returniso=False):

    tt = astime.Time(jd, format='jd', scale='utc')

    if returniso:
        return tt.iso
    else:
        return tt.datetime


def jd_now():
    '''
    Returns the JD at the current time.

    '''
    return astime.Time.now().jd



def jd_to_mjd(jd):
    '''
    Converts Julian Date to Modified Julian Date.

    MJD = JD - 2400000.5

    '''

    return jd - 2400000.5



def mjd_to_jd(mjd):
    '''
    Converts Julian Date to Modified Julian Date.

    JD = MJD + 2400000.5

    '''

    return mjd + 2400000.5


##################################
## CONVERTING JD TO HJD and BJD ##
##################################

def jd_corr(jd,
            ra,
            dec,
            obslon=None,
            obslat=None,
            obsalt=None,
            jd_type='bjd'):
     """Return BJD_TDB or HJD_TDB for input JD_UTC

     BJD_TDB = JD_UTC + JD_to_TDB_corr + romer_delay

     where:

     JD_to_TDB_corr is the difference between UTC and TDB JDs
     romer_delay is the delay caused by finite speed of light from Earth-Sun

     This is based on the code at:

     https://mail.scipy.org/pipermail/astropy/2014-April/003148.html

     Note that this does not correct for:

     1. precession of coordinates if the epoch is not 2000.0
     2. precession of coordinates if the target has a proper motion
     3. location of the observatory on the earth (this adds 20 msec error)
     4. Shapiro delay
     5. Einstein delay

     """

     if not HAVEKERNEL:
         LOGERROR('no JPL kernel available, can\'t continue!')
         return

     # Source unit-vector
     ## Assume coordinates in ICRS
     ## Set distance to unit (kilometers)

     # convert the angles to degrees
     rarad = np.radians(ra)
     decrad = np.radians(dec)
     cosra = np.cos(rarad)
     sinra = np.sin(rarad)
     cosdec = np.cos(decrad)
     sindec = np.sin(decrad)

     # this assumes that the target is very far away
     src_unitvector = np.array([cosdec*cosra,cosdec*sinra,sindec])

     # Convert epochs to astropy.time.Time
     ## Assume JD(UTC)
     if (obslon is None) or (obslat is None) or (obsalt is None):
         t = astime.Time(jd, scale='utc', format='jd')
     else:
         t = astime.Time(jd, scale='utc', format='jd',
                         location=('%.5fd' % obslon,
                                   '%.5fd' % obslat,
                                   obsalt))

     # Get Earth-Moon barycenter position
     ## NB: jplephem uses Barycentric Dynamical Time, e.g. JD(TDB)
     ## and gives positions relative to solar system barycenter
     barycenter_earthmoon = jplkernel[0,3].compute(t.tdb.jd)

     # Get Moon position vectors from the center of Earth to the Moon
     # this means we get the following vectors from the ephemerides
     # Earth Barycenter (3) -> Moon (301)
     # Earth Barycenter (3) -> Earth (399)
     # so the final vector is [3,301] - [3,399]
     # units are in km
     moonvector = (jplkernel[3,301].compute(t.tdb.jd) -
                   jplkernel[3,399].compute(t.tdb.jd))

     # Compute Earth position vectors (this is for the center of the earth with
     # respect to the solar system barycenter)
     # all these units are in km
     pos_earth = (barycenter_earthmoon - moonvector * 1.0/(1.0+EMRAT))

     if jd_type == 'bjd':

         # Compute BJD correction
         ## Assume source vectors parallel at Earth and Solar System
         ## Barycenter
         ## i.e. source is at infinity
         # the romer_delay correction is (r.dot.n)/c where:
         # r is the vector from SSB to earth center
         # n is the unit vector from
         correction_seconds = np.dot(pos_earth.T, src_unitvector)/CLIGHT_KPS
         correction_days = correction_seconds/SEC_P_DAY

     elif jd_type == 'hjd':

         # Compute HJD correction via Sun ephemeris

         # this is the position vector of the center of the sun in km
         # Solar System Barycenter (0) -> Sun (10)
         pos_sun = jplkernel[0,10].compute(t.tdb.jd)

         # this is the vector from the center of the sun to the center of the
         # earth
         sun_earth_vec = pos_earth - pos_sun

         # calculate the heliocentric correction
         correction_seconds = np.dot(sun_earth_vec.T, src_unitvector)/CLIGHT_KPS
         correction_days = correction_seconds/SEC_P_DAY

     # TDB is the appropriate time scale for these ephemerides
     new_jd = t.tdb.jd + correction_days

     return new_jd
