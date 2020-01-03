#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# timeutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Sept 2013

'''
Contains various useful tools for dealing with time in astronomical contexts.

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

import os.path
import os

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

except Exception:

    # this is so we don't download the JPL kernel when this module is imported
    # as part of a docs build on RTD
    import os
    RTD_INVOCATION = os.environ.get('RTD_IGNORE_HTLS_FAIL')

    if not RTD_INVOCATION:

        # this function is used to check progress of the download
        def on_download_chunk(transferred,blocksize,totalsize):
            progress = transferred*blocksize/float(totalsize)*100.0
            print('{progress:.1f}%'.format(progress=progress),end='\r')

        # this is the URL to the SPK
        spkurl = (
            'https://naif.jpl.nasa.gov/'
            'pub/naif/generic_kernels/spk/planets/de430.bsp'
        )

        LOGINFO('JPL kernel de430.bsp not found. Downloading from:\n\n%s\n' %
                spkurl)
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

    else:
        HAVEKERNEL = False


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
    '''Precesses target coordinates `ra`, `dec` from `epoch_one` to `epoch_two`.

    This takes into account the jd of the observations, as well as the proper
    motion of the target mu_ra, mu_dec. Adapted from J. D. Hartman's
    VARTOOLS/converttime.c [coordprecess].

    Parameters
    ----------

    ra,dec : float
        The equatorial coordinates of the object at `epoch_one` to precess in
        decimal degrees.

    epoch_one : float
        Origin epoch to precess from to target epoch. This is a float, like:
        1985.0, 2000.0, etc.

    epoch_two : float
        Target epoch to precess from origin epoch. This is a float, like:
        2000.0, 2018.0, etc.

    jd : float
        The full Julian date to use along with the propermotions in `mu_ra`, and
        `mu_dec` to handle proper motion along with the coordinate frame
        precession. If one of `jd`, `mu_ra`, or `mu_dec` is missing, the proper
        motion will not be used to calculate the final precessed coordinates.

    mu_ra,mu_dec : float
        The proper motion in mas/yr in right ascension and declination. If these
        are provided along with `jd`, the total proper motion of the object will
        be taken into account to calculate the final precessed coordinates.

    outscalar : bool
        If True, converts the output coordinates from one-element np.arrays to
        scalars.

    Returns
    -------

    precessed_ra, precessed_dec : float
        A tuple of precessed equatorial coordinates in decimal degrees at
        `epoch_two` taking into account proper motion if `jd`, `mu_ra`, and
        `mu_dec` are provided.

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


############################
## EPOCHS FOR EPHEMERIDES ##
############################

def _single_true(iterable):
    '''This returns True if only one True-ish element exists in `iterable`.

    Parameters
    ----------

    iterable : iterable

    Returns
    -------

    bool
        True if only one True-ish element exists in `iterable`. False otherwise.

    '''

    # return True if exactly one true found
    iterator = iter(iterable)

    # consume from "i" until first true or it's exhausted
    has_true = any(iterator)

    # carry on consuming until another true value / exhausted
    has_another_true = any(iterator)

    return has_true and not has_another_true


def get_epochs_given_midtimes_and_period(
        t_mid,
        period,
        err_t_mid=None,
        t0_fixed=None,
        t0_percentile=None,
        verbose=False
):
    '''This calculates the future epochs for a transit, given a period and a
    starting epoch

    The equation used is::

        t_mid = period*epoch + t0

    Default behavior if no kwargs are used is to define `t0` as the median
    finite time of the passed `t_mid` array.

    Only one of `err_t_mid` or `t0_fixed` should be passed.

    Parameters
    ----------

    t_mid : np.array
        A np.array of transit mid-time measurements

    period : float
        The period used to calculate epochs, per the equation above. For typical
        use cases, a period precise to ~1e-5 days is sufficient to get correct
        epochs.

    err_t_mid : None or np.array
        If provided, contains the errors of the transit mid-time
        measurements. The zero-point epoch is then set equal to the average of
        the transit times, weighted as `1/err_t_mid^2` . This minimizes the
        covariance between the transit epoch and the period (e.g., Gibson et
        al. 2013). For standard O-C analysis this is the best method.

    t0_fixed : None or float:
        If provided, use this t0 as the starting epoch. (Overrides all others).

    t0_percentile : None or float
        If provided, use this percentile of `t_mid` to define `t0`.

    Returns
    -------

    tuple
        This is the of the form `(integer_epoch_array, t0)`.
        `integer_epoch_array` is an array of integer epochs (float-type),
        of length equal to the number of *finite* mid-times passed.

    '''

    kwargarr = np.array([isinstance(err_t_mid,np.ndarray),
                         t0_fixed,
                         t0_percentile])
    if not _single_true(kwargarr) and not np.all(~kwargarr.astype(bool)):
        raise AssertionError(
            'can have at most one of err_t_mid, t0_fixed, t0_percentile')

    t_mid = t_mid[np.isfinite(t_mid)]
    N_midtimes = len(t_mid)

    if t0_fixed:
        t0 = t0_fixed
    elif isinstance(err_t_mid,np.ndarray):
        # get the weighted average. then round it to the nearest transit epoch.
        t0_avg = np.average(t_mid, weights=1/err_t_mid**2)
        t0_options = np.arange(min(t_mid), max(t_mid)+period, period)
        t0 = t0_options[np.argmin(np.abs(t0_options - t0_avg))]
    else:
        if not t0_percentile:
            # if there are an odd number of times, take the median time as
            # epoch=0.  elif there are an even number of times, take the lower
            # of the two middle times as epoch=0.
            if N_midtimes % 2 == 1:
                t0 = np.median(t_mid)
            else:
                t0 = t_mid[int(N_midtimes/2)]
        else:
            t0 = np.sort(t_mid)[int(N_midtimes*t0_percentile/100)]

    epoch = (t_mid - t0)/period

    # do not convert numpy entries to actual ints, because np.nan is float type
    int_epoch = np.round(epoch, 0)

    if verbose:
        LOGINFO('epochs before rounding')
        LOGINFO('\n{:s}'.format(repr(epoch)))
        LOGINFO('epochs after rounding')
        LOGINFO('\n{:s}'.format(repr(int_epoch)))

    return int_epoch, t0


###########################
## JULIAN DATE FUNCTIONS ##
###########################

def unixtime_to_jd(unix_time):
    '''This converts UNIX time in seconds to a Julian date in UTC (JD_UTC).

    Parameters
    ----------

    unix_time : float
        A UNIX time in decimal seconds since the 1970 UNIX epoch.

    Returns
    -------

    jd : float
        The Julian date corresponding to the provided UNIX time.

    '''

    # use astropy's time module
    jdutc = astime.Time(unix_time, format='unix', scale='utc')
    return jdutc.jd


def datetime_to_jd(dt):
    '''This converts a Python datetime object (naive, time in UT) to JD_UTC.

    Parameters
    ----------

    dt : datetime
        A naive Python `datetime` object (e.g. with no tz attribute) measured at
        UTC.

    Returns
    -------

    jd : float
        The Julian date corresponding to the `datetime` object.

    '''

    jdutc = astime.Time(dt, format='datetime',scale='utc')
    return jdutc.jd


def jd_to_datetime(jd, returniso=False):
    '''This converts a UTC JD to a Python `datetime` object or ISO date string.

    Parameters
    ----------

    jd : float
        The Julian date measured at UTC.

    returniso : bool
        If False, returns a naive Python `datetime` object corresponding to
        `jd`. If True, returns the ISO format string corresponding to the date
        and time at UTC from `jd`.

    Returns
    -------

    datetime or str
        Depending on the value of `returniso`.

    '''

    tt = astime.Time(jd, format='jd', scale='utc')

    if returniso:
        return tt.iso
    else:
        return tt.datetime


def jd_now():
    '''
    Gets the Julian date at the current time.

    Returns
    -------

    float
        The current Julian date in days.

    '''
    return astime.Time.now().jd


def jd_to_mjd(jd):
    '''
    Converts Julian Date to Modified Julian Date.

    Parameters
    ----------

    jd : float
        The Julian date measured at UTC.

    Returns
    -------

    mjd : float
        `mjd = jd - 2400000.5`

    '''

    return jd - 2400000.5


def mjd_to_jd(mjd):
    '''
    Converts Modified Julian date to Julian Date.

    Parameters
    ----------

    mjd : float
        The Modified Julian date measured at UTC.

    Returns
    -------

    jd : float
        `jd = mjd + 2400000.5`

    '''

    return mjd + 2400000.5


##################################
## CONVERTING JD TO HJD and BJD ##
##################################

def jd_corr(jd,
            ra, dec,
            obslon=None,
            obslat=None,
            obsalt=None,
            jd_type='bjd'):
    '''Returns BJD_TDB or HJD_TDB for input JD_UTC.

    The equation used is::

        BJD_TDB = JD_UTC + JD_to_TDB_corr + romer_delay

    where:

    - JD_to_TDB_corr is the difference between UTC and TDB JDs
    - romer_delay is the delay caused by finite speed of light from Earth-Sun

    This is based on the code at:

    https://mail.scipy.org/pipermail/astropy/2014-April/003148.html

    Note that this does not correct for:

    1. precession of coordinates if the epoch is not 2000.0
    2. precession of coordinates if the target has a proper motion
    3. Shapiro delay
    4. Einstein delay

    Parameters
    ----------

    jd : float or array-like
        The Julian date(s) measured at UTC.

    ra,dec : float
        The equatorial coordinates of the object in decimal degrees.

    obslon,obslat,obsalt : float or None
        The longitude, latitude of the observatory in decimal degrees and
        altitude of the observatory in meters. If these are not provided, the
        corrected JD will be calculated with respect to the center of the Earth.

    jd_type : {'bjd','hjd'}
        Conversion type to perform, either to Baryocentric Julian Date ('bjd')
        or to Heliocenter Julian Date ('hjd').

    Returns
    -------

    float or np.array
        The converted BJD or HJD.

    '''

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
