#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''starfeatures - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
License: MIT. See the LICENSE file for more details.

This calculates various features related to the color/proper-motion of stars.

All of the functions in this module require as input an 'objectinfo' dict with
keys outlined below. This should usually be taken from a light curve file.

{
    'ra': right ascension in degrees. REQUIRED: all functions below,
    'decl': declination in degrees. REQUIRED: all functions below,
    'pmra': propermotion in RA in mas/yr. REQUIRED: coord_features,
    'pmdecl': propermotion in DEC in mas/yr. REQUIRED: coord_features,
    'jmag': 2MASS J mag. REQUIRED: color_features,
    'hmag': 2MASS H mag. REQUIRED: color_features,
    'kmag': 2MASS Ks mag. REQUIRED: color_features,
    'bmag': Cousins B mag. optional: color_features,
    'vmag': Cousins V mag. optional: color_features,
    'sdssu': SDSS u mag. optional: color_features,
    'sdssg': SDSS g mag. optional: color_features,
    'sdssr': SDSS r mag. optional: color_features,
    'sdssi': SDSS i mag. optional: color_features,
    'sdssz': SDSS z mag. optional: color_features,
}

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


#############
## IMPORTS ##
#############

from time import time as unixtime
import pickle
import gzip
import os.path

import numpy as np
from scipy.spatial import cKDTree, KDTree

from astropy.wcs import WCS

###################
## LOCAL IMPORTS ##
###################

from .. import magnitudes, coordutils
from ..services import dust, gaia, skyview

dust.set_logger_parent(__name__)
gaia.set_logger_parent(__name__)
skyview.set_logger_parent(__name__)

#########################
## COORDINATE FEATURES ##
#########################

def coord_features(objectinfo):

    '''
    Calculates object coordinates features, including:

    - galactic coordinates
    - total proper motion from pmra, pmdecl
    - reduced J proper motion from propermotion and Jmag


    '''

    if ('pmra' in objectinfo and 'pmdecl' in objectinfo and
        'jmag' in objectinfo and
        'ra' in objectinfo and 'decl' in objectinfo and
        objectinfo['pmra'] is not None and objectinfo['pmdecl'] is not None and
        objectinfo['jmag'] is not None and
        objectinfo['ra'] is not None and objectinfo['decl'] is not None):

        gl, gb = coordutils.equatorial_to_galactic(objectinfo['ra'],
                                                   objectinfo['decl'])

        propermotion = coordutils.total_proper_motion(objectinfo['pmra'],
                                                      objectinfo['pmdecl'],
                                                      objectinfo['decl'])

        rpm = coordutils.reduced_proper_motion(objectinfo['jmag'],
                                               propermotion)

        return {'propermotion':propermotion,
                'gl':gl,
                'gb':gb,
                'rpmj':rpm}

    elif ('ra' in objectinfo and 'decl' in objectinfo and
          objectinfo['ra'] is not None and objectinfo['decl'] is not None):

        gl, gb = coordutils.equatorial_to_galactic(objectinfo['ra'],
                                                   objectinfo['decl'])

        LOGWARNING("one or more of pmra, pmdecl, jmag "
                   "are missing from the input objectinfo dict, "
                   "can't get proper motion features")

        return {'propermotion':np.nan,
                'gl':gl,
                'gb':gb,
                'rpmj':np.nan}

    else:

        LOGERROR("one or more of pmra, pmdecl, jmag, ra, decl "
                 "are missing from the input objectinfo dict, can't continue")
        return {'propermotion':np.nan,
                'gl':np.nan,
                'gb':np.nan,
                'rpmj':np.nan}




#############################
## COLOR FEATURE FUNCTIONS ##
#############################

def color_features(objectinfo, deredden=True):
    '''Stellar colors and dereddened stellar colors using 2MASS DUST API:

    http://irsa.ipac.caltech.edu/applications/DUST/docs/dustProgramInterface.html

    Requires at least ra, decl, jmag, hmag, kmag.

    If sdssu, sdssg, sdssr, sdssi, sdssz aren't provided, they'll be calculated
    using the JHK mags and the conversion functions in astrobase.magnitudes.

    '''

    # this is the default output dict
    outdict = {
        'ugcolor':np.nan,
        'grcolor':np.nan,
        'ricolor':np.nan,
        'izcolor':np.nan,
        'jhcolor':np.nan,
        'hkcolor':np.nan,
        'jkcolor':np.nan,
        'ijcolor':np.nan,
        'gjcolor':np.nan,
        'gkcolor':np.nan,
        'bvcolor':np.nan,
        'vkcolor':np.nan,
        'extinctj':np.nan,
        'extincth':np.nan,
        'extinctk':np.nan,
        'extinctu':np.nan,
        'extinctg':np.nan,
        'extinctr':np.nan,
        'extincti':np.nan,
        'extinctz':np.nan,
        'extinctb':np.nan,
        'extinctv':np.nan,
        'sdssu':np.nan,
        'sdssg':np.nan,
        'sdssr':np.nan,
        'sdssi':np.nan,
        'sdssz':np.nan,
        'jmag':np.nan,
        'hmag':np.nan,
        'kmag':np.nan,
        'bmag':np.nan,
        'vmag':np.nan,
        'deredu':np.nan,
        'deredg':np.nan,
        'deredr':np.nan,
        'deredi':np.nan,
        'deredz':np.nan,
        'deredj':np.nan,
        'deredh':np.nan,
        'deredk':np.nan,
        'deredb':np.nan,
        'deredv':np.nan,
        'bmagfromjhk':False,
        'vmagfromjhk':False,
        'sdssufromjhk':False,
        'sdssgfromjhk':False,
        'sdssrfromjhk':False,
        'sdssifromjhk':False,
        'sdsszfromjhk':False,
        'dereddened':False
    }

    # if we don't have JHK, we can't continue
    if not ('jmag' in objectinfo and objectinfo['jmag'] is not None and
            'hmag' in objectinfo and objectinfo['hmag'] is not None and
            'kmag' in objectinfo and objectinfo['kmag'] is not None):
        LOGERROR("one or more of J, H, K mags not found in "
                 "objectinfo dict, can't continue")
        return outdict


    if (not ('ra' in objectinfo and 'decl' in objectinfo)):
        LOGERROR("ra or decl not found in objectinfo dict "
                 "for dereddening, can't continue")
        return outdict


    # first, get the extinction table for this object
    extinction = dust.extinction_query(objectinfo['ra'],
                                       objectinfo['decl'],
                                       verbose=False)

    if deredden and not extinction:
        LOGERROR("could not retrieve extinction info from "
                 "2MASS DUST, can't continue")
        return outdict

    if deredden:

        outdict['extinctj'] = extinction['Amag']['2MASS J']['sf11']
        outdict['extincth'] = extinction['Amag']['2MASS H']['sf11']
        outdict['extinctk'] = extinction['Amag']['2MASS Ks']['sf11']

        outdict['extinctu'] = extinction['Amag']['SDSS u']['sf11']
        outdict['extinctg'] = extinction['Amag']['SDSS g']['sf11']
        outdict['extinctr'] = extinction['Amag']['SDSS r']['sf11']
        outdict['extincti'] = extinction['Amag']['SDSS i']['sf11']
        outdict['extinctz'] = extinction['Amag']['SDSS z']['sf11']

        outdict['extinctb'] = extinction['Amag']['CTIO B']['sf11']
        outdict['extinctv'] = extinction['Amag']['CTIO V']['sf11']

    # get the 2MASS mags
    outdict['jmag'] = objectinfo['jmag']
    outdict['hmag'] = objectinfo['hmag']
    outdict['kmag'] = objectinfo['kmag']

    # dered these if requested
    if deredden:

        # calculate the dereddened JHK mags
        outdict['deredj'] = outdict['jmag'] - outdict['extinctj']
        outdict['deredh'] = outdict['hmag'] - outdict['extincth']
        outdict['deredk'] = outdict['kmag'] - outdict['extinctk']

    else:

        outdict['deredj'] = outdict['jmag']
        outdict['deredh'] = outdict['hmag']
        outdict['deredk'] = outdict['kmag']


    #
    # get the BVugriz mags from the JHK mags if necessary
    #
    # FIXME: should these be direct dered mag_0 = f(J_0, H_0, K_0) instead?
    # Bilir+ 2008 uses dereddened colors for their transforms, should check if
    # we need to do so here

    if ('bmag' not in objectinfo or
        ('bmag' in objectinfo and objectinfo['bmag'] is None)):
        outdict['bmag'] = magnitudes.jhk_to_bmag(objectinfo['jmag'],
                                                 objectinfo['hmag'],
                                                 objectinfo['kmag'])
        outdict['bmagfromjhk'] = True
    else:
        outdict['bmag'] = objectinfo['bmag']

    if ('vmag' not in objectinfo or
        ('vmag' in objectinfo and objectinfo['vmag'] is None)):
        outdict['vmag'] = magnitudes.jhk_to_vmag(objectinfo['jmag'],
                                                 objectinfo['hmag'],
                                                 objectinfo['kmag'])
        outdict['vmagfromjhk'] = True
    else:
        outdict['vmag'] = objectinfo['vmag']

    if ('sdssu' not in objectinfo or
        ('sdssu' in objectinfo and objectinfo['sdssu'] is None)):
        outdict['sdssu'] = magnitudes.jhk_to_sdssu(objectinfo['jmag'],
                                                   objectinfo['hmag'],
                                                   objectinfo['kmag'])
        outdict['sdssufromjhk'] = True
    else:
        outdict['sdssu'] = objectinfo['sdssu']

    if ('sdssg' not in objectinfo or
        ('sdssg' in objectinfo and objectinfo['sdssg'] is None)):
        outdict['sdssg'] = magnitudes.jhk_to_sdssg(objectinfo['jmag'],
                                                   objectinfo['hmag'],
                                                   objectinfo['kmag'])
        outdict['sdssgfromjhk'] = True
    else:
        outdict['sdssg'] = objectinfo['sdssg']

    if ('sdssr' not in objectinfo or
        ('sdssr' in objectinfo and objectinfo['sdssr'] is None)):
        outdict['sdssr'] = magnitudes.jhk_to_sdssr(objectinfo['jmag'],
                                                   objectinfo['hmag'],
                                                   objectinfo['kmag'])
        outdict['sdssrfromjhk'] = True
    else:
        outdict['sdssr'] = objectinfo['sdssr']

    if ('sdssi' not in objectinfo or
        ('sdssi' in objectinfo and objectinfo['sdssi'] is None)):
        outdict['sdssi'] = magnitudes.jhk_to_sdssi(objectinfo['jmag'],
                                                   objectinfo['hmag'],
                                                   objectinfo['kmag'])
        outdict['sdssifromjhk'] = True
    else:
        outdict['sdssi'] = objectinfo['sdssi']

    if ('sdssz' not in objectinfo or
        ('sdssz' in objectinfo and objectinfo['sdssz'] is None)):
        outdict['sdssz'] = magnitudes.jhk_to_sdssz(objectinfo['jmag'],
                                                   objectinfo['hmag'],
                                                   objectinfo['kmag'])
        outdict['sdsszfromjhk'] = True
    else:
        outdict['sdssz'] = objectinfo['sdssz']


    # calculating dereddened mags:
    # A_x = m - m0_x where m is measured mag, m0 is intrinsic mag
    # m0_x = m - A_x
    #
    # so for two bands x, y:
    # intrinsic color (m_x - m_y)_0 = (m_x - m_y) - (A_x - A_y)

    if deredden:

        # calculate the dereddened SDSS mags
        outdict['deredu'] = outdict['sdssu'] - outdict['extinctu']
        outdict['deredg'] = outdict['sdssg'] - outdict['extinctg']
        outdict['deredr'] = outdict['sdssr'] - outdict['extinctr']
        outdict['deredi'] = outdict['sdssi'] - outdict['extincti']
        outdict['deredz'] = outdict['sdssz'] - outdict['extinctz']

        # calculate the dereddened B and V mags
        outdict['deredb'] = outdict['bmag'] - outdict['extinctb']
        outdict['deredv'] = outdict['vmag'] - outdict['extinctv']

        outdict['dereddened'] = True

    else:

        outdict['deredu'] = outdict['sdssu']
        outdict['deredg'] = outdict['sdssg']
        outdict['deredr'] = outdict['sdssr']
        outdict['deredi'] = outdict['sdssi']
        outdict['deredz'] = outdict['sdssz']

        outdict['deredb'] = outdict['bmag']
        outdict['deredv'] = outdict['vmag']


    # finally, calculate the colors
    outdict['ugcolor'] = outdict['deredu'] - outdict['deredg']
    outdict['grcolor'] = outdict['deredg'] - outdict['deredr']
    outdict['ricolor'] = outdict['deredr'] - outdict['deredi']
    outdict['izcolor'] = outdict['deredi'] - outdict['deredz']

    outdict['jhcolor'] = outdict['deredj'] - outdict['deredh']
    outdict['hkcolor'] = outdict['deredh'] - outdict['deredk']
    outdict['jkcolor'] = outdict['deredj'] - outdict['deredk']

    outdict['ijcolor'] = outdict['deredi'] - outdict['deredj']
    outdict['gjcolor'] = outdict['deredg'] - outdict['deredj']
    outdict['gkcolor'] = outdict['deredg'] - outdict['deredk']

    outdict['bvcolor'] = outdict['deredb'] - outdict['deredv']
    outdict['vkcolor'] = outdict['deredv'] - outdict['deredk']

    return outdict



def mdwarf_subtype_from_sdsscolor(ri_color, iz_color):

    # calculate the spectral type index and the spectral type spread of the
    # object. sti is calculated by fitting a line to the locus in r-i and i-z
    # space for M dwarfs in West+ 2007
    obj_sti = 0.875274*ri_color + 0.483628*(iz_color + 0.00438)
    obj_sts = -0.483628*ri_color + 0.875274*(iz_color + 0.00438)

    # possible M star if sti is >= 0.666 but <= 3.4559
    if ((obj_sti > 0.666) and (obj_sti < 3.4559)):

        # decide which M subclass object this is
        if ((obj_sti > 0.6660) and (obj_sti < 0.8592)):
            m_class = 'M0'

        if ((obj_sti > 0.8592) and (obj_sti < 1.0822)):
            m_class = 'M1'

        if ((obj_sti > 1.0822) and (obj_sti < 1.2998)):
            m_class = 'M2'

        if ((obj_sti > 1.2998) and (obj_sti < 1.6378)):
            m_class = 'M3'

        if ((obj_sti > 1.6378) and (obj_sti < 2.0363)):
            m_class = 'M4'

        if ((obj_sti > 2.0363) and (obj_sti < 2.2411)):
            m_class = 'M5'

        if ((obj_sti > 2.2411) and (obj_sti < 2.4126)):
            m_class = 'M6'

        if ((obj_sti > 2.4126) and (obj_sti < 2.9213)):
            m_class = 'M7'

        if ((obj_sti > 2.9213) and (obj_sti < 3.2418)):
            m_class = 'M8'

        if ((obj_sti > 3.2418) and (obj_sti < 3.4559)):
            m_class = 'M9'

    else:
        m_class = None

    return m_class, obj_sti, obj_sts


def color_classification(colorfeatures, pmfeatures):
    '''This calculates rough classifications based on star colors from ugrizJHK.

    Uses the output from color_features and coord_features. By default,
    color_features will use dereddened colors, as are expected by most relations
    here.

    Based on the color cuts from:

    - SDSS SEGUE (Yanny+ 2009)
    - SDSS QSO catalog (Schneider+ 2007)
    - SDSS RR Lyrae catalog (Sesar+ 2011)
    - SDSS M-dwarf catalog (West+ 2008)
    - Helmi+ 2003
    - Bochanski+ 2014

    '''

    possible_classes = []

    if not colorfeatures:
        LOGERROR("no color features provided, can't continue")
        return possible_classes

    if not pmfeatures:
        LOGERROR("no proper motion features provided, can't continue")
        return possible_classes

    # dered mags
    u, g, r, i, z, j, h, k = (colorfeatures['deredu'],
                              colorfeatures['deredg'],
                              colorfeatures['deredr'],
                              colorfeatures['deredi'],
                              colorfeatures['deredz'],
                              colorfeatures['deredj'],
                              colorfeatures['deredh'],
                              colorfeatures['deredk'])

    # measured mags
    um, gm, rm, im, zm = (colorfeatures['sdssu'],
                          colorfeatures['sdssg'],
                          colorfeatures['sdssr'],
                          colorfeatures['sdssi'],
                          colorfeatures['sdssz'])

    # reduced proper motion
    rpmj = pmfeatures['rpmj'] if np.isfinite(pmfeatures['rpmj']) else None

    # now generate the various color indices
    # color-gravity index
    v_color = 0.283*(u-g)-0.354*(g-r)+0.455*(r-i)+0.766*(i-z)

    # metallicity index p1
    p1_color = 0.91*(u-g)+0.415*(g-r)-1.28

    # metallicity index l
    l_color = -0.436*u + 1.129*g - 0.119*r - 0.574*i + 0.1984

    # metallicity index s
    s_color = -0.249*u + 0.794*g - 0.555*r + 0.124

    # RR Lyrae ug index
    d_ug = (u-g) + 0.67*(g-r) - 1.07

    # RR Lyrae gr index
    d_gr = 0.45*(u-g) - (g-r) - 0.12

    # check the M subtype
    m_subtype, m_sti, m_sts = mdwarf_subtype_from_sdsscolor(r-i, i-z)

    # now check if this is a likely M dwarf
    if m_subtype and rpmj and rpmj > 1.0:
        possible_classes.append('d' + m_subtype)

    # white dwarf
    if ( ((g-r) < -0.2) and ((g-r) > -1.0) and
         ((u-g) < 0.7) and ((u-g) > -1) and
         ((u-g+2*(g-r)) < -0.1) ):
        possible_classes.append('WD/sdO/sdB')

    # A/BHB/BStrg
    if ( ((u-g) < 1.5) and ((u-g) > 0.8) and
         ((g-r) < 0.2) and ((g-r) > -0.5) ):
        possible_classes.append('A/BHB/blustrg')

    # F turnoff/sub-dwarf
    if ( (p1_color < -0.25) and (p1_color > -0.7) and
         ((u-g) < 1.4) and ((u-g) > 0.4) and
         ((g-r) < 0.7) and ((g-r) > 0.2) ):
        possible_classes.append('Fturnoff/sdF')

    # low metallicity
    if ( ((g-r) < 0.75) and ((g-r) > -0.5) and
         ((u-g) < 3.0) and ((u-g) > 0.6) and
         (l_color > 0.135) ):
        possible_classes.append('lowmetal')

    # low metallicity giants from Helmi+ 2003
    if ( (-0.1 < p1_color < 0.6) and (s_color > 0.05) ):
        possible_classes.append('lowmetalgiant')

    # F/G star
    if ( ((g-r) < 0.48) and ((g-r) > 0.2) ):
        possible_classes.append('F/G')

    # G dwarf
    if ( ((g-r) < 0.55) and ((g-r) > 0.48) ):
        possible_classes.append('dG')

    # K giant
    if ( ((g-r) > 0.35) and ((g-r) < 0.7) and
         (l_color > 0.07) and ((u-g) > 0.7) and ((u-g) < 4.0) and
         ((r-i) > 0.15) and ((r-i) < 0.6) ):
        possible_classes.append('gK')

    # AGB
    if ( ((u-g) < 3.5) and ((u-g) > 2.5) and
         ((g-r) < 1.3) and ((g-r) > 0.9) and
         (s_color < -0.06) ):
        possible_classes.append('AGB')

    # K dwarf
    if ( ((g-r) < 0.75) and ((g-r) > 0.55) ):
        possible_classes.append('dK')

    # M subdwarf
    if ( ((g-r) > 1.6) and ((r-i) < 1.3) and ((r-i) > 0.95) ):
        possible_classes.append('sdM')

    # M giant colors from Bochanski+ 2014
    if ( ((j-k) > 1.02) and
         ((j-h) < (0.561*(j-k) + 0.46)) and
         ((j-h) > (0.561*(j-k) + 0.14)) and
         ((g-i) > (0.932*(i-k) - 0.872)) ):
        possible_classes.append('gM')

    # MS+WD pair
    if ( ((um-gm) < 2.25) and ((gm-rm) > -0.2) and
         ((gm-rm) < 1.2) and ((rm-im) > 0.5) and
         ((rm-im) < 2.0) and
         ((gm-rm) > (-19.78*(rm-im)+11.13)) and
         ((gm-rm) < (0.95*(rm-im)+0.5)) ):
        possible_classes.append('MSWD')

    # brown dwarf
    if ( (zm < 19.5) and (um > 21.0) and (gm > 22.0) and
         (rm > 21.0) and ((im - zm) > 1.7) ):
        possible_classes.append('BD')

    # RR Lyrae candidate
    if ( ((u-g) > 0.98) and ((u-g) < 1.3) and
         (d_ug > -0.05) and (d_ug < 0.35) and
         (d_gr > 0.06) and (d_gr < 0.55) and
         ((r-i) > -0.15) and ((r-i) < 0.22) and
         ((i-z) > -0.21) and ((i-z) < 0.25) ):
        possible_classes.append('RRL')

    # QSO color
    if ( (((u-g) > -0.1) and ((u-g) < 0.7) and
          ((g-r) > -0.3) and ((g-r) < 0.5)) or
         ((u-g) > (1.6*(g-r) + 1.34)) ):
        possible_classes.append('QSO')

    return {'color_classes':possible_classes,
            'v_color':v_color,
            'p1_color':p1_color,
            's_color':s_color,
            'l_color':l_color,
            'd_ug':d_ug,
            'd_gr':d_gr,
            'm_sti':m_sti,
            'm_sts':m_sts}



def neighbor_gaia_features(objectinfo,
                           lclist_kdtree,
                           neighbor_radius_arcsec,
                           verbose=True):
    '''Gets several neighbor and GAIA features:

    from the given light curve catalog:
    - distance to closest neighbor in arcsec
    - total number of all neighbors within 2 x neighbor_radius_arcsec

    from the GAIA DR1 catalog:
    - distance to closest neighbor in arcsec
    - total number of all neighbors within 2 x neighbor_radius_arcsec
    - gets the parallax for the object and neighbors
    - calculates the absolute GAIA mag and G-K color for use in CMDs

    objectinfo is the objectinfo dict from an object light curve

    lclist_kdtree is a scipy.spatial.cKDTree object built on the cartesian xyz
    coordinates from (ra, dec) of all objects in the same field as this
    object. It is similar to that produced by lcproc.make_lclist, and is used to
    carry out the spatial search required to find neighbors for this object.

    '''

    # kdtree search for neighbors in light curve catalog
    if ('ra' in objectinfo and 'decl' in objectinfo and
        objectinfo['ra'] is not None and objectinfo['decl'] is not None and
        (isinstance(lclist_kdtree, cKDTree) or
         isinstance(lclist_kdtree, KDTree))):

        ra, decl = objectinfo['ra'], objectinfo['decl']

        cosdecl = np.cos(np.radians(decl))
        sindecl = np.sin(np.radians(decl))
        cosra = np.cos(np.radians(ra))
        sinra = np.sin(np.radians(ra))

        # this is the search distance in xyz unit vectors
        xyzdist = 2.0 * np.sin(np.radians(neighbor_radius_arcsec/3600.0)/2.0)

        # look up the coordinates for the closest 100 objects in the kdtree
        # within 2 x neighbor_radius_arcsec
        kdt_dist, kdt_ind = lclist_kdtree.query(
            [cosra*cosdecl,
             sinra*cosdecl,
             sindecl],
            k=100,
            distance_upper_bound=xyzdist
        )

        # the first match is the object itself
        finite_distind = (np.isfinite(kdt_dist)) & (kdt_dist > 0)
        finite_dists = kdt_dist[finite_distind]
        nbrindices = kdt_ind[finite_distind]
        n_neighbors = finite_dists.size

        if n_neighbors > 0:

            closest_dist = finite_dists.min()
            closest_dist_arcsec = (
                np.degrees(2.0*np.arcsin(closest_dist/2.0))*3600.0
            )
            closest_dist_nbrind = nbrindices[finite_dists == finite_dists.min()]

            resultdict = {
                'neighbors':n_neighbors,
                'nbrindices':nbrindices,
                'distarcsec':np.degrees(2.0*np.arcsin(finite_dists/2.0))*3600.0,
                'closestdistarcsec':closest_dist_arcsec,
                'closestdistnbrind':closest_dist_nbrind,
                'searchradarcsec':neighbor_radius_arcsec,
            }

        else:

            resultdict = {
                'neighbors':0,
                'nbrindices':np.array([]),
                'distarcsec':np.array([]),
                'closestdistarcsec':np.nan,
                'closestdistnbrind':np.array([]),
                'searchradarcsec':neighbor_radius_arcsec,
            }


    else:
        if verbose:
            LOGWARNING("one of ra, decl, kdtree is missing in "
                       "objectinfo dict or lclistpkl, "
                       "can't get observed neighbors")

        resultdict = {
            'neighbors':np.nan,
            'nbrindices':np.array([]),
            'distarcsec':np.array([]),
            'closestdistarcsec':np.nan,
            'closestdistnbrind':np.array([]),
            'searchradarcsec':neighbor_radius_arcsec,
        }


    # next, search for this object in GAIA
    if ('ra' in objectinfo and 'decl' in objectinfo and
        objectinfo['ra'] is not None and objectinfo['decl'] is not None):

        gaia_result = gaia.objectlist_conesearch(objectinfo['ra'],
                                                 objectinfo['decl'],
                                                 neighbor_radius_arcsec,
                                                 verbose=False)
        if gaia_result:

            gaia_objlistf = gaia_result['result']

            with gzip.open(gaia_objlistf,'rb') as infd:

                gaia_objlist = np.genfromtxt(
                    infd,
                    names=True,
                    delimiter=',',
                    dtype='U20,f8,f8,f8,f8,f8,f8,f8,f8',
                    usecols=(0,1,2,3,4,5,6,7,8)
                )

            gaia_objlist = np.atleast_1d(gaia_objlist)

            if gaia_objlist.size > 0:

                # if we have GAIA results, we can get xypositions of all of
                # these objects on the object skyview stamp
                stampres = skyview.get_stamp(objectinfo['ra'],
                                             objectinfo['decl'])

                if (stampres and
                    'fitsfile' in stampres and
                    stampres['fitsfile'] is not None and
                    os.path.exists(stampres['fitsfile'])):

                    stampwcs = WCS(stampres['fitsfile'])

                    gaia_xypos = stampwcs.all_world2pix(
                        np.column_stack((gaia_objlist['ra'],
                                         gaia_objlist['dec'])),
                        1
                    )

                else:

                    gaia_xypos = None


                # the first object is likely the match to the object itself
                if gaia_objlist['dist_arcsec'][0] < 3.0:

                    if gaia_objlist.size > 1:

                        gaia_nneighbors = gaia_objlist[1:].size

                        gaia_status = (
                            'ok: object found with %s neighbors' %
                            gaia_nneighbors
                        )

                        # the first in each array is the object
                        gaia_ids = gaia_objlist['source_id']
                        gaia_mags = gaia_objlist['phot_g_mean_mag']
                        gaia_parallaxes = gaia_objlist['parallax']
                        gaia_parallax_errs = gaia_objlist['parallax_error']

                        gaia_absolute_mags = magnitudes.absolute_gaia_magnitude(
                            gaia_mags, gaia_parallaxes
                        )
                        gaiak_colors = gaia_mags - objectinfo['kmag']

                        gaia_dists = gaia_objlist['dist_arcsec']
                        gaia_closest_distarcsec = gaia_objlist['dist_arcsec'][1]
                        gaia_closest_gmagdiff = (
                            gaia_objlist['phot_g_mean_mag'][0] -
                            gaia_objlist['phot_g_mean_mag'][1]
                        )

                    else:

                        LOGWARNING('object found in GAIA at (%.3f,%.3f), '
                                   'but no neighbors' % (objectinfo['ra'],
                                                         objectinfo['decl']))

                        gaia_nneighbors = 0

                        gaia_status = (
                            'ok: object found but no neighbors'
                        )

                        # the first in each array is the object
                        gaia_ids = gaia_objlist['source_id']
                        gaia_mags = gaia_objlist['phot_g_mean_mag']
                        gaia_parallaxes = gaia_objlist['parallax']
                        gaia_parallax_errs = gaia_objlist['parallax_error']
                        gaia_absolute_mags = magnitudes.absolute_gaia_magnitude(
                            gaia_mags, gaia_parallaxes
                        )
                        gaiak_colors = gaia_mags - objectinfo['kmag']

                        gaia_dists = gaia_objlist['dist_arcsec']
                        gaia_closest_distarcsec = np.nan
                        gaia_closest_gmagdiff = np.nan


                # otherwise, the object wasn't found in GAIA for some reason
                else:

                    LOGWARNING('failed: no GAIA objects found within '
                               '%s of object position (%.3f, %.3f), '
                               'closest object is at %.3f arcsec away' %
                               (3.0, objectinfo['ra'], objectinfo['decl'],
                                gaia_objlist['dist_arcsec'][0]))

                    gaia_status = ('failed: no object within 3 '
                                   'arcsec, closest = %.3f arcsec' %
                                   gaia_objlist['dist_arcsec'][0])

                    gaia_nneighbors = np.nan

                    gaia_ids = gaia_objlist['source_id']
                    gaia_mags = gaia_objlist['phot_g_mean_mag']
                    gaia_parallaxes = gaia_objlist['parallax']
                    gaia_parallax_errs = gaia_objlist['parallax_error']
                    gaia_absolute_mags = magnitudes.absolute_gaia_magnitude(
                        gaia_mags, gaia_parallaxes
                    )
                    gaiak_colors = gaia_mags - objectinfo['kmag']

                    gaia_dists = gaia_objlist['dist_arcsec']
                    gaia_closest_distarcsec = np.nan
                    gaia_closest_gmagdiff = np.nan

            # if there are no neighbors within neighbor_radius_arcsec
            # or this object is not covered by GAIA. return nothing
            else:

                LOGERROR('no GAIA objects at this position')

                gaia_status = 'failed: no GAIA objects at this position'
                gaia_nneighbors = np.nan
                gaia_ids = None
                gaia_mags = None

                gaia_xypos = None
                gaia_parallaxes = None
                gaia_parallax_errs = None
                gaia_absolute_mags = None
                gaiak_colors = None

                gaia_dists = None
                gaia_closest_distarcsec = np.nan
                gaia_closest_gmagdiff = np.nan

            # update the resultdict with gaia stuff
            resultdict.update(
                {'gaia_status':gaia_status,
                 'gaia_neighbors':gaia_nneighbors,
                 'gaia_ids':gaia_ids,
                 'gaia_xypos':gaia_xypos,
                 'gaia_mags':gaia_mags,
                 'gaia_parallaxes':gaia_parallaxes,
                 'gaia_parallax_errs':gaia_parallax_errs,
                 'gaia_absolute_mags':gaia_absolute_mags,
                 'gaiak_colors':gaiak_colors,
                 'gaia_dists':gaia_dists,
                 'gaia_closest_distarcsec':gaia_closest_distarcsec,
                 'gaia_closest_gmagdiff':gaia_closest_gmagdiff}
            )

        else:

            LOGERROR('GAIA query did not return a '
                     'result for object at (%.3f, %.3f)' % (objectinfo['ra'],
                                                            objectinfo['decl']))

            resultdict.update(
                {'gaia_status':'failed: GAIA TAP query failed',
                 'gaia_neighbors':np.nan,
                 'gaia_ids':None,
                 'gaia_xypos':None,
                 'gaia_mags':None,
                 'gaia_parallaxes':None,
                 'gaia_parallax_errs':None,
                 'gaia_absolute_mags':None,
                 'gaiak_colors':None,
                 'gaia_dists':None,
                 'gaia_closest_distarcsec':np.nan,
                 'gaia_closest_gmagdiff':np.nan}
            )


    else:

        LOGERROR('objectinfo does not have ra or decl')

        resultdict.update(
            {'gaia_status':'failed: no ra/decl for object',
             'gaia_neighbors':np.nan,
             'gaia_ids':None,
             'gaia_xypos':None,
             'gaia_mags':None,
             'gaia_parallaxes':None,
             'gaia_parallax_errs':None,
             'gaia_absolute_mags':None,
             'gaiak_colors':None,
             'gaia_dists':None,
             'gaia_closest_distarcsec':np.nan,
             'gaia_closest_gmagdiff':np.nan}
        )

    return resultdict
