#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# starfeatures - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
# License: MIT. See the LICENSE file for more details.

'''This calculates various features related to the color/proper-motion of stars.

All of the functions in this module require as input an 'objectinfo' dict. This
should usually be taken from a light curve file read into an `lcdict`. The
format and the minimum keys required are::

    {'objectid': the name of the object,
     'ra': the right ascension of the object in decimal degrees,
     'decl': the declination of the object in decimal degrees,
     'ndet': the number of observations of this object}

You can also provide magnitudes and proper motions of the object using the
following keys and the appropriate values in the `objectinfo` dict. These will
be used to calculate colors, total and reduced proper motion, etc.::

    'pmra' -> the proper motion in mas/yr in right ascension,
    'pmdecl' -> the proper motion in mas/yr in declination,
    'umag'  -> U mag		 -> colors: U-B, U-V, U-g
    'bmag'  -> B mag		 -> colors: U-B, B-V
    'vmag'  -> V mag		 -> colors: U-V, B-V, V-R, V-I, V-K
    'rmag'  -> R mag		 -> colors: V-R, R-I
    'imag'  -> I mag		 -> colors: g-I, V-I, R-I, B-I
    'jmag'  -> 2MASS J mag	 -> colors: J-H, J-K, g-J, i-J
    'hmag'  -> 2MASS H mag	 -> colors: J-H, H-K
    'kmag'  -> 2MASS Ks mag	 -> colors: g-Ks, H-Ks, J-Ks, V-Ks
    'sdssu' -> SDSS u mag	 -> colors: u-g, u-V
    'sdssg' -> SDSS g mag	 -> colors: g-r, g-i, g-K, u-g, U-g, g-J
    'sdssr' -> SDSS r mag	 -> colors: r-i, g-r
    'sdssi' -> SDSS i mag	 -> colors: r-i, i-z, g-i, i-J, i-W1
    'sdssz' -> SDSS z mag	 -> colors: i-z, z-W2, g-z
    'ujmag' -> UKIRT J mag	 -> colors: J-H, H-K, J-K, g-J, i-J
    'uhmag' -> UKIRT H mag	 -> colors: J-H, H-K
    'ukmag' -> UKIRT K mag	 -> colors: g-K, H-K, J-K, V-K
    'irac1' -> Spitzer IRAC1 mag -> colors: i-I1, I1-I2
    'irac2' -> Spitzer IRAC2 mag -> colors: I1-I2, I2-I3
    'irac3' -> Spitzer IRAC3 mag -> colors: I2-I3
    'irac4' -> Spitzer IRAC4 mag -> colors: I3-I4
    'wise1' -> WISE W1 mag	 -> colors: i-W1, W1-W2
    'wise2' -> WISE W2 mag	 -> colors: W1-W2, W2-W3
    'wise3' -> WISE W3 mag	 -> colors: W2-W3
    'wise4' -> WISE W4 mag	 -> colors: W3-W4

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

import gzip
import os.path

import numpy as np
from scipy.spatial import cKDTree, KDTree

from astropy.wcs import WCS


###################
## LOCAL IMPORTS ##
###################

from .. import magnitudes, coordutils
from ..services import dust, gaia, skyview, simbad


#########################
## COORDINATE FEATURES ##
#########################

def coord_features(objectinfo):

    '''Calculates object coordinates features, including:

    - galactic coordinates
    - total proper motion from pmra, pmdecl
    - reduced J proper motion from propermotion and Jmag

    Parameters
    ----------

    objectinfo : dict
        This is an objectinfo dict from a light curve file read into an
        `lcdict`. The format and the minimum keys required are::

            {'ra': the right ascension of the object in decimal degrees,
             'decl': the declination of the object in decimal degrees,
             'pmra': the proper motion in right ascension in mas/yr,
             'pmdecl': the proper motion in declination in mas/yr,
             'jmag': the 2MASS J mag of this object}

    Returns
    -------

    dict
        A dict containing the total proper motion

    '''

    retdict = {'propermotion': np.nan,
               'gl':np.nan,
               'gb':np.nan,
               'rpmj':np.nan}

    if ('ra' in objectinfo and
        objectinfo['ra'] is not None and
        np.isfinite(objectinfo['ra']) and
        'decl' in objectinfo and
        objectinfo['decl'] is not None and
        np.isfinite(objectinfo['decl'])):

        retdict['gl'], retdict['gb'] = coordutils.equatorial_to_galactic(
            objectinfo['ra'],
            objectinfo['decl']
        )

    if ('pmra' in objectinfo and
        objectinfo['pmra'] is not None and
        np.isfinite(objectinfo['pmra']) and
        'pmdecl' in objectinfo and
        objectinfo['pmdecl'] is not None and
        np.isfinite(objectinfo['pmdecl']) and
        'decl' in objectinfo and
        objectinfo['decl'] is not None and
        np.isfinite(objectinfo['decl'])):

        retdict['propermotion'] = coordutils.total_proper_motion(
            objectinfo['pmra'],
            objectinfo['pmdecl'],
            objectinfo['decl']
        )

    if ('jmag' in objectinfo and
        objectinfo['jmag'] is not None and
        np.isfinite(objectinfo['jmag']) and
        'propermotion' in retdict and
        retdict['propermotion'] is not None and
        np.isfinite(retdict['propermotion'])):

        retdict['rpmj'] = coordutils.reduced_proper_motion(
            objectinfo['jmag'],
            retdict['propermotion']
        )

    return retdict



#############################
## COLOR FEATURE FUNCTIONS ##
#############################

BANDPASSES_COLORS = {
    'umag':{'dustkey':'CTIO U',
            'label':'U',
            'colors':[['umag-bmag','U - B'],
                      ['umag-vmag','U - V'],
                      ['umag-sdssg','U - g']]},
    'bmag':{'dustkey':'CTIO B',
            'label':'B',
            'colors':[['umag-bmag','U - B'],
                      ['bmag-vmag','B - V']]},
    'vmag':{'dustkey':'CTIO V',
            'label':'V',
            'colors':[['umag-vmag','U - V'],
                      ['bmag-vmag','B - V'],
                      ['vmag-rmag','V - R'],
                      ['vmag-imag','V - I'],
                      ['vmag-kmag','V - K']]},
    'rmag':{'dustkey':'CTIO R',
            'label':'R',
            'colors':[['vmag-rmag','V - R'],
                      ['rmag-imag','R - I']]},
    'imag':{'dustkey':'CTIO I',
            'label':'I',
            'colors':[['sdssg-imag','g - I'],
                      ['vmag-imag','V - I'],
                      ['rmag-imag','R - I'],
                      ['bmag-imag','B - I']]},
    'jmag':{'dustkey':'2MASS J',
            'label':'J',
            'colors':[['jmag-hmag','J - H'],
                      ['jmag-kmag','J - Ks'],
                      ['sdssg-jmag','g - J'],
                      ['sdssi-jmag','i - J']]},
    'hmag':{'dustkey':'2MASS H',
            'label':'H',
            'colors':[['jmag-hmag','J - H'],
                      ['hmag-kmag','H - Ks']]},
    'kmag':{'dustkey':'2MASS Ks',
            'label':'Ks',
            'colors':[['sdssg-kmag','g - Ks'],
                      ['vmag-kmag','V - Ks'],
                      ['hmag-kmag','H - Ks'],
                      ['jmag-kmag','J - Ks']]},
    'sdssu':{'dustkey':'SDSS u',
             'label':'u',
             'colors':[['sdssu-sdssg','u - g'],
                       ['sdssu-vmag','u - V']]},
    'sdssg':{'dustkey':'SDSS g',
             'label':'g',
             'colors':[['sdssg-sdssr','g - r'],
                       ['sdssg-sdssi','g - i'],
                       ['sdssg-kmag','g - Ks'],
                       ['sdssu-sdssg','u - g'],
                       ['umag-sdssg','U - g'],
                       ['sdssg-jmag','g - J']]},
    'sdssr':{'dustkey':'SDSS r',
             'label':'r',
             'colors':[['sdssr-sdssi','r - i'],
                       ['sdssg-sdssr','g - r']]},
    'sdssi':{'dustkey':'SDSS i',
             'label':'i',
             'colors':[['sdssr-sdssi','r - i'],
                       ['sdssi-sdssz','i - z'],
                       ['sdssg-sdssi','g - i'],
                       ['sdssi-jmag','i - J'],
                       ['sdssi-wise1','i - W1']]},
    'sdssz':{'dustkey':'SDSS z',
             'label':'z',
             'colors':[['sdssi-sdssz','i - z'],
                       ['sdssz-wise2','z - W2'],
                       ['sdssg-sdssz','g - z']]},
    'ujmag':{'dustkey':'UKIRT J',
             'label':'uJ',
             'colors':[['ujmag-uhmag','uJ - uH'],
                       ['ujmag-ukmag','uJ - uK'],
                       ['sdssg-ujmag','g - uJ'],
                       ['sdssi-ujmag','i - uJ']]},
    'uhmag':{'dustkey':'UKIRT H',
             'label':'uH',
             'colors':[['ujmag-uhmag','uJ - uH'],
                       ['uhmag-ukmag','uH - uK']]},
    'ukmag':{'dustkey':'UKIRT K',
             'label':'uK',
             'colors':[['sdssg-ukmag','g - uK'],
                       ['vmag-ukmag','V - uK'],
                       ['uhmag-ukmag','uH - uK'],
                       ['ujmag-ukmag','uJ - uK']]},
    'irac1':{'dustkey':'IRAC-1',
             'label':'I1',
             'colors':[['sdssi-irac1','i - I1'],
                       ['irac1-irac2','I1 - I2']]},
    'irac2':{'dustkey':'IRAC-2',
             'label':'I2',
             'colors':[['irac1-irac2','I1 - I2'],
                       ['irac2-irac3','I2 - I3']]},
    'irac3':{'dustkey':'IRAC-3',
             'label':'I3',
             'colors':[['irac3-irac4','I3 - I4']]},
    'irac4':{'dustkey':'IRAC-4',
             'label':'I4',
             'colors':[['irac3-irac4','I3 - I4']]},
    'wise1':{'dustkey':'WISE-1',
             'label':'W1',
             'colors':[['sdssi-wise1','i - W1'],
                       ['wise1-wise2','W1 - W2']]},
    'wise2':{'dustkey':'WISE-2',
             'label':'W2',
             'colors':[['wise1-wise2','W1 - W2'],
                       ['wise2-wise3','W2 - W3']]},
    'wise3':{'dustkey':None,
             'label':'W3',
             'colors':[['wise2-wise3','W2 - W3']]},
    'wise4':{'dustkey':None,
             'label':'W4',
             'colors':[['wise3-wise4','W3 - W4']]},
    'tessmag':{'dustkey':None,
               'label':'T',
               'colors':[
                   ['tessmag-jmag', 'T - J'],
                   ['tessmag-kmag', 'T - Ks'],
               ]},
    'kepmag':{'dustkey':None,
              'label':'Kp',
              'colors':[
                  ['kepmag-jmag', 'Kp - J'],
                  ['kepmag-kmag', 'Kp - Ks']
              ]}
}


BANDPASS_LIST = [
    'umag','bmag','vmag','rmag','imag',
    'jmag','hmag','kmag',
    'sdssu','sdssg','sdssr','sdssi','sdssz',
    'ujmag','uhmag','ukmag',
    'irac1','irac2','irac3','irac4',
    'wise1','wise2','wise3','wise4',
    'tessmag',
    'kepmag',
]


def color_features(in_objectinfo,
                   deredden=True,
                   custom_bandpasses=None,
                   dust_timeout=10.0):
    '''Stellar colors and dereddened stellar colors using 2MASS DUST API:

    http://irsa.ipac.caltech.edu/applications/DUST/docs/dustProgramInterface.html

    Parameters
    ----------

    in_objectinfo : dict
        This is a dict that contains the object's magnitudes and positions. This
        requires at least 'ra', and 'decl' as keys which correspond to the right
        ascension and declination of the object, and one or more of the
        following keys for object magnitudes::

            'umag'  -> U mag             -> colors: U-B, U-V, U-g
            'bmag'  -> B mag             -> colors: U-B, B-V
            'vmag'  -> V mag             -> colors: U-V, B-V, V-R, V-I, V-K
            'rmag'  -> R mag             -> colors: V-R, R-I
            'imag'  -> I mag             -> colors: g-I, V-I, R-I, B-I
            'jmag'  -> 2MASS J mag       -> colors: J-H, J-K, g-J, i-J
            'hmag'  -> 2MASS H mag       -> colors: J-H, H-K
            'kmag'  -> 2MASS Ks mag      -> colors: g-Ks, H-Ks, J-Ks, V-Ks
            'sdssu' -> SDSS u mag        -> colors: u-g, u-V
            'sdssg' -> SDSS g mag        -> colors: g-r, g-i, g-K, u-g, U-g, g-J
            'sdssr' -> SDSS r mag        -> colors: r-i, g-r
            'sdssi' -> SDSS i mag        -> colors: r-i, i-z, g-i, i-J, i-W1
            'sdssz' -> SDSS z mag        -> colors: i-z, z-W2, g-z
            'ujmag' -> UKIRT J mag       -> colors: J-H, H-K, J-K, g-J, i-J
            'uhmag' -> UKIRT H mag       -> colors: J-H, H-K
            'ukmag' -> UKIRT K mag       -> colors: g-K, H-K, J-K, V-K
            'irac1' -> Spitzer IRAC1 mag -> colors: i-I1, I1-I2
            'irac2' -> Spitzer IRAC2 mag -> colors: I1-I2, I2-I3
            'irac3' -> Spitzer IRAC3 mag -> colors: I2-I3
            'irac4' -> Spitzer IRAC4 mag -> colors: I3-I4
            'wise1' -> WISE W1 mag       -> colors: i-W1, W1-W2
            'wise2' -> WISE W2 mag       -> colors: W1-W2, W2-W3
            'wise3' -> WISE W3 mag       -> colors: W2-W3
            'wise4' -> WISE W4 mag       -> colors: W3-W4

        These are basically taken from the available reddening bandpasses from
        the 2MASS DUST service. If B, V, u, g, r, i, z aren't provided but 2MASS
        J, H, Ks are all provided, the former will be calculated using the 2MASS
        JHKs -> BVugriz conversion functions in :py:mod:`astrobase.magnitudes`.

    deredden : bool
        If True, will make sure all colors use dereddened mags where possible.

    custom_bandpasses : dict
        This is a dict used to define any custom bandpasses in the
        `in_objectinfo` dict you want to make this function aware of and
        generate colors for. Use the format below for this dict::

            {
            '<bandpass_key_1>':{'dustkey':'<twomass_dust_key_1>',
                                'label':'<band_label_1>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            .
            ...
            .
            '<bandpass_key_N>':{'dustkey':'<twomass_dust_key_N>',
                                'label':'<band_label_N>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            }

        Where:

        `bandpass_key` is a key to use to refer to this bandpass in the
        `objectinfo` dict, e.g. 'sdssg' for SDSS g band

        `twomass_dust_key` is the key to use in the 2MASS DUST result table for
        reddening per band-pass. For example, given the following DUST result
        table (using http://irsa.ipac.caltech.edu/applications/DUST/)::

            |Filter_name|LamEff |A_over_E_B_V_SandF|A_SandF|A_over_E_B_V_SFD|A_SFD|
            |char       |float  |float             |float  |float           |float|
            |           |microns|                  |mags   |                |mags |
             CTIO U       0.3734              4.107   0.209            4.968 0.253
             CTIO B       0.4309              3.641   0.186            4.325 0.221
             CTIO V       0.5517              2.682   0.137            3.240 0.165
            .
            .
            ...

        The `twomass_dust_key` for 'vmag' would be 'CTIO V'. If you want to
        skip DUST lookup and want to pass in a specific reddening magnitude
        for your bandpass, use a float for the value of
        `twomass_dust_key`. If you want to skip DUST lookup entirely for
        this bandpass, use None for the value of `twomass_dust_key`.

        `band_label` is the label to use for this bandpass, e.g. 'W1' for
        WISE-1 band, 'u' for SDSS u, etc.

        The 'colors' list contains color definitions for all colors you want
        to generate using this bandpass. this list contains elements of the
        form::

            ['<bandkey1>-<bandkey2>','<BAND1> - <BAND2>']

        where the the first item is the bandpass keys making up this color,
        and the second item is the label for this color to be used by the
        frontends. An example::

            ['sdssu-sdssg','u - g']

    dust_timeout : float
        The timeout to use when contacting the 2MASS DUST web service.

    Returns
    -------

    dict
        An `objectinfo` dict with all of the generated colors, dereddened
        magnitude,s dereddened colors, as specified in the input args is
        returned.

    '''

    objectinfo = in_objectinfo.copy()

    # this is the initial output dict
    outdict = {
        'available_bands':[],
        'available_band_labels':[],
        'available_dereddened_bands':[],
        'available_dereddened_band_labels':[],
        'available_colors':[],
        'available_color_labels':[],
        'dereddened':False
    }

    #
    # get the BVugriz mags from the JHK mags if necessary
    #
    # FIXME: should these be direct dered mag_0 = f(J_0, H_0, K_0) instead?
    # Bilir+ 2008 uses dereddened colors for their transforms, should check if
    # we need to do so here

    if ('jmag' in objectinfo and
        objectinfo['jmag'] is not None and
        np.isfinite(objectinfo['jmag']) and
        'hmag' in objectinfo and
        objectinfo['hmag'] is not None and
        np.isfinite(objectinfo['hmag']) and
        'kmag' in objectinfo and
        objectinfo['kmag'] is not None and
        np.isfinite(objectinfo['kmag'])):

        if ('bmag' not in objectinfo or
            ('bmag' in objectinfo and objectinfo['bmag'] is None) or
            ('bmag' in objectinfo and not np.isfinite(objectinfo['bmag']))):
            objectinfo['bmag'] = magnitudes.jhk_to_bmag(objectinfo['jmag'],
                                                        objectinfo['hmag'],
                                                        objectinfo['kmag'])
            outdict['bmagfromjhk'] = True
        else:
            outdict['bmagfromjhk'] = False

        if ('vmag' not in objectinfo or
            ('vmag' in objectinfo and objectinfo['vmag'] is None) or
            ('vmag' in objectinfo and not np.isfinite(objectinfo['vmag']))):
            objectinfo['vmag'] = magnitudes.jhk_to_vmag(objectinfo['jmag'],
                                                        objectinfo['hmag'],
                                                        objectinfo['kmag'])
            outdict['vmagfromjhk'] = True
        else:
            outdict['vmagfromjhk'] = False


        if ('sdssu' not in objectinfo or
            ('sdssu' in objectinfo and objectinfo['sdssu'] is None) or
            ('sdssu' in objectinfo and not np.isfinite(objectinfo['sdssu']))):
            objectinfo['sdssu'] = magnitudes.jhk_to_sdssu(objectinfo['jmag'],
                                                          objectinfo['hmag'],
                                                          objectinfo['kmag'])
            outdict['sdssufromjhk'] = True
        else:
            outdict['sdssufromjhk'] = False

        if ('sdssg' not in objectinfo or
            ('sdssg' in objectinfo and objectinfo['sdssg'] is None) or
            ('sdssg' in objectinfo and not np.isfinite(objectinfo['sdssg']))):
            objectinfo['sdssg'] = magnitudes.jhk_to_sdssg(objectinfo['jmag'],
                                                          objectinfo['hmag'],
                                                          objectinfo['kmag'])
            outdict['sdssgfromjhk'] = True
        else:
            outdict['sdssgfromjhk'] = False

        if ('sdssr' not in objectinfo or
            ('sdssr' in objectinfo and objectinfo['sdssr'] is None) or
            ('sdssr' in objectinfo and not np.isfinite(objectinfo['sdssr']))):
            objectinfo['sdssr'] = magnitudes.jhk_to_sdssr(objectinfo['jmag'],
                                                          objectinfo['hmag'],
                                                          objectinfo['kmag'])
            outdict['sdssrfromjhk'] = True
        else:
            outdict['sdssrfromjhk'] = False

        if ('sdssi' not in objectinfo or
            ('sdssi' in objectinfo and objectinfo['sdssi'] is None) or
            ('sdssi' in objectinfo and not np.isfinite(objectinfo['sdssi']))):
            objectinfo['sdssi'] = magnitudes.jhk_to_sdssi(objectinfo['jmag'],
                                                          objectinfo['hmag'],
                                                          objectinfo['kmag'])
            outdict['sdssifromjhk'] = True
        else:
            outdict['sdssifromjhk'] = False

        if ('sdssz' not in objectinfo or
            ('sdssz' in objectinfo and objectinfo['sdssz'] is None) or
            ('sdssz' in objectinfo and not np.isfinite(objectinfo['sdssz']))):
            objectinfo['sdssz'] = magnitudes.jhk_to_sdssz(objectinfo['jmag'],
                                                          objectinfo['hmag'],
                                                          objectinfo['kmag'])
            outdict['sdsszfromjhk'] = True
        else:
            outdict['sdsszfromjhk'] = False


    # now handle dereddening if possible
    if deredden:

        try:

            # first, get the extinction table for this object
            extinction = dust.extinction_query(objectinfo['ra'],
                                               objectinfo['decl'],
                                               verbose=False,
                                               timeout=dust_timeout)

        except Exception as e:

            LOGERROR("deredden = True but 'ra', 'decl' keys not present "
                     "or invalid in objectinfo dict, ignoring reddening...")
            extinction = None
            outdict['dereddened'] = False

    else:

        extinction = None
        outdict['dereddened'] = False

    # handle timeout from DUST service
    if not extinction:
        outdict['dereddened'] = False

    # go through the objectdict and pick out the mags we have available from the
    # BANDPASSES_COLORS dict

    # update our bandpasses_colors dict with any custom ones the user defined
    our_bandpasses_colors = BANDPASSES_COLORS.copy()
    our_bandpass_list = BANDPASS_LIST[::]

    if custom_bandpasses is not None and isinstance(custom_bandpasses, dict):

        our_bandpasses_colors.update(custom_bandpasses)

        # also update the list
        for key in custom_bandpasses:
            if key not in our_bandpass_list:
                our_bandpass_list.append(key)

    for mk in our_bandpass_list:

        if (mk in objectinfo and
            objectinfo[mk] is not None and
            np.isfinite(objectinfo[mk])):

            thisbandlabel = our_bandpasses_colors[mk]['label']
            thisdustkey = our_bandpasses_colors[mk]['dustkey']

            # add this to the outdict
            outdict[mk] = objectinfo[mk]

            outdict['available_bands'].append(mk)
            outdict['available_band_labels'].append(thisbandlabel)

            #
            # deredden if possible
            #
            # calculating dereddened mags:
            # A_x = m - m0_x where m is measured mag, m0 is intrinsic mag
            # m0_x = m - A_x
            #
            # so for two bands x, y:
            # intrinsic color (m_x - m_y)_0 = (m_x - m_y) - (A_x - A_y)
            if (deredden and extinction):

                outdict['dereddened'] = True

                # check if the dustkey is None, float, or str to figure out how
                # to retrieve the reddening
                if (thisdustkey is not None and
                    isinstance(thisdustkey, str) and
                    thisdustkey in extinction['Amag'] and
                    np.isfinite(extinction['Amag'][thisdustkey]['sf11'])):

                    outdict['extinction_%s' % mk] = (
                        extinction['Amag'][thisdustkey]['sf11']
                    )

                elif (thisdustkey is not None and
                      isinstance(thisdustkey, float)):

                    outdict['extinction_%s' % mk] = thisdustkey

                else:

                    outdict['extinction_%s' % mk] = 0.0

                # apply the extinction
                outdict['dered_%s' % mk] = (
                    outdict[mk] - outdict['extinction_%s' % mk]
                )
                outdict['available_dereddened_bands'].append('dered_%s' % mk)
                outdict['available_dereddened_band_labels'].append(
                    thisbandlabel
                )

                # get all the colors to generate for this bandpass
                for colorspec in BANDPASSES_COLORS[mk]['colors']:

                    # only add this if the color's not there already
                    if colorspec[0] not in outdict:

                        colorkey, colorlabel = colorspec

                        # look for the bands to make this color

                        # if it's not found now, this should work when we come
                        # around for the next bandpass for this color

                        band1, band2 = colorkey.split('-')

                        if ('dered_%s' % band1 in outdict and
                            'dered_%s' % band2 in outdict and
                            np.isfinite(outdict['dered_%s' % band1]) and
                            np.isfinite(outdict['dered_%s' % band2])):

                            outdict[colorkey] = (
                                outdict['dered_%s' % band1] -
                                outdict['dered_%s' % band2]
                            )
                            outdict['available_colors'].append(colorkey)
                            outdict['available_color_labels'].append(colorlabel)


            # handle no dereddening
            else:

                outdict['dereddened'] = False
                outdict['extinction_%s' % mk] = 0.0
                outdict['dered_%s' % mk] = np.nan

                # get all the colors to generate for this bandpass
                for colorspec in our_bandpasses_colors[mk]['colors']:

                    # only add this if the color's not there already
                    if colorspec[0] not in outdict:

                        colorkey, colorlabel = colorspec

                        # look for the bands to make this color

                        # if it's not found now, this should work when we come
                        # around for the next bandpass for this color

                        band1, band2 = colorkey.split('-')

                        if (band1 in outdict and
                            band2 in outdict and
                            outdict[band1] is not None and
                            outdict[band2] is not None and
                            np.isfinite(outdict[band1]) and
                            np.isfinite(outdict[band2])):

                            outdict[colorkey] = (
                                outdict[band1] -
                                outdict[band2]
                            )
                            outdict['available_colors'].append(colorkey)
                            outdict['available_color_labels'].append(colorlabel)


        # if this bandpass was not found in the objectinfo dict, ignore it
        else:

            outdict[mk] = np.nan


    return outdict



def mdwarf_subtype_from_sdsscolor(ri_color, iz_color):
    '''This calculates the M-dwarf subtype given SDSS `r-i` and `i-z` colors.

    Parameters
    ----------

    ri_color : float
        The SDSS `r-i` color of the object.

    iz_color : float
        The SDSS `i-z` color of the object.

    Returns
    -------

    (subtype, index1, index2) : tuple
        `subtype`: if the star appears to be an M dwarf, will return an int
        between 0 and 9 indicating its subtype, e.g. will return 4 for an M4
        dwarf. If the object isn't an M dwarf, will return None

        `index1`, `index2`: the M-dwarf color locus value and spread of this
        object calculated from the `r-i` and `i-z` colors.

    '''

    # calculate the spectral type index and the spectral type spread of the
    # object. sti is calculated by fitting a line to the locus in r-i and i-z
    # space for M dwarfs in West+ 2007
    if np.isfinite(ri_color) and np.isfinite(iz_color):
        obj_sti = 0.875274*ri_color + 0.483628*(iz_color + 0.00438)
        obj_sts = -0.483628*ri_color + 0.875274*(iz_color + 0.00438)
    else:
        obj_sti = np.nan
        obj_sts = np.nan

    # possible M star if sti is >= 0.666 but <= 3.4559
    if (np.isfinite(obj_sti) and np.isfinite(obj_sts) and
        (obj_sti > 0.666) and (obj_sti < 3.4559)):

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
    '''This calculates rough star type classifications based on star colors
    in the ugrizJHK bands.

    Uses the output from `color_features` and `coord_features`. By default,
    `color_features` will use dereddened colors, as are expected by most
    relations here.

    Based on the color cuts from:

    - SDSS SEGUE (Yanny+ 2009)
    - SDSS QSO catalog (Schneider+ 2007)
    - SDSS RR Lyrae catalog (Sesar+ 2011)
    - SDSS M-dwarf catalog (West+ 2008)
    - Helmi+ 2003
    - Bochanski+ 2014

    Parameters
    ----------

    colorfeatures : dict
        This is the dict produced by the `color_features` function.

    pmfeatures : dict
        This is the dict produced by the `coord_features` function.

    Returns
    -------

    dict
        A dict containing all of the possible classes this object can belong to
        as a list in the `color_classes` key, and values of the various color
        indices used to arrive to that conclusion as the other keys.

    '''

    possible_classes = []

    if not colorfeatures:
        return possible_classes

    if not pmfeatures:
        return possible_classes

    # dered mags
    if ( ('dered_sdssu' in colorfeatures) and
         (colorfeatures['dered_sdssu'] is not None) and
         (np.isfinite(colorfeatures['dered_sdssu'])) ):
        u = colorfeatures['dered_sdssu']
    else:
        u = np.nan

    if ( ('dered_sdssg' in colorfeatures) and
         (colorfeatures['dered_sdssg'] is not None) and
         (np.isfinite(colorfeatures['dered_sdssg'])) ):
        g = colorfeatures['dered_sdssg']
    else:
        g = np.nan

    if ( ('dered_sdssr' in colorfeatures) and
         (colorfeatures['dered_sdssr'] is not None) and
         (np.isfinite(colorfeatures['dered_sdssr'])) ):
        r = colorfeatures['dered_sdssr']
    else:
        r = np.nan

    if ( ('dered_sdssi' in colorfeatures) and
         (colorfeatures['dered_sdssi'] is not None) and
         (np.isfinite(colorfeatures['dered_sdssi'])) ):
        i = colorfeatures['dered_sdssi']
    else:
        i = np.nan

    if ( ('dered_sdssz' in colorfeatures) and
         (colorfeatures['dered_sdssz'] is not None) and
         (np.isfinite(colorfeatures['dered_sdssz'])) ):
        z = colorfeatures['dered_sdssz']
    else:
        z = np.nan

    if ( ('dered_jmag' in colorfeatures) and
         (colorfeatures['dered_jmag'] is not None) and
         (np.isfinite(colorfeatures['dered_jmag'])) ):
        j = colorfeatures['dered_jmag']
    else:
        j = np.nan

    if ( ('dered_hmag' in colorfeatures) and
         (colorfeatures['dered_hmag'] is not None) and
         (np.isfinite(colorfeatures['dered_hmag'])) ):
        h = colorfeatures['dered_hmag']
    else:
        h = np.nan

    if ( ('dered_kmag' in colorfeatures) and
         (colorfeatures['dered_kmag'] is not None) and
         (np.isfinite(colorfeatures['dered_kmag'])) ):
        k = colorfeatures['dered_kmag']
    else:
        k = np.nan


    # measured mags
    if 'sdssu' in colorfeatures and colorfeatures['sdssu'] is not None:
        um = colorfeatures['sdssu']
    else:
        um = np.nan
    if 'sdssg' in colorfeatures and colorfeatures['sdssg'] is not None:
        gm = colorfeatures['sdssg']
    else:
        gm = np.nan
    if 'sdssr' in colorfeatures and colorfeatures['sdssr'] is not None:
        rm = colorfeatures['sdssr']
    else:
        rm = np.nan
    if 'sdssi' in colorfeatures and colorfeatures['sdssi'] is not None:
        im = colorfeatures['sdssi']
    else:
        im = np.nan
    if 'sdssz' in colorfeatures and colorfeatures['sdssz'] is not None:
        zm = colorfeatures['sdssz']
    else:
        zm = np.nan
    if 'jmag' in colorfeatures and colorfeatures['jmag'] is not None:
        jm = colorfeatures['jmag']
    else:
        jm = np.nan
    if 'hmag' in colorfeatures and colorfeatures['hmag'] is not None:
        hm = colorfeatures['hmag']
    else:
        hm = np.nan
    if 'kmag' in colorfeatures and colorfeatures['kmag'] is not None:
        km = colorfeatures['kmag']
    else:
        km = np.nan


    # reduced proper motion
    rpmj = pmfeatures['rpmj'] if np.isfinite(pmfeatures['rpmj']) else None

    # now generate the various color indices
    # color-gravity index
    if (np.isfinite(u) and np.isfinite(g) and
        np.isfinite(r) and np.isfinite(i) and
        np.isfinite(z)):
        v_color = 0.283*(u-g)-0.354*(g-r)+0.455*(r-i)+0.766*(i-z)
    else:
        v_color = np.nan

    # metallicity index p1
    if (np.isfinite(u) and np.isfinite(g) and np.isfinite(r)):
        p1_color = 0.91*(u-g)+0.415*(g-r)-1.28
    else:
        p1_color = np.nan

    # metallicity index l
    if (np.isfinite(u) and np.isfinite(g) and
        np.isfinite(r) and np.isfinite(i)):
        l_color = -0.436*u + 1.129*g - 0.119*r - 0.574*i + 0.1984
    else:
        l_color = np.nan

    # metallicity index s
    if (np.isfinite(u) and np.isfinite(g) and np.isfinite(r)):
        s_color = -0.249*u + 0.794*g - 0.555*r + 0.124
    else:
        s_color = np.nan

    # RR Lyrae ug and gr indexes
    if (np.isfinite(u) and np.isfinite(g) and np.isfinite(r)):
        d_ug = (u-g) + 0.67*(g-r) - 1.07
        d_gr = 0.45*(u-g) - (g-r) - 0.12
    else:
        d_ug, d_gr = np.nan, np.nan


    # check the M subtype
    m_subtype, m_sti, m_sts = mdwarf_subtype_from_sdsscolor(r-i, i-z)

    # now check if this is a likely M dwarf
    if m_subtype and rpmj and rpmj > 1.0:
        possible_classes.append('d' + m_subtype)

    # white dwarf
    if ( np.isfinite(u) and np.isfinite(g) and np.isfinite(r) and
         ((g-r) < -0.2) and ((g-r) > -1.0) and
         ((u-g) < 0.7) and ((u-g) > -1) and
         ((u-g+2*(g-r)) < -0.1) ):
        possible_classes.append('WD/sdO/sdB')

    # A/BHB/BStrg
    if ( np.isfinite(u) and np.isfinite(g) and np.isfinite(r) and
         ((u-g) < 1.5) and ((u-g) > 0.8) and
         ((g-r) < 0.2) and ((g-r) > -0.5) ):
        possible_classes.append('A/BHB/blustrg')

    # F turnoff/sub-dwarf
    if ( (np.isfinite(p1_color) and np.isfinite(p1_color) and
          np.isfinite(u) and np.isfinite(g) and np.isfinite(r) ) and
         (p1_color < -0.25) and (p1_color > -0.7) and
         ((u-g) < 1.4) and ((u-g) > 0.4) and
         ((g-r) < 0.7) and ((g-r) > 0.2) ):
        possible_classes.append('Fturnoff/sdF')

    # low metallicity
    if ( (np.isfinite(u) and np.isfinite(g) and np.isfinite(r) and
          np.isfinite(l_color)) and
         ((g-r) < 0.75) and ((g-r) > -0.5) and
         ((u-g) < 3.0) and ((u-g) > 0.6) and
         (l_color > 0.135) ):
        possible_classes.append('lowmetal')

    # low metallicity giants from Helmi+ 2003
    if ( (np.isfinite(p1_color) and np.isfinite(s_color)) and
         (-0.1 < p1_color < 0.6) and (s_color > 0.05) ):
        possible_classes.append('lowmetalgiant')

    # F/G star
    if ( (np.isfinite(g) and np.isfinite(g) and np.isfinite(r)) and
         ((g-r) < 0.48) and ((g-r) > 0.2) ):
        possible_classes.append('F/G')

    # G dwarf
    if ( (np.isfinite(g) and np.isfinite(r)) and
         ((g-r) < 0.55) and ((g-r) > 0.48) ):
        possible_classes.append('dG')

    # K giant
    if ( (np.isfinite(u) and np.isfinite(g) and
          np.isfinite(r) and np.isfinite(i) and
          np.isfinite(l_color)) and
         ((g-r) > 0.35) and ((g-r) < 0.7) and
         (l_color > 0.07) and ((u-g) > 0.7) and ((u-g) < 4.0) and
         ((r-i) > 0.15) and ((r-i) < 0.6) ):
        possible_classes.append('gK')

    # AGB
    if ( (np.isfinite(u) and np.isfinite(g) and
          np.isfinite(r) and np.isfinite(s_color)) and
         ((u-g) < 3.5) and ((u-g) > 2.5) and
         ((g-r) < 1.3) and ((g-r) > 0.9) and
         (s_color < -0.06) ):
        possible_classes.append('AGB')

    # K dwarf
    if ( (np.isfinite(g) and np.isfinite(r)) and
         ((g-r) < 0.75) and ((g-r) > 0.55) ):
        possible_classes.append('dK')

    # M subdwarf
    if ( (np.isfinite(g) and np.isfinite(r) and np.isfinite(i)) and
         ((g-r) > 1.6) and ((r-i) < 1.3) and ((r-i) > 0.95) ):
        possible_classes.append('sdM')

    # M giant colors from Bochanski+ 2014
    if ( (np.isfinite(j) and np.isfinite(h) and np.isfinite(k) and
          np.isfinite(g) and np.isfinite(i)) and
         ((j-k) > 1.02) and
         ((j-h) < (0.561*(j-k) + 0.46)) and
         ((j-h) > (0.561*(j-k) + 0.14)) and
         ((g-i) > (0.932*(i-k) - 0.872)) ):
        possible_classes.append('gM')

    # MS+WD pair
    if ( (np.isfinite(um) and np.isfinite(gm) and
          np.isfinite(rm) and np.isfinite(im)) and
         ((um-gm) < 2.25) and ((gm-rm) > -0.2) and
         ((gm-rm) < 1.2) and ((rm-im) > 0.5) and
         ((rm-im) < 2.0) and
         ((gm-rm) > (-19.78*(rm-im)+11.13)) and
         ((gm-rm) < (0.95*(rm-im)+0.5)) ):
        possible_classes.append('MSWD')

    # brown dwarf
    if ( (np.isfinite(um) and np.isfinite(gm) and np.isfinite(rm) and
          np.isfinite(im) and np.isfinite(zm)) and
         (zm < 19.5) and (um > 21.0) and (gm > 22.0) and
         (rm > 21.0) and ((im - zm) > 1.7) ):
        possible_classes.append('BD')

    # RR Lyrae candidate
    if ( (np.isfinite(u) and np.isfinite(g) and np.isfinite(r) and
          np.isfinite(i) and np.isfinite(z) and np.isfinite(d_ug) and
          np.isfinite(d_gr)) and
         ((u-g) > 0.98) and ((u-g) < 1.3) and
         (d_ug > -0.05) and (d_ug < 0.35) and
         (d_gr > 0.06) and (d_gr < 0.55) and
         ((r-i) > -0.15) and ((r-i) < 0.22) and
         ((i-z) > -0.21) and ((i-z) < 0.25) ):
        possible_classes.append('RRL')

    # QSO color
    if ( (np.isfinite(u) and np.isfinite(g) and np.isfinite(r)) and
         ( (((u-g) > -0.1) and ((u-g) < 0.7) and
            ((g-r) > -0.3) and ((g-r) < 0.5)) or
           ((u-g) > (1.6*(g-r) + 1.34)) ) ):
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
                           gaia_matchdist_arcsec=3.0,
                           verbose=True,
                           gaia_submit_timeout=10.0,
                           gaia_submit_tries=3,
                           gaia_max_timeout=180.0,
                           gaia_mirror=None,
                           complete_query_later=True,
                           search_simbad=False):
    '''Gets several neighbor, GAIA, and SIMBAD features:

    From the KD-Tree in the given light curve catalog the object is in:
    `lclist_kdtree`:

    - distance to closest neighbor in arcsec
    - total number of all neighbors within 2 x `neighbor_radius_arcsec`

    From the GAIA DR2 catalog:

    - distance to closest neighbor in arcsec
    - total number of all neighbors within 2 x `neighbor_radius_arcsec`
    - gets the parallax for the object and neighbors
    - calculates the absolute GAIA mag and `G-K` color for use in CMDs
    - gets the proper motion in RA/Dec if available

    From the SIMBAD catalog:

    - the name of the object
    - the type of the object

    Parameters
    ----------

    objectinfo : dict
        This is the objectinfo dict from an object's light curve. This must
        contain at least the following keys::

            {'ra': the right ascension of the object,
             'decl': the declination of the object}

    lclist_kdtree : scipy.spatial.cKDTree object
        This is a KD-Tree built on the Cartesian xyz coordinates from (ra, dec)
        of all objects in the same field as this object. It is similar to that
        produced by :py:func:`astrobase.lcproc.catalogs.make_lclist`, and is
        used to carry out the spatial search required to find neighbors for this
        object.

    neighbor_radius_arcsec : float
        The maximum radius in arcseconds around this object to search for
        neighbors in both the light curve catalog and in the GAIA DR2 catalog.

    gaia_matchdist_arcsec : float
        The maximum distance in arcseconds to use for a GAIA cross-match to this
        object.

    verbose : bool
        If True, indicates progress and warns of problems.

    gaia_submit_timeout : float
        Sets the timeout in seconds to use when submitting a request to look up
        the object's information to the GAIA service. Note that if `fast_mode`
        is set, this is ignored.

    gaia_submit_tries : int
        Sets the maximum number of times the GAIA services will be contacted to
        obtain this object's information. If `fast_mode` is set, this is
        ignored, and the services will be contacted only once (meaning that a
        failure to respond will be silently ignored and no GAIA data will be
        added to the checkplot's objectinfo dict).

    gaia_max_timeout : float
        Sets the timeout in seconds to use when waiting for the GAIA service to
        respond to our request for the object's information. Note that if
        `fast_mode` is set, this is ignored.

    gaia_mirror : str
        This sets the GAIA mirror to use. This is a key in the
        `services.gaia.GAIA_URLS` dict which defines the URLs to hit for each
        mirror.

    search_simbad : bool
        If this is True, searches for objects in SIMBAD at this object's
        location and gets the object's SIMBAD main ID, type, and stellar
        classification if available.

    Returns
    -------

    dict
        Returns a dict with neighbor, GAIA, and SIMBAD features.

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

        gaia_result = gaia.objectlist_conesearch(
            objectinfo['ra'],
            objectinfo['decl'],
            neighbor_radius_arcsec,
            verbose=verbose,
            timeout=gaia_submit_timeout,
            maxtimeout=gaia_max_timeout,
            maxtries=gaia_submit_tries,
            gaia_mirror=gaia_mirror,
            complete_query_later=complete_query_later
        )

        if gaia_result:

            gaia_objlistf = gaia_result['result']

            with gzip.open(gaia_objlistf,'rb') as infd:

                try:
                    gaia_objlist = np.genfromtxt(
                        infd,
                        names=True,
                        delimiter=',',
                        dtype='U20,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8',
                        usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12)
                    )
                except Exception as e:
                    gaia_objlist = []

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
                if gaia_objlist['dist_arcsec'][0] < gaia_matchdist_arcsec:

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
                        gaia_pmra = gaia_objlist['pmra']
                        gaia_pmra_err = gaia_objlist['pmra_error']
                        gaia_pmdecl = gaia_objlist['pmdec']
                        gaia_pmdecl_err = gaia_objlist['pmdec_error']

                        gaia_absolute_mags = magnitudes.absolute_gaia_magnitude(
                            gaia_mags, gaia_parallaxes
                        )
                        if ('kmag' in objectinfo and
                            objectinfo['kmag'] is not None and
                            np.isfinite(objectinfo['kmag'])):
                            gaiak_colors = gaia_mags - objectinfo['kmag']
                        else:
                            gaiak_colors = None

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
                        gaia_pmra = gaia_objlist['pmra']
                        gaia_pmra_err = gaia_objlist['pmra_error']
                        gaia_pmdecl = gaia_objlist['pmdec']
                        gaia_pmdecl_err = gaia_objlist['pmdec_error']

                        gaia_absolute_mags = magnitudes.absolute_gaia_magnitude(
                            gaia_mags, gaia_parallaxes
                        )
                        if ('kmag' in objectinfo and
                            objectinfo['kmag'] is not None and
                            np.isfinite(objectinfo['kmag'])):
                            gaiak_colors = gaia_mags - objectinfo['kmag']
                        else:
                            gaiak_colors = None

                        gaia_dists = gaia_objlist['dist_arcsec']
                        gaia_closest_distarcsec = np.nan
                        gaia_closest_gmagdiff = np.nan


                # otherwise, the object wasn't found in GAIA for some reason
                else:

                    LOGWARNING('no GAIA objects found within '
                               '%.3f arcsec of object position (%.3f, %.3f), '
                               'closest object is at %.3f arcsec away' %
                               (gaia_matchdist_arcsec,
                                objectinfo['ra'], objectinfo['decl'],
                                gaia_objlist['dist_arcsec'][0]))

                    gaia_status = ('failed: no object within %.3f '
                                   'arcsec, closest = %.3f arcsec' %
                                   (gaia_matchdist_arcsec,
                                    gaia_objlist['dist_arcsec'][0]))

                    gaia_nneighbors = np.nan

                    gaia_ids = gaia_objlist['source_id']
                    gaia_mags = gaia_objlist['phot_g_mean_mag']
                    gaia_parallaxes = gaia_objlist['parallax']
                    gaia_parallax_errs = gaia_objlist['parallax_error']
                    gaia_pmra = gaia_objlist['pmra']
                    gaia_pmra_err = gaia_objlist['pmra_error']
                    gaia_pmdecl = gaia_objlist['pmdec']
                    gaia_pmdecl_err = gaia_objlist['pmdec_error']

                    gaia_absolute_mags = magnitudes.absolute_gaia_magnitude(
                        gaia_mags, gaia_parallaxes
                    )
                    if ('kmag' in objectinfo and
                        objectinfo['kmag'] is not None and
                        np.isfinite(objectinfo['kmag'])):
                        gaiak_colors = gaia_mags - objectinfo['kmag']
                    else:
                        gaiak_colors = None

                    gaia_dists = gaia_objlist['dist_arcsec']
                    gaia_closest_distarcsec = np.nan
                    gaia_closest_gmagdiff = np.nan

            # if there are no neighbors within neighbor_radius_arcsec
            # or this object is not covered by GAIA. return nothing
            else:

                LOGERROR('no GAIA objects at this '
                         'position or GAIA query failed')

                gaia_status = (
                    'failed: no GAIA objects at this '
                    'position or GAIA query failed.'
                )
                gaia_nneighbors = np.nan
                gaia_ids = None
                gaia_mags = None

                gaia_xypos = None
                gaia_parallaxes = None
                gaia_parallax_errs = None
                gaia_pmra = None
                gaia_pmra_err = None
                gaia_pmdecl = None
                gaia_pmdecl_err = None
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
                 'gaia_pmras':gaia_pmra,
                 'gaia_pmra_errs':gaia_pmra_err,
                 'gaia_pmdecls':gaia_pmdecl,
                 'gaia_pmdecl_errs':gaia_pmdecl_err,
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
                 'gaia_pmras':None,
                 'gaia_pmra_errs':None,
                 'gaia_pmdecls':None,
                 'gaia_pmdecl_errs':None,
                 'gaia_absolute_mags':None,
                 'gaiak_colors':None,
                 'gaia_dists':None,
                 'gaia_closest_distarcsec':np.nan,
                 'gaia_closest_gmagdiff':np.nan}
            )


    else:

        LOGERROR("one or more of the 'ra', 'decl' keys "
                 "are missing from the objectinfo dict, "
                 "can't get GAIA or LC collection neighbor features")

        resultdict.update(
            {'gaia_status':'failed: no ra/decl for object',
             'gaia_neighbors':np.nan,
             'gaia_ids':None,
             'gaia_xypos':None,
             'gaia_mags':None,
             'gaia_parallaxes':None,
             'gaia_parallax_errs':None,
             'gaia_pmras':None,
             'gaia_pmra_errs':None,
             'gaia_pmdecls':None,
             'gaia_pmdecl_errs':None,
             'gaia_absolute_mags':None,
             'gaiak_colors':None,
             'gaia_dists':None,
             'gaia_closest_distarcsec':np.nan,
             'gaia_closest_gmagdiff':np.nan}
        )


    # finally, search for this object in SIMBAD
    if ('ra' in objectinfo and 'decl' in objectinfo and
        objectinfo['ra'] is not None and objectinfo['decl'] is not None and
        search_simbad):

        simbad_result = simbad.objectnames_conesearch(
            objectinfo['ra'],
            objectinfo['decl'],
            neighbor_radius_arcsec,
            verbose=verbose,
            timeout=gaia_submit_timeout,
            maxtimeout=gaia_max_timeout,
            maxtries=gaia_submit_tries,
            complete_query_later=complete_query_later
        )

    else:

        simbad_result = None

    if (simbad_result and
        simbad_result['result'] and
        os.path.exists(simbad_result['result'])):

        with gzip.open(simbad_result['result'],'rb') as infd:

            try:
                simbad_objectnames = np.genfromtxt(
                    infd,
                    names=True,
                    delimiter=',',
                    dtype='U20,f8,f8,U20,U20,U20,i8,U600,f8',
                    usecols=(0,1,2,3,4,5,6,7,8),
                    comments='?',
                )
            except Exception as e:
                simbad_objectnames = []

            simbad_objectnames = np.atleast_1d(simbad_objectnames)

            if simbad_objectnames.size > 0:

                simbad_mainid = simbad_objectnames['main_id'].tolist()
                simbad_allids = simbad_objectnames['all_ids'].tolist()
                simbad_objtype = simbad_objectnames['otype_txt'].tolist()
                simbad_distarcsec = simbad_objectnames['dist_arcsec'].tolist()
                simbad_nmatches = len(simbad_mainid)

                simbad_mainid = [x.replace('"','') for x in simbad_mainid]
                simbad_allids = [x.replace('"','') for x in simbad_allids]
                simbad_objtype = [x.replace('"','') for x in simbad_objtype]


                resultdict.update({
                    'simbad_nmatches':simbad_nmatches,
                    'simbad_mainid':simbad_mainid,
                    'simbad_objtype':simbad_objtype,
                    'simbad_allids':simbad_allids,
                    'simbad_distarcsec':simbad_distarcsec
                })


                if simbad_nmatches > 1:

                    resultdict['simbad_status'] = (
                        'ok: multiple SIMBAD matches found'
                    )

                else:

                    resultdict['simbad_status'] = 'ok: single SIMBAD match'



                # get the closest match
                if simbad_distarcsec[0] < gaia_matchdist_arcsec:

                    resultdict.update({
                        'simbad_best_mainid':simbad_mainid[0],
                        'simbad_best_objtype':simbad_objtype[0],
                        'simbad_best_allids':simbad_allids[0],
                        'simbad_best_distarcsec':simbad_distarcsec[0],
                        'simbad_status':'ok: object found within match radius'
                    })

                else:

                    LOGWARNING('no SIMBAD objects found within '
                               '%.3f arcsec of object position (%.3f, %.3f), '
                               'closest object: %s at %.3f arcsec away' %
                               (gaia_matchdist_arcsec,
                                objectinfo['ra'],
                                objectinfo['decl'],
                                simbad_mainid[0],
                                simbad_distarcsec[0]))

                    simbad_status = ('failed: no object within %.3f '
                                     'arcsec, closest = %.3f arcsec' %
                                     (gaia_matchdist_arcsec,
                                      simbad_distarcsec[0]))


                    resultdict.update({
                        'simbad_best_mainid':None,
                        'simbad_best_objtype':None,
                        'simbad_best_allids':None,
                        'simbad_best_distarcsec':None,
                        'simbad_status':simbad_status
                    })


            else:

                resultdict.update({
                    'simbad_status':'failed: no SIMBAD matches found',
                    'simbad_nmatches':None,
                    'simbad_mainid':None,
                    'simbad_objtype':None,
                    'simbad_allids':None,
                    'simbad_distarcsec':None,
                    'simbad_best_mainid':None,
                    'simbad_best_objtype':None,
                    'simbad_best_allids':None,
                    'simbad_best_distarcsec':None,
                })

    else:

        if search_simbad:
            simbad_status = 'failed: SIMBAD query failed'
        else:
            simbad_status = 'failed: SIMBAD query not tried'

        resultdict.update({
            'simbad_status':simbad_status,
            'simbad_nmatches':None,
            'simbad_mainid':None,
            'simbad_objtype':None,
            'simbad_allids':None,
            'simbad_distarcsec':None,
            'simbad_best_mainid':None,
            'simbad_best_objtype':None,
            'simbad_best_allids':None,
            'simbad_best_distarcsec':None,
        })

    return resultdict
