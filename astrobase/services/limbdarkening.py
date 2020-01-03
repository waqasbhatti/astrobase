#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# limbdarkening.py - Luke Bouma (luke@astro.princeton.edu) - Aug 2019
# License: MIT - see the LICENSE file for the full text.

'''
Utilities to get stellar limb darkening coefficients for use during transit
fitting.
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

import numpy as np

from astropy import units as units, constants as const
try:
    from astroquery.vizier import Vizier
    vizier_dependency = True
except Exception:
    vizier_dependency = False


def get_tess_limb_darkening_guesses(teff, logg):
    '''
    Given Teff and log(g), query the Claret+2017 limb darkening coefficient
    grid. Return the nearest match.

    TODO: interpolate instead of doing the nearest match. Nearest match is
    good to maybe only ~200 K and ~0.3 in log(g).

    Parameters
    ----------

    teff : float
        The stellar effective temperature to use.

    logg : float
        The stellar log g value to use.

    Returns
    -------

    (linear_coeff, quadratic_coeff) : tuple
        Returns a tuple containing the linear and quadratic limb-darkening
        coefficients for the given effective temperature and log g.

    '''

    if not vizier_dependency:
        raise ImportError(
            'This function requires astroquery.'
            'Try: `pip [or conda -c] install astropy astroquery`'
        )

    # Get the Claret quadratic priors for TESS bandpass.  The table below is
    # good from Teff = 1500 - 12000K, logg = 2.5 to 6. We choose values
    # computed with the "r method", see
    # http://vizier.u-strasbg.fr/viz-bin/VizieR-n?-source=METAnot&catid=36000030&notid=1&-out=text
    if not 2300 < teff < 12000:
        if teff < 15000:
            LOGWARNING('using 12000K atmosphere LD coeffs even tho teff={}'.
                       format(teff))
        else:
            LOGERROR('got teff error')
    if not 2.5 < logg < 6:
        if teff < 15000:
            # Rough guess: assume star is B6V, Pecaut & Mamajek (2013) table.
            _Mstar = 4*units.Msun
            _Rstar = 2.9*units.Rsun
            logg = np.log10((const.G * _Mstar / _Rstar**2).cgs.value)
        else:
            LOGERROR('got logg error')

    Vizier.ROW_LIMIT = -1
    catalog_list = Vizier.find_catalogs('J/A+A/600/A30')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    t = catalogs[1]
    sel = t['Type'] == 'r'
    df = t[sel]

    # Each Teff has 8 tabulated logg values. First, find the best teff match.
    best_teff_match_inds = (np.abs(df['Teff'] - teff)).argsort()[:8]
    teff_matching_rows = df[best_teff_match_inds]

    # Then, among those best 8, get the best logg match.
    best_logg_match_ind = (
        np.abs(teff_matching_rows['logg'] - logg)
    ).argsort()[0]
    teff_logg_matching_row = teff_matching_rows[best_logg_match_ind]

    # TODO: should probably determine these coefficients by INTERPOLATING.
    # (especially in cases when you're FIXING them, rather than letting them
    # float).
    LOGWARNING('skipping interpolation for Claret coefficients.')
    LOGWARNING('data logg={:.3f}, teff={:.1f}'.format(logg, teff))
    LOGWARNING('Claret logg={:.3f}, teff={:.1f}'.
               format(teff_logg_matching_row['logg'],
                      teff_logg_matching_row['Teff']))

    u_linear = teff_logg_matching_row['aLSM']
    u_quad = teff_logg_matching_row['bLSM']

    return float(u_linear), float(u_quad)
