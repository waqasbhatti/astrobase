#!/usr/bin/env python
# -*- coding: utf-8 -*-
# periodbase - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''Contains various useful tools for period finding.

- :py:mod:`astrobase.periodbase.spdm`: Stellingwerf (1978) phase-dispersion
  minimization period search algorithm.

- :py:mod:`astrobase.periodbase.saov`: Schwarzenberg-Czerny (1989) analysis of
  variance period search algorithm.

- :py:mod:`astrobase.periodbase.smav`: Schwarzenberg-Czerny (1996)
  multi-harmonic AoV period search algorithm.

- :py:mod:`astrobase.periodbase.zgls`: Zechmeister & Kurster (2009) generalized
  Lomb-Scargle period search algorithm.

- :py:mod:`astrobase.periodbase.kbls`: Kovacs et al. (2002) Box-Least-Squares
  search using a wrapped `eebls.f` from G. Kovacs.

- :py:mod:`astrobase.periodbase.abls`: -> Kovacs et al. (2002) BLS using
  Astropy's implementation.

- :py:mod:`astrobase.periodbase.htls`: Hippke & Heller (2019)
  Transit-Least-Squares period search algorithm.

- :py:mod:`astrobase.periodbase.macf`: -> McQuillan et al. (2013a, 2014) ACF
  period search algorithm.

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


####################################################
## HOIST THE FINDER FUNCTIONS INTO THIS NAMESPACE ##
####################################################

from .zgls import pgen_lsp, specwindow_lsp
from .spdm import stellingwerf_pdm
from .saov import aov_periodfind
from .smav import aovhm_periodfind
from .macf import macf_period_find
from .kbls import bls_serial_pfind, bls_parallel_pfind

try:
    from .tls import tls_parallel_pfind
    HAVE_TLS = True
except Exception as e:
    HAVE_TLS = False

# used to figure out which function to run for bootstrap resampling
LSPMETHODS = {
    'bls':bls_parallel_pfind,
    'gls':pgen_lsp,
    'aov':aov_periodfind,
    'mav':aovhm_periodfind,
    'pdm':stellingwerf_pdm,
    'acf':macf_period_find,
    'win':specwindow_lsp
}
if HAVE_TLS:
    LSPMETHODS['tls'] = tls_parallel_pfind


# check if we have the astropy implementation of BLS available
import astropy
apversion = astropy.__version__
apversion = apversion.split('.')
apversion = [int(x) for x in apversion]

if len(apversion) == 2:
    apversion.append(0)

apversion = tuple(apversion)

if apversion >= (3,1,0):

    LOGINFO('An Astropy implementation of BLS is '
            'available because Astropy >= 3.1.')
    LOGINFO('If you want to use it as the default periodbase BLS runner, '
            'call the periodbase.use_astropy_bls() function.')

    def use_astropy_bls():
        '''This function can be used to switch from the default astrobase BLS
        implementation (kbls) to the Astropy version (abls).

        If this is called, subsequent calls to the BLS periodbase functions will
        use the Astropy versions instead::

            from astrobase import periodbase

            # initially points to periodbase.kbls.bls_serial_pfind
            periodbase.bls_serial_pfind(...)

            # initially points to periodbase.kbls.bls_parallel_pfind
            periodbase.bls_parallel_pfind(...)

            periodbase.use_astropy_bls()

            # now points to periodbase.abls.bls_serial_pfind
            periodbase.bls_serial_pfind(...)

            # now points to periodbase.abls.bls_parallel_pfind
            periodbase.bls_parallel_pfind(...)

        '''
        from .abls import bls_serial_pfind, bls_parallel_pfind
        globals()['bls_serial_pfind'] = bls_serial_pfind
        globals()['bls_parallel_pfind'] = bls_parallel_pfind
        globals()['LSPMETHODS']['bls'] = bls_parallel_pfind
