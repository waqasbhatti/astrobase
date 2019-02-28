#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lcfit.py
# Waqas Bhatti and Luke Bouma - Feb 2017
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''Fitting routines for light curves.

..deprecated:: 0.3.20
  This module has been broken out into its own subpackage at the top level of
  astrobase: :py:mod:`astrobase.lcfit`. This module is now just a wrapper around
  that subpackage. It will be removed in astrobase 0.4.5.

Includes the following functions:

- :py:func:`astrobase.lcfit.sinusoidal.fourier_fit_magseries`: fit an arbitrary
  order Fourier series to a magnitude/flux time series.

- :py:func:`astrobase.lcfit.nonphysical.spline_fit_magseries`: fit a univariate
  cubic spline to a magnitude/flux time series with a specified spline knot
  fraction.

- :py:func:`astrobase.lcfit.nonphysical.savgol_fit_magseries`: apply a
  Savitzky-Golay smoothing filter to a magnitude/flux time series, returning the
  resulting smoothed function as a "fit".

- :py:func:`astrobase.lcfit.nonphysical.legendre_fit_magseries`: fit a Legendre
  function of the specified order to the magnitude/flux time series.

- :py:func:`astrobase.lcfit.eclipses.gaussianeb_fit_magseries`: fit a double
  inverted gaussian eclipsing binary model to the magnitude/flux time series

- :py:func:`astrobase.lcfit.transits.traptransit_fit_magseries`: fit a
  trapezoid-shaped transit signal to the magnitude/flux time series

- :py:func:`astrobase.lcfit.transits.mandelagol_fit_magseries`: fit a Mandel &
  Agol (2002) planet transit model to the flux time series.

- :py:func:`astrobase.lcfit.transits.mandelagol_and_line_fit_magseries`: fit a
  Mandel & Agol 2002 model, + a local line to the flux time series.

'''

#########################
## DEPRECATION WARNING ##
#########################

import warnings
warnings.warn(
    "This module has been broken out into its own subpackage at the "
    "top level of astrobase: astrobase.lcfit. This module is now just a "
    "wrapper around that subpackage. It will be removed in astrobase 0.4.5.",
    FutureWarning
)

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


#####################################
## HOIST THE FIT FUNCTIONS UP HERE ##
#####################################

from astrobase.lcfit.sinusoidal import fourier_fit_magseries
from astrobase.lcfit.nonphysical import (
    spline_fit_magseries,
    savgol_fit_magseries,
    legendre_fit_magseries
)

from astrobase.lcfit.eclipses import gaussianeb_fit_magseries

try:
    import batman
    import emcee
    import corner

    if int(emcee.__version__[0]) >= 3:
        from astrobase.lcfit.transits import (
            mandelagol_fit_magseries,
            mandelagol_and_line_fit_magseries,
            traptransit_fit_magseries
        )
    else:
        from astrobase.lcfit.transits import (
            traptransit_fit_magseries
        )

except Exception as e:
        from astrobase.lcfit.transits import (
            traptransit_fit_magseries
        )
