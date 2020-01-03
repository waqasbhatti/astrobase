#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lcmodels - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
# License: MIT. See the LICENSE file for more details.

'''This contains various light curve models for variable stars. Useful for
first order fits to distinguish between variable types, and for generating these
variables' light curves for a recovery simulation.

- :py:mod:`astrobase.lcmodels.transits`: trapezoid-shaped planetary transit
  light curves.
- :py:mod:`astrobase.lcmodels.eclipses`: double inverted-gaussian shaped
  eclipsing binary light curves.
- :py:mod:`astrobase.lcmodels.flares`: stellar flare model from Pitkin+ 2014.
- :py:mod:`astrobase.lcmodels.sinusoidal`: sinusoidal light curve generation for
  pulsating variables.

'''
