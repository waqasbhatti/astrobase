#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''lcmodels - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This contains various light curve models for variable stars. Useful for first
order fits to distinguish between variable types, and for generating these
variables' light curves for a recovery simulation.

transits.py - trapezoid-shaped planetary transit light curves
eclipses.py - double inverted-gaussian shaped eclipsing binary light curves
flares.py   - stellar flare model from Pitkin+ 2014
sinusoidal.py  - sinusoidal light curve generation for pulsating variables

'''
