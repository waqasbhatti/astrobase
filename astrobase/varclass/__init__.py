#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# varclass - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
# License: MIT. See the LICENSE file for more details.

'''This contains various modules that obtain features to use in variable star
classification.

- :py:mod:`astrobase.varclass.starfeatures`: features related to color, proper
  motion, neighbor proximity, cross-matches against GAIA and SIMBAD, etc.

- :py:mod:`astrobase.varclass.varfeatures`: non-periodic light curve variability
  features

- :py:mod:`astrobase.varclass.periodicfeatures`: light curve features for phased
  light curves

- :py:mod:`astrobase.varclass.rfclass`: random forest classifier and support
  functions for variability classification

'''
