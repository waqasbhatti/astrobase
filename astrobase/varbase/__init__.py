#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# varbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2016

'''Contains functions to deal with light curve variability, fitting functions,
masking signals, autocorrelation, etc.

- :py:mod:`astrobase.varbase.autocorr`: calculating the autocorrelation function
  of light curves.
- :py:mod:`astrobase.varbase.signals`: masking periodic signals, pre-whitening
  light curves.
- :py:mod:`astrobase.varbase.transits`: light curve tools specifically for
  planetary transits.
- :py:mod:`astrobase.varbase.trends`: tools for running External Parameter
  Decorrelation (EPD) on light curves.

FIXME: finish up the :py:mod:`astrobase.varbase.flares` module to find flares in
LCs.

'''

# there's nothing else here in this top-level module
