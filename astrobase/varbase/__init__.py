#!/usr/bin/env python

'''varbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2016

Contains functions to deal with light curve variability, fitting functions,
masking signals, calculating nonperiodic and periodic features, autocorrelation,
etc.

CURRENT SUBMODULES:

varbase.autocorr: calculating the autocorrelation function of light curves
varbase.features: calculating nonperiodic and periodic variability features
varbase.lcfit: fitting Fourier series and splines to light curves
varbase.signals: masking periodic signals, pre-whitening light curves

TO BE IMPLEMENTED:

varbase.checkplot: adding variability info to checkplot pickles


'''
