#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''recoverysim - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This generates light curves of variable stars using the astrobase.lcmodels
package, adds noise and observation sampling to them based on given parameters
(or example light curves) and then runs them through variable star detection and
classification to see how well they are recovered.

TODO: random notes below for implementation

use realistic timebase, mag distribution, noise distribution and generate
variable LCs (also non-variable LCs using same distributions for false positive
rate).

generate periodic and non periodic vars with given period and amplitude
distributions:

- planets with trapezoid LC

- EBs with double inverted gaussian

- pulsators with Fourier coefficients

- flares with flare model

calculate the various non-periodic variability indices for these sim LCs and
test recall and precision. get the PCA PC1 of this for a combined var index.

tune for a false positive and false negative rate using the ROC curve and set
the sigma limit for confirmed variability per magnitude bin.

afterwards check if successful by cross validation.

run period searches and see recovery rate by period, amplitude, magnitude,
number of observations, etc.

'''
