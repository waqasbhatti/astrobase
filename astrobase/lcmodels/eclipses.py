#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''eclipses.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

This contains a double gaussian model for first order modeling of eclipsing
binaries.

'''

import numpy as np

from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, all as npall, \
    correlate as npcorrelate, nonzero as npnonzero, diag as npdiag


##################################
## MODEL AND RESIDUAL FUNCTIONS ##
##################################

def _gaussian(x, amp, loc, std):
    '''
    This is a simple gaussian.

    '''

    return amp * np.exp(-((x - loc)*(x - loc))/(2.0*std*std))



def _double_inverted_gaussian(x,
                             amp1, loc1, std1,
                             amp2, loc2, std2):
    '''
    This is a double inverted gaussian.

    '''

    gaussian1 = -_gaussian(x,amp1,loc1,stdev1)
    gaussian2 = -_gaussian(x,amp2,loc2,stdev2)
    return gaussian1 + gaussian2



def invgauss_eclipses_func(ebparams, times, mags, errs):
    '''This returns a double eclipse shaped function.

    Suitable for first order modeling of eclipsing binaries.

    FIXME: maybe convert prim/sec to ratios
    FIXME: we need a detached vs. contact parameter

    transitparams = [period (time),
                     epoch (time),
                     eccentricity,
                     primdepth (mags or flux),
                     primduration (time),
                     secdepth (mags or flux),
                     secduration (phase)]

    All of these will then have fitted values after the fit is done.

    for magnitudes -> transitdepth should be < 0
    for fluxes     -> transitdepth should be > 0

    TODO: finish this up

    '''



def invgauss_eclipses_residual(ebparams, times, mags, errs):
    '''
    This returns the residual between the modelmags and the actual mags.

    '''

    modelmags, phase, ptimes, pmags, perrs = (
        invgauss_eclipses_func(ebparams, times, mags, errs)
    )

    # this is now a weighted residual taking into account the measurement err
    return (pmags - modelmags)/perrs
