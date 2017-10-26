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

    gaussian1 = -_gaussian(x,amp1,loc1,std1)
    gaussian2 = -_gaussian(x,amp2,loc2,std2)
    return gaussian1 + gaussian2



def invgauss_eclipses_func(ebparams, times, mags, errs):
    '''This returns a double eclipse shaped function.

    Suitable for first order modeling of eclipsing binaries.

    ebparams = [period (time),
                epoch (time),
                pdepth (mags),
                pduration (phase),
                depthratio]

    period is the period in days

    epoch is the time of minimum in JD

    pdepth is the depth of the primary eclipse
    - for magnitudes -> transitdepth should be < 0
    - for fluxes     -> transitdepth should be > 0

    pduration is the length of the primary eclipse in phase

    depthratio is the ratio in the eclipse depths:
    depth_secondary/depth_primary. this is generally the same as the ratio of
    the Teffs of the two stars.

    All of these will then have fitted values after the fit is done.

    '''

    (period, epoch, pdepth, pduration, depthratio) = ebparams

    # generate the phases
    iphase = (times - epoch)/period
    iphase = iphase - npfloor(iphase)

    phasesortind = npargsort(iphase)
    phase = iphase[phasesortind]
    ptimes = times[phasesortind]
    pmags = mags[phasesortind]
    perrs = errs[phasesortind]

    zerolevel = npmedian(pmags)
    modelmags = npfull_like(phase, zerolevel)

    primaryecl_amp = -pdepth
    secondaryecl_amp = -pdepth * depthratio

    primaryecl_std = pduration/5.0 # we use 5-sigma as full-width -> duration
    secondaryecl_std = pduration/5.0 # secondary eclipse has the same duration

    halfduration = pduration/2.0


    # phase indices
    primary_eclipse_ingress = (
        (phase >= (1.0 - halfduration)) & (phase <= 1.0)
    )
    primary_eclipse_egress = (
        (phase >= 0.0) & (phase <= halfduration)
    )

    secondary_eclipse_phase = (
        (phase >= (0.5 - halfduration)) & (phase <= (0.5 + halfduration))
    )

    # put in the eclipses
    modelmags[primary_eclipse_ingress] = (
        zerolevel + _gaussian(phase[primary_eclipse_ingress],
                              primaryecl_amp,
                              1.0,
                              primaryecl_std)
    )
    modelmags[primary_eclipse_egress] = (
        zerolevel + _gaussian(phase[primary_eclipse_egress],
                              primaryecl_amp,
                              0.0,
                              primaryecl_std)
    )
    modelmags[secondary_eclipse_phase] = (
        zerolevel + _gaussian(phase[secondary_eclipse_phase],
                              secondaryecl_amp,
                              0.5,
                              secondaryecl_std)
    )

    return modelmags, phase, ptimes, pmags, perrs



def invgauss_eclipses_residual(ebparams, times, mags, errs):
    '''
    This returns the residual between the modelmags and the actual mags.

    '''

    modelmags, phase, ptimes, pmags, perrs = (
        invgauss_eclipses_func(ebparams, times, mags, errs)
    )

    # this is now a weighted residual taking into account the measurement err
    return (pmags - modelmags)/perrs
