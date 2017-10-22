#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''transits.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

This contains a trapezoid model for first order model of planetary transits
light curves.

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

def trapezoid_transit_func(transitparams, times, mags, errs):
    '''This returns a trapezoid transit-shaped function.

    Suitable for first order modeling of transit signals.

    transitparams = [transitperiod (time),
                     transitepoch (time),
                     transitdepth (flux or mags),
                     transitduration (phase),
                     ingressduration (phase)]

    All of these will then have fitted values after the fit is done.

    for magnitudes -> transitdepth should be < 0
    for fluxes     -> transitdepth should be > 0
    '''

    (transitperiod,
     transitepoch,
     transitdepth,
     transitduration,
     ingressduration) = transitparams

    # generate the phases
    iphase = (times - transitepoch)/transitperiod
    iphase = iphase - npfloor(iphase)

    phasesortind = npargsort(iphase)
    phase = iphase[phasesortind]
    ptimes = times[phasesortind]
    pmags = mags[phasesortind]
    perrs = errs[phasesortind]

    zerolevel = npmedian(pmags)
    modelmags = npfull_like(phase, zerolevel)

    halftransitduration = transitduration/2.0
    bottomlevel = zerolevel - transitdepth
    slope = transitdepth/ingressduration

    # the four contact points of the eclipse
    firstcontact = 1.0 - halftransitduration
    secondcontact = firstcontact + ingressduration
    thirdcontact = halftransitduration - ingressduration
    fourthcontact = halftransitduration

    ## the phase indices ##

    # during ingress
    ingressind = (phase > firstcontact) & (phase < secondcontact)

    # at transit bottom
    bottomind = (phase > secondcontact) | (phase < thirdcontact)

    # during egress
    egressind = (phase > thirdcontact) & (phase < fourthcontact)

    # set the mags
    modelmags[ingressind] = zerolevel - slope*(phase[ingressind] - firstcontact)
    modelmags[bottomind] = bottomlevel
    modelmags[egressind] = bottomlevel + slope*(phase[egressind] - thirdcontact)

    return modelmags, phase, ptimes, pmags, perrs



def trapezoid_transit_residual(transitparams, times, mags, errs):
    '''
    This returns the residual between the modelmags and the actual mags.

    '''

    modelmags, phase, ptimes, pmags, perrs = (
        _trapezoid_transit_func(transitparams, times, mags, errs)
    )

    # this is now a weighted residual taking into account the measurement err
    return (pmags - modelmags)/perrs
