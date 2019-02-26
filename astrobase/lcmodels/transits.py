#!/usr/bin/env python
# -*- coding: utf-8 -*-
# transits.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

'''
This contains a trapezoid model for first order model of planetary transits
light curves.

'''

import numpy as np


##################################
## MODEL AND RESIDUAL FUNCTIONS ##
##################################

def trapezoid_transit_func(transitparams, times, mags, errs,
                           get_ntransitpoints=False):
    '''This returns a trapezoid transit-shaped function.

    Suitable for first order modeling of transit signals.

    Parameters
    ----------

    transitparams : list of float
        This contains the transiting planet trapezoid model::

            transitparams = [transitperiod (time),
                             transitepoch (time),
                             transitdepth (flux or mags),
                             transitduration (phase),
                             ingressduration (phase)]

        All of these will then have fitted values after the fit is done.

        - for magnitudes -> `transitdepth` should be < 0
        - for fluxes     -> `transitdepth` should be > 0

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the transit model will be generated. The times will be used to generate
        model mags, and the input `times`, `mags`, and `errs` will be resorted
        by model phase and returned.

    Returns
    -------

    (modelmags, phase, ptimes, pmags, perrs) : tuple
        Returns the model mags and phase values. Also returns the input `times`,
        `mags`, and `errs` sorted by the model's phase.

    '''

    (transitperiod,
     transitepoch,
     transitdepth,
     transitduration,
     ingressduration) = transitparams

    # generate the phases
    iphase = (times - transitepoch)/transitperiod
    iphase = iphase - np.floor(iphase)

    phasesortind = np.argsort(iphase)
    phase = iphase[phasesortind]
    ptimes = times[phasesortind]
    pmags = mags[phasesortind]
    perrs = errs[phasesortind]

    zerolevel = np.median(pmags)
    modelmags = np.full_like(phase, zerolevel)

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

    # count the number of points in transit
    in_transit_points = ingressind | bottomind | egressind
    n_transit_points = np.sum(in_transit_points)

    # set the mags
    modelmags[ingressind] = zerolevel - slope*(phase[ingressind] - firstcontact)
    modelmags[bottomind] = bottomlevel
    modelmags[egressind] = bottomlevel + slope*(phase[egressind] - thirdcontact)

    if get_ntransitpoints:
        return modelmags, phase, ptimes, pmags, perrs, n_transit_points

    else:
        return modelmags, phase, ptimes, pmags, perrs



def trapezoid_transit_residual(transitparams, times, mags, errs):
    '''
    This returns the residual between the modelmags and the actual mags.

    Parameters
    ----------

    transitparams : list of float
        This contains the transiting planet trapezoid model::

            transitparams = [transitperiod (time),
                             transitepoch (time),
                             transitdepth (flux or mags),
                             transitduration (phase),
                             ingressduration (phase)]

        All of these will then have fitted values after the fit is done.

        - for magnitudes -> `transitdepth` should be < 0
        - for fluxes     -> `transitdepth` should be > 0

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the transit model will be generated. The times will be used to generate
        model mags, and the input `times`, `mags`, and `errs` will be resorted
        by model phase and returned.

    Returns
    -------

    np.array
        The residuals between the input `mags` and generated `modelmags`,
        weighted by the measurement errors in `errs`.


    '''

    modelmags, phase, ptimes, pmags, perrs = (
        trapezoid_transit_func(transitparams, times, mags, errs)
    )

    # this is now a weighted residual taking into account the measurement err
    return (pmags - modelmags)/perrs
