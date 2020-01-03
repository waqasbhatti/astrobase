#!/usr/bin/env python3
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


def trapezoid_transit_curvefit_func(
        times,
        period,
        epoch,
        depth,
        duration,
        ingressduration,
        zerolevel=0.0,
        fixed_params=None,
):
    '''
    This is the function used for scipy.optimize.curve_fit.

    Parameters
    ----------

    times : np.array
        The array of times used to construct the transit model.

    period : float
        The period of the transit.

    epoch : float
        The time of mid-transit (phase 0.0). Must be in the same units as times.

    depth : float
        The depth of the transit.

    duration : float
        The duration of the transit in phase units.

    ingressduration : float
        The ingress duration of the transit in phase units.

    zerolevel : float
        The level of the measurements outside transit.

    fixed_params : dict or None
        If this is provided, must be a dict containing the parameters to fix and
        their values. Should be of the form below::

            {'period': fixed value,
             'epoch': fixed value,
             'depth': fixed value,
             'duration': fixed value,
             'ingressduration': fixed value}

        Any parameter in the dict provided will have its parameter fixed to the
        provided value. This is best done with an application of
        functools.partial before passing the function to the
        scipy.optimize.curve_fit function, e.g.::

            curvefit_func = functools.partial(
                                transits.trapezoid_transit_curvefit_func,
                                zerolevel=np.median(mags),
                                fixed_params={'ingressduration':0.05})

            fit_params, fit_cov = scipy.optimize.curve_fit(
                                    curvefit_func,
                                    times, mags,
                                    p0=initial_params,
                                    sigma=errs,
                                    ...)

    Returns
    -------

    model : np.array
        Returns the transit model as an np.array. This is in the same order as
        the times input array.

    '''

    if fixed_params is not None and len(fixed_params) > 0:

        if 'period' in fixed_params:
            period = fixed_params['period']
        if 'epoch' in fixed_params:
            epoch = fixed_params['epoch']
        if 'pdepth' in fixed_params:
            depth = fixed_params['depth']
        if 'duration' in fixed_params:
            duration = fixed_params['duration']
        if 'ingressduration' in fixed_params:
            ingressduration = fixed_params['ingressduration']

    # generate the phases
    phase = (times - epoch)/period
    phase = phase - np.floor(phase)

    transitmodel = np.full_like(phase, zerolevel)

    halftransitduration = duration/2.0
    bottomlevel = zerolevel - depth
    slope = depth/ingressduration

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

    # set the transit model
    transitmodel[ingressind] = (
        zerolevel - slope*(phase[ingressind] - firstcontact)
    )
    transitmodel[bottomind] = bottomlevel
    transitmodel[egressind] = (
        bottomlevel + slope*(phase[egressind] - thirdcontact)
    )

    return transitmodel


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
