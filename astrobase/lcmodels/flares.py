#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''flares.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

This contains a stellar flare model from Pitkin+ 2014.

http://adsabs.harvard.edu/abs/2014MNRAS.445.2268P

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

def flare_model(flareparams, times, mags, errs):
    '''This is a flare model function, similar to Kowalski+ 2011.

    Model params
    ------------

    flareparams is a list:

    [amplitude, flare_peak_time, rise_gaussian_stdev, decay_time_constant]

    where:

    amplitude: the maximum flare amplitude in mags or flux. If flux, then
    amplitude should be positive. If mags, amplitude should be negative.

    flare_peak_time: time at which the flare maximum happens

    rise_gaussian_stdev: the stdev of the gaussian describing the rise of the
                         flare

    decay_time_constant: the time constant of the exponential fall of the flare


    Other args
    ----------

    times: a numpy array of times

    mags: a numpy array of magnitudes or fluxes. the flare will simply be added
    to mags at the appropriate times

    errs: a numpy array of measurement errors for each mag/flux measurement

    '''

    (amplitude, flare_peak_time,
     rise_gaussian_stdev, decay_time_constant) = flareparams

    zerolevel = npmedian(mags)
    modelmags = npfull_like(times, zerolevel)

    # before peak gaussian rise...
    modelmags[times < flare_peak_time] = (
        mags[times < flare_peak_time] +
        amplitude * np.exp(
            -((times[times < flare_peak_time] -
               flare_peak_time) *
              (times[times < flare_peak_time] -
               flare_peak_time)) /
            (2.0*rise_gaussian_stdev*rise_gaussian_stdev)
            )
    )

    # after peak exponential decay...
    modelmags[times > flare_peak_time] = (
        mags[times > flare_peak_time] +
        amplitude * np.exp(
            -((times[times > flare_peak_time] -
               flare_peak_time)) /
            (decay_time_constant)
            )
    )

    return modelmags, times, mags, errs



def flare_model_residual(flareparams, times, mags, errs):
    '''
    This just returns the residual between model mags and the actual mags.

    '''

    modelmags, _, _, _ = flare_model(flareparams, times, mags, errs)

    return (mags - modelmags)/errs
