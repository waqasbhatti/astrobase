#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flares.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

'''
This contains a stellar flare model from Pitkin+ 2014.

http://adsabs.harvard.edu/abs/2014MNRAS.445.2268P

'''

import numpy as np


##################################
## MODEL AND RESIDUAL FUNCTIONS ##
##################################

def flare_model(flareparams, times, mags, errs):
    '''This is a flare model function, similar to Kowalski+ 2011.

    From the paper by Pitkin+ 2014:
    http://adsabs.harvard.edu/abs/2014MNRAS.445.2268P

    Parameters
    ----------

    flareparams : list of float
        This defines the flare model::

            [amplitude,
             flare_peak_time,
             rise_gaussian_stdev,
             decay_time_constant]

        where:

        `amplitude`: the maximum flare amplitude in mags or flux. If flux, then
        amplitude should be positive. If mags, amplitude should be negative.

        `flare_peak_time`: time at which the flare maximum happens.

        `rise_gaussian_stdev`: the stdev of the gaussian describing the rise of
        the flare.

        `decay_time_constant`: the time constant of the exponential fall of the
        flare.

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the model will be generated. The times will be used to generate
        model mags.

    Returns
    -------

    (modelmags, times, mags, errs) : tuple
        Returns the model mags evaluated at the input time values. Also returns
        the input `times`, `mags`, and `errs`.

    '''

    (amplitude, flare_peak_time,
     rise_gaussian_stdev, decay_time_constant) = flareparams

    zerolevel = np.median(mags)
    modelmags = np.full_like(times, zerolevel)

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
    This returns the residual between model mags and the actual mags.

    Parameters
    ----------

    flareparams : list of float
        This defines the flare model::

            [amplitude,
             flare_peak_time,
             rise_gaussian_stdev,
             decay_time_constant]

        where:

        `amplitude`: the maximum flare amplitude in mags or flux. If flux, then
        amplitude should be positive. If mags, amplitude should be negative.

        `flare_peak_time`: time at which the flare maximum happens.

        `rise_gaussian_stdev`: the stdev of the gaussian describing the rise of
        the flare.

        `decay_time_constant`: the time constant of the exponential fall of the
        flare.

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the model will be generated. The times will be used to generate
        model mags.

    Returns
    -------

    np.array
        The residuals between the input `mags` and generated `modelmags`,
        weighted by the measurement errors in `errs`.

    '''

    modelmags, _, _, _ = flare_model(flareparams, times, mags, errs)

    return (mags - modelmags)/errs
