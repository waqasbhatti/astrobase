#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sinusoidal.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

'''
This contains models for sinusoidal light curves generated using Fourier
expansion.
'''

import numpy as np


##################################
## MODEL AND RESIDUAL FUNCTIONS ##
##################################

def fourier_sinusoidal_func(fourierparams, times, mags, errs):
    '''This generates a sinusoidal light curve using a Fourier cosine series.

    Parameters
    ----------

    fourierparams : list
        This MUST be a list of the following form like so::

            [period,
             epoch,
             [amplitude_1, amplitude_2, amplitude_3, ..., amplitude_X],
             [phase_1, phase_2, phase_3, ..., phase_X]]

        where X is the Fourier order.

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the model will be generated. The times will be used to generate model
        mags, and the input `times`, `mags`, and `errs` will be resorted by
        model phase and returned.

    Returns
    -------

    (modelmags, phase, ptimes, pmags, perrs) : tuple
        Returns the model mags and phase values. Also returns the input `times`,
        `mags`, and `errs` sorted by the model's phase.

    '''

    period, epoch, famps, fphases = fourierparams

    # figure out the order from the length of the Fourier param list
    forder = len(famps)

    # phase the times with this period
    iphase = (times - epoch)/period
    iphase = iphase - np.floor(iphase)

    phasesortind = np.argsort(iphase)
    phase = iphase[phasesortind]
    ptimes = times[phasesortind]
    pmags = mags[phasesortind]
    perrs = errs[phasesortind]

    # calculate all the individual terms of the series
    fseries = [famps[x]*np.cos(2.0*np.pi*x*phase + fphases[x])
               for x in range(forder)]

    # this is the zeroth order coefficient - a constant equal to median mag
    modelmags = np.median(mags)

    # sum the series
    for fo in fseries:
        modelmags += fo

    return modelmags, phase, ptimes, pmags, perrs


def fourier_curvefit_func(times,
                          period,
                          *fourier_coeffs,
                          zerolevel=0.0,
                          epoch=None,
                          fixed_period=None):
    '''
    This is a function to be used with scipy.optimize.curve_fit.

    Parameters
    ----------

    times : np.array
        An array of times at which the model will be evaluated.

    period : float
        The period of the sinusoidal variability.

    fourier_coeffs : float
        These should be the amplitudes and phases of the sinusoidal series
        sum. 2N coefficients are required for Fourier order = N. The first N
        coefficients will be used as the amplitudes and the second N
        coefficients will be used as the phases.

    zerolevel : float
        The base level of the model.

    epoch : float or None
        The epoch to use to generate the phased light curve. If None, the
        minimum value of the times array will be used.

    fixed_period : float or None
        If not None, will indicate that the period is to be held fixed at the
        provided value.

    Returns
    -------

    model : np.array
        Returns the sinusodial series sum model evaluated at each value of
        times.

    '''

    if epoch is None:
        epoch = times.min()

    if fixed_period is not None:
        period = fixed_period

    fourier_order = int(len(fourier_coeffs)/2.0)

    fourier_amplitudes, fourier_phases = (
        fourier_coeffs[:fourier_order],
        fourier_coeffs[fourier_order:]
    )

    # phase the times with this period
    phase = (times - epoch)/period
    phase = phase - np.floor(phase)

    # calculate all the individual terms of the series
    fseries = [
        fourier_amplitudes[x]*np.cos(2.0*np.pi*x*phase + fourier_phases[x])
        for x in range(fourier_order)
    ]

    model = zerolevel
    for fo in fseries:
        model += fo

    return model


def fourier_sinusoidal_residual(fourierparams, times, mags, errs):
    '''
    This returns the residual between the model mags and the actual mags.

    Parameters
    ----------

    fourierparams : list
        This MUST be a list of the following form like so::

            [period,
             epoch,
             [amplitude_1, amplitude_2, amplitude_3, ..., amplitude_X],
             [phase_1, phase_2, phase_3, ..., phase_X]]

        where X is the Fourier order.

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the model will be generated. The times will be used to generate model
        mags, and the input `times`, `mags`, and `errs` will be resorted by
        model phase and returned.

    Returns
    -------

    np.array
        The residuals between the input `mags` and generated `modelmags`,
        weighted by the measurement errors in `errs`.


    '''
    modelmags, phase, ptimes, pmags, perrs = (
        fourier_sinusoidal_func(fourierparams, times, mags, errs)
    )

    # this is now a weighted residual taking into account the measurement err
    return (pmags - modelmags)/perrs


def sine_series_sum(fourierparams, times, mags, errs):
    '''This generates a sinusoidal light curve using a Fourier sine series.

    Parameters
    ----------

    fourierparams : list
        This MUST be a list of the following form like so::

            [period,
             epoch,
             [amplitude_1, amplitude_2, amplitude_3, ..., amplitude_X],
             [phase_1, phase_2, phase_3, ..., phase_X]]

        where X is the Fourier order.

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the model will be generated. The times will be used to generate model
        mags, and the input `times`, `mags`, and `errs` will be resorted by
        model phase and returned.

    Returns
    -------

    (modelmags, phase, ptimes, pmags, perrs) : tuple
        Returns the model mags and phase values. Also returns the input `times`,
        `mags`, and `errs` sorted by the model's phase.

    '''

    period, epoch, famps, fphases = fourierparams

    # figure out the order from the length of the Fourier param list
    forder = len(famps)

    # phase the times with this period
    iphase = (times - epoch)/period
    iphase = iphase - np.floor(iphase)

    phasesortind = np.argsort(iphase)
    phase = iphase[phasesortind]
    ptimes = times[phasesortind]
    pmags = mags[phasesortind]
    perrs = errs[phasesortind]

    # calculate all the individual terms of the series
    fseries = [famps[x]*np.sin(2.0*np.pi*x*phase + fphases[x])
               for x in range(forder)]

    # this is the zeroth order coefficient - a constant equal to median mag
    modelmags = np.median(mags)

    # sum the series
    for fo in fseries:
        modelmags += fo

    return modelmags, phase, ptimes, pmags, perrs
