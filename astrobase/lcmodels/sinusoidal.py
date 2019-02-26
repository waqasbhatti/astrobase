#!/usr/bin/env python
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
