#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''sinusoidal.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

This contains models for sinusoidal light curves generated using Fourier
expansion.
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

from scipy.signal import medfilt, savgol_filter

##################################
## MODEL AND RESIDUAL FUNCTIONS ##
##################################


def fourier_sinusoidal_func(fourierparams, times, mags, errs):
    '''This generates a sinusoidal light curve using a Fourier series.

    The Fourier series is generated using the coefficients provided in
    fourierparams. This is a sequence like so:

    [period,
     epoch,
     [ampl_1, ampl_2, ampl_3, ..., ampl_X],
     [pha_1, pha_2, pha_3, ..., pha_X]]

    where X is the Fourier order.

    '''

    period, epoch, famps, fphases = fourierparams

    # figure out the order from the length of the Fourier param list
    forder = len(famps)

    # phase the times with this period
    iphase = (times - epoch)/period
    iphase = iphase - npfloor(iphase)

    phasesortind = npargsort(iphase)
    phase = iphase[phasesortind]
    ptimes = times[phasesortind]
    pmags = mags[phasesortind]
    perrs = errs[phasesortind]

    # calculate all the individual terms of the series
    fseries = [famps[x]*npcos(2.0*MPI*x*phase + fphases[x])
               for x in range(forder)]

    # this is the zeroth order coefficient - a constant equal to median mag
    modelmags = npmedian(mags)

    # sum the series
    for fo in fseries:
        modelmags += fo

    return modelmags, phase, ptimes, pmags, perrs



def fourier_sinusoidal_residual(fourierparams, times, mags, errs):
    '''
    This returns the residual between the model mags and the actual mags.

    '''
    modelmags, phase, ptimes, pmags, perrs = (
        fourier_sinusoidal_func(fourierparams, times, mags, errs)
    )

    # this is now a weighted residual taking into account the measurement err
    return (pmags - modelmags)/perrs



def sine_series_sum(fourierparams, times, mags, errs):
    '''This generates a sinusoidal light curve using a sine series.

    The series is generated using the coefficients provided in
    fourierparams. This is a sequence like so:

    [period,
     epoch,
     [ampl_1, ampl_2, ampl_3, ..., ampl_X],
     [pha_1, pha_2, pha_3, ..., pha_X]]

    where X is the Fourier order.

    '''

    period, epoch, famps, fphases = fourierparams

    # figure out the order from the length of the Fourier param list
    forder = len(famps)

    # phase the times with this period
    iphase = (times - epoch)/period
    iphase = iphase - npfloor(iphase)

    phasesortind = npargsort(iphase)
    phase = iphase[phasesortind]
    ptimes = times[phasesortind]
    pmags = mags[phasesortind]
    perrs = errs[phasesortind]

    # calculate all the individual terms of the series
    fseries = [famps[x]*npsin(2.0*MPI*x*phase + fphases[x])
               for x in range(forder)]

    # this is the zeroth order coefficient - a constant equal to median mag
    modelmags = npmedian(mags)

    # sum the series
    for fo in fseries:
        modelmags += fo

    return modelmags, phase, ptimes, pmags, perrs
