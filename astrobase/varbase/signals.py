#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# signals.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''
Contains functions to deal with masking and removing periodic signals in light
curves.

'''

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


#############
## IMPORTS ##
#############

import os.path
import os
from io import BytesIO as Strio


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

###################
## LOCAL IMPORTS ##
###################

from ..periodbase.zgls import pgen_lsp
from ..lcfit.sinusoidal import _fourier_func, fourier_fit_magseries
from ..lcmath import sigclip_magseries, phase_magseries


################################################
## REMOVING SIGNALS FROM MAGNITUDE TIMESERIES ##
################################################

def prewhiten_magseries(times, mags, errs,
                        whitenperiod,
                        whitenparams,
                        sigclip=3.0,
                        magsarefluxes=False,
                        plotfit=None,
                        plotfitphasedlconly=True,
                        rescaletomedian=True):
    '''Removes a periodic sinusoidal signal generated using whitenparams from
    the input magnitude time series.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to prewhiten.

    whitenperiod : float
        The period of the sinusoidal signal to remove.

    whitenparams : list of floats
        This contains the Fourier amplitude and phase coefficients of the
        sinusoidal signal to remove::

            [ampl_1, ampl_2, ampl_3, ..., ampl_X,
             pha_1, pha_2, pha_3, ..., pha_X]

        where `X` is the Fourier order. These are usually the output of a
        previous Fourier fit to the light curve (from
        :py:func:`astrobase.lcfit.sinusoidal.fourier_fit_magseries` for
        example).

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    magsarefluxes : bool
        If True, will treat the input values of `mags` as fluxes for purposes of
        plotting the fit and sig-clipping.

    plotfit : str or False
        If this is a string, this function will make a plot showing the effect
        of the pre-whitening on the mag/flux time-series and write the plot to
        the path specified here.

    plotfitphasedlconly : bool
        If True, will plot only the phased LC for showing the effect of
        pre-whitening, and skip plotting the unphased LC.

    rescaletomedian : bool
        If this is True, then we add back the constant median term of the
        magnitudes to the final pre-whitened mag series.

    Returns
    -------

    dict
        Returns a dict of the form::

            {'wtimes':times array after pre-whitening,
             'wphase':phase array after pre-whitening,
             'wmags':mags array after pre-whitening,
             'werrs':errs array after pre-whitening,
             'whitenparams':the input pre-whitening params used,
             'whitenperiod':the input pre-whitening period used,
             'fitplotfile':the output plot file if plotfit was set}

    '''

    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

    median_mag = np.median(smags)

    # phase the mag series using the given period and epoch = min(stimes)
    mintime = np.min(stimes)

    # calculate the unsorted phase, then sort it
    iphase = (
        (stimes - mintime)/whitenperiod -
        np.floor((stimes - mintime)/whitenperiod)
    )
    phasesortind = np.argsort(iphase)

    # these are the final quantities to use for the Fourier fits
    phase = iphase[phasesortind]
    pmags = smags[phasesortind]
    perrs = serrs[phasesortind]

    # get the times sorted in phase order (useful to get the fit mag minimum
    # with respect to phase -- the light curve minimum)
    ptimes = stimes[phasesortind]

    # now subtract the harmonic series from the phased LC
    # these are still in phase order
    wmags = pmags - _fourier_func(whitenparams, phase, pmags)

    # resort everything by time order
    wtimeorder = np.argsort(ptimes)
    wtimes = ptimes[wtimeorder]
    wphase = phase[wtimeorder]
    wmags = wmags[wtimeorder]
    werrs = perrs[wtimeorder]

    if rescaletomedian:
        wmags = wmags + median_mag

    # prepare the returndict
    returndict = {'wtimes':wtimes,  # these are in the new time order
                  'wphase':wphase,
                  'wmags':wmags,
                  'werrs':werrs,
                  'whitenparams':whitenparams,
                  'whitenperiod':whitenperiod}

    # make the fit plot if required
    if plotfit and (isinstance(plotfit, str) or isinstance(plotfit, Strio)):

        if plotfitphasedlconly:
            plt.figure(figsize=(10,4.8))
        else:
            plt.figure(figsize=(16,9.6))

        if plotfitphasedlconly:

            # phased series before whitening
            plt.subplot(121)
            plt.plot(phase,pmags,
                     marker='.',
                     color='k',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('phase')
            plt.title('phased LC before pre-whitening')

            # phased series after whitening
            plt.subplot(122)
            plt.plot(wphase,wmags,
                     marker='.',
                     color='g',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('phase')
            plt.title('phased LC after pre-whitening')

        else:

            # time series before whitening
            plt.subplot(221)
            plt.plot(stimes,smags,
                     marker='.',
                     color='k',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('JD')
            plt.title('LC before pre-whitening')

            # time series after whitening
            plt.subplot(222)
            plt.plot(wtimes,wmags,
                     marker='.',
                     color='g',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('JD')
            plt.title('LC after pre-whitening with period: %.6f' % whitenperiod)

            # phased series before whitening
            plt.subplot(223)
            plt.plot(phase,pmags,
                     marker='.',
                     color='k',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('phase')
            plt.title('phased LC before pre-whitening')

            # phased series after whitening
            plt.subplot(224)
            plt.plot(wphase,wmags,
                     marker='.',
                     color='g',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('phase')
            plt.title('phased LC after pre-whitening')

        plt.tight_layout()
        plt.savefig(plotfit, format='png', pad_inches=0.0)
        plt.close()

        if isinstance(plotfit, str) or isinstance(plotfit, Strio):
            returndict['fitplotfile'] = plotfit

    return returndict


def gls_prewhiten(times, mags, errs,
                  fourierorder=3,  # 3rd order series to start with
                  initfparams=None,
                  startp_gls=None,
                  endp_gls=None,
                  stepsize=1.0e-4,
                  autofreq=True,
                  sigclip=30.0,
                  magsarefluxes=False,
                  nbestpeaks=5,
                  nworkers=4,
                  plotfits=None):
    '''Iterative pre-whitening of a magnitude series using the L-S periodogram.

    This finds the best period, fits a fourier series with the best period, then
    whitens the time series with the best period, and repeats until `nbestpeaks`
    are done.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to iteratively pre-whiten.

    fourierorder : int
        The Fourier order of the sinusoidal signal to fit to the time-series and
        iteratively remove.

    initfparams : list or None
        If this is provided, should be a list of Fourier amplitudes and phases
        in the following format::

            [ampl_1, ampl_2, ampl_3, ..., ampl_X,
             pha_1, pha_2, pha_3, ..., pha_X]

        where `X` is the Fourier order. These are usually the output of a
        previous Fourier fit to the light curve (from
        :py:func:`astrobase.lcfit.sinusoidal.fourier_fit_magseries` for
        example). You MUST provide ONE of `fourierorder` and `initfparams`, but
        not both. If both are provided or both are None, a sinusoidal signal of
        Fourier order 3 will be used by default.

    startp_gls, endp_gls : float or None
        If these are provided, will serve as input to the Generalized
        Lomb-Scargle function that will attempt to find the best `nbestpeaks`
        periods in the time-series. These set the minimum and maximum period to
        search for in the time-series.

    stepsize : float
        The step-size in frequency to use when constructing a frequency grid for
        the period search.

    autofreq : bool
        If this is True, the value of `stepsize` will be ignored and the
        :py:func:`astrobase.periodbase.get_frequency_grid` function will be used
        to generate a frequency grid based on `startp`, and `endp`. If these are
        None as well, `startp` will be set to 0.1 and `endp` will be set to
        `times.max() - times.min()`.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    magsarefluxes : bool
        If the input measurement values in `mags` and `errs` are in fluxes, set
        this to True.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

    nworkers : int
        The number of parallel workers to use when calculating the periodogram.

    plotfits : None or str
        If this is a str, should indicate the file to which a plot of the
        successive iterations of pre-whitening will be written to. This will
        contain a row of plots indicating the before/after states of the light
        curves for each round of pre-whitening.

    Returns
    -------

    (bestperiods, plotfile) : tuple
        This returns a list of the best periods (with the "highest" peak in the
        periodogram) after each round of pre-whitening is done. If plotfit is a
        str, will also return the path to the generated plot file.

    '''

    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

    # now start the cycle by doing an GLS on the initial timeseries
    gls = pgen_lsp(stimes, smags, serrs,
                   magsarefluxes=magsarefluxes,
                   startp=startp_gls,
                   endp=endp_gls,
                   autofreq=autofreq,
                   sigclip=sigclip,
                   stepsize=stepsize,
                   nworkers=nworkers)

    LOGINFO('round %s: period = %.6f' % (0, gls['bestperiod']))

    if plotfits and isinstance(plotfits, str):

        plt.figure(figsize=(20,6*nbestpeaks))

        nplots = nbestpeaks + 1

        # periodogram
        plt.subplot(nplots,3,1)
        plt.plot(gls['periods'],gls['lspvals'])
        plt.xlabel('period [days]')
        plt.ylabel('GLS power')
        plt.xscale('log')
        plt.title('round 0, best period = %.6f' % gls['bestperiod'])

        # unphased LC
        plt.subplot(nplots,3,2)
        plt.plot(stimes, smags,
                 linestyle='none', marker='o',ms=1.0,rasterized=True)
        if not magsarefluxes:
            plt.gca().invert_yaxis()
            plt.ylabel('magnitude')
        else:
            plt.ylabel('flux')
        plt.xlabel('JD')
        plt.title('unphased LC before whitening')

        # phased LC
        plt.subplot(nplots,3,3)
        phased = phase_magseries(stimes, smags,
                                 gls['bestperiod'], stimes.min())

        plt.plot(phased['phase'], phased['mags'],
                 linestyle='none', marker='o',ms=1.0,rasterized=True)
        if not magsarefluxes:
            plt.ylabel('magnitude')
            plt.gca().invert_yaxis()
        else:
            plt.ylabel('flux')
        plt.xlabel('phase')
        plt.title('phased LC before whitening: P = %.6f' % gls['bestperiod'])

    # set up the initial times, mags, errs, period
    wtimes, wmags, werrs = stimes, smags, serrs
    wperiod = gls['bestperiod']

    # start the best periods list
    bestperiods = []

    # now go through the rest of the cycles
    for fitind in range(nbestpeaks):

        wfseries = fourier_fit_magseries(wtimes, wmags, werrs, wperiod,
                                         fourierorder=fourierorder,
                                         fourierparams=initfparams,
                                         magsarefluxes=magsarefluxes,
                                         sigclip=sigclip)

        wffitparams = wfseries['fitinfo']['finalparams']

        wseries = prewhiten_magseries(wtimes, wmags, werrs,
                                      wperiod,
                                      wffitparams,
                                      magsarefluxes=magsarefluxes,
                                      sigclip=sigclip)

        LOGINFO('round %s: period = %.6f' % (fitind+1, wperiod))
        bestperiods.append(wperiod)

        # update the mag series with whitened version
        wtimes, wmags, werrs = (
            wseries['wtimes'], wseries['wmags'], wseries['werrs']
        )

        # redo the periodogram
        wgls = pgen_lsp(wtimes, wmags, werrs,
                        magsarefluxes=magsarefluxes,
                        startp=startp_gls,
                        endp=endp_gls,
                        autofreq=autofreq,
                        sigclip=sigclip,
                        stepsize=stepsize,
                        nworkers=nworkers)
        wperiod = wgls['bestperiod']
        bestperiods.append(wperiod)

        # make plots if requested
        if plotfits and isinstance(plotfits, str):

            # periodogram
            plt.subplot(nplots,3,4+fitind*3)
            plt.plot(wgls['periods'],wgls['lspvals'])
            plt.xlabel('period [days]')
            plt.ylabel('LSP power')
            plt.xscale('log')
            plt.title('round %s, best period = %.6f' % (fitind+1,
                                                        wgls['bestperiod']))

            # unphased LC
            plt.subplot(nplots,3,5+fitind*3)
            plt.plot(wtimes, wmags,
                     linestyle='none', marker='o',ms=1.0,rasterized=True)
            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('flux')
            plt.xlabel('JD')
            plt.title('unphased LC after whitening')

            # phased LC
            plt.subplot(nplots,3,6+fitind*3)
            wphased = phase_magseries(wtimes, wmags,
                                      wperiod, stimes.min())

            plt.plot(wphased['phase'], wphased['mags'],
                     linestyle='none', marker='o',ms=1.0,rasterized=True)
            if not magsarefluxes:
                plt.ylabel('magnitude')
                plt.gca().invert_yaxis()
            else:
                plt.ylabel('flux')
            plt.xlabel('phase')
            plt.title('phased LC after whitening: P = %.6f' % wperiod)

    # in the end, write out the plot
    if plotfits and isinstance(plotfits, str):

        plt.subplots_adjust(hspace=0.2,wspace=0.4)
        plt.savefig(plotfits, bbox_inches='tight')
        plt.close('all')
        return bestperiods, os.path.abspath(plotfits)

    else:

        return bestperiods


def mask_signal(times, mags, errs,
                signalperiod,
                signalepoch,
                magsarefluxes=False,
                maskphases=(0,0,0.5,1.0),
                maskphaselength=0.1,
                plotfit=None,
                plotfitphasedlconly=True,
                sigclip=30.0):
    '''This removes repeating signals in the magnitude time series.

    Useful for masking planetary transit signals in light curves to search for
    other variability.

    A small worked example of using this and `prewhiten_magseries` above:

    https://github.com/waqasbhatti/astrobase/issues/77#issuecomment-463803558

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to run the masking on.

    signalperiod : float
        The period of the signal to mask.

    signalepoch : float
        The epoch of the signal to mask.

    magsarefluxes : bool
        Set to True if `mags` is actually an array of fluxes.

    maskphases : sequence of floats
        This defines which phase values will be masked. For each item in this
        sequence, this function will mask a length of phase given by
        `maskphaselength` centered on each `maskphases` value, and remove all LC
        points in these regions from the light curve.

    maskphaselength : float
        The length in phase to mask for each phase value provided in
        `maskphases`.

    plotfit : str or None
        If provided as a str, indicates the output plot file.

    plotfitphasedlconly : bool
        If True, will only plot the effect of masking the signal as requested on
        the phased LC. If False, will also plot the unphased LC.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    '''

    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

    # now phase the light curve using the period and epoch provided
    phases = (
        (stimes - signalepoch)/signalperiod -
        np.floor((stimes - signalepoch)/signalperiod)
    )

    # mask the requested phases using the mask length (in phase units)
    # this gets all the masks into one array
    masks = np.array([(np.abs(phases - x) > maskphaselength)
                      for x in maskphases])
    # this flattens the masks to a single array for all combinations
    masks = np.all(masks,axis=0)

    # apply the mask to the times, mags, and errs
    mphases = phases[masks]
    mtimes = stimes[masks]
    mmags = smags[masks]
    merrs = serrs[masks]

    returndict = {'mphases':mphases,
                  'mtimes':mtimes,
                  'mmags':mmags,
                  'merrs':merrs}

    # make the fit plot if required
    if plotfit and isinstance(plotfit, str) or isinstance(plotfit, Strio):

        if plotfitphasedlconly:
            plt.figure(figsize=(10,4.8))
        else:
            plt.figure(figsize=(16,9.6))

        if plotfitphasedlconly:

            # phased series before whitening
            plt.subplot(121)
            plt.plot(phases,smags,
                     marker='.',
                     color='k',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('phase')
            plt.title('phased LC before signal masking')

            # phased series after whitening
            plt.subplot(122)
            plt.plot(mphases,mmags,
                     marker='.',
                     color='g',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('phase')
            plt.title('phased LC after signal masking')

        else:

            # time series before whitening
            plt.subplot(221)
            plt.plot(stimes,smags,
                     marker='.',
                     color='k',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('JD')
            plt.title('LC before signal masking')

            # time series after whitening
            plt.subplot(222)
            plt.plot(mtimes,mmags,
                     marker='.',
                     color='g',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('JD')
            plt.title('LC after signal masking')

            # phased series before whitening
            plt.subplot(223)
            plt.plot(phases,smags,
                     marker='.',
                     color='k',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('phase')
            plt.title('phased LC before signal masking')

            # phased series after whitening
            plt.subplot(224)
            plt.plot(mphases,mmags,
                     marker='.',
                     color='g',
                     linestyle='None',
                     markersize=2.0,
                     markeredgewidth=0)

            if not magsarefluxes:
                plt.gca().invert_yaxis()
                plt.ylabel('magnitude')
            else:
                plt.ylabel('fluxes')

            plt.xlabel('phase')
            plt.title('phased LC after signal masking')

        plt.tight_layout()
        plt.savefig(plotfit, format='png', pad_inches=0.0)
        plt.close()

        if isinstance(plotfit, str) or isinstance(plotfit, Strio):
            returndict['fitplotfile'] = plotfit

    return returndict
