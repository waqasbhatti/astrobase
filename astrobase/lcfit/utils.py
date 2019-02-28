#!/usr/bin/env python
# -*- coding: utf-8 -*-
# varbase/lcfit.py
# Waqas Bhatti and Luke Bouma - Feb 2017
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''This contains utilities for fitting routines in the rest of this subpackage.

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

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


########################################
## FUNCTIONS FOR SIMPLE LC OPERATIONS ##
########################################

def get_phased_quantities(stimes, smags, serrs, period):
    '''Does phase-folding for the mag/flux time-series given a period.

    Given finite and sigma-clipped times, magnitudes, and errors, along with the
    period at which to phase-fold the data, perform the phase-folding and
    return the phase-folded values.

    Parameters
    ----------

    stimes,smags,serrs : np.array
        The sigma-clipped and finite input mag/flux time-series arrays to
        operate on.

    period : float
        The period to phase the mag/flux time-series at. stimes.min() is used as
        the epoch value to fold the times-series around.

    Returns
    -------

    (phase, pmags, perrs, ptimes, mintime) : tuple
        The tuple returned contains the following items:

        - `phase`: phase-sorted values of phase at each of stimes
        - `pmags`: phase-sorted magnitudes at each phase
        - `perrs`: phase-sorted errors
        - `ptimes`: phase-sorted times
        - `mintime`: earliest time in stimes.

    '''

    # phase the mag series using the given period and faintest mag time
    # mintime = stimes[npwhere(smags == npmax(smags))]

    # phase the mag series using the given period and epoch = min(stimes)
    mintime = np.min(stimes)

    # calculate the unsorted phase, then sort it
    iphase = (stimes - mintime)/period - np.floor((stimes - mintime)/period)
    phasesortind = np.argsort(iphase)

    # these are the final quantities to use for the Fourier fits
    phase = iphase[phasesortind]
    pmags = smags[phasesortind]
    perrs = serrs[phasesortind]

    # get the times sorted in phase order (useful to get the fit mag minimum
    # with respect to phase -- the light curve minimum)
    ptimes = stimes[phasesortind]

    return phase, pmags, perrs, ptimes, mintime


########################
## PLOTTING UTILITIES ##
########################

def make_fit_plot(phase, pmags, perrs, fitmags,
                  period, mintime, magseriesepoch,
                  plotfit,
                  magsarefluxes=False,
                  wrap=False,
                  model_over_lc=False):
    '''This makes a plot of the LC model fit.

    Parameters
    ----------

    phase,pmags,perrs : np.array
        The actual mag/flux time-series.

    fitmags : np.array
        The model fit time-series.

    period : float
        The period at which the phased LC was generated.

    mintime : float
        The minimum time value.

    magseriesepoch : float
        The value of time around which the phased LC was folded.

    plotfit : str
        The name of a file to write the plot to.

    magsarefluxes : bool
        Set this to True if the values in `pmags` and `fitmags` are actually
        fluxes.

    wrap : bool
        If True, will wrap the phased LC around 0.0 to make some phased LCs
        easier to look at.

    model_over_lc : bool
        Usually, this function will plot the actual LC over the model LC. Set
        this to True to plot the model over the actual LC; this is most useful
        when you have a very dense light curve and want to be able to see how it
        follows the model.

    Returns
    -------

    Nothing.

    '''

    # set up the figure
    plt.close('all')
    plt.figure(figsize=(8,4.8))

    if model_over_lc:
        model_z = 100
        lc_z = 0
    else:
        model_z = 0
        lc_z = 100


    if not wrap:

        plt.plot(phase, fitmags, linewidth=3.0, color='red',zorder=model_z)
        plt.plot(phase,pmags,
                 marker='o',
                 markersize=1.0,
                 linestyle='none',
                 rasterized=True, color='k',zorder=lc_z)

        # set the x axis ticks and label
        plt.gca().set_xticks(
            [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        )

    else:
        plt.plot(np.concatenate([phase-1.0,phase]),
                 np.concatenate([fitmags,fitmags]),
                 linewidth=3.0,
                 color='red',zorder=model_z)
        plt.plot(np.concatenate([phase-1.0,phase]),
                 np.concatenate([pmags,pmags]),
                 marker='o',
                 markersize=1.0,
                 linestyle='none',
                 rasterized=True, color='k',zorder=lc_z)

        plt.gca().set_xlim((-0.8,0.8))
        # set the x axis ticks and label
        plt.gca().set_xticks(
            [-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,
             0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        )

    # set the y axis limit and label
    ymin, ymax = plt.ylim()
    if not magsarefluxes:
        plt.gca().invert_yaxis()
        plt.ylabel('magnitude')
    else:
        plt.ylabel('flux')


    plt.xlabel('phase')
    plt.title('period: %.6f, folded at %.6f, fit epoch: %.6f' %
              (period, mintime, magseriesepoch))
    plt.savefig(plotfit)
    plt.close()
