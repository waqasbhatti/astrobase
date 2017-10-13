#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''macf.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

This contains the ACF period-finding algorithm from McQuillan+ 2013a and
McQuillan+ 2014.

'''


from multiprocessing import Pool, cpu_count
import logging
from datetime import datetime
from traceback import format_exc

import numpy as np

# import these to avoid lookup overhead
from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, \
    arctan as nparctan, nanargmax as npnanargmax, nanargmin as npnanargmin, \
    empty as npempty, ceil as npceil, mean as npmean, \
    digitize as npdigitize, unique as npunique, \
    argmax as npargmax, argmin as npargmin

from scipy.signal import argrelmax, argrelmin, savgol_filter
from astropy.convolution import convolve, Gaussian1DKernel


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.macf' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


###################
## LOCAL IMPORTS ##
###################

from ..lcmath import phase_magseries, sigclip_magseries, time_bin_magseries, \
    phase_bin_magseries, fill_magseries_gaps

from ..varbase.autocorr import autocorr_magseries


############
## CONFIG ##
############

NCPUS = cpu_count()



######################
## HELPER FUNCTIONS ##
######################


def _smooth_acf(acf, windowfwhm=7, windowsize=21):
    '''
    This returns a smoothed version of the ACF.

    Convolves the ACF with a Gaussian of given windowsize and windowfwhm

    '''

    convkernel = Gaussian1DKernel(windowfwhm, x_size=windowsize)
    smoothed = convolve(acf, convkernel, boundary='extend')

    return smoothed


def _smooth_acf_savgol(acf, windowsize=21, polyorder=2):
    '''
    This returns a smoothed version of the ACF.

    This version uses the Savitsky-Golay smoothing filter

    '''

    smoothed = savgol_filter(acf, windowsize, polyorder)

    return smoothed



def _get_acf_peakheights(lags, acf, npeaks=20, searchinterval=1):
    '''This calculates the relative peak heights for first npeaks in ACF.

    Usually, the first peak or the second peak (if its peak height > first peak)
    corresponds to the correct lag. When we know the correct lag, the period is
    then:

    bestperiod = time[lags == bestlag] - time[0]

    '''

    maxinds = argrelmax(acf, order=searchinterval)[0]
    maxacfs = acf[maxinds]
    maxlags = lags[maxinds]
    mininds = argrelmin(acf, order=searchinterval)[0]
    minacfs = acf[mininds]
    minlags = lags[mininds]

    relpeakheights = np.zeros(npeaks)
    relpeaklags = np.zeros(npeaks,dtype=np.int64)
    peakindices = np.zeros(npeaks,dtype=np.int64)

    for peakind, mxi in enumerate(maxinds[:npeaks]):

        # check if there are no mins to the left
        # throw away this peak because it's probably spurious
        # (FIXME: is this OK?)
        if np.all(mxi < mininds):
            continue

        leftminind = mininds[mininds < mxi][-1] # the last index to the left
        rightminind = mininds[mininds > mxi][0] # the first index to the right
        relpeakheights[peakind] = (
            acf[mxi] - (acf[leftminind] + acf[rightminind])/2.0
        )
        relpeaklags[peakind] = lags[mxi]
        peakindices[peakind] = peakind

    # figure out the bestperiod if possible
    if relpeakheights[0] > relpeakheights[1]:
        bestlag = relpeaklags[0]
        bestpeakheight = relpeakheights[0]
        bestpeakindex = peakindices[0]
    else:
        bestlag = relpeaklags[1]
        bestpeakheight = relpeakheights[1]
        bestpeakindex = peakindices[1]

    return {'maxinds':maxinds,
            'maxacfs':maxacfs,
            'maxlags':maxlags,
            'mininds':mininds,
            'minacfs':minacfs,
            'minlags':minlags,
            'relpeakheights':relpeakheights,
            'relpeaklags':relpeaklags,
            'peakindices':peakindices,
            'bestlag':bestlag,
            'bestpeakheight':bestpeakheight,
            'bestpeakindex':bestpeakindex}


def plot_acf_results(acfp, outfile, maxlags=5000, yrange=[-0.4,0.4]):
    '''
    This plots the unsmoothed/smoothed ACF vs lag.

    acfp is the resultdict from macf_period_find below.

    '''

    import matplotlib.pyplot as plt

    lags = acfp['acfresults']['lags'][:maxlags]
    smoothedacf = acfp['acf'][:maxlags]
    unsmoothedacf = acfp['acfresults']['acf'][:maxlags]

    acfparams = acfp['kwargs']['smoothfunckwargs'].copy()
    acfparams.update({'peakinterval': int(acfp['kwargs']['smoothacf']/2.0)})

    # plot the ACFs
    fig, ax1 = plt.subplots()

    # this is lags vs acf
    ax1.plot(lags, unsmoothedacf, label='unsmoothed ACF',color='#1f77b4')
    ax1.plot(lags, smoothedacf, label='smoothed ACF', color='#ff7f0e')

    ax1.set_xlim((0,maxlags))

    ax1.set_xlabel('lags')

    # overplot the identified peaks
    acfmaxinds = acfp['acfpeaks']['maxinds']

    for i, maxind in enumerate(acfmaxinds):
        if i == 0:
            ax1.axvline(maxind,
                        linewidth=2.0,
                        color='red',
                        ymin=0.2, ymax=0.3,
                        label='identified ACF peaks')
        else:
            ax1.axvline(maxind,
                        linewidth=2.0,
                        color='red',
                        ymin=0.2, ymax=0.3)

    plt.ylabel('ACF')
    plt.ylim(yrange)
    ax1.legend()
    plt.title('%s' % repr(acfparams))
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close('all')

    return outfile



############################
## PERIOD FINDER FUNCTION ##
############################

def macf_period_find(
        times,
        mags,
        errs,
        fillgaps=0.0,
        filterwindow=11,
        forcetimebin=None,
        maxlags=None,
        maxacfpeaks=10,
        smoothacf=21,
        smoothfunc=_smooth_acf_savgol,
        smoothfunckwargs={},
        magsarefluxes=False,
        sigclip=3.0,
        verbose=True,
        nworkers=None, # doesn't do anything, but for consistent external API
):
    '''This finds periods using the McQuillan+ (2013a, 2014) method.

    Args
    ----

    times, mags, errs are np.arrays.

    fillgaps is what to use to fill in gaps in the time series. If this is
    'noiselevel', will smooth the light curve using a point window size of
    filterwindow (this should be an odd integer), subtract the smoothed LC from
    the actual LC and estimate the RMS. This RMS will be used to fill in the
    gaps. Other useful values here are 0.0, and np.nan.

    forcetimebin is used to force a particular cadence in the light curve other
    than the automatically determined cadence. This effectively rebins the light
    curve to this cadence.

    maxlags is maximum number of lags to calculate. If None, will calculate all
    lags.

    maxacfpeaks is the maximum number of ACF peaks to use when finding the
    highest peak and obtaining a fit period.

    smoothacf is the number of points to use as the window size when smoothing
    the acf with the smoothfunc. This should be an odd integer value. If this is
    None, will not smooth the ACF, but this will probably lead to finding
    spurious peaks in a generally noisy ACF.

    For Kepler, a value between 21 and 51 seems to work fine. For ground based
    data, much larger values may be necessary: between 1001 and 2001 seem to
    work best for the HAT surveys. This is dependent on cadence, RMS of the
    light curve, the periods of the objects you're looking for, and finally, any
    correlated noise in the light curve. Make a plot of the smoothed/unsmoothed
    ACF vs. lag using the result dict of this function and the plot_acf_results
    function above to see the identified ACF peaks and what kind of smoothing
    might be needed.

    The value of smoothacf will also be used to figure out the interval to use
    when searching for local peaks in the ACF: this interval is 1/2 of the
    smoothacf value.

    smoothfunc is a function to use when smoothing the ACF. This should take at
    least one kwarg: 'windowsize'. Other kwargs can be passed in using a dict
    provided in smoothfunckwargs. By default, this uses a Savitsky-Golay filter,
    a Gaussian filter is also provided but not used. Another good option would
    be an actual low-pass filter (generated using scipy.signal?) to remove all
    high frequency noise from the ACF.

    magarefluxes is True if the measurements provided in mags are actually
    fluxes, False otherwise.

    sigclip is the sigma to use when sigma-clipping the magnitude time series.


    Returns
    -------

    Returns a dictionary with results. dict['bestperiod'] is the estimated best
    period and dict['fitperiodrms'] is its estimated error. Other interesting
    things in the output include:

    - dict['acfresults']: all results from calculating the ACF. in particular,
      the unsmoothed ACF might be of interest: dict['acfresults']['acf'] and
      dict['acfresults']['lags'].

    - dict['lags'] and dict['acf'] contain the ACF after smoothing was applied.

    - dict['periods'] and dict['lspvals'] can be used to construct a
      pseudo-periodogram.

    - dict['naivebestperiod'] is obtained by multiplying the lag at the highest
      ACF peak with the cadence. This is usually close to the fit period
      (dict['fitbestperiod']), which is calculated by doing a fit to the lags
      vs. peak index relation as in McQuillan+ 2014.

    '''

    # get the ACF
    acfres = autocorr_magseries(
        times,
        mags,
        errs,
        maxlags=maxlags,
        fillgaps=fillgaps,
        forcetimebin=forcetimebin,
        sigclip=sigclip,
        magsarefluxes=magsarefluxes,
        filterwindow=filterwindow,
        verbose=verbose
    )

    xlags = acfres['lags']

    # smooth the ACF if requested
    if smoothacf and isinstance(smoothacf, int) and smoothacf > 0:

        sfkwargs = smoothfunckwargs.copy()

        sfkwargs.update({'windowsize':smoothacf})
        xacf = smoothfunc(acfres['acf'], **sfkwargs)

    else:

        xacf = acfres['acf']


    # get the relative peak heights and fit best lag
    peakres = _get_acf_peakheights(xlags, xacf, npeaks=maxacfpeaks,
                                   searchinterval=int(smoothacf/2))

    # this is the best period's best ACF peak height
    bestlspval = peakres['bestpeakheight']

    try:

        # get the fit best lag from a linear fit to the peak index vs time(peak
        # lag) function as in McQillian+ (2014)
        fity = np.concatenate((
            [0.0, peakres['bestlag']],
            peakres['relpeaklags'][peakres['relpeaklags'] > peakres['bestlag']]
        ))
        fity = fity*acfres['cadence']
        fitx = np.arange(fity.size)

        fitcoeffs, fitcovar = np.polyfit(fitx, fity, 1, cov=True)

        # fit best period is the gradient of fit
        fitbestperiod = fitcoeffs[0]
        bestperiodrms = np.sqrt(fitcovar[0,0]) # from the covariance matrix

    except:

        LOGWARNING('linear fit to time at each peak lag '
                   'value vs. peak number failed, '
                   'naively calculated ACF period may not be accurate')
        fitcoeffs = np.array([np.nan, np.nan])
        fitcovar = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        fitbestperiod = np.nan
        bestperiodrms = np.nan
        raise

    # calculate the naive best period using delta_tau = lag * cadence
    naivebestperiod = peakres['bestlag']*acfres['cadence']

    if fitbestperiod < naivebestperiod:
        LOGWARNING('fit bestperiod = %.5f may be an alias, '
                   'naively calculated bestperiod is = %.5f' %
                   (fitbestperiod, naivebestperiod))

    if np.isfinite(fitbestperiod):
        bestperiod = fitbestperiod
    else:
        bestperiod = naivebestperiod


    return {'bestperiod':bestperiod,
            'bestlspval':bestlspval,
            'nbestpeaks':maxacfpeaks,
            'lspvals':xacf,
            'periods':xlags*acfres['cadence'],
            'acf':xacf,
            'lags':xlags,
            'method':'acf',
            'naivebestperiod':naivebestperiod,
            'fitbestperiod':fitbestperiod,
            'fitperiodrms':bestperiodrms,
            'periodfitcoeffs':fitcoeffs,
            'periodfitcovar':fitcovar,
            'kwargs':{'maxlags':maxlags,
                      'maxacfpeaks':maxacfpeaks,
                      'fillgaps':fillgaps,
                      'filterwindow':filterwindow,
                      'smoothacf':smoothacf,
                      'smoothfunckwargs':sfkwargs,
                      'magsarefluxes':magsarefluxes,
                      'sigclip':sigclip},
            'acfresults':acfres,
            'acfpeaks':peakres}
