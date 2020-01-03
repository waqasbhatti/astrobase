#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generation - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
# License: MIT. See the LICENSE file for more details.

'''This generates light curves of variable stars using the astrobase.lcmodels
package, adds noise and observation sampling to them based on given parameters
(or example light curves). See fakelcrecovery.py for functions that run a full
recovery simulation.

NOTE 1: the parameters for these light curves are currently all chosen from
uniform distributions, which obviously doesn't reflect reality (some of the
current parameter upper and lower limits are realistic, however). Some of these
will be updated with real-life distributions as soon as I find them, especially
for periods and amplitudes (along with references).

NOTE 2: To generate custom distributions, one can subclass
scipy.stats.rv_continuous and override the _pdf and _cdf methods (or just the
_rvs method directly to get the distributed variables if distribution's location
and scale don't really matter). This is described here:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html

and doesn't seem to be restricted to distributions described by analytic
functions only. It's probably possible to get a histogram of some complex
distribution in parameter space and use a kernel-density estimator to get the
PDF and CDF, e.g.:

http://scikit-learn.org/stable/modules/density.html#kernel-density.

NOTE 3: any distribution parameter below having to do with magnitudes/flux is by
default in MAGNITUDES. So depth, amplitude, etc. distributions and their limits
will have to be adjusted appropriately for fluxes. IT IS NOT SUFFICIENT to just
set magsarefluxes = True.

FIXME: in the future, we'll do all the amplitude, etc. distributions in
differential fluxes canonically, and then take logs where appropriate if
magsarefluxes = False.

FIXME: we should add RA/DEC values that are taken from GAIA if we provide a
radec box for the simulation to take place in. in this way, we can parameterize
blending and take it into account in the recovery as well.

FIXME: check if object coordinates end up so that two or more objects lie within
a chosen blend radius. if this happens, we should check if the blender(s) are
variable, and add in some fraction of their phased light curve to the
blendee. if the blender(s) are not variable, add in a constant fraction of the
brightness to the blendee's light curve. the blending fraction is multiplied
into the light curve of the blender(s) and the resulting flux added to the
blendee's light curve.

- given the FWHM of the instrument, figure out the overlap

- we need to calculate the pixel area for blendee and the sum of pixel areas
  covered by the blenders. this will require input kwargs for pixel size of the
  detector and FWHM of the star (this might need to be calculated based on the
  brightness of the star)

FIXME: add input from TRILEGAL produced .dat files for color and mag
information. This will let us generate pulsating variables with their actual
colors.

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

import os
import os.path
import pickle
import shutil

from hashlib import md5, sha512

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem


def _dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)


import numpy as np
import numpy.random as npr

# seed the numpy random generator
# we'll use RANDSEED for scipy.stats distribution functions as well
RANDSEED = 0xdecaff
npr.seed(RANDSEED)

import scipy.stats as sps
import scipy.interpolate as spi


###################
## LOCAL IMPORTS ##
###################

# light curve models
from ..lcmodels import transits, eclipses, flares, sinusoidal

# magnitude conversion
from ..magnitudes import jhk_to_sdssr

# get the lcformat functions
from ..lcproc import get_lcformat, _read_pklc


#############################################
## FUNCTIONS TO GENERATE FAKE LIGHT CURVES ##
#############################################

def generate_transit_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={'transitperiod':sps.uniform(loc=0.1,scale=49.9),
                    'transitdepth':sps.uniform(loc=1.0e-4,scale=2.0e-2),
                    'transitduration':sps.uniform(loc=0.01,scale=0.29)},
        magsarefluxes=False,
):
    '''This generates fake planet transit light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'transitperiod', 'transitdepth', 'transitduration'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The ingress duration will be automatically chosen from a uniform
        distribution ranging from 0.05 to 0.5 of the transitduration.

        The transitdepth will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'planet',
             'params': {'transitperiod': generated value of period,
                        'transitepoch': generated value of epoch,
                        'transitdepth': generated value of transit depth,
                        'transitduration': generated value of transit duration,
                        'ingressduration': generated value of transit ingress
                                           duration},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'transitperiod'
             'varamplitude': the generated amplitude of
                             variability == 'transitdepth'}

    '''

    if mags is None:
        mags = np.full_like(times, 0.0)

    if errs is None:
        errs = np.full_like(times, 0.0)

    # choose the epoch
    epoch = npr.random()*(times.max() - times.min()) + times.min()

    # choose the period, depth, duration
    period = paramdists['transitperiod'].rvs(size=1)
    depth = paramdists['transitdepth'].rvs(size=1)
    duration = paramdists['transitduration'].rvs(size=1)

    # figure out the ingress duration
    ingduration = npr.random()*(0.5*duration - 0.05*duration) + 0.05*duration

    # fix the transit depth if it needs to be flipped
    if magsarefluxes and depth < 0.0:
        depth = -depth
    elif not magsarefluxes and depth > 0.0:
        depth = -depth

    # generate the model
    modelmags, phase, ptimes, pmags, perrs = (
        transits.trapezoid_transit_func([period, epoch, depth,
                                         duration, ingduration],
                                        times,
                                        mags,
                                        errs)
    )

    # resort in original time order
    timeind = np.argsort(ptimes)
    mtimes = ptimes[timeind]
    mmags = modelmags[timeind]
    merrs = perrs[timeind]

    # return a dict with everything
    modeldict = {
        'vartype':'planet',
        'params':{x:np.asscalar(y) for x,y in zip(['transitperiod',
                                                   'transitepoch',
                                                   'transitdepth',
                                                   'transitduration',
                                                   'ingressduration'],
                                                  [period,
                                                   epoch,
                                                   depth,
                                                   duration,
                                                   ingduration])},
        'times':mtimes,
        'mags':mmags,
        'errs':merrs,
        # these are standard keys that help with later characterization of
        # variability as a function period, variability amplitude, object mag,
        # ndet, etc.
        'varperiod':period,
        'varamplitude':depth
    }

    return modeldict


def generate_eb_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={'period':sps.uniform(loc=0.2,scale=99.8),
                    'pdepth':sps.uniform(loc=1.0e-4,scale=0.7),
                    'pduration':sps.uniform(loc=0.01,scale=0.44),
                    'depthratio':sps.uniform(loc=0.01,scale=0.99),
                    'secphase':sps.norm(loc=0.5,scale=0.1)},
        magsarefluxes=False,
):
    '''This generates fake EB light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'period', 'pdepth', 'pduration', 'depthratio', 'secphase'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The `pdepth` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'EB',
             'params': {'period': generated value of period,
                        'epoch': generated value of epoch,
                        'pdepth': generated value of priary eclipse depth,
                        'pduration': generated value of prim eclipse duration,
                        'depthratio': generated value of prim/sec eclipse
                                      depth ratio},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'period'
             'varamplitude': the generated amplitude of
                             variability == 'pdepth'}

    '''

    if mags is None:
        mags = np.full_like(times, 0.0)

    if errs is None:
        errs = np.full_like(times, 0.0)

    # choose the epoch
    epoch = npr.random()*(times.max() - times.min()) + times.min()

    # choose the period, pdepth, duration, depthratio
    period = paramdists['period'].rvs(size=1)
    pdepth = paramdists['pdepth'].rvs(size=1)
    pduration = paramdists['pduration'].rvs(size=1)
    depthratio = paramdists['depthratio'].rvs(size=1)
    secphase = paramdists['secphase'].rvs(size=1)

    # fix the transit depth if it needs to be flipped
    if magsarefluxes and pdepth < 0.0:
        pdepth = -pdepth
    elif not magsarefluxes and pdepth > 0.0:
        pdepth = -pdepth

    # generate the model
    modelmags, phase, ptimes, pmags, perrs = (
        eclipses.invgauss_eclipses_func([period, epoch, pdepth,
                                         pduration, depthratio, secphase],
                                        times,
                                        mags,
                                        errs)
    )

    # resort in original time order
    timeind = np.argsort(ptimes)
    mtimes = ptimes[timeind]
    mmags = modelmags[timeind]
    merrs = perrs[timeind]

    # return a dict with everything
    modeldict = {
        'vartype':'EB',
        'params':{x:np.asscalar(y) for x,y in zip(['period',
                                                   'epoch',
                                                   'pdepth',
                                                   'pduration',
                                                   'depthratio'],
                                                  [period,
                                                   epoch,
                                                   pdepth,
                                                   pduration,
                                                   depthratio])},
        'times':mtimes,
        'mags':mmags,
        'errs':merrs,
        'varperiod':period,
        'varamplitude':pdepth,
    }

    return modeldict


def generate_flare_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={
            # flare peak amplitude from 0.01 mag to 1.0 mag above median.  this
            # is tuned for redder bands, flares are much stronger in bluer
            # bands, so tune appropriately for your situation.
            'amplitude':sps.uniform(loc=0.01,scale=0.99),
            # up to 5 flares per LC and at least 1
            'nflares':[1,5],
            # 10 minutes to 1 hour for rise stdev
            'risestdev':sps.uniform(loc=0.007, scale=0.04),
            # 1 hour to 4 hours for decay time constant
            'decayconst':sps.uniform(loc=0.04, scale=0.163)
        },
        magsarefluxes=False,
):
    '''This generates fake flare light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'amplitude', 'nflares', 'risestdev', 'decayconst'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The `flare_peak_time` for each flare will be generated automatically
        between `times.min()` and `times.max()` using a uniform distribution.

        The `amplitude` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'flare',
             'params': {'amplitude': generated value of flare amplitudes,
                        'nflares': generated value of number of flares,
                        'risestdev': generated value of stdev of rise time,
                        'decayconst': generated value of decay constant,
                        'peaktime': generated value of flare peak time},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varamplitude': the generated amplitude of
                             variability == 'amplitude'}

    '''

    if mags is None:
        mags = np.full_like(times, 0.0)

    if errs is None:
        errs = np.full_like(times, 0.0)

    nflares = npr.randint(paramdists['nflares'][0],
                          high=paramdists['nflares'][1])

    # generate random flare peak times based on the number of flares
    flarepeaktimes = (
        npr.random(
            size=nflares
        )*(times.max() - times.min()) + times.min()
    )

    # now add the flares to the time-series
    params = {'nflares':nflares}

    for flareind, peaktime in zip(range(nflares), flarepeaktimes):

        # choose the amplitude, rise stdev and decay time constant
        amp = paramdists['amplitude'].rvs(size=1)
        risestdev = paramdists['risestdev'].rvs(size=1)
        decayconst = paramdists['decayconst'].rvs(size=1)

        # fix the transit depth if it needs to be flipped
        if magsarefluxes and amp < 0.0:
            amp = -amp
        elif not magsarefluxes and amp > 0.0:
            amp = -amp

        # add this flare to the light curve
        modelmags, ptimes, pmags, perrs = (
            flares.flare_model(
                [amp, peaktime, risestdev, decayconst],
                times,
                mags,
                errs
            )
        )

        # update the mags
        mags = modelmags

        # add the flare params to the modeldict
        params[flareind] = {'peaktime':peaktime,
                            'amplitude':amp,
                            'risestdev':risestdev,
                            'decayconst':decayconst}

    #
    # done with all flares
    #

    # return a dict with everything
    modeldict = {
        'vartype':'flare',
        'params':params,
        'times':times,
        'mags':mags,
        'errs':errs,
        'varperiod':None,
        # FIXME: this is complicated because we can have multiple flares
        # figure out a good way to handle this upstream
        'varamplitude':[params[x]['amplitude']
                        for x in range(params['nflares'])],
    }

    return modeldict


def generate_sinusoidal_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={
            'period':sps.uniform(loc=0.04,scale=500.0),
            'fourierorder':[2,10],
            'amplitude':sps.uniform(loc=0.1,scale=0.9),
            'phioffset':0.0,
        },
        magsarefluxes=False
):
    '''This generates fake sinusoidal light curves.

    This can be used for a variety of sinusoidal variables, e.g. RRab, RRc,
    Cepheids, Miras, etc. The functions that generate these model LCs below
    implement the following table::

        ## FOURIER PARAMS FOR SINUSOIDAL VARIABLES
        #
        # type        fourier           period [days]
        #             order    dist     limits         dist

        # RRab        8 to 10  uniform  0.45--0.80     uniform
        # RRc         3 to 6   uniform  0.10--0.40     uniform
        # HADS        7 to 9   uniform  0.04--0.10     uniform
        # rotator     2 to 5   uniform  0.80--120.0    uniform
        # LPV         2 to 5   uniform  250--500.0     uniform

    FIXME: for better model LCs, figure out how scipy.signal.butter works and
    low-pass filter using scipy.signal.filtfilt.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'period', 'fourierorder', 'amplitude', 'phioffset'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The `amplitude` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'sinusoidal',
             'params': {'period': generated value of period,
                        'epoch': generated value of epoch,
                        'amplitude': generated value of amplitude,
                        'fourierorder': generated value of fourier order,
                        'fourieramps': generated values of fourier amplitudes,
                        'fourierphases': generated values of fourier phases},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'period'
             'varamplitude': the generated amplitude of
                             variability == 'amplitude'}

    '''

    if mags is None:
        mags = np.full_like(times, 0.0)

    if errs is None:
        errs = np.full_like(times, 0.0)

    # choose the epoch
    epoch = npr.random()*(times.max() - times.min()) + times.min()

    # choose the period, fourierorder, and amplitude
    period = paramdists['period'].rvs(size=1)
    fourierorder = npr.randint(paramdists['fourierorder'][0],
                               high=paramdists['fourierorder'][1])
    amplitude = paramdists['amplitude'].rvs(size=1)

    # fix the amplitude if it needs to be flipped
    if magsarefluxes and amplitude < 0.0:
        amplitude = -amplitude
    elif not magsarefluxes and amplitude > 0.0:
        amplitude = -amplitude

    # generate the amplitudes and phases of the Fourier components
    ampcomps = [abs(amplitude/2.0)/float(x)
                for x in range(1,fourierorder+1)]
    phacomps = [paramdists['phioffset']*float(x)
                for x in range(1,fourierorder+1)]

    # now that we have our amp and pha components, generate the light curve
    modelmags, phase, ptimes, pmags, perrs = sinusoidal.sine_series_sum(
        [period, epoch, ampcomps, phacomps],
        times,
        mags,
        errs
    )

    # resort in original time order
    timeind = np.argsort(ptimes)
    mtimes = ptimes[timeind]
    mmags = modelmags[timeind]
    merrs = perrs[timeind]
    mphase = phase[timeind]

    # return a dict with everything
    modeldict = {
        'vartype':'sinusoidal',
        'params':{x:y for x,y in zip(['period',
                                      'epoch',
                                      'amplitude',
                                      'fourierorder',
                                      'fourieramps',
                                      'fourierphases'],
                                     [period,
                                      epoch,
                                      amplitude,
                                      fourierorder,
                                      ampcomps,
                                      phacomps])},
        'times':mtimes,
        'mags':mmags,
        'errs':merrs,
        'phase':mphase,
        # these are standard keys that help with later characterization of
        # variability as a function period, variability amplitude, object mag,
        # ndet, etc.
        'varperiod':period,
        'varamplitude':amplitude
    }

    return modeldict


def generate_rrab_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={
            'period':sps.uniform(loc=0.45,scale=0.35),
            'fourierorder':[8,11],
            'amplitude':sps.uniform(loc=0.4,scale=0.5),
            'phioffset':np.pi,
        },
        magsarefluxes=False
):
    '''This generates fake RRab light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'period', 'fourierorder', 'amplitude'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The `amplitude` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'RRab',
             'params': {'period': generated value of period,
                        'epoch': generated value of epoch,
                        'amplitude': generated value of amplitude,
                        'fourierorder': generated value of fourier order,
                        'fourieramps': generated values of fourier amplitudes,
                        'fourierphases': generated values of fourier phases},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'period'
             'varamplitude': the generated amplitude of
                             variability == 'amplitude'}

    '''

    modeldict = generate_sinusoidal_lightcurve(times,
                                               mags=mags,
                                               errs=errs,
                                               paramdists=paramdists,
                                               magsarefluxes=magsarefluxes)
    modeldict['vartype'] = 'RRab'
    return modeldict


def generate_rrc_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={
            'period':sps.uniform(loc=0.10,scale=0.30),
            'fourierorder':[2,3],
            'amplitude':sps.uniform(loc=0.1,scale=0.3),
            'phioffset':1.5*np.pi,
        },
        magsarefluxes=False
):
    '''This generates fake RRc light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'period', 'fourierorder', 'amplitude'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The `amplitude` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'RRc',
             'params': {'period': generated value of period,
                        'epoch': generated value of epoch,
                        'amplitude': generated value of amplitude,
                        'fourierorder': generated value of fourier order,
                        'fourieramps': generated values of fourier amplitudes,
                        'fourierphases': generated values of fourier phases},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'period'
             'varamplitude': the generated amplitude of
                             variability == 'amplitude'}

    '''

    modeldict = generate_sinusoidal_lightcurve(times,
                                               mags=mags,
                                               errs=errs,
                                               paramdists=paramdists,
                                               magsarefluxes=magsarefluxes)
    modeldict['vartype'] = 'RRc'
    return modeldict


def generate_hads_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={
            'period':sps.uniform(loc=0.04,scale=0.06),
            'fourierorder':[5,10],
            'amplitude':sps.uniform(loc=0.1,scale=0.6),
            'phioffset':np.pi,
        },
        magsarefluxes=False
):
    '''This generates fake HADS light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'period', 'fourierorder', 'amplitude'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The `amplitude` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'HADS',
             'params': {'period': generated value of period,
                        'epoch': generated value of epoch,
                        'amplitude': generated value of amplitude,
                        'fourierorder': generated value of fourier order,
                        'fourieramps': generated values of fourier amplitudes,
                        'fourierphases': generated values of fourier phases},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'period'
             'varamplitude': the generated amplitude of
                             variability == 'amplitude'}

    '''

    modeldict = generate_sinusoidal_lightcurve(times,
                                               mags=mags,
                                               errs=errs,
                                               paramdists=paramdists,
                                               magsarefluxes=magsarefluxes)
    modeldict['vartype'] = 'HADS'
    return modeldict


def generate_rotator_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={
            'period':sps.uniform(loc=0.80,scale=119.20),
            'fourierorder':[2,3],
            'amplitude':sps.uniform(loc=0.01,scale=0.7),
            'phioffset':1.5*np.pi,
        },
        magsarefluxes=False
):
    '''This generates fake rotator light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'period', 'fourierorder', 'amplitude'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The `amplitude` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'rotator',
             'params': {'period': generated value of period,
                        'epoch': generated value of epoch,
                        'amplitude': generated value of amplitude,
                        'fourierorder': generated value of fourier order,
                        'fourieramps': generated values of fourier amplitudes,
                        'fourierphases': generated values of fourier phases},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'period'
             'varamplitude': the generated amplitude of
                             variability == 'amplitude'}

    '''

    modeldict = generate_sinusoidal_lightcurve(times,
                                               mags=mags,
                                               errs=errs,
                                               paramdists=paramdists,
                                               magsarefluxes=magsarefluxes)
    modeldict['vartype'] = 'rotator'
    return modeldict


def generate_lpv_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={
            'period':sps.uniform(loc=250.0,scale=250.0),
            'fourierorder':[2,3],
            'amplitude':sps.uniform(loc=0.1,scale=0.8),
            'phioffset':1.5*np.pi,
        },
        magsarefluxes=False
):
    '''This generates fake long-period-variable (LPV) light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'period', 'fourierorder', 'amplitude'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The `amplitude` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'LPV',
             'params': {'period': generated value of period,
                        'epoch': generated value of epoch,
                        'amplitude': generated value of amplitude,
                        'fourierorder': generated value of fourier order,
                        'fourieramps': generated values of fourier amplitudes,
                        'fourierphases': generated values of fourier phases},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'period'
             'varamplitude': the generated amplitude of
                             variability == 'amplitude'}

    '''

    modeldict = generate_sinusoidal_lightcurve(times,
                                               mags=mags,
                                               errs=errs,
                                               paramdists=paramdists,
                                               magsarefluxes=magsarefluxes)
    modeldict['vartype'] = 'LPV'
    return modeldict


def generate_cepheid_lightcurve(
        times,
        mags=None,
        errs=None,
        paramdists={
            'period':sps.uniform(loc=1.5,scale=108.5),
            'fourierorder':[8,11],
            'amplitude':sps.uniform(loc=0.1,scale=0.9),
            'phioffset':np.pi,
        },
        magsarefluxes=False
):
    '''This generates fake Cepheid light curves.

    Parameters
    ----------

    times : np.array
        This is an array of time values that will be used as the time base.

    mags,errs : np.array
        These arrays will have the model added to them. If either is
        None, `np.full_like(times, 0.0)` will used as a substitute and the model
        light curve will be centered around 0.0.

    paramdists : dict
        This is a dict containing parameter distributions to use for the
        model params, containing the following keys ::

            {'period', 'fourierorder', 'amplitude'}

        The values of these keys should all be 'frozen' scipy.stats distribution
        objects, e.g.:

        https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        The variability epoch will be automatically chosen from a uniform
        distribution between `times.min()` and `times.max()`.

        The `amplitude` will be flipped automatically as appropriate if
        `magsarefluxes=True`.

    magsarefluxes : bool
        If the generated time series is meant to be a flux time-series, set this
        to True to get the correct sign of variability amplitude.

    Returns
    -------

    dict
        A dict of the form below is returned::

            {'vartype': 'cepheid',
             'params': {'period': generated value of period,
                        'epoch': generated value of epoch,
                        'amplitude': generated value of amplitude,
                        'fourierorder': generated value of fourier order,
                        'fourieramps': generated values of fourier amplitudes,
                        'fourierphases': generated values of fourier phases},
             'times': the model times,
             'mags': the model mags,
             'errs': the model errs,
             'varperiod': the generated period of variability == 'period'
             'varamplitude': the generated amplitude of
                             variability == 'amplitude'}

    '''

    modeldict = generate_sinusoidal_lightcurve(times,
                                               mags=mags,
                                               errs=errs,
                                               paramdists=paramdists,
                                               magsarefluxes=magsarefluxes)
    modeldict['vartype'] = 'cepheid'
    return modeldict


# this maps functions to generate light curves to their vartype codes as put
# into the make_fakelc_collection function.
VARTYPE_LCGEN_MAP = {
    'EB': generate_eb_lightcurve,
    'RRab': generate_rrab_lightcurve,
    'RRc': generate_rrc_lightcurve,
    'rotator': generate_rotator_lightcurve,
    'flare': generate_flare_lightcurve,
    'HADS': generate_hads_lightcurve,
    'planet': generate_transit_lightcurve,
    'LPV': generate_lpv_lightcurve,
    'cepheid':generate_cepheid_lightcurve,
}


###############################################
## FUNCTIONS TO COLLECT LIGHT CURVES FOR SIM ##
###############################################

def make_fakelc(lcfile,
                outdir,
                magrms=None,
                randomizemags=True,
                randomizecoords=False,
                lcformat='hat-sql',
                lcformatdir=None,
                timecols=None,
                magcols=None,
                errcols=None):
    '''This preprocesses an input real LC and sets it up to be a fake LC.

    Parameters
    ----------

    lcfile : str
        This is an input light curve file that will be used to copy over the
        time-base. This will be used to generate the time-base for fake light
        curves to provide a realistic simulation of the observing window
        function.

    outdir : str
        The output directory where the the fake light curve will be written.

    magrms : dict
        This is a dict containing the SDSS r mag-RMS (SDSS rmag-MAD preferably)
        relation based on all light curves that the input lcfile is from. This
        will be used to generate the median mag and noise corresponding to the
        magnitude chosen for this fake LC.

    randomizemags : bool
        If this is True, then a random mag between the first and last magbin in
        magrms will be chosen as the median mag for this light curve. This
        choice will be weighted by the mag bin probability obtained from the
        magrms kwarg. Otherwise, the median mag will be taken from the input
        lcfile's lcdict['objectinfo']['sdssr'] key or a transformed SDSS r mag
        generated from the input lcfile's lcdict['objectinfo']['jmag'],
        ['hmag'], and ['kmag'] keys. The magrms relation for each magcol will be
        used to generate Gaussian noise at the correct level for the magbin this
        light curve's median mag falls into.

    randomizecoords : bool
        If this is True, will randomize the RA, DEC of the output fake object
        and not copy over the RA/DEC from the real input object.

    lcformat : str
        This is the `formatkey` associated with your input real light curve
        format, which you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curve specified in `lcfile`.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    timecols : list of str or None
        The timecol keys to use from the input lcdict in generating the fake
        light curve. Fake LCs will be generated for each each
        timecol/magcol/errcol combination in the input light curve.

    magcols : list of str or None
        The magcol keys to use from the input lcdict in generating the fake
        light curve. Fake LCs will be generated for each each
        timecol/magcol/errcol combination in the input light curve.

    errcols : list of str or None
        The errcol keys to use from the input lcdict in generating the fake
        light curve. Fake LCs will be generated for each each
        timecol/magcol/errcol combination in the input light curve.

    Returns
    -------

    tuple
        A tuple of the following form is returned::

            (fakelc_fpath,
             fakelc_lcdict['columns'],
             fakelc_lcdict['objectinfo'],
             fakelc_lcdict['moments'])

    '''

    try:
        formatinfo = get_lcformat(lcformat,
                                  use_lcformat_dir=lcformatdir)
        if formatinfo:
            (fileglob, readerfunc,
             dtimecols, dmagcols, derrcols,
             magsarefluxes, normfunc) = formatinfo
        else:
            LOGERROR("can't figure out the light curve format")
            return None
    except Exception:
        LOGEXCEPTION("can't figure out the light curve format")
        return None

    # override the default timecols, magcols, and errcols
    # using the ones provided to the function
    if timecols is None:
        timecols = dtimecols
    if magcols is None:
        magcols = dmagcols
    if errcols is None:
        errcols = derrcols

    # read in the light curve
    lcdict = readerfunc(lcfile)
    if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
        lcdict = lcdict[0]

    # set up the fakelcdict with a randomly assigned objectid
    fakeobjectid = sha512(npr.bytes(12)).hexdigest()[-8:]
    fakelcdict = {
        'objectid':fakeobjectid,
        'objectinfo':{'objectid':fakeobjectid},
        'columns':[],
        'moments':{},
        'origformat':lcformat,
    }

    # now, get the actual mag of this object and other info and use that to
    # populate the corresponding entries of the fakelcdict objectinfo
    if ('objectinfo' in lcdict and
        isinstance(lcdict['objectinfo'], dict)):

        objectinfo = lcdict['objectinfo']

        # get the RA
        if (not randomizecoords and 'ra' in objectinfo and
            objectinfo['ra'] is not None and
            np.isfinite(objectinfo['ra'])):

            fakelcdict['objectinfo']['ra'] = objectinfo['ra']

        else:

            # if there's no RA available, we'll assign a random one between 0
            # and 360.0
            LOGWARNING('%s: assigning a random right ascension' % lcfile)
            fakelcdict['objectinfo']['ra'] = npr.random()*360.0

        # get the DEC
        if (not randomizecoords and 'decl' in objectinfo and
            objectinfo['decl'] is not None and
            np.isfinite(objectinfo['decl'])):

            fakelcdict['objectinfo']['decl'] = objectinfo['decl']

        else:

            # if there's no DECL available, we'll assign a random one between
            # -90.0 and +90.0
            LOGWARNING(' %s: assigning a random declination' % lcfile)
            fakelcdict['objectinfo']['decl'] = npr.random()*180.0 - 90.0

        # get the SDSS r mag for this object
        # this will be used for getting the eventual mag-RMS relation later
        if ((not randomizemags) and 'sdssr' in objectinfo and
            objectinfo['sdssr'] is not None and
            np.isfinite(objectinfo['sdssr'])):

            fakelcdict['objectinfo']['sdssr'] = objectinfo['sdssr']

        # if the SDSS r is unavailable, but we have J, H, K: use those to get
        # the SDSS r by using transformations
        elif ((not randomizemags) and ('jmag' in objectinfo and
                                       objectinfo['jmag'] is not None and
                                       np.isfinite(objectinfo['jmag'])) and
              ('hmag' in objectinfo and
               objectinfo['hmag'] is not None and
               np.isfinite(objectinfo['hmag'])) and
              ('kmag' in objectinfo and
               objectinfo['kmag'] is not None and
               np.isfinite(objectinfo['kmag']))):

            LOGWARNING('used JHK mags to generate an SDSS r mag for %s' %
                       lcfile)
            fakelcdict['objectinfo']['sdssr'] = jhk_to_sdssr(
                objectinfo['jmag'],
                objectinfo['hmag'],
                objectinfo['kmag']
            )

        # if there are no mags available or we're specically told to randomize
        # them, generate a random mag between 8 and 16.0
        elif randomizemags and magrms:

            LOGWARNING(' %s: assigning a random mag weighted by mag '
                       'bin probabilities' % lcfile)

            magbins = magrms[magcols[0]]['binned_sdssr_median']
            binprobs = magrms[magcols[0]]['magbin_probabilities']

            # this is the center of the magbin chosen
            magbincenter = npr.choice(magbins,size=1,p=binprobs)

            # in this magbin, choose between center and -+ 0.25 mag
            chosenmag = (
                npr.random()*((magbincenter+0.25) - (magbincenter-0.25)) +
                (magbincenter-0.25)
            )

            fakelcdict['objectinfo']['sdssr'] = np.asscalar(chosenmag)

        # if there are no mags available at all, generate a random mag
        # between 8 and 16.0
        else:

            LOGWARNING(' %s: assigning a random mag from '
                       'uniform distribution between 8.0 and 16.0' % lcfile)

            fakelcdict['objectinfo']['sdssr'] = npr.random()*8.0 + 8.0

    # if there's no info available, generate fake info
    else:

        LOGWARNING('no object information found in %s, '
                   'generating random ra, decl, sdssr' %
                   lcfile)
        fakelcdict['objectinfo']['ra'] = npr.random()*360.0
        fakelcdict['objectinfo']['decl'] = npr.random()*180.0 - 90.0
        fakelcdict['objectinfo']['sdssr'] = npr.random()*8.0 + 8.0

    #
    # NOW FILL IN THE TIMES, MAGS, ERRS
    #

    # get the time columns
    for tcind, tcol in enumerate(timecols):

        if '.' in tcol:
            tcolget = tcol.split('.')
        else:
            tcolget = [tcol]

        if tcol not in fakelcdict:
            fakelcdict[tcol] = _dict_get(lcdict, tcolget)
            fakelcdict['columns'].append(tcol)

            # update the ndet with the first time column's size. it's possible
            # that different time columns have different lengths, but that would
            # be weird and we won't deal with it for now
            if tcind == 0:
                fakelcdict['objectinfo']['ndet'] = fakelcdict[tcol].size

    # get the mag columns
    for mcol in magcols:

        if '.' in mcol:
            mcolget = mcol.split('.')
        else:
            mcolget = [mcol]

        # put the mcol in only once
        if mcol not in fakelcdict:

            measuredmags = _dict_get(lcdict, mcolget)
            measuredmags = measuredmags[np.isfinite(measuredmags)]

            # if we're randomizing, get the mags from the interpolated mag-RMS
            # relation
            if (randomizemags and magrms and mcol in magrms and
                'interpolated_magmad' in magrms[mcol] and
                magrms[mcol]['interpolated_magmad'] is not None):

                interpfunc = magrms[mcol]['interpolated_magmad']
                lcmad = interpfunc(fakelcdict['objectinfo']['sdssr'])

                fakelcdict['moments'][mcol] = {
                    'median': fakelcdict['objectinfo']['sdssr'],
                    'mad': lcmad
                }

            # if we're not randomizing, get the median and MAD from the light
            # curve itself
            else:

                # we require at least 10 finite measurements
                if measuredmags.size > 9:

                    measuredmedian = np.median(measuredmags)
                    measuredmad = np.median(
                        np.abs(measuredmags - measuredmedian)
                    )
                    fakelcdict['moments'][mcol] = {'median':measuredmedian,
                                                   'mad':measuredmad}

                # if there aren't enough measurements in this LC, try to get the
                # median and RMS from the interpolated mag-RMS relation first
                else:

                    if (magrms and mcol in magrms and
                        'interpolated_magmad' in magrms[mcol] and
                        magrms[mcol]['interpolated_magmad'] is not None):

                        LOGWARNING(
                            'input LC %s does not have enough '
                            'finite measurements, '
                            'generating mag moments from '
                            'fakelc sdssr and the mag-RMS relation' % lcfile
                        )

                        interpfunc = magrms[mcol]['interpolated_magmad']
                        lcmad = interpfunc(fakelcdict['objectinfo']['sdssr'])

                        fakelcdict['moments'][mcol] = {
                            'median': fakelcdict['objectinfo']['sdssr'],
                            'mad': lcmad
                        }

                    # if we don't have the mag-RMS relation either, then we
                    # can't do anything for this light curve, generate a random
                    # MAD between 5e-4 and 0.1
                    else:

                        LOGWARNING(
                            'input LC %s does not have enough '
                            'finite measurements and '
                            'no mag-RMS relation provided '
                            'assigning a random MAD between 5.0e-4 and 0.1'
                            % lcfile
                        )

                        fakelcdict['moments'][mcol] = {
                            'median':fakelcdict['objectinfo']['sdssr'],
                            'mad':npr.random()*(0.1 - 5.0e-4) + 5.0e-4
                        }

            # the magnitude column is set to all zeros initially. this will be
            # filled in by the add_fakelc_variability function below
            fakelcdict[mcol] = np.full_like(_dict_get(lcdict, mcolget), 0.0)
            fakelcdict['columns'].append(mcol)

    # get the err columns
    for mcol, ecol in zip(magcols, errcols):

        if '.' in ecol:
            ecolget = ecol.split('.')
        else:
            ecolget = [ecol]

        if ecol not in fakelcdict:

            measurederrs = _dict_get(lcdict, ecolget)
            measurederrs = measurederrs[np.isfinite(measurederrs)]

            # if we're randomizing, get the errs from the interpolated mag-RMS
            # relation
            if (randomizemags and magrms and mcol in magrms and
                'interpolated_magmad' in magrms[mcol] and
                magrms[mcol]['interpolated_magmad'] is not None):

                interpfunc = magrms[mcol]['interpolated_magmad']
                lcmad = interpfunc(fakelcdict['objectinfo']['sdssr'])

                # the median of the errs = lcmad
                # the mad of the errs is 0.1 x lcmad
                fakelcdict['moments'][ecol] = {
                    'median': lcmad,
                    'mad': 0.1*lcmad
                }

            else:

                # we require at least 10 finite measurements
                # we'll calculate the median and MAD of the errs to use later on
                if measurederrs.size > 9:
                    measuredmedian = np.median(measurederrs)
                    measuredmad = np.median(
                        np.abs(measurederrs - measuredmedian)
                    )
                    fakelcdict['moments'][ecol] = {'median':measuredmedian,
                                                   'mad':measuredmad}
                else:

                    if (magrms and mcol in magrms and
                        'interpolated_magmad' in magrms[mcol] and
                        magrms[mcol]['interpolated_magmad'] is not None):

                        LOGWARNING(
                            'input LC %s does not have enough '
                            'finite measurements, '
                            'generating err moments from '
                            'the mag-RMS relation' % lcfile
                        )

                        interpfunc = magrms[mcol]['interpolated_magmad']
                        lcmad = interpfunc(fakelcdict['objectinfo']['sdssr'])

                        fakelcdict['moments'][ecol] = {
                            'median': lcmad,
                            'mad': 0.1*lcmad
                        }

                    # if we don't have the mag-RMS relation either, then we
                    # can't do anything for this light curve, generate a random
                    # MAD between 5e-4 and 0.1
                    else:

                        LOGWARNING(
                            'input LC %s does not have '
                            'enough finite measurements and '
                            'no mag-RMS relation provided, '
                            'generating errs randomly' % lcfile
                        )
                        fakelcdict['moments'][ecol] = {
                            'median':npr.random()*(0.01 - 5.0e-4) + 5.0e-4,
                            'mad':npr.random()*(0.01 - 5.0e-4) + 5.0e-4
                        }

            # the errors column is set to all zeros initially. this will be
            # filled in by the add_fakelc_variability function below.
            fakelcdict[ecol] = np.full_like(_dict_get(lcdict, ecolget), 0.0)
            fakelcdict['columns'].append(ecol)

    # add the timecols, magcols, errcols to the lcdict
    fakelcdict['timecols'] = timecols
    fakelcdict['magcols'] = magcols
    fakelcdict['errcols'] = errcols

    # generate an output file name
    fakelcfname = '%s-fakelc.pkl' % fakelcdict['objectid']
    fakelcfpath = os.path.abspath(os.path.join(outdir, fakelcfname))

    # write this out to the output directory
    with open(fakelcfpath,'wb') as outfd:
        pickle.dump(fakelcdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    # return the fakelc path, its columns, info, and moments so we can put them
    # into a collection DB later on
    LOGINFO('real LC %s -> fake LC %s OK' % (lcfile, fakelcfpath))

    return (fakelcfpath, fakelcdict['columns'],
            fakelcdict['objectinfo'], fakelcdict['moments'])


##########################
## COLLECTION FUNCTIONS ##
##########################

def collection_worker(task):
    '''
    This wraps `process_fakelc` for `make_fakelc_collection` below.

    Parameters
    ----------

    task : tuple
        This is of the form::

            task[0] = lcfile
            task[1] = outdir
            task[2] = magrms
            task[3] = dict with keys: {'lcformat', 'timecols', 'magcols',
                                       'errcols', 'randomizeinfo'}

    Returns
    -------

    tuple
        This returns a tuple of the form::

            (fakelc_fpath,
             fakelc_lcdict['columns'],
             fakelc_lcdict['objectinfo'],
             fakelc_lcdict['moments'])
    '''

    lcfile, outdir, kwargs = task

    try:

        fakelcresults = make_fakelc(
            lcfile,
            outdir,
            **kwargs
        )

        return fakelcresults

    except Exception:

        LOGEXCEPTION('could not process %s into a fakelc' % lcfile)
        return None


def make_fakelc_collection(lclist,
                           simbasedir,
                           magrmsfrom,
                           magrms_interpolate='quadratic',
                           magrms_fillvalue='extrapolate',
                           maxlcs=25000,
                           maxvars=2000,
                           randomizemags=True,
                           randomizecoords=False,
                           vartypes=('EB','RRab','RRc','cepheid',
                                     'rotator','flare','HADS',
                                     'planet','LPV'),
                           lcformat='hat-sql',
                           lcformatdir=None,
                           timecols=None,
                           magcols=None,
                           errcols=None):

    '''This prepares light curves for the recovery sim.

    Collects light curves from `lclist` using a uniform sampling among
    them. Copies them to the `simbasedir`, zeroes out their mags and errs but
    keeps their time bases, also keeps their RMS and median mags for later
    use. Calculates the mag-rms relation for the entire collection and writes
    that to the `simbasedir` as well.

    The purpose of this function is to copy over the time base and mag-rms
    relation of an existing light curve collection to use it as the basis for a
    variability recovery simulation.

    This returns a pickle written to the `simbasedir` that contains all the
    information for the chosen ensemble of fake light curves and writes all
    generated light curves to the `simbasedir/lightcurves` directory. Run the
    `add_variability_to_fakelc_collection` function after this function to add
    variability of the specified type to these generated light curves.

    Parameters
    ----------

    lclist : list of str
        This is a list of existing project light curves. This can be generated
        from :py:func:`astrobase.lcproc.catalogs.make_lclist` or similar.

    simbasedir : str
        This is the directory to where the fake light curves and their
        information will be copied to.

    magrmsfrom : str or dict
        This is used to generate magnitudes and RMSes for the objects in the
        output collection of fake light curves. This arg is either a string
        pointing to an existing pickle file that must contain a dict or a dict
        variable that MUST have the following key-vals at a minimum::

            {'<magcol1_name>': {
                  'binned_sdssr_median': array of median mags for each magbin
                  'binned_lcmad_median': array of LC MAD values per magbin
             },
             '<magcol2_name>': {
                  'binned_sdssr_median': array of median mags for each magbin
                  'binned_lcmad_median': array of LC MAD values per magbin
             },
             .
             .
             ...}

        where `magcol1_name`, etc. are the same as the `magcols` listed in the
        magcols kwarg (or the default magcols for the specified
        lcformat). Examples of the magrmsfrom dict (or pickle) required can be
        generated by the
        :py:func:`astrobase.lcproc.varthreshold.variability_threshold` function.

    magrms_interpolate,magrms_fillvalue : str
        These are arguments that will be passed directly to the
        scipy.interpolate.interp1d function to generate interpolating functions
        for the mag-RMS relation. See:

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

        for details.

    maxlcs : int
        This is the total number of light curves to choose from `lclist` and
        generate as fake LCs.

    maxvars : int
        This is the total number of fake light curves that will be marked as
        variable.

    vartypes : list of str
        This is a list of variable types to put into the collection. The
        vartypes for each fake variable star will be chosen uniformly from this
        list.

    lcformat : str
        This is the `formatkey` associated with your input real light curves'
        format, which you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curves specified in `lclist`.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    timecols : list of str or None
        The timecol keys to use from the input lcdict in generating the fake
        light curve. Fake LCs will be generated for each each
        timecol/magcol/errcol combination in the input light curves.

    magcols : list of str or None
        The magcol keys to use from the input lcdict in generating the fake
        light curve. Fake LCs will be generated for each each
        timecol/magcol/errcol combination in the input light curves.

    errcols : list of str or None
        The errcol keys to use from the input lcdict in generating the fake
        light curve. Fake LCs will be generated for each each
        timecol/magcol/errcol combination in the input light curves.

    Returns
    -------

    str
        Returns the string file name of a pickle containing all of the
        information for the fake LC collection that has been generated.

    '''

    try:
        formatinfo = get_lcformat(lcformat,
                                  use_lcformat_dir=lcformatdir)
        if formatinfo:
            (fileglob, readerfunc,
             dtimecols, dmagcols, derrcols,
             magsarefluxes, normfunc) = formatinfo
        else:
            LOGERROR("can't figure out the light curve format")
            return None
    except Exception:
        LOGEXCEPTION("can't figure out the light curve format")
        return None

    # override the default timecols, magcols, and errcols
    # using the ones provided to the function
    if timecols is None:
        timecols = dtimecols
    if magcols is None:
        magcols = dmagcols
    if errcols is None:
        errcols = derrcols

    if not isinstance(lclist, np.ndarray):
        lclist = np.array(lclist)

    chosenlcs = npr.choice(lclist, maxlcs, replace=False)

    fakelcdir = os.path.join(simbasedir, 'lightcurves')
    if not os.path.exists(fakelcdir):
        os.makedirs(fakelcdir)

    # get the magrms relation needed from the pickle or input dict
    if isinstance(magrmsfrom, str) and os.path.exists(magrmsfrom):
        with open(magrmsfrom,'rb') as infd:
            xmagrms = pickle.load(infd)
    elif isinstance(magrmsfrom, dict):
        xmagrms = magrmsfrom

    magrms = {}

    # get the required items from the magrms dict. interpolate the mag-rms
    # relation for the magcol so the make_fake_lc function can use it directly.
    for magcol in magcols:

        if (magcol in xmagrms and
            'binned_sdssr_median' in xmagrms[magcol] and
            'binned_lcmad_median' in xmagrms[magcol]):

            magrms[magcol] = {
                'binned_sdssr_median':np.array(
                    xmagrms[magcol]['binned_sdssr_median']
                ),
                'binned_lcmad_median':np.array(
                    xmagrms[magcol]['binned_lcmad_median']
                ),
            }

            # interpolate the mag-MAD relation
            interpolated_magmad = spi.interp1d(
                xmagrms[magcol]['binned_sdssr_median'],
                xmagrms[magcol]['binned_lcmad_median'],
                kind=magrms_interpolate,
                fill_value=magrms_fillvalue,
            )

            # save the magrms
            magrms[magcol]['interpolated_magmad'] = interpolated_magmad

            # generate the probability distribution in magbins. this is needed
            # to correctly sample the objects in this population
            bincounts = np.array(xmagrms[magcol]['binned_count'])
            binprobs = bincounts/np.sum(bincounts)

            # save the bin probabilities as well
            magrms[magcol]['magbin_probabilities'] = binprobs

        else:

            LOGWARNING('input magrms dict does not have '
                       'required info for magcol: %s' % magcol)

            magrms[magcol] = {
                'binned_sdssr_median':None,
                'binned_lcmad_median':None,
                'interpolated_magmad':None,
                'magbin_probabilities':None,
            }

    tasks = [(x, fakelcdir, {'lcformat':lcformat,
                             'timecols':timecols,
                             'magcols':magcols,
                             'errcols':errcols,
                             'magrms':magrms,
                             'randomizemags':randomizemags,
                             'randomizecoords':randomizecoords})
             for x in chosenlcs]

    # we can't parallelize because it messes up the random number generation,
    # causing all the IDs to clash. FIXME: figure out a way around this
    # (probably initial a seed in each worker process?)
    fakeresults = [collection_worker(task) for task in tasks]

    fakedb = {'simbasedir':simbasedir,
              'lcformat':lcformat,
              'timecols':timecols,
              'magcols':magcols,
              'errcols':errcols,
              'magsarefluxes':magsarefluxes}

    fobjects, fpaths = [], []
    fras, fdecls, fndets = [], [], []

    fmags, fmagmads = [], []
    ferrmeds, ferrmads = [], []

    # these are the indices for the variable objects chosen randomly
    isvariableind = npr.randint(0,high=len(fakeresults), size=maxvars)
    isvariable = np.full(len(fakeresults), False, dtype=np.bool)
    isvariable[isvariableind] = True
    fakedb['isvariable'] = isvariable

    LOGINFO('added %s variable stars' % maxvars)

    # these are the variable types for each variable object
    vartypeind = npr.randint(0,high=len(vartypes), size=maxvars)
    vartypearr = np.array([vartypes[x] for x in vartypeind])
    fakedb['vartype'] = vartypearr

    for vt in sorted(vartypes):
        LOGINFO('%s: %s stars' % (vt, vartypearr[vartypearr == vt].size))

    # now go through the collection and get the mag/rms and err/rms for each
    # star. these will be used later to add noise to light curves
    LOGINFO('collecting info...')

    for fr in fakeresults:

        if fr is not None:

            fpath, fcols, finfo, fmoments = fr

            fobjects.append(finfo['objectid'])
            fpaths.append(fpath)

            fras.append(finfo['ra'])
            fdecls.append(finfo['decl'])
            fndets.append(finfo['ndet'])

            fmags.append(finfo['sdssr'])
            # this is per magcol
            fmagmads.append([fmoments[x]['mad'] for x in magcols])

            # these are per errcol
            ferrmeds.append([fmoments[x]['median'] for x in errcols])
            ferrmads.append([fmoments[x]['mad'] for x in errcols])

    # convert to nparrays
    fobjects = np.array(fobjects)
    fpaths = np.array(fpaths)

    fras = np.array(fras)
    fdecls = np.array(fdecls)
    fndets = np.array(fndets)

    fmags = np.array(fmags)
    fmagmads = np.array(fmagmads)
    ferrmeds = np.array(ferrmeds)
    ferrmads = np.array(ferrmads)

    # put these in the fakedb
    fakedb['objectid'] = fobjects
    fakedb['lcfpath'] = fpaths

    fakedb['ra'] = fras
    fakedb['decl'] = fdecls
    fakedb['ndet'] = fndets

    fakedb['sdssr'] = fmags
    fakedb['mad'] = fmagmads
    fakedb['errmedian'] = ferrmeds
    fakedb['errmad'] = ferrmads

    # get the mag-RMS curve for this light curve collection for each magcol
    fakedb['magrms'] = magrms

    # finally, write the collection DB to a pickle in simbasedir
    dboutfname = os.path.join(simbasedir,'fakelcs-info.pkl')
    with open(dboutfname, 'wb') as outfd:
        pickle.dump(fakedb, outfd)

    LOGINFO('wrote %s fake LCs to: %s' % (len(fakeresults), simbasedir))
    LOGINFO('fake LC info written to: %s' % dboutfname)

    return dboutfname


########################################################
## FUNCTIONS TO ADD VARIABLE LCS TO FAKELC COLLECTION ##
########################################################

def add_fakelc_variability(fakelcfile,
                           vartype,
                           override_paramdists=None,
                           magsarefluxes=False,
                           overwrite=False):
    '''This adds variability of the specified type to the fake LC.

    The procedure is (for each `magcol`):

    - read the fakelcfile, get the stored moments and vartype info

    - add the periodic variability specified in vartype and varparamdists. if
      `vartype == None`, then do nothing in this step. If `override_vartype` is
      not None, override stored vartype with specified vartype. If
      `override_varparamdists` provided, override with specified
      `varparamdists`. NOTE: the varparamdists must make sense for the vartype,
      otherwise, weird stuff will happen.

    - add the median mag level stored in `fakelcfile` to the time series

    - add Gaussian noise to the light curve as specified in `fakelcfile`

    - add a varinfo key and dict to the lcdict with `varperiod`, `varepoch`,
      `varparams`

    - write back to fake LC pickle

    - return the `varinfo` dict to the caller

    Parameters
    ----------

    fakelcfile : str
        The name of the fake LC file to process.

    vartype : str
        The type of variability to add to this fake LC file.

    override_paramdists : dict
        A parameter distribution dict as in the `generate_XX_lightcurve`
        functions above. If provided, will override the distribution stored in
        the input fake LC file itself.

    magsarefluxes : bool
        Sets if the variability amplitude is in fluxes and not magnitudes.

    overwite : bool
        This overwrites the input fake LC file with a new variable LC even if
        it's been processed before.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'objectid':lcdict['objectid'],
             'lcfname':fakelcfile,
             'actual_vartype':vartype,
             'actual_varparams':lcdict['actual_varparams']}

    '''

    # read in the fakelcfile
    lcdict = _read_pklc(fakelcfile)

    # make sure to bail out if this light curve already has fake variability
    # added
    if ('actual_vartype' in lcdict and
        'actual_varparams' in lcdict and
        not overwrite):
        LOGERROR('%s has existing variability type: %s '
                 'and params: %s and overwrite = False, '
                 'skipping this file...' %
                 (fakelcfile, lcdict['actual_vartype'],
                  repr(lcdict['actual_varparams'])))
        return None

    # get the times, mags, errs from this LC
    timecols, magcols, errcols = (lcdict['timecols'],
                                  lcdict['magcols'],
                                  lcdict['errcols'])

    # get the correct function to apply variability
    if vartype in VARTYPE_LCGEN_MAP:
        vargenfunc = VARTYPE_LCGEN_MAP[vartype]
    elif vartype is None:
        vargenfunc = None
    else:
        LOGERROR('unknown variability type: %s, choose from: %s' %
                 (vartype, repr(list(VARTYPE_LCGEN_MAP.keys()))))
        return None

    # 1. generate the variability, including the overrides if provided we do
    # this outside the loop below to get the period, etc. distributions once
    # only per object. NOTE: in doing so, we're assuming that the difference
    # between magcols is just additive and the timebases for each magcol are the
    # same; this is not strictly correct
    if vargenfunc is not None:
        if (override_paramdists is not None and
            isinstance(override_paramdists,dict)):

            variablelc = vargenfunc(lcdict[timecols[0]],
                                    paramdists=override_paramdists,
                                    magsarefluxes=magsarefluxes)

        else:

            variablelc = vargenfunc(lcdict[timecols[0]],
                                    magsarefluxes=magsarefluxes)

    # for nonvariables, don't execute vargenfunc, but return a similar dict
    # so we can add the required noise to it
    else:
        variablelc = {'vartype':None,
                      'params':None,
                      'times':lcdict[timecols[0]],
                      'mags':np.full_like(lcdict[timecols[0]], 0.0),
                      'errs':np.full_like(lcdict[timecols[0]], 0.0)}

    # now iterate over the time, mag, err columns
    for mcol, ecol in zip(magcols, errcols):

        # 2. get the moments for this magcol
        mag_median = lcdict['moments'][mcol]['median']
        mag_mad = lcdict['moments'][mcol]['mad']

        # add up to 5 mmag of extra RMS for systematics and red-noise
        mag_rms = mag_mad*1.483

        err_median = lcdict['moments'][ecol]['median']
        err_mad = lcdict['moments'][ecol]['mad']
        err_rms = err_mad*1.483

        # 3. add the median level + gaussian noise
        magnoise = npr.normal(size=variablelc['mags'].size)*mag_rms
        errnoise = npr.normal(size=variablelc['errs'].size)*err_rms

        finalmags = mag_median + (variablelc['mags'] + magnoise)
        finalerrs = err_median + (variablelc['errs'] + errnoise)

        # 4. update these tcol, mcol, ecol values in the lcdict
        lcdict[mcol] = finalmags
        lcdict[ecol] = finalerrs

    #
    # all done with updating mags and errs
    #

    # 5. update the light curve with the variability info
    lcdict['actual_vartype'] = variablelc['vartype']
    lcdict['actual_varparams'] = variablelc['params']

    # these standard keys are set to help out later with characterizing recovery
    # rates by magnitude, period, amplitude, ndet, etc.
    if vartype is not None:
        lcdict['actual_varperiod'] = variablelc['varperiod']
        lcdict['actual_varamplitude'] = variablelc['varamplitude']
    else:
        lcdict['actual_varperiod'] = np.nan
        lcdict['actual_varamplitude'] = np.nan

    # 6. write back, making sure to do it safely
    tempoutf = '%s.%s' % (fakelcfile, md5(npr.bytes(4)).hexdigest()[-8:])
    with open(tempoutf, 'wb') as outfd:
        pickle.dump(lcdict, outfd, pickle.HIGHEST_PROTOCOL)

    if os.path.exists(tempoutf):
        shutil.copy(tempoutf, fakelcfile)
        os.remove(tempoutf)
    else:
        LOGEXCEPTION('could not write output light curve file to dir: %s' %
                     os.path.dirname(tempoutf))
        # fail here
        raise

    LOGINFO('object: %s, vartype: %s -> %s OK' % (
        lcdict['objectid'],
        vartype,
        fakelcfile)
    )

    return {'objectid':lcdict['objectid'],
            'lcfname':fakelcfile,
            'actual_vartype':vartype,
            'actual_varparams':lcdict['actual_varparams']}


def add_variability_to_fakelc_collection(simbasedir,
                                         override_paramdists=None,
                                         overwrite_existingvar=False):
    '''This adds variability and noise to all fake LCs in `simbasedir`.

    If an object is marked as variable in the `fakelcs-info`.pkl file in
    `simbasedir`, a variable signal will be added to its light curve based on
    its selected type, default period and amplitude distribution, the
    appropriate params, etc. the epochs for each variable object will be chosen
    uniformly from its time-range (and may not necessarily fall on a actual
    observed time). Nonvariable objects will only have noise added as determined
    by their params, but no variable signal will be added.

    Parameters
    ----------

    simbasedir : str
        The directory containing the fake LCs to process.

    override_paramdists : dict
        This can be used to override the stored variable parameters in each fake
        LC. It should be a dict of the following form::

            {'<vartype1>': {'<param1>: a scipy.stats distribution function or
                                       the np.random.randint function,
                            .
                            .
                            .
                            '<paramN>: a scipy.stats distribution function
                                       or the np.random.randint function}

        for any vartype in VARTYPE_LCGEN_MAP. These are used to override the
        default parameter distributions for each variable type.

    overwrite_existingvar : bool
        If this is True, then will overwrite any existing variability in the
        input fake LCs in `simbasedir`.

    Returns
    -------

    dict
        This returns a dict containing the fake LC filenames as keys and
        variability info for each as values.

    '''

    # open the fakelcs-info.pkl
    infof = os.path.join(simbasedir,'fakelcs-info.pkl')
    with open(infof, 'rb') as infd:
        lcinfo = pickle.load(infd)

    lclist = lcinfo['lcfpath']
    varflag = lcinfo['isvariable']
    vartypes = lcinfo['vartype']

    vartind = 0

    varinfo = {}

    # go through all the LCs and add the required type of variability
    for lc, varf, _lcind in zip(lclist, varflag, range(len(lclist))):

        # if this object is variable, add variability
        if varf:

            thisvartype = vartypes[vartind]

            if (override_paramdists and
                isinstance(override_paramdists, dict) and
                thisvartype in override_paramdists and
                isinstance(override_paramdists[thisvartype], dict)):

                thisoverride_paramdists = override_paramdists[thisvartype]
            else:
                thisoverride_paramdists = None

            varlc = add_fakelc_variability(
                lc, thisvartype,
                override_paramdists=thisoverride_paramdists,
                overwrite=overwrite_existingvar
            )
            varinfo[varlc['objectid']] = {'params': varlc['actual_varparams'],
                                          'vartype': varlc['actual_vartype']}

            # update vartind
            vartind = vartind + 1

        else:

            varlc = add_fakelc_variability(
                lc, None,
                overwrite=overwrite_existingvar
            )
            varinfo[varlc['objectid']] = {'params': varlc['actual_varparams'],
                                          'vartype': varlc['actual_vartype']}
    #
    # done with all objects
    #

    # write the varinfo back to the dict and fakelcs-info.pkl
    lcinfo['varinfo'] = varinfo

    tempoutf = '%s.%s' % (infof, md5(npr.bytes(4)).hexdigest()[-8:])
    with open(tempoutf, 'wb') as outfd:
        pickle.dump(lcinfo, outfd, pickle.HIGHEST_PROTOCOL)

    if os.path.exists(tempoutf):
        shutil.copy(tempoutf, infof)
        os.remove(tempoutf)
    else:
        LOGEXCEPTION('could not write output light curve file to dir: %s' %
                     os.path.dirname(tempoutf))
        # fail here
        raise

    return lcinfo
