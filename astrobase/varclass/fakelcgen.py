#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''fakelcgen - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This generates light curves of variable stars using the astrobase.lcmodels
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

'''
import os
import os.path
import pickle
import shutil

import multiprocessing as mp
import logging
from datetime import datetime
from traceback import format_exc
from concurrent.futures import ProcessPoolExecutor
from hashlib import md5

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem
def dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)

import numpy as np
import numpy.random as npr

# seed the numpy random generator
# we'll use RANDSEED for scipy.stats distribution functions as well
RANDSEED = 0xdecaff
npr.seed(RANDSEED)

import scipy.stats as sps
import scipy.interpolate as spi
import scipy.signal as sig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.fakelcgen' % parent_name)

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

# LC reading functions
from astrobase.hatlc import read_and_filter_sqlitecurve, read_csvlc, \
    normalize_lcdict_byinst
from astrobase.hplc import read_hatpi_textlc, read_hatpi_pklc
from astrobase.astrokep import read_kepler_fitslc, read_kepler_pklc

from ..lcmodels import transits, eclipses, flares, sinusoidal
from ..varbase.features import all_nonperiodic_features

from ..magnitudes import jhk_to_sdssr


#######################
## LC FORMATS SET UP ##
#######################

def read_pklc(lcfile):
    '''
    This just reads a pickle.

    '''

    try:
        with open(lcfile,'rb') as infd:
            lcdict = pickle.load(infd)
    except UnicodeDecodeError:
        with open(lcfile,'rb') as infd:
            lcdict = pickle.load(infd, encoding='latin1')

    return lcdict


# LC format -> [default fileglob,  function to read LC format]
LCFORM = {
    'hat-sql':[
        '*-hatlc.sqlite*',           # default fileglob
        read_and_filter_sqlitecurve,   # function to read this LC
        ['rjd','rjd'],                 # default timecols to use for period/var
        ['aep_000','atf_000'],         # default magcols to use for period/var
        ['aie_000','aie_000'],         # default errcols to use for period/var
        False,                         # default magsarefluxes = False
        normalize_lcdict_byinst,       # default special normalize function
    ],
    'hat-csv':[
        '*-hatlc.csv*',
        read_csvlc,
        ['rjd','rjd'],
        ['aep_000','atf_000'],
        ['aie_000','aie_000'],
        False,
        normalize_lcdict_byinst,
    ],
    'hp-txt':[
        'HAT-*tfalc.TF1*',
        read_hatpi_textlc,
        ['rjd','rjd'],
        ['iep1','itf1'],
        ['ire1','ire1'],
        False,
        None,
    ],
    'hp-pkl':[
        '*-pklc.pkl',
        read_hatpi_pklc,
        ['rjd','rjd'],
        ['iep1','itf1'],
        ['ire1','ire1'],
        False,
        None,
    ],
    'kep-fits':[
        '*_llc.fits',
        read_kepler_fitslc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdc_sapflux'],
        ['sap.sap_flux_err','pdc.pdc_sapflux_err'],
        True,
        None,
    ],
    'kep-pkl':[
        '-keplc.pkl',
        read_kepler_pklc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdc_sapflux'],
        ['sap.sap_flux_err','pdc.pdc_sapflux_err'],
        True,
        None,
    ],
    # binned light curve format
    'binned-hat':[
        '*binned-*hat*.pkl',
        read_pklc,
        ['binned.aep_000.times','binned.atf_000.times'],
        ['binned.aep_000.mags','binned.atf_000.mags'],
        ['binned.aep_000.errs','binned.atf_000.errs'],
        False,
        None,
    ],
    'binned-hp':[
        '*binned-*hp*.pkl',
        read_pklc,
        ['binned.iep1.times','binned.itf1.times'],
        ['binned.iep1.mags','binned.itf1.mags'],
        ['binned.iep1.errs','binned.itf1.errs'],
        False,
        None,
    ],
    'binned-kep':[
        '*binned-*kep*.pkl',
        read_pklc,
        ['binned.sap_flux.times','binned.pdc_sapflux.times'],
        ['binned.sap_flux.mags','binned.pdc_sapflux.mags'],
        ['binned.sap_flux.errs','binned.pdc_sapflux.errs'],
        True,
        None,
    ],
}



def register_custom_lcformat(formatkey,
                             fileglob,
                             readerfunc,
                             timecols,
                             magcols,
                             errcols,
                             magsarefluxes=False,
                             specialnormfunc=None):
    '''This adds a custom format LC to the dict above.

    Allows handling of custom format light curves that are input to the fake LC
    generators below. Once the format is successfully registered, light curves
    should work transparently with all of the functions below, by simply calling
    them with the formatkey in the lcformat keyword argument where appropriate.

    Args
    ----

    formatkey: <string>: what to use as the key for your light curve format


    fileglob: <string>: the default fileglob to use to search for light curve
    files in this custom format. This is a string like
    '*-whatever-???-*.*??-.lc'.


    readerfunc: <function>: this is the function to use to read light curves in
    the custom format. This should return a dictionary (the 'lcdict') with the
    following signature (the keys listed below are required, but others are
    allowed):

    {'objectid':'<this object's name>',
     'objectinfo':{'ra':<this object's right ascension>
                   'decl':<this object's declination>},
     ...time columns, mag columns, etc.}


    timecols, magcols, errcols: <list>: these are all lists of strings
    indicating which keys in the lcdict to use for processing. The lists must
    all have the same dimensions, e.g. if timecols = ['timecol1','timecol2'],
    then magcols must be something like ['magcol1','magcol2'] and errcols must
    be something like ['errcol1', 'errcol2']. This allows you to process
    multiple apertures or multiple types of measurements in one go.

    Each element in these lists can be a simple key, e.g. 'time' (which would
    correspond to lcdict['time']), or a composite key,
    e.g. 'aperture1.times.rjd' (which would correspond to
    lcdict['aperture1']['times']['rjd']). See the LCFORM dict above for
    examples.


    magsarefluxes: <boolean>: if this is True, then all functions will treat the
    magnitude columns as flux instead, so things like default normalization and
    sigma-clipping will be done correctly. If this is False, magnitudes will be
    treated as magnitudes.


    specialnormfunc: <function>: if you intend to use a special normalization
    function for your lightcurves, indicate it here. If None, the default
    normalization method used by lcproc is to find gaps in the time-series,
    normalize measurements grouped by these gaps to zero, then normalize the
    entire magnitude time series to global time series median using the
    astrobase.lcmath.normalize_magseries function. The function should take and
    return an lcdict of the same form as that produced by readerfunc above. For
    an example of a special normalization function, see normalize_lcdict_by_inst
    in the astrobase.hatlc module.

    '''

    globals()['LCFORM'][formatkey] = [
        fileglob,
        readerfunc,
        timecols,
        magcols,
        errcols,
        magsarefluxes,
        specialnormfunc
    ]

    LOGINFO('added %s to registry' % formatkey)


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
    '''This generates fake transit light curves.

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'transitperiod', 'transitdepth', 'transitduration'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The transit epoch will be automatically chosen from a uniform distribution
    between times.min() and times.max().

    The ingress duration will be automatically chosen from a uniform
    distribution ranging from 0.05 to 0.5 of the transitduration.

    The transitdepth will be flipped automatically as appropriate if
    magsarefluxes=True.

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
                    'depthratio':sps.uniform(loc=0.01,scale=0.99)},
        magsarefluxes=False,
):
    '''This generates fake EB light curves.

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'period', 'pdepth', 'pduration','depthratio'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The EB epoch will be automatically chosen from a uniform distribution
    between times.min() and times.max().

    The pdepth will be flipped automatically as appropriate if
    magsarefluxes=True.

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

    # fix the transit depth if it needs to be flipped
    if magsarefluxes and pdepth < 0.0:
        pdepth = -pdepth
    elif not magsarefluxes and pdepth > 0.0:
        pdepth = -pdepth

    # generate the model
    modelmags, phase, ptimes, pmags, perrs = (
        eclipses.invgauss_eclipses_func([period, epoch, pdepth,
                                         pduration, depthratio],
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

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'amplitude', 'nflares', 'risestdev','decayconst'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The flare_peak_time for each flare will be generated automatically between
    times.min() and times.max() using a uniform distribution.

    The amplitude will be flipped automatically as appropriate if
    magsarefluxes=True.

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



## FOURIER PARAMS FOR SINUSOIDAL VARIABLES
#
# type        fourier           period [days]
#             order    dist     limits         dist

# RRab        8 to 10  uniform  0.45--0.80     uniform
# RRc         3 to 6   uniform  0.10--0.40     uniform
# HADS        7 to 9   uniform  0.04--0.10     uniform
# rotator     2 to 5   uniform  0.80--120.0    uniform
# LPV         2 to 5   uniform  250--500.0     uniform

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

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'period', 'fourierorder', 'amplitude'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The minimum light curve epoch will be automatically chosen from a uniform
    distribution between times.min() and times.max().

    The amplitude will be flipped automatically as appropriate if
    magsarefluxes=True.

    FIXME: figure out how scipy.signal.butter works and low-pass filter using
    scipy.signal.filtfilt.

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

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'period', 'fourierorder', 'amplitude'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The minimum light curve epoch will be automatically chosen from a uniform
    distribution between times.min() and times.max().

    The amplitude will be flipped automatically as appropriate if
    magsarefluxes=True.

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

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'period', 'fourierorder', 'amplitude'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The minimum light curve epoch will be automatically chosen from a uniform
    distribution between times.min() and times.max().

    The amplitude will be flipped automatically as appropriate if
    magsarefluxes=True.

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

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'period', 'fourierorder', 'amplitude'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The minimum light curve epoch will be automatically chosen from a uniform
    distribution between times.min() and times.max().

    The amplitude will be flipped automatically as appropriate if
    magsarefluxes=True.

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

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'period', 'fourierorder', 'amplitude'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The minimum light curve epoch will be automatically chosen from a uniform
    distribution between times.min() and times.max().

    The amplitude will be flipped automatically as appropriate if
    magsarefluxes=True.

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
    '''This generates fake LPV light curves.

    times is an array of time values that will be used as the time base.

    mags and errs will have the model mags applied to them. If either is None,
    np.full_like(times, 0.0) will used as a substitute.

    paramdists is a dict containing parameter distributions to use for the
    transitparams, in order:

    {'period', 'fourierorder', 'amplitude'}

    These are all 'frozen' scipy.stats distribution objects, e.g.:

    https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

    The minimum light curve epoch will be automatically chosen from a uniform
    distribution between times.min() and times.max().

    The amplitude will be flipped automatically as appropriate if
    magsarefluxes=True.

    '''

    modeldict = generate_sinusoidal_lightcurve(times,
                                               mags=mags,
                                               errs=errs,
                                               paramdists=paramdists,
                                               magsarefluxes=magsarefluxes)
    modeldict['vartype'] = 'LPV'
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
                timecols=None,
                magcols=None,
                errcols=None):
    '''This preprocesses the light curve and sets it up to be a sim light curve.

    Args
    ----

    lcfile is the input light curve file to copy the time base from.

    outdir is the directory to which the fake LC will be written.

    magrms is a dict containing the SDSS r mag-RMS (SDSS rmag-MAD preferably)
    relations for all light curves that the input lcfile is from. This will be
    used to generate the median mag and noise corresponding to the magnitude
    chosen for this fake LC. If randomizemags is True, then a random mag between
    the first and last magbin in magrms will be chosen as the median mag for
    this light curve. This choice will be weighted by the mag bin probability
    obtained from the magrms kwarg. Otherwise, the median mag will be taken from
    the input lcfile's lcdict['objectinfo']['sdssr'] key or a transformed SDSS r
    mag generated from the input lcfile's lcdict['objectinfo']['jmag'],
    ['hmag'], and ['kmag'] keys. The magrms relation for each magcol will be
    used to generate Gaussian noise at the correct level for the magbin this
    light curve's median mag falls into.

    If randomizecoords is True, will randomize the RA, DEC of the object.

    lcformat is one of the entries in the LCFORMATS dict. This is used to set
    the light curve reader function for lcfile, and the time, mag, err cols to
    use by default if timecols, magcols, or errcols are None.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

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
    fakeobjectid = md5(npr.bytes(12)).hexdigest()[-8:]
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
            fakelcdict[tcol] = dict_get(lcdict, tcolget)
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

            measuredmags = dict_get(lcdict, mcolget)
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
            fakelcdict[mcol] = np.full_like(dict_get(lcdict, mcolget), 0.0)
            fakelcdict['columns'].append(mcol)


    # get the err columns
    for mcol, ecol in zip(magcols, errcols):

        if '.' in ecol:
            ecolget = ecol.split('.')
        else:
            ecolget = [ecol]

        if ecol not in fakelcdict:

            measurederrs = dict_get(lcdict, ecolget)
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
            fakelcdict[ecol] = np.full_like(dict_get(lcdict, ecolget), 0.0)
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
    This wraps process_fakelc for collect_and_index_fakelcs below.

    task[0] = lcfile
    task[1] = outdir
    task[2] = magrms
    task[3] = {'lcformat', 'timecols', 'magcols', 'errcols', 'randomizeinfo'}

    '''

    lcfile, outdir, kwargs = task

    try:

        fakelcresults = make_fakelc(
            lcfile,
            outdir,
            **kwargs
        )

        return fakelcresults

    except Exception as e:

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
                           vartypes=['EB','RRab','RRc',
                                     'rotator','flare','HADS',
                                     'planet','LPV'],
                           lcformat='hat-sql',
                           timecols=None,
                           magcols=None,
                           errcols=None):

    '''This prepares light curves for the recovery sim.

    Collects light curves from lclist using a uniform sampling among
    them. Copies them to the simbasedir, zeroes out their mags and errs but
    keeps their time bases, also keeps their rms and median mags for later
    use. Calculates the mag-rms relation for the entire collection and writes
    that to the simbasedir as well.

    The purpose of this function is to copy over the time base and mag-rms
    relation of an existing light curve collection to use it as the basis for a
    variability recovery simulation.

    This returns a pickle written to the simbasedir that contains all the
    information for the chosen ensemble of fake light curves and writes all
    generated light curves to the simbasedir/lightucrves directory. Run the
    add_variability_to_fakelc_collection function after this function to add
    variability of the specified type to these generated light curves.

    Args
    ----

    lclist is a list of existing project light curves. This can be generated
    from lcproc.getlclist or similar.

    simbasedir is the directory to where the fake light curves and their
    information will be copied to.

    magrmsfrom is used to generate magnitudes and RMSes for the objects in the
    output collection of fake light curves. This arg is either a string pointing
    to an existing pickle file that must contain a dict or a dict variable that
    MUST have the following key-vals at a minimum:

    {'<magcol1_name>': {
          'binned_sdssr_median': list/array of median mags for each magbin
          'binned_lcmad_median': list/array of LC MAD values per magbin
     },
     '<magcol2_name>': {
          'binned_sdssr_median': list/array of median mags for each magbin
          'binned_lcmad_median': list/array of LC MAD values per magbin
     },
     .
     .
     ...}

    where magcol1_name, etc. are the same as the magcols liste in the magcols
    kwarg (or the default magcols for the specified lcformat). Examples of the
    magrmsfrom dict (or pickle) required can be generated by the
    astrobase.lcproc.variability_threshold function.


    magrms_interpolate and magrms_fillvalue will be passed to the
    scipy.interpolate.interp1d function that generates interpolators for the
    mag-RMS relation. See:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

    for details.


    maxlcs is the total number of light curves to choose from lclist and
    generate as fake LCs.


    maxvars is the total number of fake light curves that will be marked as
    variable.


    vartypes is a list of variable types to put into the collection. The
    vartypes for each fake variable star will be chosen uniformly from this
    list.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

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
            magbins = np.array(xmagrms[magcol]['binned_sdssr_median'])
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
              'errcols':errcols}

    fobjects, fpaths = [], []
    fras, fdecls, fndets = [], [], []

    fmags, fmagmads = [], []
    ferrmeds, ferrmads = [], []

    totalvars = 0

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

    The procedure is (for each magcol):

    - read the fakelcfile, get the stored moments and vartype info

    - add the periodic variability specified in vartype and varparamdists. if
     vartype == None, then do nothing in this step. If override_vartype is not
     None, override stored vartype with specified vartype. If
     override_varparamdists is not None, override with specified
     varparamdists. NOTE: the varparamdists must make sense for the vartype,
     otherwise, weird stuff will happen.

    - add the median mag level stored in fakelcfile to the time series

    - add gaussian noise to the light curve as specified in fakelcfile

    - add a varinfo key and dict to the lcdict with varperiod, varepoch,
      varparams

    - write back to pickle

    - return the varinfo dict to the caller

    '''

    # read in the fakelcfile
    lcdict = read_pklc(fakelcfile)

    # make sure to bail out if this light curve already has fake variability
    # added
    if ('actual_vartype' in lcdict and
        'actual_varparams' in lcdict
        and not overwrite):
        LOGERROR('%s has existing variability type: %s '
                 'and params: %s and overwrite = False, '
                 'skipping this file...' %(fakelcfile, lcdict['actual_vartype'],
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
    for tcol, mcol, ecol in zip(timecols, magcols, errcols):

        times, mags, errs = lcdict[tcol], lcdict[mcol], lcdict[ecol]

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
    '''This adds variability and noise to all fakelcs in simbasedir.

    If an object is marked as variable in the fakelcs-info.pkl file in
    simbasedir, a variable signal will be added to its light curve based on its
    selected type, default period and amplitude distribution, the appropriate
    params, etc. the epochs for each variable object will be chosen uniformly
    from its time-range (and may not necessarily fall on a actual observed
    time). Nonvariable objects will only have noise added as determined by their
    params, but no variable signal will be added.

    override_paramdists is a dict like so:

    {'<vartype1>': {'<param1>: scipy.stats distribution or npr.randint function,
                    .
                    .
                    .
                    '<paramN>: scipy.stats distribution or npr.randint function}

    for any vartype in VARTYPE_LCGEN_MAP. These are used to override the default
    parameter distributions for each variable type.

    If overwrite_existingvar is True, then

    This will get back the varinfo from the add_fakelc_variability function and
    writes that info back to the simbasedir/fakelcs-info.pkl file for each
    object.

    TODO: finish this

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
    for lc, varf, lcind in zip(lclist, varflag, range(len(lclist))):

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
