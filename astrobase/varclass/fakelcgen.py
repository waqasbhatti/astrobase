#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''fakelcgen - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This generates light curves of variable stars using the astrobase.lcmodels
package, adds noise and observation sampling to them based on given parameters
(or example light curves). See fakelcrecovery.py for functions that run a full
recovery simulation.

'''
import os
import os.path
import pickle

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
npr.seed(0xdecaff)

import scipy.stats as sps
import scipy.interpolate as spi

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

    Allows handling of custom format light curves for astrobase lcproc
    drivers. Once the format is successfully registered, light curves should
    work transparently with all of the functions below, by simply calling them
    with the formatkey in the lcformat keyword argument.

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

def generate_transit_lightcurve(fakelcfile, transitparams):
    '''
    This generates fake transit light curves.

    transitparams = [transitperiod (time),
                     transitepoch (time),
                     transitdepth (flux or mags),
                     transitduration (phase),
                     ingressduration (phase)]

    for magnitudes -> transitdepth should be < 0
    for fluxes     -> transitdepth should be > 0

    TODO: finish this

    '''


def generate_eb_lightcurve(fakelcfile, ebparams):
    '''
    This generates fake EB light curves.

    ebparams = [period (time),
                epoch (time),
                pdepth (mags),
                pduration (phase),
                depthratio]

    TODO: finish this

    '''


def generate_flare_lightcurve(fakelcfile, flareparams):
    '''
    This generates fake flare light curves.

    flareparams = [amplitude,
                   flare_peak_time,
                   rise_gaussian_stdev,
                   decay_time_constant]

    TODO: finish this

    '''


def generate_sinusoidal_lightcurve(fakelcfile,
                                   sintype,
                                   fourierparams):
    '''This generates fake sinusoidal light curves.

    sintype is one of 'RRab', 'RRc', 'HADS', 'rotation', 'LPV', which sets the
    fourier order and period limits like so:

    type        fourier           period [days]
                order    dist     limits         dist

    RRab        7 to 10  uniform  0.45--0.80     uniform
    RRc         2 to 4   uniform  0.10--0.40     uniform
    HADS        5 to 9   uniform  0.04--0.10     uniform
    rotator     1 to 3   uniform  0.80--120.0    uniform
    LPV         1 to 3   uniform  250--500.0     uniform

    fourierparams = [epoch,
                     [ampl_1, ampl_2, ampl_3, ..., ampl_X],
                     [phas_1, phas_2, phas_3, ..., phas_X]]

    The period will be set using the table above.

    TODO: finish this

    '''



def generate_rrab_lightcurve(fakelcfile, rrabparams):
    '''This wraps generate_sinusoidal_lightcurves for RRab LCs.

    '''

    return generate_sinusoidal_lightcurve(fakelcfile, 'RRab', rrabparams)



def generate_rrc_lightcurve(fakelcfile, rrcparams):
    '''This wraps generate_sinusoidal_lightcurves for RRc LCs.

    '''

    return generate_sinusoidal_lightcurve(fakelcfile, 'RRc', rrcparams)



def generate_hads_lightcurve(fakelcfile, hadsparams):
    '''This wraps generate_sinusoidal_lightcurves for HADS LCs.

    '''

    return generate_sinusoidal_lightcurve(fakelcfile, 'HADS', hadsparams)



def generate_rotator_lightcurve(fakelcfile, rotparams):
    '''This wraps generate_sinusoidal_lightcurves for rotation LCs.

    '''

    return generate_sinusoidal_lightcurve(fakelcfile, 'rotator', rotparams)



def generate_lpv_lightcurve(fakelcfile, lpvparams):
    '''This wraps generate_sinusoidal_lightcurves for LPV LCs.

    '''

    return generate_sinusoidal_lightcurve(fakelcfile, 'LPV', lpvparams)



# this maps functions to generate light curves to their vartype codes as put
# into the make_fakelc_collection function.
VARTYPE_LCGEN_MAP = {
    'EB': generate_eb_lightcurve,
    'RRab': generate_rrab_lightcurve,
    'RRc': generate_rrc_lightcurve,
    'rotator': generate_rotation_lightcurve,
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
    chosen for this fake LC. If randomizeinfo is True, then a random mag between
    the first and last magbin in magrms will be chosen as the median mag for
    this light curve. Otherwise, the median mag will be taken from the input
    lcfile's lcdict['objectinfo']['sdssr'] key or a transformed SDSS r mag
    generated from the input lcfile's lcdict['objectinfo']['jmag'], ['hmag'],
    and ['kmag'] keys. The magrms relation for each magcol will be used to
    generate Gaussian noise at the correct level for the magbin this light
    curve's median mag falls into.

    If randomizemags is True, will generate random SDSS r in the output fakelc
    even if these values are available from the input LC. If randomizecoords is
    True, will do the same for RA, DEC.

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
            fakelcdict['objectinfo']['decl'] = npr.random()*90.0 - 90.0

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

        # if there are no mags available at all, generate a random mag
        # between 8 and 16.0
        else:

            LOGWARNING(' %s: assigning a random mag' % lcfile)
            fakelcdict['objectinfo']['sdssr'] = npr.random()*8.0 + 8.0

    # if there's no info available, generate fake info
    else:

        LOGWARNING('no object information found in %s, '
                   'generating random ra, decl, sdssr' %
                   lcfile)
        fakelcdict['objectinfo']['ra'] = npr.random()*360.0
        fakelcdict['objectinfo']['decl'] = npr.random()*90.0 - 90.0
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



    # generate an output file name
    fakelcfname = '%s-fakelc.pkl' % fakelcdict['objectid']
    fakelcfpath = os.path.join(outdir, fakelcfname)

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
                'binned_sdssr_median':xmagrms[magcol]['binned_sdssr_median'],
                'binned_lcmad_median':xmagrms[magcol]['binned_lcmad_median'],
            }

            # interpolate the mag-MAD relation
            interpolated_magmad = spi.interp1d(
                xmagrms[magcol]['binned_sdssr_median'],
                xmagrms[magcol]['binned_lcmad_median'],
                kind=magrms_interpolate,
                fill_value=magrms_fillvalue,
            )
            # save this as well
            magrms[magcol]['interpolated_magmad'] = interpolated_magmad

        else:

            LOGWARNING('input magrms dict does not have '
                       'required info for magcol: %s' % magcol)

            magrms[magcol] = {
                'binned_sdssr_median':None,
                'binned_lcmad_median':None,
                'interpolated_magmad':None,
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
                           varparams):
    '''This adds variability of the specified type to the fake LC.

    The procedure is (for each magcol):

    - add the periodic variability specified in vartype and varparams. if not
      periodic variable, then do nothing.

    - add the median mag level stored in fakelcfile to the time series

    - add gaussian noise to the light curve as specified in fakelcfile

    - add a varinfo key and dict to the lcdict with varperiod, varepoch,
      varparams

    - write back to pickle

    - return the varinfo dict to the caller

    TODO: finish this

    '''



def add_variability_to_fakelc_collection(simbasedir, customparams=None):
    '''This adds variability and noise to all fakelcs in simbasedir.

    If an object is marked as variable in the fakelcs-info.pkl file in
    simbasedir, a variable signal will be added to its light curve based on its
    selected type, default period and amplitude distribution, the appropriate
    params, etc. the epochs for each variable object will be chosen uniformly
    from its time-range (and may not necessarily fall on a actual observed
    time). Nonvariable objects will only have noise added as determined by their
    params, but no variable signal will be added.

    customparams is a dict like so:

    {'<vartype1>': {'periodrange': [startp, endp],
                    'perioddist': <a scipy.stats distribution object>,
                    'amplrange': [startampl, endampl],
                    'ampldist': <a scipy.stats distribution object>},
     ...}

    for any vartype in VARTYPE_LCGEN_MAP. These are used to override the default
    period and amplitude distributions for each variable type.

    This will get back the varinfo from the add_fakelc_variability function and
    writes that info back to the simbasedir/fakelcs-info.pkl file for each
    object.

    TODO: finish this

    '''
