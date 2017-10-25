#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''recoverysim - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This generates light curves of variable stars using the astrobase.lcmodels
package, adds noise and observation sampling to them based on given parameters
(or example light curves) and then runs them through variable star detection and
classification to see how well they are recovered.

TODO: random notes below for implementation

use realistic timebase, mag distribution, noise distribution and generate
variable LCs (also non-variable LCs using same distributions for false positive
rate).

generate periodic and non periodic vars with given period and amplitude
distributions:

- planets with trapezoid LC

- EBs with double inverted gaussian

- pulsators with Fourier coefficients

- flares with flare model

calculate the various non-periodic variability indices for these sim LCs and
test recall and precision. get the PCA PC1 of this for a combined var index.

tune for a true positive and false positive rate using the ROC curve and set the
sigma limit for confirmed variability per magnitude bin.

afterwards check if successful by cross validation.

run period searches and see recovery rate by period, amplitude, magnitude,
number of observations, etc.

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.lcproc' % parent_name)

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

    '''


def generate_eb_lightcurve(fakelcfile, ebparams):
    '''
    This generates fake EB light curves.

    '''


def generate_flare_lightcurve(fakelcfile, flareparams):
    '''
    This generates fake flare light curves.

    '''


def generate_sinusoidal_lightcurve(fakelcfile,
                                   sintype,
                                   fourierparams):
    '''This generates fake sinusoidal light curves.

    sintype is 'RRab', 'RRc', 'HADS', 'rotation', which sets the fourier order
    and period limits like so:

    type        fourier order limits        period limit

    RRab        5 to 8                      0.45 to 0.80 days
    RRc         2 to 4                      0.10 to 0.40 days
    HADS        5 to 8                      0.04 to 0.10 days
    rotation    1 to 3                      0.8 to 120.0 days
    LPV         1 to 3                      250 to 500.0 days

    '''

def generate_rrab_lightcurve(fakelcfile, rrabparams):
    '''This wraps generate_sinusoidal_lightcurves for RRab LCs.

    '''

def generate_rrc_lightcurve(fakelcfile, rrcparams):
    '''This wraps generate_sinusoidal_lightcurves for RRc LCs.

    '''

def generate_hads_lightcurve(fakelcfile, hadsparams):
    '''This wraps generate_sinusoidal_lightcurves for HADS LCs.

    '''

def generate_rotation_lightcurve(fakelcfile, rotparams):
    '''This wraps generate_sinusoidal_lightcurves for rotation LCs.

    '''

def generate_lpv_lightcurve(fakelcfile, lpvparams):
    '''This wraps generate_sinusoidal_lightcurves for LPV LCs.

    '''


# this maps functions to generate light curves to their vartype codes as put
# into the make_fakelc_collection function.
VARTYPE_LCGEN_MAP = {
    'EB': generate_eb_lightcurve,
    'RRAB': generate_rrab_lightcurve,
    'RRC': generate_rrc_lightcurve,
    'ROT': generate_rotation_lightcurve,
    'FLR': generate_flare_lightcurve,
    'HADS': generate_hads_lightcurve,
    'PLT': generate_transit_lightcurve,
    'LPV': generate_lpv_lightcurve,
}



###############################################
## FUNCTIONS TO COLLECT LIGHT CURVES FOR SIM ##
###############################################

def make_fakelc(lcfile,
                outdir,
                lcformat='hat-sql',
                timecols=None,
                magcols=None,
                errcols=None,
                randomizeinfo=False):
    '''This preprocesses the light curve and sets it up to be a sim light curve.

    If randomizeinfo is True, will generate random RA, DEC, and SDSS r in the
    output fakelc even if these values are available from the input LC.

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
    fakeobjectid = md5(npr.bytes(12)).hexdigest()
    fakelcdict = {
        'objectid':fakeobjectid,
        'objectinfo':{'objectid':fakeobjectid},
        'columns':[],
        'moments':{},
        'origformat':lcformat,
    }

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

        if mcol not in fakelcdict:

            measuredmags = dict_get(lcdict, mcolget)
            measuredmags = measuredmags[np.isfinite(measuredmags)]

            # we require at least 10 finite measurements
            if measuredmags.size > 9:
                measuredmedian = np.median(measuredmags)
                measuredmad = np.median(np.abs(measuredmags - measuredmedian))
                fakelcdict['moments'][mcol] = {'median':measuredmedian,
                                               'mad':measuredmad}
            else:
                LOGWARNING(
                    'input LC %s does not have enough finite measurements, '
                    'no mag moments calculated' % lcfile
                )
                fakelcdict['moments'][mcol] = {'median':np.nan,
                                               'mad':np.nan}

            # the magnitude column is set to all zeros initially
            fakelcdict[mcol] = np.full_like(dict_get(lcdict, mcolget), 0.0)
            fakelcdict['columns'].append(mcol)


    # get the err columns
    for ecol in errcols:

        if '.' in ecol:
            ecolget = ecol.split('.')
        else:
            ecolget = [ecol]

        if ecol not in fakelcdict:

            measurederrs = dict_get(lcdict, ecolget)
            measurederrs = measurederrs[np.isfinite(measurederrs)]

            # we require at least 10 finite measurements
            # we'll calculate the median and MAD of the errs to use later on
            if measurederrs.size > 9:
                measuredmedian = np.median(measurederrs)
                measuredmad = np.median(np.abs(measurederrs - measuredmedian))
                fakelcdict['moments'][ecol] = {'median':measuredmedian,
                                               'mad':measuredmad}
            else:
                LOGWARNING(
                    'input LC %s does not have enough finite measurements, '
                    'no err moments calculated' % lcfile
                )
                fakelcdict['moments'][ecol] = {'median':np.nan,
                                               'mad':np.nan}

            # the errors column is set to all zeros initially
            fakelcdict[ecol] = np.full_like(dict_get(lcdict, ecolget), 0.0)
            fakelcdict['columns'].append(ecol)


    # now, get the actual mag of this object and other info and use that to
    # populate the corresponding entries of the fakelcdict objectinfo
    if (not randomizeinfo and
        'objectinfo' in lcdict and
        isinstance(lcdict['objectinfo'], dict)):

        objectinfo = lcdict['objectinfo']

        # get the RA
        if ('ra' in objectinfo and

            objectinfo['ra'] is not None and
            np.isfinite(objectinfo['ra'])):

            fakelcdict['objectinfo']['ra'] = objectinfo['ra']

        else:

            # if there's no RA available, we'll assign a random one between 0
            # and 360.0
            LOGWARNING('no "ra" key available in objectinfo dict for %s, '
                       'assigning a random right ascension' % lcfile)
            fakelcdict['objectinfo']['ra'] = npr.random()*360.0

        # get the DEC
        if ('decl' in objectinfo and
            objectinfo['decl'] is not None and
            np.isfinite(objectinfo['decl'])):

            fakelcdict['objectinfo']['decl'] = objectinfo['decl']

        else:

            # if there's no DECL available, we'll assign a random one between
            # -90.0 and +90.0
            LOGWARNING('no "decl" key available in objectinfo dict for %s, '
                       'assigning a random declination' % lcfile)
            fakelcdict['objectinfo']['decl'] = npr.random()*90.0 - 90.0

        # get the SDSS r mag for this object
        # this will be used for getting the eventual mag-RMS relation later
        if ('sdssr' in objectinfo and
            objectinfo['sdssr'] is not None and
            np.isfinite(objectinfo['sdssr'])):

            fakelcdict['objectinfo']['sdssr'] = objectinfo['sdssr']

        # if the SDSS r is unavailable, but we have J, H, K: use those to get
        # the SDSS r by using transformations
        elif (('jmag' in objectinfo and
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

            fakelcdict['objectinfo']['sdssr'] = npr.random()*8.0 + 8.0

    # if there's no info available, generate fake info
    else:

        LOGWARNING('no object information found in %s or randomizeinfo = True, '
                   'generating random ra, decl, sdssr' %
                   lcfile)
        fakelcdict['objectinfo']['ra'] = npr.random()*360.0
        fakelcdict['objectinfo']['decl'] = npr.random()*90.0 - 90.0
        fakelcdict['objectinfo']['sdssr'] = npr.random()*8.0 + 8.0

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
    task[2] = {'lcformat', 'timecols', 'magcols', 'errcols', 'randomizeinfo'}

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
                           maxlcs=25000,
                           maxvars=2000,
                           vartypes=['EB','RRAB','RRC',
                                     'ROT','FLR','HADS',
                                     'PLT','LPV'],
                           randomizeinfo=False,
                           lcformat='hat-sql',
                           timecols=None,
                           magcols=None,
                           errcols=None):

    '''This prepares light curves for the recovery sim.

    Args
    ----

    lclist is a list of existing project light curves. This can be generated
    from lcproc.getlclist or similar.

    simbasedir is the directory to where the fake light curves and their
    information will be copied to.

    maxlcs is the total number of light curves to choose from lclist and
    generate as fake LCs.

    maxvars is the total number of fake light curves that will be marked as
    variable.

    vartypes is a list of variable types to put into the collection. The
    vartypes for each fake variable star will be chosen uniformly from this
    list.


    Collects light curves from lclist using a uniform sampling among
    them. Copies them to the simbasedir, zeroes out their mags and errs but
    keeps their time bases, also keeps their rms and median mags for later
    use. Calculates the mag-rms relation for the entire collection and writes
    that to the simbasedir as well.

    The purpose of this function is to copy over the time base and mag-rms
    relation of an existing light curve collection to use it as the basis for a
    variability recovery simulation.

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

    tasks = [(x, fakelcdir, {'lcformat':lcformat,
                             'timecols':timecols,
                             'magcols':magcols,
                             'errcols':errcols,
                             'randomizeinfo':randomizeinfo})
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

    fakedb['magrms'] = {}

    for mcolind, mcol in enumerate(magcols):

        LOGINFO('characterizing mag-RMS for %s' % mcol)

        thisrms = fakedb['mad'][:,mcolind]*1.483
        finind = np.isfinite(thisrms) & np.isfinite(fmags)
        thisrms = thisrms[finind]
        thismags = fmags[finind]

        # do a polyfit - make sure to use the unsaturated star range
        fitcoeffs = np.polyfit(thismags, thisrms, 2)

        # get the poly function with these coeffs
        magrmspoly = np.poly1d(fitcoeffs)

        # write it to the output dict
        fakedb['magrms'][mcol] = magrmspoly

        # make a plot of the mag-rms relation and the fit
        plt.figure(figsize=(10,8))
        plt.plot(thismags, thisrms,
                 linestyle='none', marker='.', ms=1.0, rasterized=True)
        thismodelmags = np.linspace(8.0,16.0,num=2000)
        plt.plot(thismodelmags, magrmspoly(thismodelmags))
        plt.xlabel('SDSS r [mag]')
        plt.ylabel(r'RMS (1.483 $\times$ MAD)')
        plt.title('SDSS r vs. RMS for magcol: %s' % mcol)
        plt.yscale('log')
        plt.tight_layout()

        plotfname = os.path.join(simbasedir,'mag-rms-%s.png' % mcol)
        plt.savefig(plotfname, bbox_inches='tight')
        plt.close('all')


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


def add_variability_to_fakelc(fakelcfile,
                              vartype,
                              varparams,
                              noiseparams):
    '''
    This adds variability of the specified type to the fake LC.

    Also adds noise as specified by noiseparams.

    '''
