#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''lcformat.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

This package contains functions that help drive large batch jobs processing HAT
light curves.

This top level module contains functions to import various light curve formats.

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

try:
    import cPickle as pickle
except Exception as e:
    import pickle
import gzip


# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce, partial
from operator import getitem
def dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)


###################
## LOCAL IMPORTS ##
###################

# LC reading functions
from astrobase.hatsurveys.hatlc import read_and_filter_sqlitecurve, \
    read_csvlc, normalize_lcdict_byinst
from astrobase.hatsurveys.hplc import read_hatpi_textlc, read_hatpi_pklc
from astrobase.astrokep import read_kepler_fitslc, read_kepler_pklc, \
    filter_kepler_lcdict
from astrobase.astrotess import read_tess_fitslc, read_tess_pklc, \
    filter_tess_lcdict


############################################
## MAPS FOR LCFORMAT TO LCREADER FUNCTIONS ##
#############################################

def read_pklc(lcfile):
    '''
    This just reads a pickle.

    '''

    if lcfile.endswith('.gz'):

        try:
            with gzip.open(lcfile,'rb') as infd:
                lcdict = pickle.load(infd)
        except UnicodeDecodeError:
            with gzip.open(lcfile,'rb') as infd:
                lcdict = pickle.load(infd, encoding='latin1')

    else:

        try:
            with open(lcfile,'rb') as infd:
                lcdict = pickle.load(infd)
        except UnicodeDecodeError:
            with open(lcfile,'rb') as infd:
                lcdict = pickle.load(infd, encoding='latin1')

    return lcdict



# This is the lcproc dictionary to store registered light curve formats and the
# means to read and normalize light curve files associated with each format. The
# format spec for a light curve format is a list with the elements outlined
# below. To register a new light curve format, use the register_custom_lcformat
# function below.
LCFORM = {
    'hat-sql':[
        '*-hatlc.sqlite*',             # default fileglob
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
        '*-pklc.pkl*',
        read_hatpi_pklc,
        ['rjd','rjd'],
        ['iep1','itf1'],
        ['ire1','ire1'],
        False,
        None,
    ],
    'kep-fits':[
        '*_llc.fits',
        partial(read_kepler_fitslc,normalize=True),
        ['time','time'],
        ['sap.sap_flux','pdc.pdc_sapflux'],
        ['sap.sap_flux_err','pdc.pdcsap_flux_err'],
        True,
        filter_kepler_lcdict,
    ],
    'kep-pkl':[
        '-keplc.pkl',
        read_kepler_pklc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdcsap_flux'],
        ['sap.sap_flux_err','pdc.pdcsap_flux_err'],
        True,
        filter_kepler_lcdict,
    ],
    'tess-fits':[
        '*_lc.fits',
        partial(read_tess_fitslc,normalize=True),
        ['time','time'],
        ['sap.sap_flux','pdc.pdcsap_flux'],
        ['sap.sap_flux_err','pdc.pdcsap_flux_err'],
        True,
        filter_tess_lcdict,
    ],
    'tess-pkl':[
        '-tesslc.pkl',
        read_tess_pklc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdcsap_flux'],
        ['sap.sap_flux_err','pdc.pdcsap_flux_err'],
        True,
        filter_tess_lcdict,
    ],
    # binned light curve format
    'binned-hat':[
        '*binned*hat*.pkl',
        read_pklc,
        ['binned.aep_000.times','binned.atf_000.times'],
        ['binned.aep_000.mags','binned.atf_000.mags'],
        ['binned.aep_000.errs','binned.atf_000.errs'],
        False,
        None,
    ],
    'binned-hp':[
        '*binned*hp*.pkl',
        read_pklc,
        ['binned.iep1.times','binned.itf1.times'],
        ['binned.iep1.mags','binned.itf1.mags'],
        ['binned.iep1.errs','binned.itf1.errs'],
        False,
        None,
    ],
    'binned-kep':[
        '*binned*kep*.pkl',
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
                             readerfunc_kwargs=None,
                             specialnormfunc=None,
                             normfunc_kwargs=None,
                             magsarefluxes=False):
    '''This adds a custom format LC to the dict above.

    Allows handling of custom format light curves for astrobase lcproc
    drivers. Once the format is successfully registered, light curves should
    work transparently with all of the functions in this module, by simply
    calling them with the formatkey in the lcformat keyword argument.

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


    readerfunc_kwargs is a dictionary containing any kwargs to pass through to
    the light curve reader function.


    specialnormfunc: <function>: if you intend to use a special normalization
    function for your lightcurves, indicate it here. If None, the default
    normalization method used by lcproc is to find gaps in the time-series,
    normalize measurements grouped by these gaps to zero, then normalize the
    entire magnitude time series to global time series median using the
    astrobase.lcmath.normalize_magseries function. The function should take and
    return an lcdict of the same form as that produced by readerfunc above. For
    an example of a special normalization function, see normalize_lcdict_by_inst
    in the astrobase.hatlc module.


    normfunc_kwargs is a dictionary containing any kwargs to pass through to
    the special light curve normalization function.


    magsarefluxes: <boolean>: if this is True, then all functions will treat the
    magnitude columns as flux instead, so things like default normalization and
    sigma-clipping will be done correctly. If this is False, magnitudes will be
    treated as magnitudes.

    '''

    #
    # generate the partials
    #

    if isinstance(readerfunc_kwargs, dict):
        lcrfunc = partial(readerfunc, **readerfunc_kwargs)
    else:
        lcrfunc = readerfunc

    if specialnormfunc is not None and isinstance(normfunc_kwargs, dict):
        lcnfunc = partial(specialnormfunc, **normfunc_kwargs)
    else:
        lcnfunc = specialnormfunc


    globals()['LCFORM'][formatkey] = [
        fileglob,
        lcrfunc,
        timecols,
        magcols,
        errcols,
        magsarefluxes,
        lcnfunc
    ]

    LOGINFO('added %s to registry' % formatkey)
