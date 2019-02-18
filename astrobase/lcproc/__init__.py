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
import os.path
import os
import importlib
import sys
import json


# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce, partial
from operator import getitem
def dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)


#################################
## PICKLE LC READING FUNCTIONS ##
#################################

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



#################################
## LIGHT CURVE FORMAT HANDLING ##
#################################

def check_extmodule(module, formatkey):
    '''This just imports the module specified.

    '''

    try:

        if os.path.exists(module):

            sys.path.append(os.path.dirname(module))
            importedok = importlib.import_module(
                os.path.basename(module.replace('.py',''))
            )

        else:
            importedok = importlib.import_module(module)

    except Exception as e:

        LOGEXCEPTION('could not import the module: %s for LC format: %s. '
                     'check the file path or fully qualified module name?'
                     % (module, formatkey))
        importedok = False

    return importedok



def register_lcformat(formatkey,
                      fileglob,
                      timecols,
                      magcols,
                      errcols,
                      readerfunc_module,
                      readerfunc,
                      readerfunc_kwargs=None,
                      normfunc_module=None,
                      normfunc=None,
                      normfunc_kwargs=None,
                      magsarefluxes=False,
                      overwrite_existing=False,
                      lcformat_dir='~/.astrobase/lcformat-jsons'):
    '''This adds a new LC format to the astrobase LC format registry.

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

    LOGINFO('adding %s to LC format registry...' % formatkey)

    # search for the lcformat_dir and create it if it doesn't exist
    lcformat_dpath = os.path.abspath(
        os.path.expanduser(lcformat_dir)
    )
    if not os.path.exists(lcformat_dpath):
        os.makedirs(lcformat_dpath)

    lcformat_jsonpath = os.path.join(lcformat_dpath,'%.json' % formatkey)

    if os.path.exists(lcformat_jsonpath) and not overwrite_existing:
        LOGERROR('There is an existing lcformat JSON: %s '
                 'for this formatkey: %s and '
                 'overwrite_existing = False, skipping...'
                 % (lcformat_jsonpath, formatkey))
        return None

    # see if we can import the reader module
    readermodule = check_extmodule(readerfunc_module, formatkey)

    if not readermodule:
        LOGERROR("could not import the required "
                 "module: %s to read %s light curves" %
                 (readerfunc_module, formatkey))
        return None

    # then, get the function we need to read the light curve
    try:
        getattr(readermodule, readerfunc)
        readerfunc_in = readerfunc
    except AttributeError:
        LOGEXCEPTION('Could not get the specified reader '
                     'function: %s for lcformat: %s '
                     'from module: %s'
                     % (formatkey, readerfunc_module, readerfunc))
        raise

    # see if we can import the normalization module
    if normfunc_module:
        normmodule = check_extmodule(normfunc_module, formatkey)
        if not normmodule:
            LOGERROR("could not import the required "
                     "module: %s to normalize %s light curves" %
                     (normfunc_module, formatkey))
            return None

    else:
        normmodule = None

    # finally, get the function we need to normalize the light curve
    if normfunc_module and normfunc:
        try:
            getattr(normmodule, normfunc)
            normfunc_in = normfunc
        except AttributeError:
            LOGEXCEPTION('Could not get the specified norm '
                         'function: %s for lcformat: %s '
                         'from module: %s'
                         % (formatkey, normfunc_module, normfunc))
            raise

    else:
        normfunc_in = None


    # if we made it to here, then everything's good. generate the JSON
    # structure
    formatdict = {'fileglob':fileglob,
                  'timecols':timecols,
                  'magcols':magcols,
                  'errcols':errcols,
                  'magsarefluxes':magsarefluxes,
                  'lcreader_module':readermodule,
                  'lcreader_func':readerfunc_in,
                  'lcreader_kwargs':readerfunc_kwargs,
                  'lcnorm_module':normmodule,
                  'lcnorm_func':normfunc_in,
                  'lcnorm_kwargs':normfunc_kwargs}

    # write this to the lcformat directory
    with open(lcformat_jsonpath,'w') as outfd:
        json.dump(formatdict, outfd, indent=4)

    return lcformat_jsonpath



def get_lcformat(formatkey, use_lcformat_dir=None):
    '''This loads an LC format description from a previously-saved JSON file.

    Parameters
    ----------

    formatkey : str
        The key used to refer to the LC format. This is part of the JSON file's
        name, e.g. the format key 'hat-csv' maps to the format JSON file:
        '<astrobase install path>/data/lcformats/hat-csv.json'.

    use_lcformat_dir : str or None
        If provided, must be the path to a directory that contains the
        corresponding lcformat JSON file for `formatkey`. If this is None, this
        function will look for lcformat JSON files corresponding to the given
        `formatkey`:

        - first, in the directory specified in this kwarg,
        - if not found there, in the home directory: ~/.astrobase/lcformat-jsons
        - if not found there, in: <astrobase install path>/data/lcformats


    Returns
    -------

    tuple
        A tuple of the following form is returned:

        (fileglob       : the file glob of the associated LC files,
         readerfunc_in  : the imported Python function for reading LCs,
         timecols       : list of time col keys to get from the lcdict,
         magcols        : list of mag col keys to get from the lcdict ,
         errcols        : list of err col keys to get from the lcdict,
         magsarefluxes  : True if the measurements are fluxes not mags,
         normfunc_in    : the imported Python function for normalizing LCs)

        All `astrobase.lcproc` functions can then use this tuple to dynamically
        import your LC reader and normalization functions to work with your LC
        format transparently.

    '''

    if isinstance(use_lcformat_dir, str):

        # look for the lcformat JSON
        lcformat_jsonpath = os.path.join(
            use_lcformat_dir,
            '%s.json' % formatkey
        )

        if not os.path.exists(lcformat_jsonpath):

            lcformat_jsonpath = os.path.join(
                os.path.expanduser('~/.astrobase/lcformat-jsons'),
                '%s.json' % formatkey
            )

            if not os.path.exists(lcformat_jsonpath):

                install_path = os.path.dirname(__file__)
                install_path = os.path.abspath(
                    os.path.join(install_path, '..', 'data','lcformats')
                )

                lcformat_jsonpath = os.path.join(
                    install_path,
                    '%s.json' % formatkey
                )

                if not os.path.exists(lcformat_jsonpath):
                    LOGERROR('could not find an lcformat JSON '
                             'for formatkey: %s in any of: '
                             'use_lcformat_dir, home directory, '
                             'astrobase installed data directory'
                             % formatkey)
                    return None

    else:

        lcformat_jsonpath = os.path.join(
            os.path.expanduser('~/.astrobase/lcformat-jsons'),
            '%s.json' % formatkey
        )

        if not os.path.exists(lcformat_jsonpath):

            install_path = os.path.dirname(__file__)
            install_path = os.path.abspath(
                os.path.join(install_path, '..', 'data','lcformats')
            )

            lcformat_jsonpath = os.path.join(
                install_path,
                '%s.json' % formatkey
            )

            if not os.path.exists(lcformat_jsonpath):
                LOGERROR('could not find an lcformat JSON '
                         'for formatkey: %s in any of: '
                         'use_lcformat_dir, home directory, '
                         'astrobase installed data directory'
                         % formatkey)
                return None

    # load the found lcformat JSON
    with open(lcformat_jsonpath) as infd:
        lcformatdict = json.load(infd)

    readerfunc_module = lcformatdict['lcreader_module']
    readerfunc = lcformatdict['lcreader_func']
    readerfunc_kwargs = lcformatdict['lcreader_kwargs']
    normfunc_module = lcformatdict['lcnorm_module']
    normfunc = lcformatdict['lcnorm_func']
    normfunc_kwargs = lcformatdict['lcnorm_kwargs']

    fileglob = lcformatdict['fileglob']
    timecols = lcformatdict['timecols']
    magcols = lcformatdict['magcols']
    errcols = lcformatdict['errcols']
    magsarefluxes = lcformatdict['magsarefluxes']

    # import all the required bits
    # see if we can import the reader module
    readermodule = check_extmodule(readerfunc_module, formatkey)

    if not readermodule:
        LOGERROR("could not import the required "
                 "module: %s to read %s light curves" %
                 (readerfunc_module, formatkey))
        return None

    # then, get the function we need to read the light curve
    try:
        readerfunc_in = getattr(readermodule, readerfunc)
    except AttributeError:
        LOGEXCEPTION('Could not get the specified reader '
                     'function: %s for lcformat: %s '
                     'from module: %s'
                     % (formatkey, readerfunc_module, readerfunc))
        raise

    # see if we can import the normalization module
    if normfunc_module:
        normmodule = check_extmodule(normfunc_module, formatkey)
        if not normmodule:
            LOGERROR("could not import the required "
                     "module: %s to normalize %s light curves" %
                     (normfunc_module, formatkey))
            return None

    else:
        normmodule = None

    # finally, get the function we need to normalize the light curve
    if normfunc_module and normfunc:
        try:
            normfunc_in = getattr(normmodule, normfunc)
        except AttributeError:
            LOGEXCEPTION('Could not get the specified norm '
                         'function: %s for lcformat: %s '
                         'from module: %s'
                         % (formatkey, normfunc_module, normfunc))
            raise

    else:
        normfunc_in = None


    # add in any optional kwargs that need to be there for readerfunc
    if isinstance(readerfunc_kwargs, dict):
        readerfunc_in = partial(readerfunc_in, **readerfunc_kwargs)

    # add in any optional kwargs that need to be there for normfunc
    if normfunc_in is not None:
        if isinstance(normfunc_kwargs, dict):
            normfunc_in = partial(normfunc_in, **normfunc_kwargs)

    # assemble the return tuple
    # this can be used directly by other lcproc functions
    returntuple = (
        fileglob,
        readerfunc_in,
        timecols,
        magcols,
        errcols,
        magsarefluxes,
        normfunc_in,
    )

    return returntuple
