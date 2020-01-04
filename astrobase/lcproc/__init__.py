#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lcformat.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

'''This package contains functions that help drive large batch-processing jobs
for light curves.

This top level module contains functions to import custom light curve
formats. Once you have your own LC format registered with `lcproc`, all of the
submodules in this package can be used to process these LCs:

- :py:mod:`astrobase.lcproc.awsrun`: contains driver functions that run
  batch-processing of light curve period-finding and checkplot making using
  resources from Amazon AWS: EC2 for processing, S3 for storage, and SQS for
  queuing work.

- :py:mod:`astrobase.lcproc.catalogs`: contains functions that generate catalogs
  from collections of light curves, make KD-Trees for fast spatial matching, and
  augment these catalogs from the rich object information contained in checkplot
  pickles.

- :py:mod:`astrobase.lcproc.checkplotgen`: contains functions that drive
  batch-jobs to make checkplot pickles for a large collection of light curves
  (and optional period-finding results).

- :py:mod:`astrobase.lcproc.checkplotproc`: contains functions that add extra
  information to checkplot pickles, including color-magnitude diagrams, updating
  neighbor light curves, and cross-matches to external catalogs.

- :py:mod:`astrobase.lcproc.epd`: contains functions that drive batch-jobs for
  External Parameter Decorrelation on collections of light curves.

- :py:mod:`astrobase.lcproc.lcbin`: contains functions that drive batch-jobs
  for time-binning collections of light curves to a specified cadence.

- :py:mod:`astrobase.lcproc.lcpfeatures`: contains functions that drive
  batch-jobs to calculate features of phased light curves, if period-finding
  results for these are available. These periodic light curve features can be
  used later to do variable star classification.

- :py:mod:`astrobase.lcproc.lcsfeatures`: contains functions that drive
  batch-jobs to calculate color, coordinate, and neighbor proximity features for
  a collection of light curves. These can be used later to do variable star
  classification.

- :py:mod:`astrobase.lcproc.lcvfeatures`: contains functions that drive
  batch-jobs to calculate non-periodic features of unphased light curves
  (e.g. time-series moments and variability indices). These can be used later to
  do variable star classification.

- :py:mod:`astrobase.lcproc.periodsearch`: contains functions that drive
  batch-jobs to run period-finding using any of the methods in
  :py:mod:`astrobase.periodbase` on collections of light curves. These produce
  period-finder result pickles that can be used transparently by the functions
  in :py:mod:`astrobase.lcproc.checkplotgen` and
  :py:mod:`astrobase.lcproc.checkplotproc` to generate and update checkplot
  pickles.

- :py:mod:`astrobase.lcproc.tfa`: contains functions that drive the application
  of the Trend Filtering Algorithm (TFA) to large collections of light curves.

- :py:mod:`astrobase.lcproc.varthreshold`: contains functions that help decide
  where to place thresholds on several variability indices for a collection of
  light curves to maximize recovery of actual variable stars.

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
except Exception:
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


def _dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)


#################################
## PICKLE LC READING FUNCTIONS ##
#################################

def _read_pklc(lcfile):
    '''
    This just reads a light curve pickle file.

    Parameters
    ----------

    lcfile : str
        The file name of the pickle to open.

    Returns
    -------

    dict
        This returns an lcdict.

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

def _check_extmodule(module, formatkey):
    '''This imports the module specified.

    Used to dynamically import Python modules that are needed to support LC
    formats not natively supported by astrobase.

    Parameters
    ----------

    module : str
        This is either:

        - a Python module import path, e.g. 'astrobase.lcproc.catalogs' or
        - a path to a Python file, e.g. '/astrobase/hatsurveys/hatlc.py'

        that contains the Python module that contains functions used to open
        (and optionally normalize) a custom LC format that's not natively
        supported by astrobase.

    formatkey : str
        A str used as the unique ID of this LC format for all lcproc functions
        and can be used to look it up later and import the correct functions
        needed to support it for lcproc operations. For example, we use
        'kep-fits' as a the specifier for Kepler FITS light curves, which can be
        read by the `astrobase.astrokep.read_kepler_fitslc` function as
        specified by the `<astrobase install path>/data/lcformats/kep-fits.json`
        LC format specification JSON.

    Returns
    -------

    Python module
        This returns a Python module if it's able to successfully import it.

    '''

    try:

        if os.path.exists(module):

            sys.path.append(os.path.dirname(module))
            importedok = importlib.import_module(
                os.path.basename(module.replace('.py',''))
            )

        else:
            importedok = importlib.import_module(module)

    except Exception:

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
    calling them with the `formatkey` in the `lcformat` keyword argument.

    LC format specifications are generated as JSON files. astrobase comes with
    several of these in `<astrobase install path>/data/lcformats`. LC formats
    you add by using this function will have their specifiers written to the
    `~/.astrobase/lcformat-jsons` directory in your home directory.

    Parameters
    ----------

    formatkey : str
        A str used as the unique ID of this LC format for all lcproc functions
        and can be used to look it up later and import the correct functions
        needed to support it for lcproc operations. For example, we use
        'kep-fits' as a the specifier for Kepler FITS light curves, which can be
        read by the `astrobase.astrokep.read_kepler_fitslc` function as
        specified by the `<astrobase install path>/data/lcformats/kep-fits.json`
        LC format specification JSON produced by `register_lcformat`.

    fileglob : str
        The default UNIX fileglob to use to search for light curve files in this
        LC format. This is a string like '*-whatever-???-*.*??-.lc'.

    timecols,magcols,errcols : list of str
        These are all lists of strings indicating which keys in the lcdict
        produced by your `lcreader_func` that will be extracted and used by
        lcproc functions for processing. The lists must all have the same
        dimensions, e.g. if timecols = ['timecol1','timecol2'], then magcols
        must be something like ['magcol1','magcol2'] and errcols must be
        something like ['errcol1', 'errcol2']. This allows you to process
        multiple apertures or multiple types of measurements in one go.

        Each element in these lists can be a simple key, e.g. 'time' (which
        would correspond to lcdict['time']), or a composite key,
        e.g. 'aperture1.times.rjd' (which would correspond to
        lcdict['aperture1']['times']['rjd']). See the examples in the lcformat
        specification JSON files in `<astrobase install path>/data/lcformats`.

    readerfunc_module : str
        This is either:

        - a Python module import path, e.g. 'astrobase.lcproc.catalogs' or
        - a path to a Python file, e.g. '/astrobase/hatsurveys/hatlc.py'

        that contains the Python module that contains functions used to open
        (and optionally normalize) a custom LC format that's not natively
        supported by astrobase.

    readerfunc : str
        This is the function name in `readerfunc_module` to use to read light
        curves in the custom format. This MUST always return a dictionary (the
        'lcdict') with the following signature (the keys listed below are
        required, but others are allowed)::

            {'objectid': this object's identifier as a string,
             'objectinfo':{'ra': this object's right ascension in decimal deg,
                           'decl': this object's declination in decimal deg,
                           'ndet': the number of observations in this LC,
                           'objectid': the object ID again for legacy reasons},
             ...other time columns, mag columns go in as their own keys}

    normfunc_kwargs : dict or None
        This is a dictionary containing any kwargs to pass through to
        the light curve norm function.

    normfunc_module : str or None
        This is either:

        - a Python module import path, e.g. 'astrobase.lcproc.catalogs' or
        - a path to a Python file, e.g. '/astrobase/hatsurveys/hatlc.py'
        - None, in which case we'll use default normalization

        that contains the Python module that contains functions used to
        normalize a custom LC format that's not natively supported by astrobase.

    normfunc : str or None
        This is the function name in `normfunc_module` to use to normalize light
        curves in the custom format. If None, the default normalization method
        used by lcproc is to find gaps in the time-series, normalize
        measurements grouped by these gaps to zero, then normalize the entire
        magnitude time series to global time series median using the
        `astrobase.lcmath.normalize_magseries` function.

        If this is provided, the normalization function should take and return
        an lcdict of the same form as that produced by `readerfunc` above. For
        an example of a specific normalization function, see
        `normalize_lcdict_by_inst` in the `astrobase.hatsurveys.hatlc` module.

    normfunc_kwargs : dict or None
        This is a dictionary containing any kwargs to pass through to
        the light curve normalization function.

    magsarefluxes : bool
        If this is True, then all lcproc functions will treat the measurement
        columns in the lcdict produced by your `readerfunc` as flux instead of
        mags, so things like default normalization and sigma-clipping will be
        done correctly. If this is False, magnitudes will be treated as
        magnitudes.

    overwrite_existing : bool
        If this is True, this function will overwrite any existing LC format
        specification JSON with the same name as that provided in the
        `formatkey` arg. This can be used to update LC format specifications
        while keeping the `formatkey` the same.

    lcformat_dir : str
        This specifies the directory where the the LC format specification JSON
        produced by this function will be written. By default, this goes to the
        `.astrobase/lcformat-jsons` directory in your home directory.

    Returns
    -------

    str
        Returns the file path to the generated LC format specification JSON
        file.

    '''

    LOGINFO('adding %s to LC format registry...' % formatkey)

    # search for the lcformat_dir and create it if it doesn't exist
    lcformat_dpath = os.path.abspath(
        os.path.expanduser(lcformat_dir)
    )
    if not os.path.exists(lcformat_dpath):
        os.makedirs(lcformat_dpath)

    lcformat_jsonpath = os.path.join(lcformat_dpath,'%s.json' % formatkey)

    if os.path.exists(lcformat_jsonpath) and not overwrite_existing:
        LOGERROR('There is an existing lcformat JSON: %s '
                 'for this formatkey: %s and '
                 'overwrite_existing = False, skipping...'
                 % (lcformat_jsonpath, formatkey))
        return None

    # see if we can import the reader module
    readermodule = _check_extmodule(readerfunc_module, formatkey)

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
        normmodule = _check_extmodule(normfunc_module, formatkey)
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
                         % (normfunc, formatkey, normfunc_module))
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
                  'lcreader_module':readerfunc_module,
                  'lcreader_func':readerfunc_in,
                  'lcreader_kwargs':readerfunc_kwargs,
                  'lcnorm_module':normfunc_module,
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
        A tuple of the following form is returned::

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
    readermodule = _check_extmodule(readerfunc_module, formatkey)

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
        normmodule = _check_extmodule(normfunc_module, formatkey)
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
