#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lcvfeatures.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

'''
This contains functions to generate variability features for large collections
of light curves. Useful later for variable star classification.

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

import pickle
import os
import os.path
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from tornado.escape import squeeze

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem


def _dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)


import numpy as np

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False
    pass


############
## CONFIG ##
############

NCPUS = mp.cpu_count()


###################
## LOCAL IMPORTS ##
###################

from astrobase.lcmath import normalize_magseries
from astrobase.varclass import varfeatures

from astrobase.lcproc import get_lcformat


##########################
## VARIABILITY FEATURES ##
##########################

def get_varfeatures(lcfile,
                    outdir,
                    timecols=None,
                    magcols=None,
                    errcols=None,
                    mindet=1000,
                    lcformat='hat-sql',
                    lcformatdir=None):
    '''This runs :py:func:`astrobase.varclass.varfeatures.all_nonperiodic_features`
    on a single LC file.

    Parameters
    ----------

    lcfile : str
        The input light curve to process.

    outfile : str
        The filename of the output variable features pickle that will be
        generated.

    timecols : list of str or None
        The timecol keys to use from the lcdict in calculating the features.

    magcols : list of str or None
        The magcol keys to use from the lcdict in calculating the features.

    errcols : list of str or None
        The errcol keys to use from the lcdict in calculating the features.

    mindet : int
        The minimum number of LC points required to generate variability
        features.

    lcformat : str
        This is the `formatkey` associated with your light curve format, which
        you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curves specified in `basedir` or `use_list_of_filenames`.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    Returns
    -------

    str
        The generated variability features pickle for the input LC, with results
        for each magcol in the input `magcol` or light curve format's default
        `magcol` list.

    '''

    try:
        formatinfo = get_lcformat(lcformat,
                                  use_lcformat_dir=lcformatdir)
        if formatinfo:
            (dfileglob, readerfunc,
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

    try:

        # get the LC into a dict
        lcdict = readerfunc(lcfile)

        # this should handle lists/tuples being returned by readerfunc
        # we assume that the first element is the actual lcdict
        # FIXME: figure out how to not need this assumption
        if ( (isinstance(lcdict, (list, tuple))) and
             (isinstance(lcdict[0], dict)) ):
            lcdict = lcdict[0]

        resultdict = {'objectid':lcdict['objectid'],
                      'info':lcdict['objectinfo'],
                      'lcfbasename':os.path.basename(lcfile)}

        # normalize using the special function if specified
        if normfunc is not None:
            lcdict = normfunc(lcdict)

        for tcol, mcol, ecol in zip(timecols, magcols, errcols):

            # dereference the columns and get them from the lcdict
            if '.' in tcol:
                tcolget = tcol.split('.')
            else:
                tcolget = [tcol]
            times = _dict_get(lcdict, tcolget)

            if '.' in mcol:
                mcolget = mcol.split('.')
            else:
                mcolget = [mcol]
            mags = _dict_get(lcdict, mcolget)

            if '.' in ecol:
                ecolget = ecol.split('.')
            else:
                ecolget = [ecol]
            errs = _dict_get(lcdict, ecolget)

            # normalize here if not using special normalization
            if normfunc is None:
                ntimes, nmags = normalize_magseries(
                    times, mags,
                    magsarefluxes=magsarefluxes
                )

                times, mags, errs = ntimes, nmags, errs

            # make sure we have finite values
            finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)

            # make sure we have enough finite values
            if mags[finind].size < mindet:

                LOGINFO('not enough LC points: %s in normalized %s LC: %s' %
                        (mags[finind].size, mcol, os.path.basename(lcfile)))
                resultdict[mcol] = None

            else:

                # get the features for this magcol
                lcfeatures = varfeatures.all_nonperiodic_features(
                    times, mags, errs
                )
                resultdict[mcol] = lcfeatures

        # now that we've collected all the magcols, we can choose which is the
        # "best" magcol. this is defined as the magcol that gives us the
        # smallest LC MAD.

        try:
            magmads = np.zeros(len(magcols))
            for mind, mcol in enumerate(magcols):
                if '.' in mcol:
                    mcolget = mcol.split('.')
                else:
                    mcolget = [mcol]

                magmads[mind] = resultdict[mcol]['mad']

            # smallest MAD index
            bestmagcolind = np.where(magmads == np.min(magmads))[0]
            resultdict['bestmagcol'] = magcols[bestmagcolind]

        except Exception:
            resultdict['bestmagcol'] = None

        outfile = os.path.join(outdir,
                               'varfeatures-%s.pkl' %
                               squeeze(resultdict['objectid']).replace(' ','-'))

        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=4)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to get LC features for %s because: %s' %
                     (os.path.basename(lcfile), e))
        return None


def _varfeatures_worker(task):
    '''
    This wraps varfeatures.

    '''

    try:
        (lcfile, outdir, timecols, magcols, errcols,
         mindet, lcformat, lcformatdir) = task
        return get_varfeatures(lcfile, outdir,
                               timecols=timecols,
                               magcols=magcols,
                               errcols=errcols,
                               mindet=mindet,
                               lcformat=lcformat,
                               lcformatdir=lcformatdir)

    except Exception:
        return None


def serial_varfeatures(lclist,
                       outdir,
                       maxobjects=None,
                       timecols=None,
                       magcols=None,
                       errcols=None,
                       mindet=1000,
                       lcformat='hat-sql',
                       lcformatdir=None):
    '''This runs variability feature extraction for a list of LCs.

    Parameters
    ----------

    lclist : list of str
        The list of light curve file names to process.

    outdir : str
        The directory where the output varfeatures pickle files will be written.

    maxobjects : int
        The number of LCs to process from `lclist`.

    timecols : list of str or None
        The timecol keys to use from the lcdict in calculating the features.

    magcols : list of str or None
        The magcol keys to use from the lcdict in calculating the features.

    errcols : list of str or None
        The errcol keys to use from the lcdict in calculating the features.

    mindet : int
        The minimum number of LC points required to generate variability
        features.

    lcformat : str
        This is the `formatkey` associated with your light curve format, which
        you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curves specified in `basedir` or `use_list_of_filenames`.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    Returns
    -------

    list of str
        List of the generated variability features pickles for the input LCs,
        with results for each magcol in the input `magcol` or light curve
        format's default `magcol` list.

    '''

    if maxobjects:
        lclist = lclist[:maxobjects]

    tasks = [(x, outdir, timecols, magcols, errcols,
              mindet, lcformat, lcformatdir)
             for x in lclist]

    for task in tqdm(tasks):
        result = _varfeatures_worker(task)

    return result


def parallel_varfeatures(lclist,
                         outdir,
                         maxobjects=None,
                         timecols=None,
                         magcols=None,
                         errcols=None,
                         mindet=1000,
                         lcformat='hat-sql',
                         lcformatdir=None,
                         nworkers=NCPUS):
    '''This runs variable feature extraction in parallel for all LCs in `lclist`.

    Parameters
    ----------

    lclist : list of str
        The list of light curve file names to process.

    outdir : str
        The directory where the output varfeatures pickle files will be written.

    maxobjects : int
        The number of LCs to process from `lclist`.

    timecols : list of str or None
        The timecol keys to use from the lcdict in calculating the features.

    magcols : list of str or None
        The magcol keys to use from the lcdict in calculating the features.

    errcols : list of str or None
        The errcol keys to use from the lcdict in calculating the features.

    mindet : int
        The minimum number of LC points required to generate variability
        features.

    lcformat : str
        This is the `formatkey` associated with your light curve format, which
        you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curves specified in `basedir` or `use_list_of_filenames`.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    nworkers : int
        The number of parallel workers to launch.

    Returns
    -------

    dict
        A dict with key:val pairs of input LC file name : the generated
        variability features pickles for each of the input LCs, with results for
        each magcol in the input `magcol` or light curve format's default
        `magcol` list.

    '''
    # make sure to make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maxobjects:
        lclist = lclist[:maxobjects]

    tasks = [(x, outdir, timecols, magcols, errcols, mindet,
              lcformat, lcformatdir) for x in lclist]

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(_varfeatures_worker, tasks)

    results = list(resultfutures)
    resdict = {os.path.basename(x):y for (x,y) in zip(lclist, results)}

    return resdict


def parallel_varfeatures_lcdir(lcdir,
                               outdir,
                               fileglob=None,
                               maxobjects=None,
                               timecols=None,
                               magcols=None,
                               errcols=None,
                               recursive=True,
                               mindet=1000,
                               lcformat='hat-sql',
                               lcformatdir=None,
                               nworkers=NCPUS):
    '''This runs parallel variable feature extraction for a directory of LCs.

    Parameters
    ----------

    lcdir : str
        The directory of light curve files to process.

    outdir : str
        The directory where the output varfeatures pickle files will be written.

    fileglob : str or None
        The file glob to use when looking for light curve files in `lcdir`. If
        None, the default file glob associated for this LC format will be used.

    maxobjects : int
        The number of LCs to process from `lclist`.

    timecols : list of str or None
        The timecol keys to use from the lcdict in calculating the features.

    magcols : list of str or None
        The magcol keys to use from the lcdict in calculating the features.

    errcols : list of str or None
        The errcol keys to use from the lcdict in calculating the features.

    mindet : int
        The minimum number of LC points required to generate variability
        features.

    lcformat : str
        This is the `formatkey` associated with your light curve format, which
        you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curves specified in `basedir` or `use_list_of_filenames`.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    nworkers : int
        The number of parallel workers to launch.

    Returns
    -------

    dict
        A dict with key:val pairs of input LC file name : the generated
        variability features pickles for each of the input LCs, with results for
        each magcol in the input `magcol` or light curve format's default
        `magcol` list.

    '''

    try:
        formatinfo = get_lcformat(lcformat,
                                  use_lcformat_dir=lcformatdir)
        if formatinfo:
            (dfileglob, readerfunc,
             dtimecols, dmagcols, derrcols,
             magsarefluxes, normfunc) = formatinfo
        else:
            LOGERROR("can't figure out the light curve format")
            return None
    except Exception:
        LOGEXCEPTION("can't figure out the light curve format")
        return None

    if not fileglob:
        fileglob = dfileglob

    # now find the files
    LOGINFO('searching for %s light curves in %s ...' % (lcformat, lcdir))

    if recursive is False:
        matching = glob.glob(os.path.join(lcdir, fileglob))

    else:
        matching = glob.glob(os.path.join(lcdir,
                                          '**',
                                          fileglob),
                             recursive=True)

    # now that we have all the files, process them
    if matching and len(matching) > 0:

        LOGINFO('found %s light curves, getting varfeatures...' %
                len(matching))

        return parallel_varfeatures(matching,
                                    outdir,
                                    maxobjects=maxobjects,
                                    timecols=timecols,
                                    magcols=magcols,
                                    errcols=errcols,
                                    mindet=mindet,
                                    lcformat=lcformat,
                                    lcformatdir=lcformatdir,
                                    nworkers=nworkers)

    else:

        LOGERROR('no light curve files in %s format found in %s' % (lcformat,
                                                                    lcdir))
        return None
