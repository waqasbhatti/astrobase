#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lcbin.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

'''
This contains parallelized functions to bin large numbers of light curves in
time.

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

from tornado.escape import squeeze

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem


def _dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)


############
## CONFIG ##
############

NCPUS = mp.cpu_count()


###################
## LOCAL IMPORTS ##
###################

from astrobase.lcproc import get_lcformat
from astrobase.lcmath import (
    normalize_magseries,
    time_bin_magseries_with_errs,
)


###################################
## FUNCTIONS TO BIN LIGHT CURVES ##
###################################

def timebinlc(lcfile,
              binsizesec,
              outdir=None,
              lcformat='hat-sql',
              lcformatdir=None,
              timecols=None,
              magcols=None,
              errcols=None,
              minbinelems=7):

    '''This bins the given light curve file in time using the specified bin size.

    Parameters
    ----------

    lcfile : str
        The file name to process.

    binsizesec : float
        The time bin-size in seconds.

    outdir : str or None
        If this is a str, the output LC will be written to `outdir`. If this is
        None, the output LC will be written to the same directory as `lcfile`.

    lcformat : str
        This is the `formatkey` associated with your light curve format, which
        you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curve file.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    timecols,magcols,errcols : lists of str
        The keys in the lcdict produced by your light curve reader function that
        correspond to the times, mags/fluxes, and associated measurement errors
        that will be used as inputs to the binning process. If these are None,
        the default values for `timecols`, `magcols`, and `errcols` for your
        light curve format will be used here.

    minbinelems : int
        The minimum number of time-bin elements required to accept a time-bin as
        valid for the output binned light curve.

    Returns
    -------

    str
        The name of the output pickle file with the binned LC.

        Writes the output binned light curve to a pickle that contains the
        lcdict with an added `lcdict['binned'][magcol]` key, which contains the
        binned times, mags/fluxes, and errs as
        `lcdict['binned'][magcol]['times']`, `lcdict['binned'][magcol]['mags']`,
        and `lcdict['epd'][magcol]['errs']` for each `magcol` provided in the
        input or default `magcols` value for this light curve format.

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

    # get the LC into a dict
    lcdict = readerfunc(lcfile)

    # this should handle lists/tuples being returned by readerfunc
    # we assume that the first element is the actual lcdict
    # FIXME: figure out how to not need this assumption
    if ( (isinstance(lcdict, (list, tuple))) and
         (isinstance(lcdict[0], dict)) ):
        lcdict = lcdict[0]

    # skip already binned light curves
    if 'binned' in lcdict:
        LOGERROR('this light curve appears to be binned already, skipping...')
        return None

    lcdict['binned'] = {}

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

        # now bin the mag series as requested
        binned = time_bin_magseries_with_errs(times,
                                              mags,
                                              errs,
                                              binsize=binsizesec,
                                              minbinelems=minbinelems)

        # put this into the special binned key of the lcdict
        lcdict['binned'][mcol] = {'times':binned['binnedtimes'],
                                  'mags':binned['binnedmags'],
                                  'errs':binned['binnederrs'],
                                  'nbins':binned['nbins'],
                                  'timebins':binned['jdbins'],
                                  'binsizesec':binsizesec}

    # done with binning for all magcols, now generate the output file
    # this will always be a pickle

    if outdir is None:
        outdir = os.path.dirname(lcfile)

    outfile = os.path.join(outdir, '%s-binned%.1fsec-%s.pkl' %
                           (squeeze(lcdict['objectid']).replace(' ','-'),
                            binsizesec, lcformat))

    with open(outfile, 'wb') as outfd:
        pickle.dump(lcdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    return outfile


def timebinlc_worker(task):
    '''
    This is a parallel worker for the function below.

    Parameters
    ----------

    task : tuple
        This is of the form::

            task[0] = lcfile
            task[1] = binsizesec
            task[3] = {'outdir','lcformat','lcformatdir',
                       'timecols','magcols','errcols','minbinelems'}

    Returns
    -------

    str
        The output pickle file with the binned LC if successful. None otherwise.

    '''

    lcfile, binsizesec, kwargs = task

    try:
        binnedlc = timebinlc(lcfile, binsizesec, **kwargs)
        LOGINFO('%s binned using %s sec -> %s OK' %
                (lcfile, binsizesec, binnedlc))
        return binnedlc
    except Exception:
        LOGEXCEPTION('failed to bin %s using binsizesec = %s' % (lcfile,
                                                                 binsizesec))
        return None


def parallel_timebin(lclist,
                     binsizesec,
                     maxobjects=None,
                     outdir=None,
                     lcformat='hat-sql',
                     lcformatdir=None,
                     timecols=None,
                     magcols=None,
                     errcols=None,
                     minbinelems=7,
                     nworkers=NCPUS,
                     maxworkertasks=1000):
    '''This time-bins all the LCs in the list using the specified bin size.

    Parameters
    ----------

    lclist : list of str
        The input LCs to process.

    binsizesec : float
        The time bin size to use in seconds.

    maxobjects : int or None
        If provided, LC processing will stop at `lclist[maxobjects]`.

    outdir : str or None
        The directory where output LCs will be written. If None, will write to
        the same directory as the input LCs.

    lcformat : str
        This is the `formatkey` associated with your light curve format, which
        you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curve file.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    timecols,magcols,errcols : lists of str
        The keys in the lcdict produced by your light curve reader function that
        correspond to the times, mags/fluxes, and associated measurement errors
        that will be used as inputs to the binning process. If these are None,
        the default values for `timecols`, `magcols`, and `errcols` for your
        light curve format will be used here.

    minbinelems : int
        The minimum number of time-bin elements required to accept a time-bin as
        valid for the output binned light curve.

    nworkers : int
        Number of parallel workers to launch.

    maxworkertasks : int
        The maximum number of tasks a parallel worker will complete before being
        replaced to guard against memory leaks.

    Returns
    -------

    dict
        The returned dict contains keys = input LCs, vals = output LCs.

    '''

    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    if maxobjects is not None:
        lclist = lclist[:maxobjects]

    tasks = [(x, binsizesec, {'outdir':outdir,
                              'lcformat':lcformat,
                              'lcformatdir':lcformatdir,
                              'timecols':timecols,
                              'magcols':magcols,
                              'errcols':errcols,
                              'minbinelems':minbinelems}) for x in lclist]

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
    results = pool.map(timebinlc_worker, tasks)
    pool.close()
    pool.join()

    resdict = {os.path.basename(x):y for (x,y) in zip(lclist, results)}

    return resdict


def parallel_timebin_lcdir(lcdir,
                           binsizesec,
                           maxobjects=None,
                           outdir=None,
                           lcformat='hat-sql',
                           lcformatdir=None,
                           timecols=None,
                           magcols=None,
                           errcols=None,
                           minbinelems=7,
                           nworkers=NCPUS,
                           maxworkertasks=1000):
    '''
    This time bins all the light curves in the specified directory.

    Parameters
    ----------

    lcdir : list of str
        Directory containing the input LCs to process.

    binsizesec : float
        The time bin size to use in seconds.

    maxobjects : int or None
        If provided, LC processing will stop at `lclist[maxobjects]`.

    outdir : str or None
        The directory where output LCs will be written. If None, will write to
        the same directory as the input LCs.

    lcformat : str
        This is the `formatkey` associated with your light curve format, which
        you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curve file.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    timecols,magcols,errcols : lists of str
        The keys in the lcdict produced by your light curve reader function that
        correspond to the times, mags/fluxes, and associated measurement errors
        that will be used as inputs to the binning process. If these are None,
        the default values for `timecols`, `magcols`, and `errcols` for your
        light curve format will be used here.

    minbinelems : int
        The minimum number of time-bin elements required to accept a time-bin as
        valid for the output binned light curve.

    nworkers : int
        Number of parallel workers to launch.

    maxworkertasks : int
        The maximum number of tasks a parallel worker will complete before being
        replaced to guard against memory leaks.

    Returns
    -------

    dict
        The returned dict contains keys = input LCs, vals = output LCs.

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

    lclist = sorted(glob.glob(os.path.join(lcdir, fileglob)))

    return parallel_timebin(lclist,
                            binsizesec,
                            maxobjects=maxobjects,
                            outdir=outdir,
                            lcformat=lcformat,
                            timecols=timecols,
                            magcols=magcols,
                            errcols=errcols,
                            minbinelems=minbinelems,
                            nworkers=nworkers,
                            maxworkertasks=maxworkertasks)
