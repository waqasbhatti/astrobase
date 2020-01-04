#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# periodsearch.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

'''
This contains functions to run period-finding in a parallelized manner on large
collections of light curves.

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

###################
## LOCAL IMPORTS ##
###################

from astrobase.lcmath import normalize_magseries
from astrobase import periodbase
from astrobase.periodbase.kbls import bls_snr

from astrobase.lcproc import get_lcformat


############
## CONFIG ##
############

NCPUS = mp.cpu_count()

# used to figure out which period finder to run given a list of methods
PFMETHODS = {'bls':periodbase.bls_parallel_pfind,
             'gls':periodbase.pgen_lsp,
             'aov':periodbase.aov_periodfind,
             'mav':periodbase.aovhm_periodfind,
             'pdm':periodbase.stellingwerf_pdm,
             'acf':periodbase.macf_period_find,
             'win':periodbase.specwindow_lsp}

PFMETHOD_NAMES = {
    'gls':'Generalized Lomb-Scargle periodogram',
    'pdm':'Stellingwerf phase-dispersion minimization',
    'aov':'Schwarzenberg-Czerny AoV',
    'mav':'Schwarzenberg-Czerny AoV multi-harmonic',
    'bls':'Box Least-squared Search',
    'acf':'McQuillan+ ACF Period Search',
    'win':'Timeseries Sampling Lomb-Scargle periodogram'
}


#############################
## RUNNING PERIOD SEARCHES ##
#############################

def runpf(lcfile,
          outdir,
          timecols=None,
          magcols=None,
          errcols=None,
          lcformat='hat-sql',
          lcformatdir=None,
          pfmethods=('gls','pdm','mav','win'),
          pfkwargs=({},{},{},{}),
          sigclip=10.0,
          getblssnr=False,
          nworkers=NCPUS,
          minobservations=500,
          excludeprocessed=False,
          raiseonfail=False):
    '''This runs the period-finding for a single LC.

    Parameters
    ----------

    lcfile : str
        The light curve file to run period-finding on.

    outdir : str
        The output directory where the result pickle will go.

    timecols : list of str or None
        The timecol keys to use from the lcdict in calculating the features.

    magcols : list of str or None
        The magcol keys to use from the lcdict in calculating the features.

    errcols : list of str or None
        The errcol keys to use from the lcdict in calculating the features.

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

    pfmethods : list of str
        This is a list of period finding methods to run. Each element is a
        string matching the keys of the `PFMETHODS` dict above. By default, this
        runs GLS, PDM, AoVMH, and the spectral window Lomb-Scargle periodogram.

    pfkwargs : list of dicts
        This is used to provide any special kwargs as dicts to each
        period-finding method function specified in `pfmethods`.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    getblssnr : bool
        If this is True and BLS is one of the methods specified in `pfmethods`,
        will also calculate the stats for each best period in the BLS results:
        transit depth, duration, ingress duration, refit period and epoch, and
        the SNR of the transit.

    nworkers : int
        The number of parallel period-finding workers to launch.

    minobservations : int
        The minimum number of finite LC points required to process a light
        curve.

    excludeprocessed : bool
        If this is True, light curves that have existing period-finding result
        pickles in `outdir` will not be processed.

        FIXME: currently, this uses a dumb method of excluding already-processed
        files. A smarter way to do this is to (i) generate a SHA512 cachekey
        based on a repr of `{'lcfile', 'timecols', 'magcols', 'errcols',
        'lcformat', 'pfmethods', 'sigclip', 'getblssnr', 'pfkwargs'}`, (ii) make
        sure all list kwargs in the dict are sorted, (iii) check if the output
        file has the same cachekey in its filename (last 8 chars of cachekey
        should work), so the result was processed in exactly the same way as
        specifed in the input to this function, and can therefore be
        ignored. Will implement this later.

    raiseonfail : bool
        If something fails and this is True, will raise an Exception instead of
        returning None at the end.

    Returns
    -------

    str
        The path to the output period-finding result pickle.

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

        outfile = os.path.join(outdir, 'periodfinding-%s.pkl' %
                               squeeze(lcdict['objectid']).replace(' ', '-'))

        # if excludeprocessed is True, return the output file if it exists and
        # has a size that is at least 100 kilobytes (this should be enough to
        # contain the minimal results of this function).
        if excludeprocessed:

            test_outfile = os.path.exists(outfile)
            test_outfile_gz = os.path.exists(outfile+'.gz')

            if (test_outfile and os.stat(outfile).st_size > 102400):

                LOGWARNING('periodfinding result for %s already exists at %s, '
                           'skipping because excludeprocessed=True'
                           % (lcfile, outfile))
                return outfile

            elif (test_outfile_gz and os.stat(outfile+'.gz').st_size > 102400):

                LOGWARNING(
                    'gzipped periodfinding result for %s already '
                    'exists at %s, skipping because excludeprocessed=True'
                    % (lcfile, outfile+'.gz')
                )
                return outfile+'.gz'

        # this is the final returndict
        resultdict = {
            'objectid':lcdict['objectid'],
            'lcfbasename':os.path.basename(lcfile),
            'kwargs':{'timecols':timecols,
                      'magcols':magcols,
                      'errcols':errcols,
                      'lcformat':lcformat,
                      'lcformatdir':lcformatdir,
                      'pfmethods':pfmethods,
                      'pfkwargs':pfkwargs,
                      'sigclip':sigclip,
                      'getblssnr':getblssnr}
        }

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

            # run each of the requested period-finder functions
            resultdict[mcol] = {}

            # check if we have enough non-nan observations to proceed
            finmags = mags[np.isfinite(mags)]

            if finmags.size < minobservations:

                LOGERROR('not enough non-nan observations for '
                         'this LC. have: %s, required: %s, '
                         'magcol: %s, skipping...' %
                         (finmags.size, minobservations, mcol))
                continue

            pfmkeys = []

            for pfmind, pfm, pfkw in zip(range(len(pfmethods)),
                                         pfmethods,
                                         pfkwargs):

                pf_func = PFMETHODS[pfm]

                # get any optional kwargs for this function
                pf_kwargs = pfkw
                pf_kwargs.update({'verbose':False,
                                  'nworkers':nworkers,
                                  'magsarefluxes':magsarefluxes,
                                  'sigclip':sigclip})

                # we'll always prefix things with their index to allow multiple
                # invocations and results from the same period-finder (for
                # different period ranges, for example).
                pfmkey = '%s-%s' % (pfmind, pfm)
                pfmkeys.append(pfmkey)

                # run this period-finder and save its results to the output dict
                resultdict[mcol][pfmkey] = pf_func(
                    times, mags, errs,
                    **pf_kwargs
                )

            #
            # done with running the period finders
            #
            # append the pfmkeys list to the magcol dict
            resultdict[mcol]['pfmethods'] = pfmkeys

            # check if we need to get the SNR from any BLS pfresults
            if 'bls' in pfmethods and getblssnr:

                # we need to scan thru the pfmethods to get to any BLS pfresults
                for pfmk in resultdict[mcol]['pfmethods']:

                    if 'bls' in pfmk:

                        try:

                            bls = resultdict[mcol][pfmk]

                            # calculate the SNR for the BLS as well
                            blssnr = bls_snr(bls, times, mags, errs,
                                             magsarefluxes=magsarefluxes,
                                             verbose=False)

                            # add the SNR results to the BLS result dict
                            resultdict[mcol][pfmk].update({
                                'snr':blssnr['snr'],
                                'transitdepth':blssnr['transitdepth'],
                                'transitduration':blssnr['transitduration'],
                            })

                            # update the BLS result dict with the refit periods
                            # and epochs using the results from bls_snr
                            resultdict[mcol][pfmk].update({
                                'nbestperiods':blssnr['period'],
                                'epochs':blssnr['epoch']
                            })

                        except Exception:

                            LOGEXCEPTION('could not calculate BLS SNR for %s' %
                                         lcfile)
                            # add the SNR null results to the BLS result dict
                            resultdict[mcol][pfmk].update({
                                'snr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                                'transitdepth':[np.nan,np.nan,np.nan,
                                                np.nan,np.nan],
                                'transitduration':[np.nan,np.nan,np.nan,
                                                   np.nan,np.nan],
                            })

            elif 'bls' in pfmethods:

                # we need to scan thru the pfmethods to get to any BLS pfresults
                for pfmk in resultdict[mcol]['pfmethods']:

                    if 'bls' in pfmk:

                        # add the SNR null results to the BLS result dict
                        resultdict[mcol][pfmk].update({
                            'snr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                            'transitdepth':[np.nan,np.nan,np.nan,
                                            np.nan,np.nan],
                            'transitduration':[np.nan,np.nan,np.nan,
                                               np.nan,np.nan],
                        })

        # once all mag cols have been processed, write out the pickle
        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to run for %s, because: %s' % (lcfile, e))

        if raiseonfail:
            raise

        return None


def _runpf_worker(task):
    '''
    This runs the runpf function.

    '''

    (lcfile, outdir, timecols, magcols, errcols, lcformat, lcformatdir,
     pfmethods, pfkwargs, getblssnr, sigclip, nworkers, minobservations,
     excludeprocessed) = task

    if os.path.exists(lcfile):
        pfresult = runpf(lcfile,
                         outdir,
                         timecols=timecols,
                         magcols=magcols,
                         errcols=errcols,
                         lcformat=lcformat,
                         lcformatdir=lcformatdir,
                         pfmethods=pfmethods,
                         pfkwargs=pfkwargs,
                         getblssnr=getblssnr,
                         sigclip=sigclip,
                         nworkers=nworkers,
                         minobservations=minobservations,
                         excludeprocessed=excludeprocessed)
        return pfresult
    else:
        LOGERROR('LC does not exist for requested file %s' % lcfile)
        return None


def parallel_pf(lclist,
                outdir,
                timecols=None,
                magcols=None,
                errcols=None,
                lcformat='hat-sql',
                lcformatdir=None,
                pfmethods=('gls','pdm','mav','win'),
                pfkwargs=({},{},{},{}),
                sigclip=10.0,
                getblssnr=False,
                nperiodworkers=NCPUS,
                ncontrolworkers=1,
                liststartindex=None,
                listmaxobjects=None,
                minobservations=500,
                excludeprocessed=True):
    '''This drives the overall parallel period processing for a list of LCs.

    As a rough benchmark, 25000 HATNet light curves with up to 50000 points per
    LC take about 26 days in total for an invocation of this function using
    GLS+PDM+BLS, 10 periodworkers, and 4 controlworkers (so all 40 'cores') on a
    2 x Xeon E5-2660v3 machine.

    Parameters
    ----------

    lclist : list of str
        The list of light curve file to process.

    outdir : str
        The output directory where the period-finding result pickles will go.

    timecols : list of str or None
        The timecol keys to use from the lcdict in calculating the features.

    magcols : list of str or None
        The magcol keys to use from the lcdict in calculating the features.

    errcols : list of str or None
        The errcol keys to use from the lcdict in calculating the features.

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

    pfmethods : list of str
        This is a list of period finding methods to run. Each element is a
        string matching the keys of the `PFMETHODS` dict above. By default, this
        runs GLS, PDM, AoVMH, and the spectral window Lomb-Scargle periodogram.

    pfkwargs : list of dicts
        This is used to provide any special kwargs as dicts to each
        period-finding method function specified in `pfmethods`.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    getblssnr : bool
        If this is True and BLS is one of the methods specified in `pfmethods`,
        will also calculate the stats for each best period in the BLS results:
        transit depth, duration, ingress duration, refit period and epoch, and
        the SNR of the transit.

    nperiodworkers : int
        The number of parallel period-finding workers to launch per object task.

    ncontrolworkers : int
        The number of controlling processes to launch. This effectively sets how
        many objects from `lclist` will be processed in parallel.

    liststartindex : int or None
        This sets the index from where to start in `lclist`.

    listmaxobjects : int or None
        This sets the maximum number of objects in `lclist` to run
        period-finding for in this invocation. Together with `liststartindex`,
        `listmaxobjects` can be used to distribute processing over several
        independent machines if the number of light curves is very large.

    minobservations : int
        The minimum number of finite LC points required to process a light
        curve.

    excludeprocessed : bool
        If this is True, light curves that have existing period-finding result
        pickles in `outdir` will not be processed.

        FIXME: currently, this uses a dumb method of excluding already-processed
        files. A smarter way to do this is to (i) generate a SHA512 cachekey
        based on a repr of `{'lcfile', 'timecols', 'magcols', 'errcols',
        'lcformat', 'pfmethods', 'sigclip', 'getblssnr', 'pfkwargs'}`, (ii) make
        sure all list kwargs in the dict are sorted, (iii) check if the output
        file has the same cachekey in its filename (last 8 chars of cachekey
        should work), so the result was processed in exactly the same way as
        specifed in the input to this function, and can therefore be
        ignored. Will implement this later.

    Returns
    -------

    list of str
        A list of the period-finding pickles created for all of input LCs
        processed.

    '''

    # make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if (liststartindex is not None) and (listmaxobjects is None):
        lclist = lclist[liststartindex:]

    elif (liststartindex is None) and (listmaxobjects is not None):
        lclist = lclist[:listmaxobjects]

    elif (liststartindex is not None) and (listmaxobjects is not None):
        lclist = lclist[liststartindex:liststartindex+listmaxobjects]

    tasklist = [(x, outdir, timecols, magcols, errcols, lcformat, lcformatdir,
                 pfmethods, pfkwargs, getblssnr, sigclip, nperiodworkers,
                 minobservations,
                 excludeprocessed)
                for x in lclist]

    with ProcessPoolExecutor(max_workers=ncontrolworkers) as executor:
        resultfutures = executor.map(_runpf_worker, tasklist)

    results = list(resultfutures)
    return results


def parallel_pf_lcdir(lcdir,
                      outdir,
                      fileglob=None,
                      recursive=True,
                      timecols=None,
                      magcols=None,
                      errcols=None,
                      lcformat='hat-sql',
                      lcformatdir=None,
                      pfmethods=('gls','pdm','mav','win'),
                      pfkwargs=({},{},{},{}),
                      sigclip=10.0,
                      getblssnr=False,
                      nperiodworkers=NCPUS,
                      ncontrolworkers=1,
                      liststartindex=None,
                      listmaxobjects=None,
                      minobservations=500,
                      excludeprocessed=True):
    '''This runs parallel light curve period finding for directory of LCs.

    Parameters
    ----------

    lcdir : str
        The directory containing the LCs to process.

    outdir : str
        The directory where the resulting period-finding pickles will go.

    fileglob : str or None
        The UNIX file glob to use to search for LCs in `lcdir`. If None, the
        default file glob associated with the registered LC format will be used
        instead.

    recursive : bool
        If True, will search recursively in `lcdir` for light curves to process.

    timecols : list of str or None
        The timecol keys to use from the lcdict in calculating the features.

    magcols : list of str or None
        The magcol keys to use from the lcdict in calculating the features.

    errcols : list of str or None
        The errcol keys to use from the lcdict in calculating the features.

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

    pfmethods : list of str
        This is a list of period finding methods to run. Each element is a
        string matching the keys of the `PFMETHODS` dict above. By default, this
        runs GLS, PDM, AoVMH, and the spectral window Lomb-Scargle periodogram.

    pfkwargs : list of dicts
        This is used to provide any special kwargs as dicts to each
        period-finding method function specified in `pfmethods`.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    getblssnr : bool
        If this is True and BLS is one of the methods specified in `pfmethods`,
        will also calculate the stats for each best period in the BLS results:
        transit depth, duration, ingress duration, refit period and epoch, and
        the SNR of the transit.

    nperiodworkers : int
        The number of parallel period-finding workers to launch per object task.

    ncontrolworkers : int
        The number of controlling processes to launch. This effectively sets how
        many objects from `lclist` will be processed in parallel.

    liststartindex : int or None
        This sets the index from where to start in `lclist`.

    listmaxobjects : int or None
        This sets the maximum number of objects in `lclist` to run
        period-finding for in this invocation. Together with `liststartindex`,
        `listmaxobjects` can be used to distribute processing over several
        independent machines if the number of light curves is very large.

    minobservations : int
        The minimum number of finite LC points required to process a light
        curve.

    excludeprocessed : bool
        If this is True, light curves that have existing period-finding result
        pickles in `outdir` will not be processed.

        FIXME: currently, this uses a dumb method of excluding already-processed
        files. A smarter way to do this is to (i) generate a SHA512 cachekey
        based on a repr of `{'lcfile', 'timecols', 'magcols', 'errcols',
        'lcformat', 'pfmethods', 'sigclip', 'getblssnr', 'pfkwargs'}`, (ii) make
        sure all list kwargs in the dict are sorted, (iii) check if the output
        file has the same cachekey in its filename (last 8 chars of cachekey
        should work), so the result was processed in exactly the same way as
        specifed in the input to this function, and can therefore be
        ignored. Will implement this later.

    Returns
    -------

    list of str
        A list of the period-finding pickles created for all of input LCs
        processed.

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
                                          fileglob),recursive=True)

    # now that we have all the files, process them
    if matching and len(matching) > 0:

        # this helps us process things in deterministic order when we distribute
        # processing over several machines
        matching = sorted(matching)

        LOGINFO('found %s light curves, running pf...' % len(matching))

        return parallel_pf(matching,
                           outdir,
                           timecols=timecols,
                           magcols=magcols,
                           errcols=errcols,
                           lcformat=lcformat,
                           lcformatdir=lcformatdir,
                           pfmethods=pfmethods,
                           pfkwargs=pfkwargs,
                           getblssnr=getblssnr,
                           sigclip=sigclip,
                           nperiodworkers=nperiodworkers,
                           ncontrolworkers=ncontrolworkers,
                           liststartindex=liststartindex,
                           listmaxobjects=listmaxobjects,
                           minobservations=minobservations,
                           excludeprocessed=excludeprocessed)

    else:

        LOGERROR('no light curve files in %s format found in %s' % (lcformat,
                                                                    lcdir))
        return None
