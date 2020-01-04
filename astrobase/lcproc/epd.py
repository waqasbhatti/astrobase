#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# epd.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

'''
This contains functions to run External Parameter Decorrelation (EPD) on a large
collection of light curves.

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
from astrobase.varbase.trends import epd_magseries, smooth_magseries_savgol


##################################
## LIGHT CURVE DETRENDING - EPD ##
##################################

def apply_epd_magseries(lcfile,
                        timecol,
                        magcol,
                        errcol,
                        externalparams,
                        lcformat='hat-sql',
                        lcformatdir=None,
                        epdsmooth_sigclip=3.0,
                        epdsmooth_windowsize=21,
                        epdsmooth_func=smooth_magseries_savgol,
                        epdsmooth_extraparams=None):

    '''This applies external parameter decorrelation (EPD) to a light curve.

    Parameters
    ----------

    lcfile : str
        The filename of the light curve file to process.

    timecol,magcol,errcol : str
        The keys in the lcdict produced by your light curve reader function that
        correspond to the times, mags/fluxes, and associated measurement errors
        that will be used as input to the EPD process.

    externalparams : dict or None
        This is a dict that indicates which keys in the lcdict obtained from the
        lcfile correspond to the required external parameters. As with timecol,
        magcol, and errcol, these can be simple keys (e.g. 'rjd') or compound
        keys ('magaperture1.mags'). The dict should look something like::

          {'fsv':'<lcdict key>' array: S values for each observation,
           'fdv':'<lcdict key>' array: D values for each observation,
           'fkv':'<lcdict key>' array: K values for each observation,
           'xcc':'<lcdict key>' array: x coords for each observation,
           'ycc':'<lcdict key>' array: y coords for each observation,
           'bgv':'<lcdict key>' array: sky background for each observation,
           'bge':'<lcdict key>' array: sky background err for each observation,
           'iha':'<lcdict key>' array: hour angle for each observation,
           'izd':'<lcdict key>' array: zenith distance for each observation}

        Alternatively, if these exact keys are already present in the lcdict,
        indicate this by setting externalparams to None.

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

    epdsmooth_sigclip : float or int or sequence of two floats/ints or None
        This specifies how to sigma-clip the input LC before fitting the EPD
        function to it.

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

    epdsmooth_windowsize : int
        This is the number of LC points to smooth over to generate a smoothed
        light curve that will be used to fit the EPD function.

    epdsmooth_func : Python function
        This sets the smoothing filter function to use. A Savitsky-Golay filter
        is used to smooth the light curve by default. The functions that can be
        used with this kwarg are listed in `varbase.trends`. If you want to use
        your own function, it MUST have the following signature::

                def smoothfunc(mags_array, window_size, **extraparams)

        and return a numpy array of the same size as `mags_array` with the
        smoothed time-series. Any extra params can be provided using the
        `extraparams` dict.

    epdsmooth_extraparams : dict
        This is a dict of any extra filter params to supply to the smoothing
        function.

    Returns
    -------

    str
        Writes the output EPD light curve to a pickle that contains the lcdict
        with an added `lcdict['epd']` key, which contains the EPD times,
        mags/fluxes, and errs as `lcdict['epd']['times']`,
        `lcdict['epd']['mags']`, and `lcdict['epd']['errs']`. Returns the
        filename of this generated EPD LC pickle file.

    Notes
    -----

    - S -> measure of PSF sharpness (~1/sigma^2 sosmaller S = wider PSF)
    - D -> measure of PSF ellipticity in xy direction
    - K -> measure of PSF ellipticity in cross direction

    S, D, K are related to the PSF's variance and covariance, see eqn 30-33 in
    A. Pal's thesis: https://arxiv.org/abs/0906.3486

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

    lcdict = readerfunc(lcfile)
    if ((isinstance(lcdict, (tuple, list))) and
        isinstance(lcdict[0], dict)):
        lcdict = lcdict[0]

    objectid = lcdict['objectid']
    times, mags, errs = lcdict[timecol], lcdict[magcol], lcdict[errcol]

    if externalparams is not None:

        fsv = lcdict[externalparams['fsv']]
        fdv = lcdict[externalparams['fdv']]
        fkv = lcdict[externalparams['fkv']]

        xcc = lcdict[externalparams['xcc']]
        ycc = lcdict[externalparams['ycc']]

        bgv = lcdict[externalparams['bgv']]
        bge = lcdict[externalparams['bge']]

        iha = lcdict[externalparams['iha']]
        izd = lcdict[externalparams['izd']]

    else:

        fsv = lcdict['fsv']
        fdv = lcdict['fdv']
        fkv = lcdict['fkv']

        xcc = lcdict['xcc']
        ycc = lcdict['ycc']

        bgv = lcdict['bgv']
        bge = lcdict['bge']

        iha = lcdict['iha']
        izd = lcdict['izd']

    # apply the corrections for EPD
    epd = epd_magseries(
        times,
        mags,
        errs,
        fsv, fdv, fkv, xcc, ycc, bgv, bge, iha, izd,
        magsarefluxes=magsarefluxes,
        epdsmooth_sigclip=epdsmooth_sigclip,
        epdsmooth_windowsize=epdsmooth_windowsize,
        epdsmooth_func=epdsmooth_func,
        epdsmooth_extraparams=epdsmooth_extraparams
    )

    # save the EPD magseries to a pickle LC
    lcdict['epd'] = epd
    outfile = os.path.join(
        os.path.dirname(lcfile),
        '%s-epd-%s-pklc.pkl' % (
            squeeze(objectid).replace(' ','-'),
            magcol
        )
    )
    with open(outfile,'wb') as outfd:
        pickle.dump(lcdict, outfd,
                    protocol=pickle.HIGHEST_PROTOCOL)

    return outfile


def parallel_epd_worker(task):
    '''This is a parallel worker for the function below.

    Parameters
    ----------

    task : tuple
        - task[0] = lcfile
        - task[1] = timecol
        - task[2] = magcol
        - task[3] = errcol
        - task[4] = externalparams
        - task[5] = lcformat
        - task[6] = lcformatdir
        - task[7] = epdsmooth_sigclip
        - task[8] = epdsmooth_windowsize
        - task[9] = epdsmooth_func
        - task[10] = epdsmooth_extraparams

    Returns
    -------

    str or None
        If EPD succeeds for an input LC, returns the filename of the output EPD
        LC pickle file. If it fails, returns None.

    '''

    (lcfile, timecol, magcol, errcol,
     externalparams, lcformat, lcformatdir, magsarefluxes,
     epdsmooth_sigclip, epdsmooth_windowsize,
     epdsmooth_func, epdsmooth_extraparams) = task

    try:

        epd = apply_epd_magseries(lcfile,
                                  timecol,
                                  magcol,
                                  errcol,
                                  externalparams,
                                  lcformat=lcformat,
                                  lcformatdir=lcformatdir,
                                  epdsmooth_sigclip=epdsmooth_sigclip,
                                  epdsmooth_windowsize=epdsmooth_windowsize,
                                  epdsmooth_func=epdsmooth_func,
                                  epdsmooth_extraparams=epdsmooth_extraparams)
        if epd is not None:
            LOGINFO('%s -> %s EPD OK' % (lcfile, epd))
            return epd
        else:
            LOGERROR('EPD failed for %s' % lcfile)
            return None

    except Exception:

        LOGEXCEPTION('EPD failed for %s' % lcfile)
        return None


def parallel_epd_lclist(lclist,
                        externalparams,
                        timecols=None,
                        magcols=None,
                        errcols=None,
                        lcformat='hat-sql',
                        lcformatdir=None,
                        epdsmooth_sigclip=3.0,
                        epdsmooth_windowsize=21,
                        epdsmooth_func=smooth_magseries_savgol,
                        epdsmooth_extraparams=None,
                        nworkers=NCPUS,
                        maxworkertasks=1000):
    '''This applies EPD in parallel to all LCs in the input list.

    Parameters
    ----------

    lclist : list of str
        This is the list of light curve files to run EPD on.

    externalparams : dict or None
        This is a dict that indicates which keys in the lcdict obtained from the
        lcfile correspond to the required external parameters. As with timecol,
        magcol, and errcol, these can be simple keys (e.g. 'rjd') or compound
        keys ('magaperture1.mags'). The dict should look something like::

          {'fsv':'<lcdict key>' array: S values for each observation,
           'fdv':'<lcdict key>' array: D values for each observation,
           'fkv':'<lcdict key>' array: K values for each observation,
           'xcc':'<lcdict key>' array: x coords for each observation,
           'ycc':'<lcdict key>' array: y coords for each observation,
           'bgv':'<lcdict key>' array: sky background for each observation,
           'bge':'<lcdict key>' array: sky background err for each observation,
           'iha':'<lcdict key>' array: hour angle for each observation,
           'izd':'<lcdict key>' array: zenith distance for each observation}

        Alternatively, if these exact keys are already present in the lcdict,
        indicate this by setting externalparams to None.

    timecols,magcols,errcols : lists of str
        The keys in the lcdict produced by your light curve reader function that
        correspond to the times, mags/fluxes, and associated measurement errors
        that will be used as inputs to the EPD process. If these are None, the
        default values for `timecols`, `magcols`, and `errcols` for your light
        curve format will be used here.

    lcformat : str
        This is the `formatkey` associated with your light curve format, which
        you previously passed in to the `lcproc.register_lcformat`
        function. This will be used to look up how to find and read the light
        curve files.

    lcformatdir : str or None
        If this is provided, gives the path to a directory when you've stored
        your lcformat description JSONs, other than the usual directories lcproc
        knows to search for them in. Use this along with `lcformat` to specify
        an LC format JSON file that's not currently registered with lcproc.

    epdsmooth_sigclip : float or int or sequence of two floats/ints or None
        This specifies how to sigma-clip the input LC before fitting the EPD
        function to it.

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

    epdsmooth_windowsize : int
        This is the number of LC points to smooth over to generate a smoothed
        light curve that will be used to fit the EPD function.

    epdsmooth_func : Python function
        This sets the smoothing filter function to use. A Savitsky-Golay filter
        is used to smooth the light curve by default. The functions that can be
        used with this kwarg are listed in `varbase.trends`. If you want to use
        your own function, it MUST have the following signature::

                def smoothfunc(mags_array, window_size, **extraparams)

        and return a numpy array of the same size as `mags_array` with the
        smoothed time-series. Any extra params can be provided using the
        `extraparams` dict.

    epdsmooth_extraparams : dict
        This is a dict of any extra filter params to supply to the smoothing
        function.

    nworkers : int
        The number of parallel workers to launch when processing the LCs.

    maxworkertasks : int
        The maximum number of tasks a parallel worker will complete before it is
        replaced with a new one (sometimes helps with memory-leaks).

    Returns
    -------

    dict
        Returns a dict organized by all the keys in the input `magcols` list,
        containing lists of EPD pickle light curves for that `magcol`.

    Notes
    -----

    - S -> measure of PSF sharpness (~1/sigma^2 sosmaller S = wider PSF)
    - D -> measure of PSF ellipticity in xy direction
    - K -> measure of PSF ellipticity in cross direction

    S, D, K are related to the PSF's variance and covariance, see eqn 30-33 in
    A. Pal's thesis: https://arxiv.org/abs/0906.3486

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

    outdict = {}

    # run by magcol
    for t, m, e in zip(timecols, magcols, errcols):

        tasks = [(x, t, m, e, externalparams, lcformat, lcformatdir,
                  epdsmooth_sigclip, epdsmooth_windowsize,
                  epdsmooth_func, epdsmooth_extraparams) for
                 x in lclist]

        pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
        results = pool.map(parallel_epd_worker, tasks)
        pool.close()
        pool.join()

        outdict[m] = results

    return outdict


def parallel_epd_lcdir(
        lcdir,
        externalparams,
        lcfileglob=None,
        timecols=None,
        magcols=None,
        errcols=None,
        lcformat='hat-sql',
        lcformatdir=None,
        epdsmooth_sigclip=3.0,
        epdsmooth_windowsize=21,
        epdsmooth_func=smooth_magseries_savgol,
        epdsmooth_extraparams=None,
        nworkers=NCPUS,
        maxworkertasks=1000
):
    '''This applies EPD in parallel to all LCs in a directory.

    Parameters
    ----------

    lcdir : str
        The light curve directory to process.

    externalparams : dict or None
        This is a dict that indicates which keys in the lcdict obtained from the
        lcfile correspond to the required external parameters. As with timecol,
        magcol, and errcol, these can be simple keys (e.g. 'rjd') or compound
        keys ('magaperture1.mags'). The dict should look something like::

          {'fsv':'<lcdict key>' array: S values for each observation,
           'fdv':'<lcdict key>' array: D values for each observation,
           'fkv':'<lcdict key>' array: K values for each observation,
           'xcc':'<lcdict key>' array: x coords for each observation,
           'ycc':'<lcdict key>' array: y coords for each observation,
           'bgv':'<lcdict key>' array: sky background for each observation,
           'bge':'<lcdict key>' array: sky background err for each observation,
           'iha':'<lcdict key>' array: hour angle for each observation,
           'izd':'<lcdict key>' array: zenith distance for each observation}

    lcfileglob : str or None
        A UNIX fileglob to use to select light curve files in `lcdir`. If this
        is not None, the value provided will override the default fileglob for
        your light curve format.

    timecols,magcols,errcols : lists of str
        The keys in the lcdict produced by your light curve reader function that
        correspond to the times, mags/fluxes, and associated measurement errors
        that will be used as inputs to the EPD process. If these are None, the
        default values for `timecols`, `magcols`, and `errcols` for your light
        curve format will be used here.

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

    epdsmooth_sigclip : float or int or sequence of two floats/ints or None
        This specifies how to sigma-clip the input LC before fitting the EPD
        function to it.

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

    epdsmooth_windowsize : int
        This is the number of LC points to smooth over to generate a smoothed
        light curve that will be used to fit the EPD function.

    epdsmooth_func : Python function
        This sets the smoothing filter function to use. A Savitsky-Golay filter
        is used to smooth the light curve by default. The functions that can be
        used with this kwarg are listed in `varbase.trends`. If you want to use
        your own function, it MUST have the following signature::

                def smoothfunc(mags_array, window_size, **extraparams)

        and return a numpy array of the same size as `mags_array` with the
        smoothed time-series. Any extra params can be provided using the
        `extraparams` dict.

    epdsmooth_extraparams : dict
        This is a dict of any extra filter params to supply to the smoothing
        function.

    nworkers : int
        The number of parallel workers to launch when processing the LCs.

    maxworkertasks : int
        The maximum number of tasks a parallel worker will complete before it is
        replaced with a new one (sometimes helps with memory-leaks).

    Returns
    -------

    dict
        Returns a dict organized by all the keys in the input `magcols` list,
        containing lists of EPD pickle light curves for that `magcol`.

    Notes
    -----

    - S -> measure of PSF sharpness (~1/sigma^2 sosmaller S = wider PSF)
    - D -> measure of PSF ellipticity in xy direction
    - K -> measure of PSF ellipticity in cross direction

    S, D, K are related to the PSF's variance and covariance, see eqn 30-33 in
    A. Pal's thesis: https://arxiv.org/abs/0906.3486

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

    # find all the files matching the lcglob in lcdir
    if lcfileglob is None:
        lcfileglob = fileglob

    lclist = sorted(glob.glob(os.path.join(lcdir, lcfileglob)))

    return parallel_epd_lclist(
        lclist,
        externalparams,
        timecols=timecols,
        magcols=magcols,
        errcols=errcols,
        lcformat=lcformat,
        epdsmooth_sigclip=epdsmooth_sigclip,
        epdsmooth_windowsize=epdsmooth_windowsize,
        epdsmooth_func=epdsmooth_func,
        epdsmooth_extraparams=epdsmooth_extraparams,
        nworkers=nworkers,
        maxworkertasks=maxworkertasks
    )
