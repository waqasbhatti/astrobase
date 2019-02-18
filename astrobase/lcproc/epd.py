#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''epd.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

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

try:
    import cPickle as pickle
except Exception as e:
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
def dict_get(datadict, keylist):
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

    '''This applies EPD to a light curve.

    lcfile is the name of the file to read for times, mags, errs.

    timecol, magcol, errcol are the columns in the lcdict to use for EPD.

    externalparams is a dict that indicates which keys in the lcdict obtained
    from the lcfile correspond to the required external parameters. As with
    timecol, magcol, and errcol, these can be simple keys (e.g. 'rjd') or
    compound keys ('magaperture1.mags'). The dict should look something like:

    {'fsv':'<lcdict key>' -> ndarray: S values for each observation,
     'fdv':'<lcdict key>' -> ndarray: D values for each observation,
     'fkv':'<lcdict key>' -> ndarray: K values for each observation,
     'xcc':'<lcdict key>' -> ndarray: x coords for each observation,
     'ycc':'<lcdict key>' -> ndarray: y coords for each observation,
     'bgv':'<lcdict key>' -> ndarray: sky background for each observation,
     'bge':'<lcdict key>' -> ndarray: sky background err for each observation,
     'iha':'<lcdict key>' -> ndarray: hour angle for each observation,
     'izd':'<lcdict key>' -> ndarray: zenith distance for each observation}

    Alternatively, if these exact keys are already present in the lcdict,
    indicate this by setting externalparams to None.

    Note: S -> measure of PSF sharpness (~ 1/sigma^2 -> smaller S -> wider PSF)
          D -> measure of PSF ellipticity in xy direction
          K -> measure of PSF ellipticity in cross direction

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
    except Exception as e:
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
    '''
    This is a parallel worker for the function below.

    task[0] = lcfile
    task[1] = timecol
    task[2] = magcol
    task[3] = errcol
    task[4] = externalparams
    task[5] = lcformat
    task[6] = lcformatdir
    task[7] = epdsmooth_sigclip
    task[8] = epdsmooth_windowsize
    task[9] = epdsmooth_func
    task[10] = epdsmooth_extraparams

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

    except Exception as e:

        LOGEXCEPTION('EPD failed for %s' % lcfile)
        return None



def parallel_epd_lclist(lclist,
                        externalparams,
                        timecols=None,
                        magcols=None,
                        errcols=None,
                        lcformat='hat-sql',
                        lcformatdir=None,
                        magsarefluxes=False,
                        epdsmooth_sigclip=3.0,
                        epdsmooth_windowsize=21,
                        epdsmooth_func=smooth_magseries_savgol,
                        epdsmooth_extraparams=None,
                        nworkers=NCPUS,
                        maxworkertasks=1000):
    '''
    This applies EPD in parallel to all LCs in lclist.

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
    except Exception as e:
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
        lcfileglob,
        externalparams,
        timecols=None,
        magcols=None,
        errcols=None,
        lcformat='hat-sql',
        lcformatdir=None,
        magsarefluxes=False,
        epdsmooth_sigclip=3.0,
        epdsmooth_windowsize=21,
        epdsmooth_func=smooth_magseries_savgol,
        epdsmooth_extraparams=None,
        nworkers=NCPUS,
        maxworkertasks=1000
):
    '''
    This applies EPD in parallel to all LCs in lcdir.

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
    except Exception as e:
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
        magsarefluxes=magsarefluxes,
        epdsmooth_sigclip=epdsmooth_sigclip,
        epdsmooth_windowsize=epdsmooth_windowsize,
        epdsmooth_func=epdsmooth_func,
        epdsmooth_extraparams=epdsmooth_extraparams,
        nworkers=nworkers,
        maxworkertasks=maxworkertasks
    )
