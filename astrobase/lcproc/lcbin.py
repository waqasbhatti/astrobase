#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''lcformat.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

This contains functions to import various light curve formats.

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

from astrobase.lcproc import LCFORM
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
              timecols=None,
              magcols=None,
              errcols=None,
              minbinelems=7):

    '''
    This bins the given light curve file in time using binsizesec.

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
        times = dict_get(lcdict, tcolget)

        if '.' in mcol:
            mcolget = mcol.split('.')
        else:
            mcolget = [mcol]
        mags = dict_get(lcdict, mcolget)

        if '.' in ecol:
            ecolget = ecol.split('.')
        else:
            ecolget = [ecol]
        errs = dict_get(lcdict, ecolget)

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

    task[0] = lcfile
    task[1] = binsizesec
    task[3] = {'outdir','lcformat','timecols','magcols','errcols','minbinelems'}

    '''

    lcfile, binsizesec, kwargs = task

    try:
        binnedlc = timebinlc(lcfile, binsizesec, **kwargs)
        LOGINFO('%s binned using %s sec -> %s OK' %
                (lcfile, binsizesec, binnedlc))
        return binnedlc
    except Exception as e:
        LOGEXCEPTION('failed to bin %s using binsizesec = %s' % (lcfile,
                                                                 binsizesec))
        return None



def parallel_timebin(lclist,
                     binsizesec,
                     maxobjects=None,
                     outdir=None,
                     lcformat='hat-sql',
                     timecols=None,
                     magcols=None,
                     errcols=None,
                     minbinelems=7,
                     nworkers=NCPUS,
                     maxworkertasks=1000):
    '''
    This bins all the light curves in lclist using binsizesec.

    '''

    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

    if maxobjects is not None:
        lclist = lclist[:maxobjects]

    tasks = [(x, binsizesec, {'outdir':outdir,
                              'lcformat':lcformat,
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
                           timecols=None,
                           magcols=None,
                           errcols=None,
                           minbinelems=7,
                           nworkers=NCPUS,
                           maxworkertasks=1000):
    '''
    This bins all the light curves in lcdir using binsizesec.

    '''

    # get the light curve glob associated with specified lcformat
    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

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
