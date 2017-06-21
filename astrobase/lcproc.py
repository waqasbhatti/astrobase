#!/usr/bin/env python

'''lcproc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - May 2017

This contains functions that serve as examples for running large batch jobs
processing HAT light curves.

'''

import os.path
import pickle
import gzip
import glob
import multiprocessing as mp
import logging
from datetime import datetime
from traceback import format_exc


import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    from tqdm import tqdm
    TQDM = True
except:
    TQDM = False
    pass


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

from astrobase import hatlc, periodbase, checkplot
from astrobase.varbase import features



#######################
## UTILITY FUNCTIONS ##
#######################

def getlclist(listfile,
              basedir,
              minndet=999):
    '''
    This gets the list of HATIDs from the file.

    The file should be a CSV with the following columns:

    hatid,ra,decl,ndet,sdssr,lcfpath,...other columns,...

    '''

    lclist = np.genfromtxt(listfile,
                           usecols=(0,1,2,3,4,5),
                           delimiter=',',
                           names=['hatid','ra','decl',
                                  'ndet','sdssr','lcfpath'],
                           dtype='U20,f8,f8,i8,f8,U100')

    goodind = np.where(lclist['ndet'] > minndet)

    LOGINFO('objects with at least %s detections = %s' % (minndet,
                                                        goodind[0].size))

    goodlcs = lclist['lcfpath'][goodind[0]].tolist()
    goodlcs = [os.path.join(basedir, x) for x in goodlcs]

    return goodlcs


##################################
## GETTING VARIABILITY FEATURES ##
##################################

def varfeatures(lcfile,
                outdir,
                magcols=['aep_000','atf_000'],
                errcol='aie_000'):
    '''
    This runs varfeatures on a single LC file.

    '''

    try:

        lcd, msg = hatlc.read_and_filter_sqlitecurve(lcfile)

        resultdict = {'objectid':lcd['objectid'],
                      'info':lcd['objectinfo']}

        # normalize by instrument
        normlc = hatlc.normalize_lcdict_byinst(lcd,
                                               magcols=','.join(magcols),
                                               normto='sdssr')

        for col in magcols:

            times, mags, errs = normlc['rjd'], normlc[col], normlc[errcol]
            finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)

            if mags[finind].size < 1000:

                LOGINFO('not enough LC points: %s in normalized %s LC: %s' %
                      (mags[finind].size, col, os.path.basename(lcfile)))
                resultdict[col] = None

            else:

                lcfeatures = features.all_nonperiodic_features(
                    times, mags, errs
                )
                resultdict[col] = lcfeatures

        outfile = os.path.join(outdir,
                               'varfeatures-%s.pkl' % lcd['objectid'])

        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=4)

        return outfile

    except Exception as e:

        LOGINFO('failed to get LC features for %s because: %s' %
              (os.path.basename(lcfile), e))
        return None



def varfeatures_worker(task):
    '''
    This wraps varfeatures.

    '''

    lcfile, outdir = task
    return varfeatures(lcfile, outdir)



def parallel_varfeatures(lclist,
                         outdir,
                         nworkers=None):
    '''
    This runs varfeatures in parallel for all light curves in lclist.

    '''

    pool = mp.Pool(nworkers)

    tasks = [(x,outdir) for x in lclist]

    results = pool.map(varfeatures_worker, tasks)
    pool.close()
    pool.join()

    resdict = {os.path.basename(x):y for (x,y) in zip(lclist, results)}

    return resdict



def stetson_threshold(featuresdir,
                      magcol='aep_000',
                      minstetstdev=2.0,
                      outfile=None):
    '''This generates a list of objects with J > minstetj.

    Use this to pare down the objects to review and put through
    period-finding.

    '''

    pklist = glob.glob(os.path.join(featuresdir, 'varfeatures-HAT*.pkl'))

    varfeatures = {}
    objids = []
    objmags = []
    objstets = []

    LOGINFO('getting all objects...')

    # fancy progress bar with tqdm if present
    if TQDM:
        listiterator = tqdm(pklist)
    else:
        listiterator = pklist

    for pkl in listiterator:

        with open(pkl,'rb') as infd:
            thisfeatures = pickle.load(infd)

        hatid = thisfeatures['info']['hatid']
        varfeatures[hatid] = thisfeatures
        objids.append(hatid)

        if thisfeatures['info']['sdssr']:
            objmags.append(thisfeatures['info']['sdssr'])
        else:
            objmags.append(np.nan)

        if (magcol in thisfeatures and
            thisfeatures[magcol] and
            thisfeatures[magcol]['stetsonj']):
            objstets.append(thisfeatures[magcol]['stetsonj'])
        else:
            objstets.append(np.nan)


    objids = np.array(objids)
    objmags = np.array(objmags)
    objstets = np.array(objstets)

    medstet = np.nanmedian(objstets)
    madstet = np.nanmedian(np.abs(objstets - np.nanmedian(objstets)))
    stdstet = 1.483*madstet

    threshind = objstets > (minstetstdev*stdstet + medstet)

    goodobjs = objids[threshind]

    LOGINFO('median %s stetson J = %.5f, stdev = %s, '
          'total objects %s sigma > median = %s' %
          (magcol, medstet, stdstet, minstetstdev, goodobjs.size))

    return varfeatures, objmags, objstets, goodobjs



#############################
## RUNNING PERIOD SEARCHES ##
#############################

def runpf(lcfile, resultdir,
          magcols=['aep_000','atf_000'],
          errcol='aie_000',
          nworkers=10):
    '''
    This runs the period-finding for a single LC.

    '''

    try:

        lcd, msg = hatlc.read_and_filter_sqlitecurve(lcfile)
        outfile = os.path.join(resultdir, 'pfresult-%s.pkl' % lcd['objectid'])
        resultdict = {'objectid':lcd['objectid']}

        # normalize by instrument
        normlc = hatlc.normalize_lcdict_byinst(lcd,
                                               magcols=','.join(magcols),
                                               normto='sdssr')

        for col in magcols:

            times, mags, errs = normlc['rjd'], normlc[col], normlc[errcol]

            gls = periodbase.pgen_lsp(times, mags, errs,
                                      verbose=False,
                                      nworkers=nworkers)
            pdm = periodbase.stellingwerf_pdm(times, mags, errs,
                                              verbose=False,
                                              nworkers=nworkers)

            # specifically for planet type signals
            bls = periodbase.bls_parallel_pfind(times, mags, errs,
                                                startp=1.0,
                                                maxtransitduration=0.3,
                                                verbose=False,
                                                nworkers=nworkers)

            resultdict[col] = {'gls':gls,
                               'bls':bls,
                               'pdm':pdm}

        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=4)

        return outfile

    except Exception as e:

        LOGERROR('failed to run for %s, because: %s' % (lcfile, e))



def runpf_worker(task):
    '''
    This runs the runpf function.

    '''

    hatid, lcbasedir, outdir, magcols, errcol, nworkers = task

    hatfield = hatid.split('-')[1]

    # find the light curve for this object
    lcfpath = os.path.join(lcbasedir,
                           hatfield,
                           '%s-V0-DR0-hatlc.sqlite.gz' % hatid)

    if os.path.exists(lcfpath):
        pfresult = runpf(lcfpath, outdir,
                         magcols=magcols,
                         errcol=errcol,
                         nworkers=nworkers)
        return pfresult
    else:
        LOGERROR('LC does not exist for %s' % hatid)
        return None



def parallel_pf(hatidlistfile,
                outdir,
                lcbasedir,
                magcols=['aep_000'],
                errcol='aie_000',
                nperiodworkers=10,
                nthisworkers=4):
    '''
    This drives the overall parallel period processing.

    '''

    with open(hatidlistfile,'r') as infd:
        hatidlist = infd.readlines()
        hatidlist = [x.strip('\n') for x in hatidlist]

    tasklist = [(x, lcbasedir, outdir, magcols, errcol, nperiodworkers) for
                x in hatidlist]

    with ProcessPoolExecutor(max_workers=nthisworkers) as executor:
        resultfutures = executor.map(runpf_worker, tasklist)

    results = [x.result() for x in resultfutures]
    return results


########################
## RUNNING CHECKPLOTS ##
########################

def runcp(pfpickle,
          resultdir,
          lcbasedir,
          usehatfielddir=False,
          magcols=['aep_000'],
          errcol='aie_000'):
    '''This runs a checkplot for the given period-finding result pickle
    produced by runpf.

    '''

    with open(pfpickle,'rb') as infd:
        pfresults = pickle.load(infd)

    hatid = pfresults['objectid']
    hatfield = hatid.split('-')[1]

    if usehatfielddir:

        # find the light curve for this object
        lcfpath = os.path.join(lcbasedir,
                               hatfield,
                               '%s-V0-DR0-hatlc.sqlite.gz' % hatid)

    else:

        lcfpath = os.path.join(lcbasedir,
                               '%s-V0-DR0-hatlc.sqlite.gz' % hatid)


    if os.path.exists(lcfpath):

        lcd, msg = hatlc.read_and_filter_sqlitecurve(lcfpath)

        # normalize by instrument
        normlc = hatlc.normalize_lcdict_byinst(lcd,
                                               magcols=','.join(magcols),
                                               normto='sdssr')

        cpfs = []

        for col in magcols:

            times, mags, errs = normlc['rjd'], normlc[col], normlc[errcol]

            gls = pfresults[col]['gls']
            pdm = pfresults[col]['pdm']
            bls = pfresults[col]['bls']

            outfile = os.path.join(resultdir,
                                   'checkplot-%s-%s.pkl' % (hatid, col))

            cpf = checkplot.checkplot_pickle(
                [gls,pdm,bls],
                times, mags, errs,
                objectinfo=lcd['objectinfo'],
                outfile=outfile,
                verbose=False
            )
            cpfs.append(cpf)

        LOGINFO('done with %s -> %s' % (hatid, repr(cpfs)))
        return cpfs

    else:

        LOGERROR('LC does not exist for %s' % hatid)
        return None


def runcp_worker(task):
    '''
    This is the worker for running checkplots.

    '''

    pfpickle, resultdir, kwargs = task

    try:

        return runcp(pfpickle, resultdir, **kwargs)

    except Exception as e:

        LOGERROR(' could not make checkplots for %s: %s' % (pfpickle,
                                                            e))
        return None



def parallel_cp(pfpickledir,
                outdir,
                lcbasedir,
                pfpickleglob='pfresult*.pkl',
                magcols=['aep_000'],
                errcol='aie_000',
                nworkers=32):
    '''
    This drives the parallel execution of runcp.

    '''

    pfpicklelist = sorted(glob.glob(os.path.join(pfpickledir, pfpickleglob)))

    tasklist = [(x, outdir, {'lcbasedir':lcbasedir,
                             'magcols':magcols,
                             'errcol':errcol}) for
                x in pfpicklelist]

    resultfutures = []
    results = []

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(runcp_worker, tasklist)

    results = [x.result() for x in resultfutures]

    executor.shutdown()
    return results
