#!/usr/bin/env python

'''lcproc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - May 2017

This contains functions that serve as examples for running large batch jobs
processing HAT light curves.

'''

import os
import os.path
import sys
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

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem
def dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)



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

# LC reading functions
from astrobase.hatlc import read_and_filter_sqlitecurve, read_csvlc, \
    normalize_lcdict_byinst
from astrobase.hplc import read_hatpi_textlc, read_hatpi_pklc
from astrobase.astrokep import read_kepler_fitslc, read_kepler_pklc

from astrobase import hatlc, periodbase, checkplot
from astrobase.varbase import features
from astrobase.lcmath import normalize_magseries, time_bin_magseries_with_errs
from astrobase.periodbase.kbls import bls_snr


#############################################
## MAPS FOR LCFORMAT TO LCREADER FUNCTIONS ##
#############################################

# LC format -> [default fileglob,  function to read LC format]
LCFORM = {
    'hat-sql':[
        '*-hatlc.sqlite.gz',           # default fileglob
        read_and_filter_sqlitecurve,   # function to read this LC
        ['rjd','rjd'],                 # default timecols to use for period/var
        ['aep_000','atf_000'],         # default magcols to use for period/var
        ['aie_000','aie_000'],         # default errcols to use for period/var
        False,                         # default magsarefluxes = False
        normalize_lcdict_byinst,       # default special normalize function
    ],
    'hat-csv':[
        '*-hatlc.csv.gz',
        read_csvlc,
        ['rjd','rjd'],
        ['aep_000','atf_000'],
        ['aie_000','aie_000'],
        False,
        normalize_lcdict_byinst,
    ],
    'hp-txt':[
        '*TF1.gz',
        read_hatpi_textlc,
        ['rjd','rjd'],
        ['iep1','itf1'],
        ['ire1','ire1'],
        False,
        None,
    ],
    'hp-pkl':[
        '*-pklc.pkl',
        read_hatpi_pklc,
        ['rjd','rjd'],
        ['iep1','itf1'],
        ['ire1','ire1'],
        False,
        None,
    ],
    'kep-fits':[
        '*_llc.fits',
        read_kepler_fitslc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdc_sapflux'],
        ['sap.sap_flux_err','pdc.pdc_sapflux_err'],
        True,
        None,
    ],
    'kep-pkl':[
        '-keplc.pkl',
        read_kepler_pklc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdc_sapflux'],
        ['sap.sap_flux_err','pdc.pdc_sapflux_err'],
        True,
        None,
    ]
}


#######################
## UTILITY FUNCTIONS ##
#######################


def makelclist(basedir,
               outfile,
               lcformat='hat-sql',
               fileglob=None,
               recursive=True,
               columns=['objectid',
                        'objectinfo.ra','objectinfo.decl',
                        'objectinfo.ndet','objectinfo.sdssr'],
               colformats=['%s','%.5f','%.5f','%d','%.3f']):
    '''This generates a list file compatible with getlclist below.

    Given a base directory where all the files are, and a light curve format,
    this will find all light curves, pull out the columns requested, and write
    them to the requested output CSV file.

    fileglob is a shell glob to use to select the filenames. If None, then the
    default one for the provided lcformat will be used.

    If recursive is True, then the function will search recursively in basedir
    for any light curves matching the specified criteria. This may take a while,
    especially on network filesystems.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR("can't figure out the light curve format")
        return

    if not fileglob:
        fileglob = LCFORM[lcformat][0]

    readerfunc = LCFORM[lcformat][1]

    # now find the files
    LOGINFO('searching for %s light curves in %s ...' % (lcformat, basedir))

    if recursive == False:
        matching = glob.glob(os.path.join(basedir, fileglob))

    else:
        # use recursive glob for Python 3.5+
        if sys.version_info[:2] > (3,4):

            matching = glob.glob(os.path.join(basedir,
                                              '**',
                                              fileglob),recursive=True)

        # otherwise, use os.walk and glob
        else:

            # use os.walk to go through the directories
            walker = os.walk(basedir)
            matching = []

            for root, dirs, files in walker:
                for sdir in dirs:
                    searchpath = os.path.join(root,
                                              sdir,
                                              fileglob)
                    foundfiles = glob.glob(searchpath)

                    if foundfiles:
                        matching.extend(foundfiles)


    # now that we have all the files, process them
    if matching and len(matching) > 0:

        LOGINFO('collecting light curve info...')

        # open the output file
        outfd = open(outfile, 'w')
        outfd.write(' '.join(columns) + ' lcfpath\n')

        # generate the column format for each line
        lineform = ' '.join(colformats)

        if TQDM:
            lciter = tqdm(matching)
        else:
            lciter = matching

        for lcf in lciter:

            lcdict = readerfunc(lcf)

            thisline = []

            for colkey in columns:
                if '.' in colkey:
                    getkey = colkey.split('.')
                else:
                    getkey = [colkey]

                try:
                    thiscolval = dict_get(lcdict, getkey)
                except:
                    thiscolval = np.nan

                thisline.append(thiscolval)

            outfd.write('%s %s\n' % (lineform % tuple(thisline),
                                     os.path.basename(lcf)))

        # done with collecting info
        outfd.close()

        LOGINFO('done. LC info -> %s' % outfile)
        return outfile

    else:

        LOGERROR('no files found in %s matching %s' % (basedir, fileglob))
        return None



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
                mindet=1000,
                lcformat='hat-sql'):
    '''
    This runs varfeatures on a single LC file.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    (readerfunc, timecols, magcols,
     errcols, magsarefluxes, normfunc) = LCFORM[lcformat][1:]

    try:

        # get the LC into a dict
        lcdict = readerfunc(lcfile)
        if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
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


            # make sure we have finite values
            finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)

            # make sure we have enough finite values
            if mags[finind].size < mindet:

                LOGINFO('not enough LC points: %s in normalized %s LC: %s' %
                      (mags[finind].size, mcol, os.path.basename(lcfile)))
                resultdict[mcol] = None

            else:

                lcfeatures = features.all_nonperiodic_features(
                    times, mags, errs
                )
                resultdict[mcol] = lcfeatures

        outfile = os.path.join(outdir,
                               'varfeatures-%s.pkl' % resultdict['objectid'])

        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=4)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to get LC features for %s because: %s' %
              (os.path.basename(lcfile), e))
        return None



def varfeatures_worker(task):
    '''
    This wraps varfeatures.

    '''

    lcfile, outdir, mindet, lcformat = task
    return varfeatures(lcfile, outdir, mindet=mindet, lcformat=lcformat)



def parallel_varfeatures(lclist,
                         outdir,
                         mindet=1000,
                         lcformat='hat-sql',
                         nworkers=None):
    '''
    This runs varfeatures in parallel for all light curves in lclist.

    '''

    pool = mp.Pool(nworkers)

    tasks = [(x,outdir, mindet, lcformat) for x in lclist]

    results = pool.map(varfeatures_worker, tasks)
    pool.close()
    pool.join()

    resdict = {os.path.basename(x):y for (x,y) in zip(lclist, results)}

    return resdict



def parallel_varfeatures_lcdir(lcdir,
                               outdir,
                               recursive=True,
                               mindet=1000,
                               lcformat='hat-sql',
                               nworkers=None):
    '''
    This runs parallel variable feature extraction for a directory of LCs.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    fileglob = LCFORM[lcformat][0]

    # now find the files
    LOGINFO('searching for %s light curves in %s ...' % (lcformat, lcdir))

    if recursive == False:
        matching = glob.glob(os.path.join(lcdir, fileglob))

    else:
        # use recursive glob for Python 3.5+
        if sys.version_info[:2] > (3,4):

            matching = glob.glob(os.path.join(lcdir,
                                              '**',
                                              fileglob),recursive=True)

        # otherwise, use os.walk and glob
        else:

            # use os.walk to go through the directories
            walker = os.walk(lcdir)
            matching = []

            for root, dirs, files in walker:
                for sdir in dirs:
                    searchpath = os.path.join(root,
                                              sdir,
                                              fileglob)
                    foundfiles = glob.glob(searchpath)

                    if foundfiles:
                        matching.extend(foundfiles)


    # now that we have all the files, process them
    if matching and len(matching) > 0:

        return parallel_varfeatures(matching,
                                    outdir,
                                    mindet=mindet,
                                    lcformat=lcformat,
                                    nworkers=nworkers)

    else:

        LOGERROR('no light curve files in %s format found in %s' % (lcformat,
                                                                    lcdir))
        return None



def stetson_threshold(featuresdir,
                      lcformat='hat-sql',
                      minstetstdev=2.0,
                      outfile=None):
    '''This generates a list of objects with J > minstetj.

    Use this to pare down the objects to review and put through
    period-finding.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    # get the magnitude columns to use from the lcformat
    magcols = LCFORM[lcformat][3]

    # list of input pickles generated by varfeatures functions above
    pklist = glob.glob(os.path.join(featuresdir, 'varfeatures-*.pkl'))

    allobjects = {}

    for magcol in magcols:

        LOGINFO('getting all objects with stet J > %s x sigma for %s' %
                (minstetstdev, magcol))

        allobjects[magcol] = {'objectid':[], 'stetsonj':[]}

        # fancy progress bar with tqdm if present
        if TQDM:
            listiterator = tqdm(pklist)
        else:
            listiterator = pklist

        for pkl in listiterator:

            with open(pkl,'rb') as infd:
                thisfeatures = pickle.load(infd)

            objectid = thisfeatures['objectid']

            if (magcol in thisfeatures and
                thisfeatures[magcol] and
                thisfeatures[magcol]['stetsonj']):
                stetsonj = thisfeatures[magcol]['stetsonj']
            else:
                stetsonj = np.nan

            allobjects[magcol]['objectid'].append(objectid)
            allobjects[magcol]['stetsonj'].append(stetsonj)

        allobjects[magcol]['objectid'] = np.array(allobjects[magcol]['objectid'])
        allobjects[magcol]['stetsonj'] = np.array(allobjects[magcol]['stetsonj'])

        medstet = np.nanmedian(allobjects[magcol]['stetsonj'])
        madstet = np.nanmedian(
            np.abs(allobjects[magcol]['stetsonj'] -
                   np.nanmedian(allobjects[magcol]['stetsonj']))
        )
        stdstet = 1.483*madstet

        threshind = (
            (np.isfinite(allobjects[magcol]['stetsonj']) &
             (allobjects[magcol]['stetsonj'] > (minstetstdev*stdstet + medstet)))
        )

        allobjects[magcol]['thresholdobjects'] = (
            allobjects[magcol]['objectid'][threshind]
        )
        allobjects[magcol]['median_stetj'] = medstet
        allobjects[magcol]['mad_stetj'] = madstet
        allobjects[magcol]['stdev_stetj'] = stdstet

        LOGINFO('median %s stetson J = %.5f, stdev = %s, '
              'total objects %s sigma > median = %s' %
              (magcol, medstet, stdstet, minstetstdev,
               allobjects[magcol]['thresholdobjects'].size))

    # get the overall stetson threshold objects too
    allobjects['overallthreshold'] = allobjects[magcols[0]]['thresholdobjects']

    for magcol in magcols[1:]:
        allobjects['overallthreshold'] = (
            np.intersect1d(allobjects['overallthreshold'],
                           allobjects[magcol]['thresholdobjects'])
        )

    LOGINFO('objects above stetson threshold across all magcols: %s' %
            allobjects['overallthreshold'].size)

    return allobjects



#############################
## RUNNING PERIOD SEARCHES ##
#############################

def runpf(lcfile,
          outdir,
          lcformat='hat-sql',
          bls_startp=1.0,
          bls_maxtransitduration=0.3,
          nworkers=10):
    '''
    This runs the period-finding for a single LC.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    (readerfunc, timecols, magcols,
     errcols, magsarefluxes, normfunc) = LCFORM[lcformat][1:]


    try:

        # get the LC into a dict
        lcdict = readerfunc(lcfile)
        if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
            lcdict = lcdict[0]

        outfile = os.path.join(outdir, 'periodfinding-%s.pkl' %
                               lcdict['objectid'])
        resultdict = {'objectid':lcdict['objectid'],
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


            # run the three period-finders
            gls = periodbase.pgen_lsp(times, mags, errs,
                                      verbose=False,
                                      nworkers=nworkers,
                                      magsarefluxes=magsarefluxes)

            pdm = periodbase.stellingwerf_pdm(times, mags, errs,
                                              verbose=False,
                                              nworkers=nworkers,
                                              magsarefluxes=magsarefluxes)

            # specifically for planet type signals
            bls = periodbase.bls_parallel_pfind(
                times, mags, errs,
                startp=bls_startp,
                maxtransitduration=bls_maxtransitduration,
                verbose=False,
                nworkers=nworkers,
                magsarefluxes=magsarefluxes
            )

            # calculate the SNR for the BLS as well
            blssnr = bls_snr(bls, times, mags, errs,
                             magsarefluxes=magsarefluxes,
                             verbose=False)

            # save the results
            resultdict[mcol] = {'gls':gls,
                                'bls':bls,
                                'pdm':pdm}

            # add the SNR results to the BLS result dict
            resultdict[mcol]['bls'].update({
                'snr':blssnr['snr'],
                'altsnr':blssnr['altsnr'],
                'transitdepth':blssnr['transitdepth'],
                'transitduration':blssnr['transitduration'],
            })

        # once all mag cols have been processed, write out the pickle
        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to run for %s, because: %s' % (lcfile, e))



def runpf_worker(task):
    '''
    This runs the runpf function.

    '''

    (lcfile, outdir, lcformat,
     bls_startp, bls_maxtransitduration, nworkers) = task

    if os.path.exists(lcfile):
        pfresult = runpf(lcfile,
                         outdir,
                         lcformat=lcformat,
                         bls_startp=bls_startp,
                         bls_maxtransitduration=bls_maxtransitduration,
                         nworkers=nworkers)
        return pfresult
    else:
        LOGERROR('LC does not exist for requested file %s' % lcfile)
        return None



def parallel_pf(lclist,
                outdir,
                lcformat='hat-sql',
                bls_startp=1.0,
                bls_maxtransitduration=0.3,
                nperiodworkers=10,
                nthisworkers=4):
    '''
    This drives the overall parallel period processing.

    '''

    tasklist = [(x, outdir, lcformat,
                 bls_startp, bls_maxtransitduration, nperiodworkers)
                for x in lclist]

    with ProcessPoolExecutor(max_workers=nthisworkers) as executor:
        resultfutures = executor.map(runpf_worker, tasklist)

    results = [x for x in resultfutures]
    return results



def parallel_pf_lcdir(lcdir,
                      outdir,
                      recursive=True,
                      lcformat='hat-sql',
                      bls_startp=1.0,
                      bls_maxtransitduration=0.3,
                      nperiodworkers=10,
                      nthisworkers=4):
    '''
    This runs parallel light curve period finding for directory of LCs.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    fileglob = LCFORM[lcformat][0]

    # now find the files
    LOGINFO('searching for %s light curves in %s ...' % (lcformat, lcdir))

    if recursive == False:
        matching = glob.glob(os.path.join(lcdir, fileglob))

    else:
        # use recursive glob for Python 3.5+
        if sys.version_info[:2] > (3,4):

            matching = glob.glob(os.path.join(lcdir,
                                              '**',
                                              fileglob),recursive=True)

        # otherwise, use os.walk and glob
        else:

            # use os.walk to go through the directories
            walker = os.walk(lcdir)
            matching = []

            for root, dirs, files in walker:
                for sdir in dirs:
                    searchpath = os.path.join(root,
                                              sdir,
                                              fileglob)
                    foundfiles = glob.glob(searchpath)

                    if foundfiles:
                        matching.extend(foundfiles)


    # now that we have all the files, process them
    if matching and len(matching) > 0:

        return parallel_pf(matching,
                           outdir,
                           lcformat=lcformat,
                           bls_startp=bls_startp,
                           bls_maxtransitduration=bls_maxtransitduration,
                           nperiodworkers=nperiodworkers,
                           nthisworkers=nthisworkers)

    else:

        LOGERROR('no light curve files in %s format found in %s' % (lcformat,
                                                                    lcdir))
        return None



########################
## RUNNING CHECKPLOTS ##
########################

def runcp(pfpickle,
          outdir,
          lcbasedir,
          lcformat='hat-sql',
          timecols=None,
          magcols=None,
          errcols=None):
    '''This runs a checkplot for the given period-finding result pickle
    produced by runpf.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    # get the pickled period-finding results
    with open(pfpickle,'rb') as infd:
        pfresults = pickle.load(infd)

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

    objectid = pfresults['objectid']


    # find the light curve in lcbasedir
    lcfsearchpath = os.path.join(lcbasedir,
                                 '%s-%s' % (objectid, fileglob))

    matching = glob.glob(lcfsearchpath)

    if matching and len(matching) > 0:
        lcfpath = matching[0]
    else:
        LOGERROR('could not find light curve for pfresult %s, objectid %s' %
                 (pfpickle, objectid))
        return None


    lcdict = readerfunc(lcfpath)
    if isinstance(lcdict, tuple) and isinstance(lcdict[0], dict):
        lcdict = lcdict[0]

    cpfs = []

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

        gls = pfresults[mcol]['gls']
        pdm = pfresults[mcol]['pdm']
        bls = pfresults[mcol]['bls']

        outfile = os.path.join(outdir,
                               'checkplot-%s-%s.pkl' % (objectid, mcol))

        # make sure the checkplot has a valid objectid
        if 'objectid' not in lcdict['objectinfo']:
            lcdict['objectinfo']['objectid'] = objectid

        cpf = checkplot.checkplot_pickle(
            [gls,pdm,bls],
            times, mags, errs,
            objectinfo=lcdict['objectinfo'],
            outfile=outfile,
            verbose=False
        )
        cpfs.append(cpf)

    LOGINFO('done with %s -> %s' % (objectid, repr(cpfs)))
    return cpfs



def runcp_worker(task):
    '''
    This is the worker for running checkplots.

    '''

    pfpickle, outdir, lcbasedir, kwargs = task

    try:

        return runcp(pfpickle, outdir, lcbasedir, **kwargs)

    except Exception as e:

        LOGEXCEPTION(' could not make checkplots for %s: %s' % (pfpickle, e))
        return None



def parallel_cp(pfpickledir,
                outdir,
                lcbasedir,
                pfpickleglob='periodfinding-*.pkl',
                lcformat='hat-sql',
                timecols=None,
                magcols=None,
                errcols=None,
                nworkers=32):
    '''
    This drives the parallel execution of runcp.

    '''

    pfpicklelist = sorted(glob.glob(os.path.join(pfpickledir, pfpickleglob)))

    tasklist = [(x, outdir, lcbasedir,
                 {'lcformat':lcformat,
                  'timecols':timecols,
                  'magcols':magcols,
                  'errcols':errcols}) for
                x in pfpicklelist]

    resultfutures = []
    results = []

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(runcp_worker, tasklist)

    results = [x for x in resultfutures]

    executor.shutdown()
    return results


##########################
## BINNING LIGHT CURVES ##
##########################

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
    if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
        lcdict = lcdict[0]

    # skip already binned light curves
    if 'binned' in lcdict:
        LOGERROR('this light curve appears to be binned already, skipping...')
        return None

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
        if 'binned' not in lcdict:
            lcdict['binned'] = {mcol: {'times':binned['binnedtimes'],
                                       'mags':binned['binnedmags'],
                                       'errs':binned['binnederrs'],
                                       'nbins':binned['nbins'],
                                       'timebins':binned['jdbins'],
                                       'ndet':binned['binnedtimes'].size,
                                       'binsizesec':binsizesec}}

        else:
            lcdict['binned'][mcol] = {'times':binned['binnedtimes'],
                                      'mags':binned['binnedmags'],
                                      'errs':binned['binnederrs'],
                                      'nbins':binned['nbins'],
                                      'timebins':binned['jdbins'],
                                      'ndet':binned['binnedtimes'].size,
                                      'binsizesec':binsizesec}


    # done with binning for all magcols, now generate the output file
    # this will always be a pickle

    if outdir is None:
        outdir = os.path.dirname(lcfile)

    outfile = os.path.join(outdir, '%s-binned%.1fsec-%s.pkl' %
                           (lcdict['objectid'], binsizesec, lcformat))

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
    except Exception as e:
        LOGEXCEPTION('failed to bin %s using binsizesec = %s' % (lcfile,
                                                                 binsizesec))
        return None



def parallel_timebin_lclist(lclist,
                            binsizesec,
                            outdir=None,
                            lcformat='hat-sql',
                            timecols=None,
                            magcols=None,
                            errcols=None,
                            minbinelems=7,
                            nworkers=32,
                            maxworkertasks=1000):
    '''
    This bins all the light curves in lclist using binsizesec.

    '''

    if outdir and not os.path.exists(outdir):
        os.mkdir(outdir)

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
                           outdir=None,
                           lcformat='hat-sql',
                           timecols=None,
                           magcols=None,
                           errcols=None,
                           minbinelems=7,
                           nworkers=32,
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

    return parallel_timebin_lclist(lclist,
                                   binsizesec,
                                   outdir=outdir,
                                   lcformat=lcformat,
                                   timecols=timecols,
                                   magcols=magcols,
                                   errcols=errcols,
                                   minbinelems=minbinelems,
                                   nworkers=nworkers,
                                   maxworkertasks=maxworkertasks)
