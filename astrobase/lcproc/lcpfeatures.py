#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lcpfeatures.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

'''
This contains functions to generate periodic light curve features for later
variable star classification.

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
import sys
import glob
import gzip
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
except Exception as e:
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
from astrobase.varclass import periodicfeatures

from astrobase.lcproc import get_lcformat
from astrobase.lcproc.periodsearch import PFMETHODS



#######################
## PERIODIC FEATURES ##
#######################

def get_periodicfeatures(
        pfpickle,
        lcbasedir,
        outdir,
        fourierorder=5,
        # these are depth, duration, ingress duration
        transitparams=[-0.01,0.1,0.1],
        # these are depth, duration, depth ratio, secphase
        ebparams=[-0.2,0.3,0.7,0.5],
        pdiff_threshold=1.0e-4,
        sidereal_threshold=1.0e-4,
        sampling_peak_multiplier=5.0,
        sampling_startp=None,
        sampling_endp=None,
        starfeatures=None,
        timecols=None,
        magcols=None,
        errcols=None,
        lcformat='hat-sql',
        lcformatdir=None,
        sigclip=10.0,
        verbose=True,
        raiseonfail=False
):
    '''This gets all periodic features for the object.

    The following periodic features are obtained:

    - For all best periods from all periodogram methods in `pfpickle`,
      calculates the number of these with peaks that are at least
      `sampling_peak_multiplier` x time-sampling periodogram peak at the same
      period. This indicates how likely the `pfpickle` periodogram peaks are to
      being real as opposed to just being caused by time-sampling
      window-function of the observations.

    - For all best periods from all periodogram methods in `pfpickle`,
      calculates the number of best periods which are consistent with a sidereal
      day (1.0027379 and 0.9972696), likely indicating that they're not real.

    - For all best periods from all periodogram methods in `pfpickle`,
      calculates the number of cross-wise period differences for all of these
      that fall below the `pdiff_threshold` value. If this is high, most of the
      period-finders in `pfpickle` agree on their best period results, so it's
      likely the periods found are real.

    - The `R_ij` amplitude and `phi_ij` phase ratios for Fourier fit
      amplitudes and phases.

    - The reduced chi-sq values of fits to the phased light curves for each best
      period from each periodogram using Fourier, EB, and planet transit fits.

    - The reduced chi-sq values of fits to the phased light curves for each best
      period x 2 from each periodogram using Fourier, EB, and planet transit
      fits (to check specifically for EBs)

    - The ratio of the MAD of the residuals after both kinds of fits above to
      the unphased light curve MAD (to indicate if one of these models is a
      better fit than one at all).

    - For each phased light curve for each best period from each periodogram,
      calculates the point-to-point scatter using the phased LC for the best
      period itself, as well as that for best period x 2.

    - For each phased light curve for each best period from each periodogram,
      calculates the 1/eta variability index, the same for best period x 2, and
      finally the ratio between the phased LC 1/eta variability index and
      unphased LC 1/eta variability index.


    Parameters
    ----------

    pfpickle : str

    lcbasedir : str

    outdir : str

    fourierorder : int

    transitparams : list of float

    ebparams : list of float

    pdiff_threshold : float or None

    sidereal_threshold : float or None

    sampling_peak_multiplier : float or None

    sampling_startp : float or None

    sampling_endp : float or None

    starfeatures : str or None
        If not None, this should be the filename of the
        `starfeatures-<objectid>.pkl` created by
        :py:func:`astrobase.lcproc.lcsfeatures.get_starfeatures` for this
        object. This is used to get the neighbor's light curve and phase it with
        this object's period to see if this object is blended.

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

    verbose : bool
        If True, will indicate progress while working.

    raiseonfail : bool
        If True, will raise an Exception if something goes wrong.

    Returns
    -------

    str
        Returns a filename for the output pickle containing all of the periodic
        features for the input object's LC.

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

    # open the pfpickle
    if pfpickle.endswith('.gz'):
        infd = gzip.open(pfpickle)
    else:
        infd = open(pfpickle)
    pf = pickle.load(infd)
    infd.close()

    lcfile = os.path.join(lcbasedir, pf['lcfbasename'])
    objectid = pf['objectid']

    if 'kwargs' in pf:
        kwargs = pf['kwargs']
    else:
        kwargs = None

    # override the default timecols, magcols, and errcols
    # using the ones provided to the periodfinder
    # if those don't exist, use the defaults from the lcformat def
    if kwargs and 'timecols' in kwargs and timecols is None:
        timecols = kwargs['timecols']
    elif not kwargs and not timecols:
        timecols = dtimecols

    if kwargs and 'magcols' in kwargs and magcols is None:
        magcols = kwargs['magcols']
    elif not kwargs and not magcols:
        magcols = dmagcols

    if kwargs and 'errcols' in kwargs and errcols is None:
        errcols = kwargs['errcols']
    elif not kwargs and not errcols:
        errcols = derrcols

    # check if the light curve file exists
    if not os.path.exists(lcfile):
        LOGERROR("can't find LC %s for object %s" % (lcfile, objectid))
        return None


    # check if we have neighbors we can get the LCs for
    if starfeatures is not None and os.path.exists(starfeatures):

        with open(starfeatures,'rb') as infd:
            starfeat = pickle.load(infd)

        if starfeat['closestnbrlcfname'].size > 0:

            nbr_full_lcf = starfeat['closestnbrlcfname'][0]

            # check for this LC in the lcbasedir
            if os.path.exists(os.path.join(lcbasedir,
                                           os.path.basename(nbr_full_lcf))):
                nbrlcf = os.path.join(lcbasedir,
                                      os.path.basename(nbr_full_lcf))
            # if it's not there, check for this file at the full LC location
            elif os.path.exists(nbr_full_lcf):
                nbrlcf = nbr_full_lcf
            # otherwise, we can't find it, so complain
            else:
                LOGWARNING("can't find neighbor light curve file: %s in "
                           "its original directory: %s, or in this object's "
                           "lcbasedir: %s, skipping neighbor processing..." %
                           (os.path.basename(nbr_full_lcf),
                            os.path.dirname(nbr_full_lcf),
                            lcbasedir))
                nbrlcf = None

        else:
            nbrlcf = None

    else:
        nbrlcf = None


    # now, start processing for periodic feature extraction
    try:

        # get the object LC into a dict
        lcdict = readerfunc(lcfile)

        # this should handle lists/tuples being returned by readerfunc
        # we assume that the first element is the actual lcdict
        # FIXME: figure out how to not need this assumption
        if ( (isinstance(lcdict, (list, tuple))) and
             (isinstance(lcdict[0], dict)) ):
            lcdict = lcdict[0]

        # get the nbr object LC into a dict if there is one
        if nbrlcf is not None:

            nbrlcdict = readerfunc(nbrlcf)

            # this should handle lists/tuples being returned by readerfunc
            # we assume that the first element is the actual lcdict
            # FIXME: figure out how to not need this assumption
            if ( (isinstance(nbrlcdict, (list, tuple))) and
                 (isinstance(nbrlcdict[0], dict)) ):
                nbrlcdict = nbrlcdict[0]

        # this will be the output file
        outfile = os.path.join(outdir, 'periodicfeatures-%s.pkl' %
                               squeeze(objectid).replace(' ','-'))

        # normalize using the special function if specified
        if normfunc is not None:
            lcdict = normfunc(lcdict)

            if nbrlcf:
                nbrlcdict = normfunc(nbrlcdict)


        resultdict = {}

        for tcol, mcol, ecol in zip(timecols, magcols, errcols):

            # dereference the columns and get them from the lcdict
            if '.' in tcol:
                tcolget = tcol.split('.')
            else:
                tcolget = [tcol]
            times = _dict_get(lcdict, tcolget)

            if nbrlcf:
                nbrtimes = _dict_get(nbrlcdict, tcolget)
            else:
                nbrtimes = None


            if '.' in mcol:
                mcolget = mcol.split('.')
            else:
                mcolget = [mcol]

            mags = _dict_get(lcdict, mcolget)

            if nbrlcf:
                nbrmags = _dict_get(nbrlcdict, mcolget)
            else:
                nbrmags = None


            if '.' in ecol:
                ecolget = ecol.split('.')
            else:
                ecolget = [ecol]

            errs = _dict_get(lcdict, ecolget)

            if nbrlcf:
                nbrerrs = _dict_get(nbrlcdict, ecolget)
            else:
                nbrerrs = None

            #
            # filter out nans, etc. from the object and any neighbor LC
            #

            # get the finite values
            finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
            ftimes, fmags, ferrs = times[finind], mags[finind], errs[finind]

            if nbrlcf:

                nfinind = (np.isfinite(nbrtimes) &
                           np.isfinite(nbrmags) &
                           np.isfinite(nbrerrs))
                nbrftimes, nbrfmags, nbrferrs = (nbrtimes[nfinind],
                                                 nbrmags[nfinind],
                                                 nbrerrs[nfinind])

            # get nonzero errors
            nzind = np.nonzero(ferrs)
            ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

            if nbrlcf:

                nnzind = np.nonzero(nbrferrs)
                nbrftimes, nbrfmags, nbrferrs = (nbrftimes[nnzind],
                                                 nbrfmags[nnzind],
                                                 nbrferrs[nnzind])

            # normalize here if not using special normalization
            if normfunc is None:

                ntimes, nmags = normalize_magseries(
                    ftimes, fmags,
                    magsarefluxes=magsarefluxes
                )

                times, mags, errs = ntimes, nmags, ferrs

                if nbrlcf:
                    nbrntimes, nbrnmags = normalize_magseries(
                        nbrftimes, nbrfmags,
                        magsarefluxes=magsarefluxes
                    )
                    nbrtimes, nbrmags, nbrerrs = nbrntimes, nbrnmags, nbrferrs
                else:
                    nbrtimes, nbrmags, nbrerrs = None, None, None

            else:
                times, mags, errs = ftimes, fmags, ferrs


            if times.size > 999:

                #
                # now we have times, mags, errs (and nbrtimes, nbrmags, nbrerrs)
                #
                available_pfmethods = []
                available_pgrams = []
                available_bestperiods = []

                for k in pf[mcol].keys():

                    if k in PFMETHODS:

                        available_pgrams.append(pf[mcol][k])

                        if k != 'win':
                            available_pfmethods.append(
                                pf[mcol][k]['method']
                            )
                            available_bestperiods.append(
                                pf[mcol][k]['bestperiod']
                            )

                #
                # process periodic features for this magcol
                #
                featkey = 'periodicfeatures-%s' % mcol
                resultdict[featkey] = {}

                # first, handle the periodogram features
                pgramfeat = periodicfeatures.periodogram_features(
                    available_pgrams, times, mags, errs,
                    sigclip=sigclip,
                    pdiff_threshold=pdiff_threshold,
                    sidereal_threshold=sidereal_threshold,
                    sampling_peak_multiplier=sampling_peak_multiplier,
                    sampling_startp=sampling_startp,
                    sampling_endp=sampling_endp,
                    verbose=verbose
                )
                resultdict[featkey].update(pgramfeat)

                resultdict[featkey]['pfmethods'] = available_pfmethods

                # then for each bestperiod, get phasedlc and lcfit features
                for _ind, pfm, bp in zip(range(len(available_bestperiods)),
                                         available_pfmethods,
                                         available_bestperiods):

                    resultdict[featkey][pfm] = periodicfeatures.lcfit_features(
                        times, mags, errs, bp,
                        fourierorder=fourierorder,
                        transitparams=transitparams,
                        ebparams=ebparams,
                        sigclip=sigclip,
                        magsarefluxes=magsarefluxes,
                        verbose=verbose
                    )

                    phasedlcfeat = periodicfeatures.phasedlc_features(
                        times, mags, errs, bp,
                        nbrtimes=nbrtimes,
                        nbrmags=nbrmags,
                        nbrerrs=nbrerrs
                    )

                    resultdict[featkey][pfm].update(phasedlcfeat)


            else:

                LOGERROR('not enough finite measurements in magcol: %s, for '
                         'pfpickle: %s, skipping this magcol'
                         % (mcol, pfpickle))
                featkey = 'periodicfeatures-%s' % mcol
                resultdict[featkey] = None

        #
        # end of per magcol processing
        #
        # write resultdict to pickle
        outfile = os.path.join(outdir, 'periodicfeatures-%s.pkl' %
                               squeeze(objectid).replace(' ','-'))
        with open(outfile,'wb') as outfd:
            pickle.dump(resultdict, outfd, pickle.HIGHEST_PROTOCOL)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to run for pf: %s, lcfile: %s' %
                     (pfpickle, lcfile))
        if raiseonfail:
            raise
        else:
            return None



def periodicfeatures_worker(task):
    '''
    This is a parallel worker for the drivers below.

    '''

    pfpickle, lcbasedir, outdir, starfeatures, kwargs = task

    try:

        return get_periodicfeatures(pfpickle,
                                    lcbasedir,
                                    outdir,
                                    starfeatures=starfeatures,
                                    **kwargs)

    except Exception as e:

        LOGEXCEPTION('failed to get periodicfeatures for %s' % pfpickle)



def serial_periodicfeatures(pfpkl_list,
                            lcbasedir,
                            outdir,
                            starfeaturesdir=None,
                            fourierorder=5,
                            # these are depth, duration, ingress duration
                            transitparams=[-0.01,0.1,0.1],
                            # these are depth, duration, depth ratio, secphase
                            ebparams=[-0.2,0.3,0.7,0.5],
                            pdiff_threshold=1.0e-4,
                            sidereal_threshold=1.0e-4,
                            sampling_peak_multiplier=5.0,
                            sampling_startp=None,
                            sampling_endp=None,
                            starfeatures=None,
                            timecols=None,
                            magcols=None,
                            errcols=None,
                            lcformat='hat-sql',
                            lcformatdir=None,
                            sigclip=10.0,
                            verbose=False,
                            maxobjects=None,
                            nworkers=NCPUS):
    '''This drives the periodicfeatures collection for a list of periodfinding
    pickles.

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

    # make sure to make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maxobjects:
        pfpkl_list = pfpkl_list[:maxobjects]

    LOGINFO('%s periodfinding pickles to process' % len(pfpkl_list))

    # if the starfeaturedir is provided, try to find a starfeatures pickle for
    # each periodfinding pickle in pfpkl_list
    if starfeaturesdir and os.path.exists(starfeaturesdir):

        starfeatures_list = []

        LOGINFO('collecting starfeatures pickles...')

        for pfpkl in pfpkl_list:

            sfpkl1 = os.path.basename(pfpkl).replace('periodfinding',
                                                     'starfeatures')
            sfpkl2 = sfpkl1.replace('.gz','')

            sfpath1 = os.path.join(starfeaturesdir, sfpkl1)
            sfpath2 = os.path.join(starfeaturesdir, sfpkl2)

            if os.path.exists(sfpath1):
                starfeatures_list.append(sfpkl1)
            elif os.path.exists(sfpath2):
                starfeatures_list.append(sfpkl2)
            else:
                starfeatures_list.append(None)

    else:

        starfeatures_list = [None for x in pfpkl_list]

    # generate the task list
    kwargs = {'fourierorder':fourierorder,
              'transitparams':transitparams,
              'ebparams':ebparams,
              'pdiff_threshold':pdiff_threshold,
              'sidereal_threshold':sidereal_threshold,
              'sampling_peak_multiplier':sampling_peak_multiplier,
              'sampling_startp':sampling_startp,
              'sampling_endp':sampling_endp,
              'timecols':timecols,
              'magcols':magcols,
              'errcols':errcols,
              'lcformat':lcformat,
              'lcformatdir':lcformatdir,
              'sigclip':sigclip,
              'verbose':verbose}

    tasks = [(x, lcbasedir, outdir, y, kwargs) for (x,y) in
             zip(pfpkl_list, starfeatures_list)]

    LOGINFO('processing periodfinding pickles...')

    for task in tqdm(tasks):
        periodicfeatures_worker(task)



def parallel_periodicfeatures(pfpkl_list,
                              lcbasedir,
                              outdir,
                              starfeaturesdir=None,
                              fourierorder=5,
                              # these are depth, duration, ingress duration
                              transitparams=[-0.01,0.1,0.1],
                              # these are depth, duration, depth ratio, secphase
                              ebparams=[-0.2,0.3,0.7,0.5],
                              pdiff_threshold=1.0e-4,
                              sidereal_threshold=1.0e-4,
                              sampling_peak_multiplier=5.0,
                              sampling_startp=None,
                              sampling_endp=None,
                              timecols=None,
                              magcols=None,
                              errcols=None,
                              lcformat='hat-sql',
                              lcformatdir=None,
                              sigclip=10.0,
                              verbose=False,
                              maxobjects=None,
                              nworkers=NCPUS):
    '''
    This runs periodicfeatures in parallel for all periodfinding pickles.

    '''
    # make sure to make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maxobjects:
        pfpkl_list = pfpkl_list[:maxobjects]

    LOGINFO('%s periodfinding pickles to process' % len(pfpkl_list))

    # if the starfeaturedir is provided, try to find a starfeatures pickle for
    # each periodfinding pickle in pfpkl_list
    if starfeaturesdir and os.path.exists(starfeaturesdir):

        starfeatures_list = []

        LOGINFO('collecting starfeatures pickles...')

        for pfpkl in pfpkl_list:

            sfpkl1 = os.path.basename(pfpkl).replace('periodfinding',
                                                     'starfeatures')
            sfpkl2 = sfpkl1.replace('.gz','')

            sfpath1 = os.path.join(starfeaturesdir, sfpkl1)
            sfpath2 = os.path.join(starfeaturesdir, sfpkl2)

            if os.path.exists(sfpath1):
                starfeatures_list.append(sfpkl1)
            elif os.path.exists(sfpath2):
                starfeatures_list.append(sfpkl2)
            else:
                starfeatures_list.append(None)

    else:

        starfeatures_list = [None for x in pfpkl_list]

    # generate the task list
    kwargs = {'fourierorder':fourierorder,
              'transitparams':transitparams,
              'ebparams':ebparams,
              'pdiff_threshold':pdiff_threshold,
              'sidereal_threshold':sidereal_threshold,
              'sampling_peak_multiplier':sampling_peak_multiplier,
              'sampling_startp':sampling_startp,
              'sampling_endp':sampling_endp,
              'timecols':timecols,
              'magcols':magcols,
              'errcols':errcols,
              'lcformat':lcformat,
              'lcformatdir':lcformat,
              'sigclip':sigclip,
              'verbose':verbose}

    tasks = [(x, lcbasedir, outdir, y, kwargs) for (x,y) in
             zip(pfpkl_list, starfeatures_list)]

    LOGINFO('processing periodfinding pickles...')

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(periodicfeatures_worker, tasks)

    results = [x for x in resultfutures]
    resdict = {os.path.basename(x):y for (x,y) in zip(pfpkl_list, results)}

    return resdict



def parallel_periodicfeatures_lcdir(
        pfpkl_dir,
        lcbasedir,
        outdir,
        pfpkl_glob='periodfinding-*.pkl*',
        starfeaturesdir=None,
        fourierorder=5,
        # these are depth, duration, ingress duration
        transitparams=[-0.01,0.1,0.1],
        # these are depth, duration, depth ratio, secphase
        ebparams=[-0.2,0.3,0.7,0.5],
        pdiff_threshold=1.0e-4,
        sidereal_threshold=1.0e-4,
        sampling_peak_multiplier=5.0,
        sampling_startp=None,
        sampling_endp=None,
        timecols=None,
        magcols=None,
        errcols=None,
        lcformat='hat-sql',
        lcformatdir=None,
        sigclip=10.0,
        verbose=False,
        maxobjects=None,
        nworkers=NCPUS,
        recursive=True,
):
    '''This runs parallel periodicfeature extraction for a directory of
    periodfinding result pickles.

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

    fileglob = pfpkl_glob

    # now find the files
    LOGINFO('searching for periodfinding pickles in %s ...' % pfpkl_dir)

    if recursive is False:
        matching = glob.glob(os.path.join(pfpkl_dir, fileglob))

    else:
        # use recursive glob for Python 3.5+
        if sys.version_info[:2] > (3,4):

            matching = glob.glob(os.path.join(pfpkl_dir,
                                              '**',
                                              fileglob),recursive=True)

        # otherwise, use os.walk and glob
        else:

            # use os.walk to go through the directories
            walker = os.walk(pfpkl_dir)
            matching = []

            for root, dirs, _files in walker:
                for sdir in dirs:
                    searchpath = os.path.join(root,
                                              sdir,
                                              fileglob)
                    foundfiles = glob.glob(searchpath)

                    if foundfiles:
                        matching.extend(foundfiles)


    # now that we have all the files, process them
    if matching and len(matching) > 0:

        LOGINFO('found %s periodfinding pickles, getting periodicfeatures...' %
                len(matching))

        return parallel_periodicfeatures(
            matching,
            lcbasedir,
            outdir,
            starfeaturesdir=starfeaturesdir,
            fourierorder=fourierorder,
            transitparams=transitparams,
            ebparams=ebparams,
            pdiff_threshold=pdiff_threshold,
            sidereal_threshold=sidereal_threshold,
            sampling_peak_multiplier=sampling_peak_multiplier,
            sampling_startp=sampling_startp,
            sampling_endp=sampling_endp,
            timecols=timecols,
            magcols=magcols,
            errcols=errcols,
            lcformat=lcformat,
            lcformatdir=lcformatdir,
            sigclip=sigclip,
            verbose=verbose,
            maxobjects=maxobjects,
            nworkers=nworkers,
        )

    else:

        LOGERROR('no periodfinding pickles found in %s' % (pfpkl_dir))
        return None
