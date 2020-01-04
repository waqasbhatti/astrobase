#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# varthreshold.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
'''
This contains functions to investigate where to set a threshold for
several variability indices to distinguish between variable and non-variable
stars.

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

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem


def _dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)


import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

from astrobase.magnitudes import jhk_to_sdssr
from astrobase.lcproc import get_lcformat


###########################
## VARIABILITY THRESHOLD ##
###########################

DEFAULT_MAGBINS = np.arange(8.0,16.25,0.25)


def variability_threshold(featuresdir,
                          outfile,
                          magbins=DEFAULT_MAGBINS,
                          maxobjects=None,
                          timecols=None,
                          magcols=None,
                          errcols=None,
                          lcformat='hat-sql',
                          lcformatdir=None,
                          min_lcmad_stdev=5.0,
                          min_stetj_stdev=2.0,
                          min_iqr_stdev=2.0,
                          min_inveta_stdev=2.0,
                          verbose=True):
    '''This generates a list of objects with stetson J, IQR, and 1.0/eta
    above some threshold value to select them as potential variable stars.

    Use this to pare down the objects to review and put through
    period-finding. This does the thresholding per magnitude bin; this should be
    better than one single cut through the entire magnitude range. Set the
    magnitude bins using the magbins kwarg.

    FIXME: implement a voting classifier here. this will choose variables based
    on the thresholds in IQR, stetson, and inveta based on weighting carried
    over from the variability recovery sims.

    Parameters
    ----------

    featuresdir : str
        This is the directory containing variability feature pickles created by
        :py:func:`astrobase.lcproc.lcpfeatures.parallel_varfeatures` or similar.

    outfile : str
        This is the output pickle file that will contain all the threshold
        information.

    magbins : np.array of floats
        This sets the magnitude bins to use for calculating thresholds.

    maxobjects : int or None
        This is the number of objects to process. If None, all objects with
        feature pickles in `featuresdir` will be processed.

    timecols : list of str or None
        The timecol keys to use from the lcdict in calculating the thresholds.

    magcols : list of str or None
        The magcol keys to use from the lcdict in calculating the thresholds.

    errcols : list of str or None
        The errcol keys to use from the lcdict in calculating the thresholds.

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

    min_lcmad_stdev,min_stetj_stdev,min_iqr_stdev,min_inveta_stdev : float or np.array
        These are all the standard deviation multiplier for the distributions of
        light curve standard deviation, Stetson J variability index, the light
        curve interquartile range, and 1/eta variability index
        respectively. These multipliers set the minimum values of these measures
        to use for selecting variable stars. If provided as floats, the same
        value will be used for all magbins. If provided as np.arrays of `size =
        magbins.size - 1`, will be used to apply possibly different sigma cuts
        for each magbin.

    verbose : bool
        If True, will report progress and warn about any problems.

    Returns
    -------

    dict
        Contains all of the variability threshold information along with indices
        into the array of the object IDs chosen as variables.

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

    # list of input pickles generated by varfeatures functions above
    pklist = glob.glob(os.path.join(featuresdir, 'varfeatures-*.pkl'))

    if maxobjects:
        pklist = pklist[:maxobjects]

    allobjects = {}

    for magcol in magcols:

        # keep local copies of these so we can fix them independently in case of
        # nans
        if (isinstance(min_stetj_stdev, list) or
            isinstance(min_stetj_stdev, np.ndarray)):
            magcol_min_stetj_stdev = min_stetj_stdev[::]
        else:
            magcol_min_stetj_stdev = min_stetj_stdev

        if (isinstance(min_iqr_stdev, list) or
            isinstance(min_iqr_stdev, np.ndarray)):
            magcol_min_iqr_stdev = min_iqr_stdev[::]
        else:
            magcol_min_iqr_stdev = min_iqr_stdev

        if (isinstance(min_inveta_stdev, list) or
            isinstance(min_inveta_stdev, np.ndarray)):
            magcol_min_inveta_stdev = min_inveta_stdev[::]
        else:
            magcol_min_inveta_stdev = min_inveta_stdev

        LOGINFO('getting all object sdssr, LC MAD, stet J, IQR, eta...')

        # we'll calculate the sigma per magnitude bin, so get the mags as well
        allobjects[magcol] = {
            'objectid':[],
            'sdssr':[],
            'lcmad':[],
            'stetsonj':[],
            'iqr':[],
            'eta':[]
        }

        # fancy progress bar with tqdm if present
        if TQDM and verbose:
            listiterator = tqdm(pklist)
        else:
            listiterator = pklist

        for pkl in listiterator:

            with open(pkl,'rb') as infd:
                thisfeatures = pickle.load(infd)

            objectid = thisfeatures['objectid']

            # the object magnitude
            if ('info' in thisfeatures and
                thisfeatures['info'] and
                'sdssr' in thisfeatures['info']):

                if (thisfeatures['info']['sdssr'] and
                    thisfeatures['info']['sdssr'] > 3.0):

                    sdssr = thisfeatures['info']['sdssr']

                elif (magcol in thisfeatures and
                      thisfeatures[magcol] and
                      'median' in thisfeatures[magcol] and
                      thisfeatures[magcol]['median'] > 3.0):

                    sdssr = thisfeatures[magcol]['median']

                elif (thisfeatures['info']['jmag'] and
                      thisfeatures['info']['hmag'] and
                      thisfeatures['info']['kmag']):

                    sdssr = jhk_to_sdssr(thisfeatures['info']['jmag'],
                                         thisfeatures['info']['hmag'],
                                         thisfeatures['info']['kmag'])

                else:
                    sdssr = np.nan

            else:
                sdssr = np.nan

            # the MAD of the light curve
            if (magcol in thisfeatures and
                thisfeatures[magcol] and
                thisfeatures[magcol]['mad']):
                lcmad = thisfeatures[magcol]['mad']
            else:
                lcmad = np.nan

            # stetson index
            if (magcol in thisfeatures and
                thisfeatures[magcol] and
                thisfeatures[magcol]['stetsonj']):
                stetsonj = thisfeatures[magcol]['stetsonj']
            else:
                stetsonj = np.nan

            # IQR
            if (magcol in thisfeatures and
                thisfeatures[magcol] and
                thisfeatures[magcol]['mag_iqr']):
                iqr = thisfeatures[magcol]['mag_iqr']
            else:
                iqr = np.nan

            # eta
            if (magcol in thisfeatures and
                thisfeatures[magcol] and
                thisfeatures[magcol]['eta_normal']):
                eta = thisfeatures[magcol]['eta_normal']
            else:
                eta = np.nan

            allobjects[magcol]['objectid'].append(objectid)
            allobjects[magcol]['sdssr'].append(sdssr)
            allobjects[magcol]['lcmad'].append(lcmad)
            allobjects[magcol]['stetsonj'].append(stetsonj)
            allobjects[magcol]['iqr'].append(iqr)
            allobjects[magcol]['eta'].append(eta)

        #
        # done with collection of info
        #
        LOGINFO('finding objects above thresholds per magbin...')

        # turn the info into arrays
        allobjects[magcol]['objectid'] = np.ravel(np.array(
            allobjects[magcol]['objectid']
        ))
        allobjects[magcol]['sdssr'] = np.ravel(np.array(
            allobjects[magcol]['sdssr']
        ))
        allobjects[magcol]['lcmad'] = np.ravel(np.array(
            allobjects[magcol]['lcmad']
        ))
        allobjects[magcol]['stetsonj'] = np.ravel(np.array(
            allobjects[magcol]['stetsonj']
        ))
        allobjects[magcol]['iqr'] = np.ravel(np.array(
            allobjects[magcol]['iqr']
        ))
        allobjects[magcol]['eta'] = np.ravel(np.array(
            allobjects[magcol]['eta']
        ))

        # only get finite elements everywhere
        thisfinind = (
            np.isfinite(allobjects[magcol]['sdssr']) &
            np.isfinite(allobjects[magcol]['lcmad']) &
            np.isfinite(allobjects[magcol]['stetsonj']) &
            np.isfinite(allobjects[magcol]['iqr']) &
            np.isfinite(allobjects[magcol]['eta'])
        )
        allobjects[magcol]['objectid'] = allobjects[magcol]['objectid'][
            thisfinind
        ]
        allobjects[magcol]['sdssr'] = allobjects[magcol]['sdssr'][thisfinind]
        allobjects[magcol]['lcmad'] = allobjects[magcol]['lcmad'][thisfinind]
        allobjects[magcol]['stetsonj'] = allobjects[magcol]['stetsonj'][
            thisfinind
        ]
        allobjects[magcol]['iqr'] = allobjects[magcol]['iqr'][thisfinind]
        allobjects[magcol]['eta'] = allobjects[magcol]['eta'][thisfinind]

        # invert eta so we can threshold the same way as the others
        allobjects[magcol]['inveta'] = 1.0/allobjects[magcol]['eta']

        # do the thresholding by magnitude bin
        magbininds = np.digitize(allobjects[magcol]['sdssr'],
                                 magbins)

        binned_objectids = []
        binned_sdssr = []
        binned_sdssr_median = []

        binned_lcmad = []
        binned_stetsonj = []
        binned_iqr = []
        binned_inveta = []
        binned_count = []

        binned_objectids_thresh_stetsonj = []
        binned_objectids_thresh_iqr = []
        binned_objectids_thresh_inveta = []
        binned_objectids_thresh_all = []

        binned_lcmad_median = []
        binned_lcmad_stdev = []

        binned_stetsonj_median = []
        binned_stetsonj_stdev = []

        binned_inveta_median = []
        binned_inveta_stdev = []

        binned_iqr_median = []
        binned_iqr_stdev = []

        # go through all the mag bins and get the thresholds for J, inveta, IQR
        for mbinind, magi in zip(np.unique(magbininds),
                                 range(len(magbins)-1)):

            thisbinind = np.where(magbininds == mbinind)
            thisbin_sdssr_median = (magbins[magi] + magbins[magi+1])/2.0
            binned_sdssr_median.append(thisbin_sdssr_median)

            thisbin_objectids = allobjects[magcol]['objectid'][thisbinind]
            thisbin_sdssr = allobjects[magcol]['sdssr'][thisbinind]
            thisbin_lcmad = allobjects[magcol]['lcmad'][thisbinind]
            thisbin_stetsonj = allobjects[magcol]['stetsonj'][thisbinind]
            thisbin_iqr = allobjects[magcol]['iqr'][thisbinind]
            thisbin_inveta = allobjects[magcol]['inveta'][thisbinind]
            thisbin_count = thisbin_objectids.size

            if thisbin_count > 4:

                thisbin_lcmad_median = np.median(thisbin_lcmad)
                thisbin_lcmad_stdev = np.median(
                    np.abs(thisbin_lcmad - thisbin_lcmad_median)
                ) * 1.483
                binned_lcmad_median.append(thisbin_lcmad_median)
                binned_lcmad_stdev.append(thisbin_lcmad_stdev)

                thisbin_stetsonj_median = np.median(thisbin_stetsonj)
                thisbin_stetsonj_stdev = np.median(
                    np.abs(thisbin_stetsonj - thisbin_stetsonj_median)
                ) * 1.483
                binned_stetsonj_median.append(thisbin_stetsonj_median)
                binned_stetsonj_stdev.append(thisbin_stetsonj_stdev)

                # now get the objects above the required stdev threshold
                if isinstance(magcol_min_stetj_stdev, float):

                    thisbin_objectids_thresh_stetsonj = thisbin_objectids[
                        thisbin_stetsonj > (
                            thisbin_stetsonj_median +
                            magcol_min_stetj_stdev*thisbin_stetsonj_stdev
                        )
                    ]

                elif (isinstance(magcol_min_stetj_stdev, np.ndarray) or
                      isinstance(magcol_min_stetj_stdev, list)):

                    thisbin_min_stetj_stdev = magcol_min_stetj_stdev[magi]

                    if not np.isfinite(thisbin_min_stetj_stdev):
                        LOGWARNING('provided threshold stetson J stdev '
                                   'for magbin: %.3f is nan, using 2.0' %
                                   thisbin_sdssr_median)
                        thisbin_min_stetj_stdev = 2.0
                        # update the input list/array as well, since we'll be
                        # saving it to the output dict and using it to plot the
                        # variability thresholds
                        magcol_min_stetj_stdev[magi] = 2.0

                    thisbin_objectids_thresh_stetsonj = thisbin_objectids[
                        thisbin_stetsonj > (
                            thisbin_stetsonj_median +
                            thisbin_min_stetj_stdev*thisbin_stetsonj_stdev
                        )
                    ]

                thisbin_iqr_median = np.median(thisbin_iqr)
                thisbin_iqr_stdev = np.median(
                    np.abs(thisbin_iqr - thisbin_iqr_median)
                ) * 1.483
                binned_iqr_median.append(thisbin_iqr_median)
                binned_iqr_stdev.append(thisbin_iqr_stdev)

                # get the objects above the required stdev threshold
                if isinstance(magcol_min_iqr_stdev, float):

                    thisbin_objectids_thresh_iqr = thisbin_objectids[
                        thisbin_iqr > (thisbin_iqr_median +
                                       magcol_min_iqr_stdev*thisbin_iqr_stdev)
                    ]

                elif (isinstance(magcol_min_iqr_stdev, np.ndarray) or
                      isinstance(magcol_min_iqr_stdev, list)):

                    thisbin_min_iqr_stdev = magcol_min_iqr_stdev[magi]

                    if not np.isfinite(thisbin_min_iqr_stdev):
                        LOGWARNING('provided threshold IQR stdev '
                                   'for magbin: %.3f is nan, using 2.0' %
                                   thisbin_sdssr_median)
                        thisbin_min_iqr_stdev = 2.0
                        # update the input list/array as well, since we'll be
                        # saving it to the output dict and using it to plot the
                        # variability thresholds
                        magcol_min_iqr_stdev[magi] = 2.0

                    thisbin_objectids_thresh_iqr = thisbin_objectids[
                        thisbin_iqr > (thisbin_iqr_median +
                                       thisbin_min_iqr_stdev*thisbin_iqr_stdev)
                    ]

                thisbin_inveta_median = np.median(thisbin_inveta)
                thisbin_inveta_stdev = np.median(
                    np.abs(thisbin_inveta - thisbin_inveta_median)
                ) * 1.483
                binned_inveta_median.append(thisbin_inveta_median)
                binned_inveta_stdev.append(thisbin_inveta_stdev)

                if isinstance(magcol_min_inveta_stdev, float):

                    thisbin_objectids_thresh_inveta = thisbin_objectids[
                        thisbin_inveta > (
                            thisbin_inveta_median +
                            magcol_min_inveta_stdev*thisbin_inveta_stdev
                        )
                    ]

                elif (isinstance(magcol_min_inveta_stdev, np.ndarray) or
                      isinstance(magcol_min_inveta_stdev, list)):

                    thisbin_min_inveta_stdev = magcol_min_inveta_stdev[magi]

                    if not np.isfinite(thisbin_min_inveta_stdev):
                        LOGWARNING('provided threshold inveta stdev '
                                   'for magbin: %.3f is nan, using 2.0' %
                                   thisbin_sdssr_median)

                        thisbin_min_inveta_stdev = 2.0
                        # update the input list/array as well, since we'll be
                        # saving it to the output dict and using it to plot the
                        # variability thresholds
                        magcol_min_inveta_stdev[magi] = 2.0

                    thisbin_objectids_thresh_inveta = thisbin_objectids[
                        thisbin_inveta > (
                            thisbin_inveta_median +
                            thisbin_min_inveta_stdev*thisbin_inveta_stdev
                        )
                    ]

            else:

                thisbin_objectids_thresh_stetsonj = (
                    np.array([],dtype=np.unicode_)
                )
                thisbin_objectids_thresh_iqr = (
                    np.array([],dtype=np.unicode_)
                )
                thisbin_objectids_thresh_inveta = (
                    np.array([],dtype=np.unicode_)
                )

            #
            # done with check for enough objects in the bin
            #

            # get the intersection of all threshold objects to get objects that
            # lie above the threshold for all variable indices
            thisbin_objectids_thresh_all = reduce(
                np.intersect1d,
                (thisbin_objectids_thresh_stetsonj,
                 thisbin_objectids_thresh_iqr,
                 thisbin_objectids_thresh_inveta)
            )

            binned_objectids.append(thisbin_objectids)
            binned_sdssr.append(thisbin_sdssr)
            binned_lcmad.append(thisbin_lcmad)
            binned_stetsonj.append(thisbin_stetsonj)
            binned_iqr.append(thisbin_iqr)
            binned_inveta.append(thisbin_inveta)
            binned_count.append(thisbin_objectids.size)

            binned_objectids_thresh_stetsonj.append(
                thisbin_objectids_thresh_stetsonj
            )
            binned_objectids_thresh_iqr.append(
                thisbin_objectids_thresh_iqr
            )
            binned_objectids_thresh_inveta.append(
                thisbin_objectids_thresh_inveta
            )
            binned_objectids_thresh_all.append(
                thisbin_objectids_thresh_all
            )

        #
        # done with magbins
        #

        # update the output dict for this magcol
        allobjects[magcol]['magbins'] = magbins
        allobjects[magcol]['binned_objectids'] = binned_objectids
        allobjects[magcol]['binned_sdssr_median'] = binned_sdssr_median
        allobjects[magcol]['binned_sdssr'] = binned_sdssr
        allobjects[magcol]['binned_count'] = binned_count

        allobjects[magcol]['binned_lcmad'] = binned_lcmad
        allobjects[magcol]['binned_lcmad_median'] = binned_lcmad_median
        allobjects[magcol]['binned_lcmad_stdev'] = binned_lcmad_stdev

        allobjects[magcol]['binned_stetsonj'] = binned_stetsonj
        allobjects[magcol]['binned_stetsonj_median'] = binned_stetsonj_median
        allobjects[magcol]['binned_stetsonj_stdev'] = binned_stetsonj_stdev

        allobjects[magcol]['binned_iqr'] = binned_iqr
        allobjects[magcol]['binned_iqr_median'] = binned_iqr_median
        allobjects[magcol]['binned_iqr_stdev'] = binned_iqr_stdev

        allobjects[magcol]['binned_inveta'] = binned_inveta
        allobjects[magcol]['binned_inveta_median'] = binned_inveta_median
        allobjects[magcol]['binned_inveta_stdev'] = binned_inveta_stdev

        allobjects[magcol]['binned_objectids_thresh_stetsonj'] = (
            binned_objectids_thresh_stetsonj
        )
        allobjects[magcol]['binned_objectids_thresh_iqr'] = (
            binned_objectids_thresh_iqr
        )
        allobjects[magcol]['binned_objectids_thresh_inveta'] = (
            binned_objectids_thresh_inveta
        )
        allobjects[magcol]['binned_objectids_thresh_all'] = (
            binned_objectids_thresh_all
        )

        # get the common selected objects thru all measures
        try:
            allobjects[magcol]['objectids_all_thresh_all_magbins'] = np.unique(
                np.concatenate(
                    allobjects[magcol]['binned_objectids_thresh_all']
                )
            )
        except ValueError:
            LOGWARNING('not enough variable objects matching all thresholds')
            allobjects[magcol]['objectids_all_thresh_all_magbins'] = (
                np.array([])
            )

        allobjects[magcol]['objectids_stetsonj_thresh_all_magbins'] = np.unique(
            np.concatenate(
                allobjects[magcol]['binned_objectids_thresh_stetsonj']
            )
        )
        allobjects[magcol]['objectids_inveta_thresh_all_magbins'] = np.unique(
            np.concatenate(allobjects[magcol]['binned_objectids_thresh_inveta'])
        )
        allobjects[magcol]['objectids_iqr_thresh_all_magbins'] = np.unique(
            np.concatenate(allobjects[magcol]['binned_objectids_thresh_iqr'])
        )

        # turn these into np.arrays for easier plotting if they're lists
        if isinstance(min_stetj_stdev, list):
            allobjects[magcol]['min_stetj_stdev'] = np.array(
                magcol_min_stetj_stdev
            )
        else:
            allobjects[magcol]['min_stetj_stdev'] = magcol_min_stetj_stdev

        if isinstance(min_iqr_stdev, list):
            allobjects[magcol]['min_iqr_stdev'] = np.array(
                magcol_min_iqr_stdev
            )
        else:
            allobjects[magcol]['min_iqr_stdev'] = magcol_min_iqr_stdev

        if isinstance(min_inveta_stdev, list):
            allobjects[magcol]['min_inveta_stdev'] = np.array(
                magcol_min_inveta_stdev
            )
        else:
            allobjects[magcol]['min_inveta_stdev'] = magcol_min_inveta_stdev

        # this one doesn't get touched (for now)
        allobjects[magcol]['min_lcmad_stdev'] = min_lcmad_stdev

    #
    # done with all magcols
    #

    allobjects['magbins'] = magbins

    with open(outfile,'wb') as outfd:
        pickle.dump(allobjects, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    return allobjects


def plot_variability_thresholds(varthreshpkl,
                                xmin_lcmad_stdev=5.0,
                                xmin_stetj_stdev=2.0,
                                xmin_iqr_stdev=2.0,
                                xmin_inveta_stdev=2.0,
                                lcformat='hat-sql',
                                lcformatdir=None,
                                magcols=None):
    '''This makes plots for the variability threshold distributions.

    Parameters
    ----------

    varthreshpkl : str
        The pickle produced by the function above.

    xmin_lcmad_stdev,xmin_stetj_stdev,xmin_iqr_stdev,xmin_inveta_stdev : float or np.array
        Values of the threshold values to override the ones in the
        `vartresholdpkl`. If provided, will plot the thresholds accordingly
        instead of using the ones in the input pickle directly.

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

    magcols : list of str or None
        The magcol keys to use from the lcdict.

    Returns
    -------

    str
        The file name of the threshold plot generated.

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

    if magcols is None:
        magcols = dmagcols

    with open(varthreshpkl,'rb') as infd:
        allobjects = pickle.load(infd)

    magbins = allobjects['magbins']

    for magcol in magcols:

        min_lcmad_stdev = (
            xmin_lcmad_stdev or allobjects[magcol]['min_lcmad_stdev']
        )
        min_stetj_stdev = (
            xmin_stetj_stdev or allobjects[magcol]['min_stetj_stdev']
        )
        min_iqr_stdev = (
            xmin_iqr_stdev or allobjects[magcol]['min_iqr_stdev']
        )
        min_inveta_stdev = (
            xmin_inveta_stdev or allobjects[magcol]['min_inveta_stdev']
        )

        fig = plt.figure(figsize=(20,16))

        # the mag vs lcmad
        plt.subplot(221)
        plt.plot(allobjects[magcol]['sdssr'],
                 allobjects[magcol]['lcmad']*1.483,
                 marker='.',ms=1.0, linestyle='none',
                 rasterized=True)
        plt.plot(allobjects[magcol]['binned_sdssr_median'],
                 np.array(allobjects[magcol]['binned_lcmad_median'])*1.483,
                 linewidth=3.0)
        plt.plot(
            allobjects[magcol]['binned_sdssr_median'],
            np.array(allobjects[magcol]['binned_lcmad_median'])*1.483 +
            min_lcmad_stdev*np.array(
                allobjects[magcol]['binned_lcmad_stdev']
            ),
            linewidth=3.0, linestyle='dashed'
        )
        plt.xlim((magbins.min()-0.25, magbins.max()))
        plt.xlabel('SDSS r')
        plt.ylabel(r'lightcurve RMS (MAD $\times$ 1.483)')
        plt.title('%s - SDSS r vs. light curve RMS' % magcol)
        plt.yscale('log')
        plt.tight_layout()

        # the mag vs stetsonj
        plt.subplot(222)
        plt.plot(allobjects[magcol]['sdssr'],
                 allobjects[magcol]['stetsonj'],
                 marker='.',ms=1.0, linestyle='none',
                 rasterized=True)
        plt.plot(allobjects[magcol]['binned_sdssr_median'],
                 allobjects[magcol]['binned_stetsonj_median'],
                 linewidth=3.0)
        plt.plot(
            allobjects[magcol]['binned_sdssr_median'],
            np.array(allobjects[magcol]['binned_stetsonj_median']) +
            min_stetj_stdev*np.array(
                allobjects[magcol]['binned_stetsonj_stdev']
            ),
            linewidth=3.0, linestyle='dashed'
        )
        plt.xlim((magbins.min()-0.25, magbins.max()))
        plt.xlabel('SDSS r')
        plt.ylabel('Stetson J index')
        plt.title('%s - SDSS r vs. Stetson J index' % magcol)
        plt.yscale('log')
        plt.tight_layout()

        # the mag vs IQR
        plt.subplot(223)
        plt.plot(allobjects[magcol]['sdssr'],
                 allobjects[magcol]['iqr'],
                 marker='.',ms=1.0, linestyle='none',
                 rasterized=True)
        plt.plot(allobjects[magcol]['binned_sdssr_median'],
                 allobjects[magcol]['binned_iqr_median'],
                 linewidth=3.0)
        plt.plot(
            allobjects[magcol]['binned_sdssr_median'],
            np.array(allobjects[magcol]['binned_iqr_median']) +
            min_iqr_stdev*np.array(
                allobjects[magcol]['binned_iqr_stdev']
            ),
            linewidth=3.0, linestyle='dashed'
        )
        plt.xlabel('SDSS r')
        plt.ylabel('IQR')
        plt.title('%s - SDSS r vs. IQR' % magcol)
        plt.xlim((magbins.min()-0.25, magbins.max()))
        plt.yscale('log')
        plt.tight_layout()

        # the mag vs IQR
        plt.subplot(224)
        plt.plot(allobjects[magcol]['sdssr'],
                 allobjects[magcol]['inveta'],
                 marker='.',ms=1.0, linestyle='none',
                 rasterized=True)
        plt.plot(allobjects[magcol]['binned_sdssr_median'],
                 allobjects[magcol]['binned_inveta_median'],
                 linewidth=3.0)
        plt.plot(
            allobjects[magcol]['binned_sdssr_median'],
            np.array(allobjects[magcol]['binned_inveta_median']) +
            min_inveta_stdev*np.array(
                allobjects[magcol]['binned_inveta_stdev']
            ),
            linewidth=3.0, linestyle='dashed'
        )
        plt.xlabel('SDSS r')
        plt.ylabel(r'$1/\eta$')
        plt.title(r'%s - SDSS r vs. $1/\eta$' % magcol)
        plt.xlim((magbins.min()-0.25, magbins.max()))
        plt.yscale('log')
        plt.tight_layout()

        plt.savefig('varfeatures-%s-%s-distributions.png' % (varthreshpkl,
                                                             magcol),
                    bbox_inches='tight')
        plt.close('all')
