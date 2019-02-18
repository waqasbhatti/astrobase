#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''starfeatures.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

This contains functions to obtain various star magnitude and color features for
large numbers of light curves. Useful later for variable star classification.

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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from tornado.escape import squeeze

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce
from operator import getitem
def dict_get(datadict, keylist):
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

from astrobase.varclass import starfeatures
from astrobase.lcproc import get_lcformat



###################
## STAR FEATURES ##
###################

def get_starfeatures(lcfile,
                     outdir,
                     kdtree,
                     objlist,
                     lcflist,
                     neighbor_radius_arcsec,
                     deredden=True,
                     custom_bandpasses=None,
                     lcformat='hat-sql',
                     lcformatdir=None):
    '''This runs the functions from astrobase.varclass.starfeatures on a single
    light curve file.

    lcfile is the LC file to extract star features for

    outdir is the directory to write the output pickle to

    kdtree is a scipy.spatial KDTree or cKDTree

    objlist is a numpy array of object IDs in the same order as KDTree.data

    lcflist is a numpy array of light curve filenames in the same order as
    KDTree.data

    neighbor_radius_arcsec indicates the radius in arcsec to search for
    neighbors for this object using the light curve catalog's kdtree, objlist,
    and lcflist and in GAIA.

    deredden controls if the colors and any color classifications will be
    dereddened using 2MASS DUST.

    lcformat is a key in LCFORM specifying the type of light curve lcfile is

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

        # run the coord features first
        coordfeat = starfeatures.coord_features(lcdict['objectinfo'])

        # next, run the color features
        colorfeat = starfeatures.color_features(
            lcdict['objectinfo'],
            deredden=deredden,
            custom_bandpasses=custom_bandpasses
        )

        # run a rough color classification
        colorclass = starfeatures.color_classification(colorfeat,
                                                       coordfeat)

        # finally, run the neighbor features
        nbrfeat = starfeatures.neighbor_gaia_features(lcdict['objectinfo'],
                                                      kdtree,
                                                      neighbor_radius_arcsec)

        # get the objectids of the neighbors found if any
        if nbrfeat['nbrindices'].size > 0:
            nbrfeat['nbrobjectids'] = objlist[nbrfeat['nbrindices']]
            nbrfeat['closestnbrobjectid'] = objlist[
                nbrfeat['closestdistnbrind']
            ]
            nbrfeat['closestnbrlcfname'] = lcflist[
                nbrfeat['closestdistnbrind']
            ]

        else:
            nbrfeat['nbrobjectids'] = np.array([])
            nbrfeat['closestnbrobjectid'] = np.array([])
            nbrfeat['closestnbrlcfname'] = np.array([])

        # update the result dict
        resultdict.update(coordfeat)
        resultdict.update(colorfeat)
        resultdict.update(colorclass)
        resultdict.update(nbrfeat)

        outfile = os.path.join(outdir,
                               'starfeatures-%s.pkl' %
                               squeeze(resultdict['objectid']).replace(' ','-'))

        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=4)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to get star features for %s because: %s' %
                     (os.path.basename(lcfile), e))
        return None



def starfeatures_worker(task):
    '''
    This wraps starfeatures.

    '''

    try:
        (lcfile, outdir, kdtree, objlist,
         lcflist, neighbor_radius_arcsec,
         deredden, custom_bandpasses, lcformat, lcformatdir) = task

        return get_starfeatures(lcfile, outdir,
                                kdtree, objlist, lcflist,
                                neighbor_radius_arcsec,
                                deredden=deredden,
                                custom_bandpasses=custom_bandpasses,
                                lcformat=lcformat,
                                lcformatdir=lcformatdir)
    except Exception as e:
        return None


def serial_starfeatures(lclist,
                        outdir,
                        lclistpickle,
                        neighbor_radius_arcsec,
                        maxobjects=None,
                        deredden=True,
                        custom_bandpasses=None,
                        lcformat='hat-sql',
                        lcformatdir=None,
                        nworkers=NCPUS):
    '''This drives the starfeatures function for a collection of LCs.

    lclistpickle is a pickle containing at least:

    - an object ID array accessible with dict keys ['objects']['objectid']

    - an LC filename array accessible with dict keys ['objects']['lcfname']

    - a scipy.spatial.KDTree or cKDTree object to use for finding neighbors for
      each object accessible with dict key ['kdtree']

    This pickle can be produced using lcproc.make_lclist.

    '''
    # make sure to make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maxobjects:
        lclist = lclist[:maxobjects]

    # read in the kdtree pickle
    with open(lclistpickle, 'rb') as infd:
        kdt_dict = pickle.load(infd)

    kdt = kdt_dict['kdtree']
    objlist = kdt_dict['objects']['objectid']
    objlcfl = kdt_dict['objects']['lcfname']

    tasks = [(x, outdir, kdt, objlist, objlcfl,
              neighbor_radius_arcsec,
              deredden, custom_bandpasses,
              lcformat, lcformatdir) for x in lclist]

    for task in tqdm(tasks):
        result = starfeatures_worker(task)

    return result



def parallel_starfeatures(lclist,
                          outdir,
                          lclistpickle,
                          neighbor_radius_arcsec,
                          maxobjects=None,
                          deredden=True,
                          custom_bandpasses=None,
                          lcformat='hat-sql',
                          lcformatdir=None,
                          nworkers=NCPUS):
    '''
    This runs starfeatures in parallel for all light curves in lclist.

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

    # make sure to make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maxobjects:
        lclist = lclist[:maxobjects]

    # read in the kdtree pickle
    with open(lclistpickle, 'rb') as infd:
        kdt_dict = pickle.load(infd)

    kdt = kdt_dict['kdtree']
    objlist = kdt_dict['objects']['objectid']
    objlcfl = kdt_dict['objects']['lcfname']

    tasks = [(x, outdir, kdt, objlist, objlcfl,
              neighbor_radius_arcsec,
              deredden, custom_bandpasses, lcformat) for x in lclist]

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(starfeatures_worker, tasks)

    results = [x for x in resultfutures]
    resdict = {os.path.basename(x):y for (x,y) in zip(lclist, results)}

    return resdict



def parallel_starfeatures_lcdir(lcdir,
                                outdir,
                                lclistpickle,
                                neighbor_radius_arcsec,
                                fileglob=None,
                                maxobjects=None,
                                deredden=True,
                                custom_bandpasses=None,
                                lcformat='hat-sql',
                                lcformatdir=None,
                                nworkers=NCPUS,
                                recursive=True):
    '''
    This runs parallel star feature extraction for a directory of LCs.

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

    if not fileglob:
        fileglob = dfileglob

    # now find the files
    LOGINFO('searching for %s light curves in %s ...' % (lcformat, lcdir))

    if recursive is False:
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

        LOGINFO('found %s light curves, getting starfeatures...' %
                len(matching))

        return parallel_starfeatures(matching,
                                     outdir,
                                     lclistpickle,
                                     neighbor_radius_arcsec,
                                     deredden=deredden,
                                     custom_bandpasses=custom_bandpasses,
                                     maxobjects=maxobjects,
                                     lcformat=lcformat,
                                     lcformatdir=lcformatdir,
                                     nworkers=nworkers)

    else:

        LOGERROR('no light curve files in %s format found in %s' % (lcformat,
                                                                    lcdir))
        return None
