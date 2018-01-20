#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''lcproc.batch.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - May 2017

This contains functions that work on single light curve files at a time. The
intended uses are:

- directly in a worker container that gets deployed to AWS batch, GKE, or Mesos,
  etc., and can be parallelized across compute nodes.

- as a base for single-compute-node driver functions to be used in the lcproc
  CLI, which is implemented in lcproc_cli.py.

The structure of this file is a bit weird because we've moved most of the heavy
imports into the functions themselves. This is to make the CLI start as fast as
possible, and defer the imports until we've decided which function to run.


List of CLI commands
--------------------

lcpbatch help

lcpbatch register-lcformat --lcformat specifications

lcpbatch make-lclist /path/to/lc/basedir outfile.pkl --options

lcpbatch filter-lclist /path/to/lc/list.pkl outfile.pkl --options
                                                        --copyto

lcpbatch varfeatures /path/to/lc.file outfile.pkl --options

lcpbatch varthreshold /path/to/varfeature/dir --options
                                              --plot to plot all selections
                                              --copyto dir to copy selected vars

lcpbatch periodfind /path/to/lc.file outdir --options

lcpbatch checkplot /path/to/lc.file outdir --options
                                           --all-lclist to pick up neighbors

lcpbatch cp-png /path/to/checkplot.pkl

lcpbatch timebin /path/to/lc.file outdir --options

'''
#############
## LOGGING ##
#############

import logging
from datetime import datetime
from traceback import format_exc

# setup a logger
LOGGER = None
LOGMOD = __name__
DEBUG = False

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.%s' % (parent_name, LOGMOD))

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('[%s - DBUG] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('[%s - INFO] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('[%s - ERR!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('[%s - WRN!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '[%s - EXC!] %s\nexception was: %s' % (
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                message, format_exc()
                )
            )


#############
## IMPORTS ##
#############

import os
import os.path
import sys

try:
    import cPickle as pickle
except:
    import pickle

import gzip
import glob
import fnmatch
import shutil
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import importlib

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


###################
## LOCAL IMPORTS ##
###################

# things that import fairly fast
# everything else is deferred to the functions themselves
from astrobase.varclass import varfeatures
from astrobase.lcmath import normalize_magseries, \
    time_bin_magseries_with_errs, sigclip_magseries
from astrobase.magnitudes import jhk_to_sdssr


#############################################
## MAPS FOR LCFORMAT TO LCREADER FUNCTIONS ##
#############################################

def read_pklc(lcfile):
    '''
    This just reads a pickle.

    '''

    try:
        with open(lcfile,'rb') as infd:
            lcdict = pickle.load(infd)
    except UnicodeDecodeError:
        with open(lcfile,'rb') as infd:
            lcdict = pickle.load(infd, encoding='latin1')

    return lcdict



# these translate filter operators given as strings to Python operators
FILTEROPS = {'eq':'==',
             'gt':'>',
             'ge':'>=',
             'lt':'<',
             'le':'<=',
             'ne':'!='}


######################################
## DEALING WITH LIGHT CURVE FORMATS ##
######################################

def check_extmodule(module, formatkey):
    '''This just imports the module specified.

    '''

    try:

        if os.path.exists(module):

            sys.path.append(os.path.dirname(module))
            importlib.import_module(
                os.path.basename(module.replace('.py',''))
            )
        else:
            importlib.import_module(module)

    except Exception as e:

        LOGEXCEPTION('could not import the module: %s for LC format: %s. '
                     'check the file path or fully qualified module name?'
                     % (module, formatkey))
        raise



def get_lcformat_spec(lcformatkey, lcformatdir=None):
    '''This gets info for LC formats from the format JSON file corresponding to
    lcformatkey in the config directory lcformatdir.

    If lcformatdir is None, then this function will search in the data subdir of
    the astrobase installation.

    '''

    if lcformatdir is None:
        lcformatdir = os.path.join(os.path.dirname(__file__),
                                   'data',
                                   'lcformats')

    # find the corresponding lcformat JSON file
    lcformatjson = os.path.join(lcformatdir, '%s.json' % lcformatkey)

    if not os.path.exists(lcformatjson):

         LOGERROR("expected JSON file %s with light curve format "
                  "specification for %s not found in %s, can't continue" %
                  (lcformatjson, lcformatkey, lcformatdir))
         return None

    # open the JSON
    with open(lcformatjson,'rb') as infd:
        formatspec = json.load(infd)

    return formatspec



def register_custom_lcformat(formatkey,
                             fileglob,
                             timecols,
                             magcols,
                             errcols,
                             readermodule,
                             readerfunc,
                             readerkwargs=None,
                             normmodule=None,
                             normfunc=None,
                             normkwargs=None,
                             magsarefluxes=False,
                             lcformatdir=None):
    '''This adds a custom format spec JSON to the lcformats directory.

    Allows handling of custom format light curves for astrobase lcproc
    drivers. Once the format is successfully registered, light curves should
    work transparently with all of the functions below, by simply calling them
    with the formatkey in the lcformat keyword argument.

    Args
    ----

    formatkey: <string>: what to use as the key for your light curve format


    fileglob: <string>: the default fileglob to use to search for light curve
    files in this custom format. This is a string like
    '*-whatever-???-*.*??-.lc'.


    readerfunc: <function>: this is the function to use to read light curves in
    the custom format. This should return a dictionary (the 'lcdict') with the
    following signature (the keys listed below are required, but others are
    allowed):

    {'objectid':'<this object's name>',
     'objectinfo':{'ra':<this object's right ascension>
                   'decl':<this object's declination>},
     ...time columns, mag columns, etc.}


    timecols, magcols, errcols: <list>: these are all lists of strings
    indicating which keys in the lcdict to use for processing. The lists must
    all have the same dimensions, e.g. if timecols = ['timecol1','timecol2'],
    then magcols must be something like ['magcol1','magcol2'] and errcols must
    be something like ['errcol1', 'errcol2']. This allows you to process
    multiple apertures or multiple types of measurements in one go.

    Each element in these lists can be a simple key, e.g. 'time' (which would
    correspond to lcdict['time']), or a composite key,
    e.g. 'aperture1.times.rjd' (which would correspond to
    lcdict['aperture1']['times']['rjd']). See the LCFORM dict above for
    examples.


    magsarefluxes: <boolean>: if this is True, then all functions will treat the
    magnitude columns as flux instead, so things like default normalization and
    sigma-clipping will be done correctly. If this is False, magnitudes will be
    treated as magnitudes.


    specialnormfunc: <function>: if you intend to use a special normalization
    function for your lightcurves, indicate it here. If None, the default
    normalization method used by lcproc is to find gaps in the time-series,
    normalize measurements grouped by these gaps to zero, then normalize the
    entire magnitude time series to global time series median using the
    astrobase.lcmath.normalize_magseries function. The function should take and
    return an lcdict of the same form as that produced by readerfunc above. For
    an example of a special normalization function, see normalize_lcdict_by_inst
    in the astrobase.hatlc module.

    '''
    #
    # make sure the provided modules and functions are importable
    #

    check_extmodule(readermodule, formatkey)

    if normmodule:
        check_extmodule(normmodule, formatkey)

    # get the lcformatdir
    if lcformatdir is None:
        lcformatdir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                   '../data',
                                                   'lcformats'))

    formatdict = {'fileglob':fileglob,
                  'timecols':timecols,
                  'magcols':magcols,
                  'errcols':errcols,
                  'magsarefluxes':magsarefluxes,
                  'lcreader_module':readermodule,
                  'lcreader_func':readerfunc,
                  'lcreader_kwargs':readerkwargs,
                  'lcnorm_module':normmodule,
                  'lcnorm_func':normfunc,
                  'lcnorm_kwargs':normkwargs}

    formatfile = os.path.join(lcformatdir,'%s.json' % formatkey)
    with open(formatfile,'w') as outfd:
        json.dump(formatdict, outfd, indent=2)

    return formatfile



#######################
## UTILITY FUNCTIONS ##
#######################

def lclist_parallel_worker(task):
    '''
    This is a parallel worker for makelclist.

    task[0] = lcf
    task[1] = columns
    task[2] = readerfunc
    task[3] = lcndetkey
    task[4] = readerkwargs


    '''

    lcf, columns, readerfunc, lcndetkey, readerkwargs = task

    # we store the full path of the light curve
    lcobjdict = {'lcfname':lcf}

    try:

        # read the light curve in
        if readerkwargs:
            lcdict = readerfunc(lcf, **readerkwargs)
        else:
            lcdict = readerfunc(lcf)


        if len(lcdict) == 2:
            lcdict = lcdict[0]

        # insert all of the columns
        for colkey in columns:

            if '.' in colkey:
                getkey = colkey.split('.')
            else:
                getkey = [colkey]

            try:
                thiscolval = dict_get(lcdict, getkey)
            except:
                LOGWARNING('column %s does not exist for %s' %
                           (colkey, lcf))
                thiscolval = np.nan

            # update the lcobjdict with this value
            lcobjdict[getkey[-1]] = thiscolval

    except Exception as e:

        LOGEXCEPTION('could not figure out columns for %s' % lcf)

        # insert all of the columns as nans
        for colkey in columns:

            if '.' in colkey:
                getkey = colkey.split('.')
            else:
                getkey = [colkey]

            thiscolval = np.nan

            # update the lclistdict with this value
            lcobjdict[getkey[-1]] = thiscolval

    # now get the actual ndets; this excludes nans and infs
    for dk in lcndetkey:

        try:

            if '.' in dk:
                getdk = dk.split('.')
            else:
                getdk = [dk]

            ndetcol = dict_get(lcdict, getdk)
            actualndets = ndetcol[np.isfinite(ndetcol)].size
            lcobjdict['ndet_%s' % getdk[-1]] = actualndets

        except:
            lcobjdict['ndet_%s' % getdk[-1]] = np.nan


    return lcobjdict



def make_lclist(basedir,
                outfile,
                lcformat='hat-sql',
                fileglob=None,
                recursive=True,
                columns=['objectid',
                         'objectinfo.ra','objectinfo.decl',
                         'objectinfo.ndet','objectinfo.sdssr'],
                makecoordindex=['objectinfo.ra','objectinfo.decl'],
                maxlcs=None,
                nworkers=20):

    '''This generates a list file compatible with getlclist below.

    Given a base directory where all the files are, and a light curve format,
    this will find all light curves, pull out the keys in each lcdict requested
    in the columns kwarg for each object, and write them to the requested output
    pickle file. These keys should be pointers to scalar values (i.e. something
    like objectinfo.ra is OK, but something like rjd won't work because it's a
    vector).

    If basedir is a list of directories, all of these will be searched
    recursively to find the matching light curve files.

    All of the keys in the columns kwarg should be present in the lcdict
    generated by the reader function for the specified lcformat.

    fileglob is a shell glob to use to select the filenames. If None, then the
    default one for the provided lcformat will be used.

    If recursive is True, then the function will search recursively in basedir
    for any light curves matching the specified criteria. This may take a while,
    especially on network filesystems.

    If makecoordindex is not None, it must be a two-element list of the lcdict
    keys for the right ascension and declination for each object. These will be
    used to make a kdtree for fast look-up by position later by getlclist.

    This returns a pickle file.

    '''

    #
    # handle the lcformat
    #
    formatspec = get_lcformat_spec(lcformat)

    # first, import the reader module
    readermod = importlib.import_module(formatspec['lcreader_module'])

    # then, get the function we need to read the lightcurve
    # NOTE: this is literally black magic
    readerfunc = getattr(readermod, formatspec['lcreader_func'])

    if formatspec['lcreader_kwargs'] is not None:
        readerkwargs = formatspec['lcreader_kwargs']
    else:
        readerkwargs = None

    # get the fileglob if necessary
    if not fileglob:
        fileglob = formatspec['fileglob']

    # this is to get the actual ndet
    # set to the magnitudes column
    lcndetkey = formatspec['magcols']

    # handle the case where basedir is a list of directories
    if isinstance(basedir, list):

        matching = []

        for bdir in basedir:

            # now find the files
            LOGINFO('searching for %s light curves in %s ...' % (lcformat,
                                                                 bdir))

            if recursive == False:
                matching.extend(glob.glob(os.path.join(bdir, fileglob)))

            else:
                # use recursive glob for Python 3.5+
                if sys.version_info[:2] > (3,4):

                    matching.extend(glob.glob(os.path.join(bdir,
                                                           '**',
                                                           fileglob),
                                              recursive=True))

                # otherwise, use os.walk and glob
                else:

                    # use os.walk to go through the directories
                    walker = os.walk(bdir)

                    for root, dirs, files in walker:
                        for sdir in dirs:
                            searchpath = os.path.join(root,
                                                      sdir,
                                                      fileglob)
                            foundfiles = glob.glob(searchpath)

                            if foundfiles:
                                matching.extend(foundfiles)


    # otherwise, handle the usual case of one basedir to search in
    else:

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

        LOGINFO('found %s light curves' % len(matching))

        # cut down matching to maxlcs
        if maxlcs:
            matching = matching[:maxlcs]

        # prepare the output dict
        lclistdict = {
            'basedir':basedir,
            'lcformat':lcformat,
            'fileglob':fileglob,
            'recursive':recursive,
            'columns':columns,
            'makecoordindex':makecoordindex,
            'nfiles':len(matching),
            'objects': {
            }
        }

        # columns that will always be present in the output lclistdict
        derefcols = ['lcfname']
        derefcols.extend(['ndet_%s' % x.split('.')[-1] for x in lcndetkey])

        for dc in derefcols:
            lclistdict['objects'][dc] = []

        # fill in the rest of the lclist columns from the columns kwarg
        for col in columns:

            # dereference the column
            thiscol = col.split('.')
            thiscol = thiscol[-1]
            lclistdict['objects'][thiscol] = []
            derefcols.append(thiscol)

        # start collecting info
        LOGINFO('collecting light curve info...')

        tasks = [(x, columns, readerfunc, lcndetkey, readerkwargs)
                 for x in matching]

        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            results = executor.map(lclist_parallel_worker, tasks)

        results = [x for x in results]

        # update the columns in the overall dict from the results of the
        # parallel map
        for result in results:
            for xcol in derefcols:
                lclistdict['objects'][xcol].append(result[xcol])

        executor.shutdown()

        # done with collecting info
        # turn all of the lists in the lclistdict into arrays
        for col in lclistdict['objects']:
            lclistdict['objects'][col] = np.array(lclistdict['objects'][col])

        # if we're supposed to make a spatial index, do so
        if (makecoordindex and
            isinstance(makecoordindex, list) and
            len(makecoordindex) == 2):

            import scipy.spatial as sps

            try:

                # deref the column names
                racol, declcol = makecoordindex
                racol = racol.split('.')[-1]
                declcol = declcol.split('.')[-1]

                # get the ras and decls
                objra, objdecl = (lclistdict['objects'][racol],
                                  lclistdict['objects'][declcol])

                # get the xyz unit vectors from ra,decl
                # since i had to remind myself:
                # https://en.wikipedia.org/wiki/Equatorial_coordinate_system
                cosdecl = np.cos(np.radians(objdecl))
                sindecl = np.sin(np.radians(objdecl))
                cosra = np.cos(np.radians(objra))
                sinra = np.sin(np.radians(objra))
                xyz = np.column_stack((cosra*cosdecl,sinra*cosdecl, sindecl))

                # generate the kdtree
                kdt = sps.cKDTree(xyz,copy_data=True)

                # put the tree into the dict
                lclistdict['kdtree'] = kdt

                LOGINFO('kdtree generated for (ra, decl): %s' %
                        makecoordindex)

            except Exception as e:
                LOGEXCEPTION('could not make kdtree for (ra, decl): %s' %
                             makecoordindex)
                raise


        # write the pickle
        with open(outfile,'wb') as outfd:
            pickle.dump(lclistdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

        LOGINFO('done. LC info -> %s' % outfile)
        return outfile

    else:

        LOGERROR('no files found in %s matching %s' % (basedir, fileglob))
        return None



def filter_lclist(listpickle,
                  objectidcol='objectid',
                  xmatchexternal=None,
                  xmatchdistarcsec=3.0,
                  externalcolnums=(0,1,2),
                  externalcolnames=['objectid','ra','decl'],
                  externalcoldtypes='U20,f8,f8',
                  externalcolsep=None,
                  conesearch=None,
                  columnfilters=None,
                  conesearchworkers=1,
                  copylcsto=None,
                  outfile=None):

    '''This is used to collect light curves based on selection criteria.

    Uses the output of makelclist above. This function returns a list of light
    curves matching various criteria speciifed by the xmatchexternal,
    conesearch, and columnfilters kwargs. Use this function to generate input
    lists for the parallel_varfeatures, parallel_pf, and parallel_timebin
    functions below.

    The filter operations are applied in this order if more than one is
    specified: xmatchexternal -> conesearch -> columnfilters. All results from
    these filter operations are joined using a logical AND operation.

    Returns a two elem tuple: (matching_object_lcfiles, matching_objectids) if
    conesearch and/or column filters are used. If xmatchexternal is also used, a
    three-elem tuple is returned: (matching_object_lcfiles, matching_objectids,
    extcat_matched_objectids).

    Args
    ----

    objectidcol is the name of the object ID column in the listpickle file.


    If not None, xmatchexternal is a filename containing objectids, ras and decs
    to match the objects in this listpickle to by their positions. Use the other
    external* kwargs to provide the remaining info required:

    xmatchdistarcsec is the distance to use when matching in arcseconds.

    externalcolnums are the zero-indexed column numbers in the file containing
    objectid, ra, dec values.

    externalcolnames are the names of the columns to pull out from the external
    catalog file.

    externalcoldtypes are numpy dtype specifications for the objectid, ra, decl
    columns in the external catalog file.

    externalcolsep is the separator character to use to slice the external
    catalog file into columns. If None, will use blank space (space/tab) as the
    separator.


    conesearch is a three-element list:

    [center_ra_deg, center_decl_deg, search_radius_deg]

    This is used with the kdtree in the lclist pickle to only return objects
    that are in the specified region. consearchworkers specifies the number of
    parallel workers that can be launched by scipy to search for objects in the
    kdtree.


    columnfilters is a list of strings indicating how to filter on columns in
    the lclist pickle. All column filters are applied in the specified sequence
    and are combined with a logical AND operator. The format of each filter
    string should be:

    '<lclist column>|<operator>|<operand>'

    where:

    <lclist column> is a column in the lclist dict

    <operator> is one of: 'lt', 'gt', 'le', 'ge', 'eq', 'ne', which correspond
    to the usual operators: <, >, <=, >=, ==, != respectively.

    <operand> is a float, int, or string.


    If copylcsto is not None, it is interpreted as a directory target to copy
    all the light curves that match the specified conditions.

    '''
    import scipy.spatial as sps

    with open(listpickle,'rb') as infd:
        lclist = pickle.load(infd)

    # generate numpy arrays of the matching object indexes. we do it this way so
    # we can AND everything at the end, instead of having to look up the objects
    # at these indices and running the columnfilter on them
    xmatch_matching_index = np.full_like(lclist['objects'][objectidcol],
                                         False,
                                         dtype=np.bool)
    conesearch_matching_index = np.full_like(lclist['objects'][objectidcol],
                                             False,
                                             dtype=np.bool)
    # do the xmatch first
    ext_matches = []
    ext_matching_objects = []

    if (xmatchexternal and
        isinstance(xmatchexternal, str) and
        os.path.exists(xmatchexternal)):


        try:

            # read in the external file
            extcat = np.genfromtxt(xmatchexternal,
                                   usecols=externalcolnums,
                                   delimiter=externalcolsep,
                                   names=externalcolnames,
                                   dtype=externalcoldtypes)

            ext_cosdecl = np.cos(np.radians(extcat['decl']))
            ext_sindecl = np.sin(np.radians(extcat['decl']))
            ext_cosra = np.cos(np.radians(extcat['ra']))
            ext_sinra = np.sin(np.radians(extcat['ra']))

            ext_xyz = np.column_stack((ext_cosra*ext_cosdecl,
                                       ext_sinra*ext_cosdecl,
                                       ext_sindecl))
            ext_xyzdist = 2.0 * np.sin(np.radians(xmatchdistarcsec/3600.0)/2.0)

            # get our kdtree
            our_kdt = lclist['kdtree']

            # get the external kdtree
            ext_kdt = sps.cKDTree(ext_xyz)

            # do a query_ball_tree
            extkd_matchinds = ext_kdt.query_ball_tree(our_kdt, ext_xyzdist)

            for extind, mind in enumerate(extkd_matchinds):
                if len(mind) > 0:
                    ext_matches.append(mind[0])
                    ext_matching_objects.append(extcat['objectid'][extind])

            ext_matches = np.array(ext_matches)

            if ext_matches.size > 0:

                # update the xmatch_matching_index
                xmatch_matching_index[ext_matches] = True

                LOGINFO('xmatch: objects matched to %s within %.1f arcsec: %s' %
                        (extfile, extmatchdist, ext_matches.size))

            else:

                LOGERROR("xmatch: no objects were cross-matched to external "
                         "catalog spec: %s, can't continue" % xmatchexternal)
                return None, None, None


        except Exception as e:

            LOGEXCEPTION('could not match to external catalog spec: %s' %
                         repr(xmatchexternal))
            raise


    # do the cone search next
    if (conesearch and isinstance(conesearch, list) and len(conesearch) == 3):

        try:

            racenter, declcenter, searchradius = conesearch
            cosdecl = np.cos(np.radians(declcenter))
            sindecl = np.sin(np.radians(declcenter))
            cosra = np.cos(np.radians(racenter))
            sinra = np.sin(np.radians(racenter))

            # this is the search distance in xyz unit vectors
            xyzdist = 2.0 * np.sin(np.radians(searchradius)/2.0)

            # get the kdtree
            our_kdt = lclist['kdtree']

            # look up the coordinates
            kdtindices = our_kdt.query_ball_point([cosra*cosdecl,
                                                   sinra*cosdecl,
                                                   sindecl],
                                                  xyzdist,
                                                  n_jobs=conesearchworkers)

            if kdtindices and len(kdtindices) > 0:

                LOGINFO('cone search: objects within %.4f deg '
                        'of (%.3f, %.3f): %s' %
                        (searchradius, racenter, declcenter, len(kdtindices)))

                # update the conesearch_matching_index
                matchingind = kdtindices
                conesearch_matching_index[np.array(matchingind)] = True

            # we fail immediately if we found nothing. this assumes the user
            # cares more about the cone-search than the regular column filters
            else:

                LOGERROR("cone-search: no objects were found within "
                         "%.4f deg of (%.3f, %.3f): %s, can't continue" %
                        (searchradius, racenter, declcenter, len(kdtindices)))
                return None, None


        except Exception as e:

            LOGEXCEPTION('cone-search: could not run a cone-search, '
                         'is there a kdtree present in %s?' % listpickle)
            raise


    # now that we're done with cone-search, do the column filtering
    allfilterinds = []
    if columnfilters and isinstance(columnfilters, list):

        # go through each filter
        for cfilt in columnfilters:

            try:

                fcol, foperator, foperand = cfilt.split('|')
                foperator = FILTEROPS[foperator]

                # generate the evalstring
                filterstr = (
                    "np.isfinite(lclist['objects']['%s']) & "
                    "(lclist['objects']['%s'] %s %s)"
                    ) % (fcol, fcol, foperator, foperand)
                filterind = eval(filterstr)

                ngood = lclist['objects'][objectidcol][filterind].size
                LOGINFO('filter: %s -> objects matching: %s ' % (cfilt, ngood))

                allfilterinds.append(filterind)

            except Exception as e:

                LOGEXCEPTION('filter: could not understand filter spec: %s'
                             % cfilt)
                LOGWARNING('filter: not applying this broken filter')


    # now that we have all the filter indices good to go
    # logical-AND all the things

    # make sure we only do filtering if we were told to do so
    if (xmatchexternal or conesearch or columnfilters):

        filterstack = []
        if xmatchexternal:
            filterstack.append(xmatch_matching_index)
        if conesearch:
            filterstack.append(conesearch_matching_index)
        if columnfilters:
            filterstack.extend(allfilterinds)

        finalfilterind = np.column_stack(filterstack)
        finalfilterind = np.all(finalfilterind, axis=1)

        # get the filtered object light curves and object names
        filteredobjectids = lclist['objects'][objectidcol][finalfilterind]
        filteredlcfnames = lclist['objects']['lcfname'][finalfilterind]

    else:

        filteredobjectids = lclist['objects'][objectidcol]
        filteredlcfnames = lclist['objects']['lcfname']


    # if copylcsto is not None, copy LCs over to it
    if copylcsto is not None:

        if not os.path.exists(copylcsto):
            os.mkdir(copylcsto)

        if TQDM:
            lciter = tqdm(filteredlcfnames)
        else:
            lciter = filteredlcfnames

        LOGINFO('copying matching light curves to %s' % copylcsto)

        for lc in lciter:
            shutil.copy(lc, copylcsto)

    LOGINFO('done. objects matching all filters: %s' % filteredobjectids.size)

    if xmatchexternal and len(ext_matching_objects) > 0:

        if outfile is not None:

            with open(outfile, 'wb') as outfd:
                outdict = {
                    'inputlist':listpickle,
                    'kwargs':{},
                    'filtered_lcfnames':filteredlcfnames,
                    'filtered_objectids':filteredobjectids,
                    'filtered_extmatches':ext_matching_objects,
                }
                pickle.dump(outdict, outfd, pickle.HIGHEST_PROTOCOL)

        return filteredlcfnames, filteredobjectids, ext_matching_objects


    else:

        if outfile is not None:

            with open(outfile, 'wb') as outfd:
                outdict = {
                    'inputlist':listpickle,
                    'kwargs':{},
                    'filtered_lcfnames':filteredlcfnames,
                    'filtered_objectids':filteredobjectids,
                    'filtered_extmatches':None,
                }
                pickle.dump(outdict, outfd, pickle.HIGHEST_PROTOCOL)


        return filteredlcfnames, filteredobjectids


##################################
## GETTING VARIABILITY FEATURES ##
##################################

def get_varfeatures(lcfile,
                    outdir,
                    timecols=None,
                    magcols=None,
                    errcols=None,
                    mindet=1000,
                    lcformat='hat-sql'):
    '''
    This runs varfeatures on a single LC file.

    '''
    #
    # handle the lcformat
    #
    formatspec = get_lcformat_spec(lcformat)

    # first, import the reader module
    readermod = importlib.import_module(formatspec['lcreader_module'])

    # then, get the function we need to read the lightcurve
    readerfunc = getattr(readermod, formatspec['lcreader_func'])

    # get any default kwargs we've set for this LC format
    if formatspec['lcreader_kwargs'] is not None:
        readerkwargs = formatspec['lcreader_kwargs']
    else:
        readerkwargs = None

    # get the default timecols, magcols, errcols
    if timecols is None:
        timecols = formatspec['timecols']
    if magcols is None:
        magcols = formatspec['magcols']
    if errcols is None:
        errcols = formatspec['errcols']

    # get the normalization module and function if provided
    if formatspec['lcnorm_module'] is not None:
        normmod = importlib.import_module(formatspec['lcnorm_module'])
        normfunc = getattr(normmod, formatspec['lcnorm_func'])
    else:
        normfunc = None

    # get any normalization function kwargs we've set
    if formatspec['lcnorm_kwargs'] is not None:
        normkwargs = formatspec['lcnorm_kwargs']
    else:
        normkwargs = None

    # finally, get the magsarefluxes key
    magsarefluxes = formatspec['magsarefluxes']

    try:

        # get the LC into a dict

        if readerkwargs:
            lcdict = readerfunc(lcfile, **readerkwargs)
        else:
            lcdict = readerfunc(lcfile)


        if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
            lcdict = lcdict[0]

        resultdict = {'objectid':lcdict['objectid'],
                      'info':lcdict['objectinfo'],
                      'lcfbasename':os.path.basename(lcfile)}


        # normalize using the special function if specified
        if normfunc is not None:

            if normkwargs is not None:
                lcdict = normfunc(lcdict, **normkwargs)
            else:
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
                resultdict[mcolget[-1]] = None

            else:

                # get the features for this magcol
                lcfeatures = features.all_nonperiodic_features(
                    times, mags, errs
                )
                resultdict[mcolget[-1]] = lcfeatures

        # now that we've collected all the magcols, we can choose which is the
        # "best" magcol. this is defined as the magcol that gives us the
        # smallest LC MAD.

        try:
            magmads = np.zeros(len(magcols))
            for mind, mcol in enumerate(magcols):
                if '.' in mcol:
                    mcolget = mcol.split('.')
                else:
                    mcolget = [mcol]

                magmads[mind] = resultdict[mcolget[-1]]['mad']

            # smallest MAD index
            bestmagcolind = np.where(magmads == np.min(magmads))[0]
            resultdict['bestmagcol'] = magcols[bestmagcolind]

        except:
            resultdict['bestmagcol'] = None

        outfile = os.path.join(outdir,
                               'varfeatures-%s.pkl' % resultdict['objectid'])

        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=4)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to get LC features for %s because: %s' %
              (os.path.basename(lcfile), e))
        return None



def variability_threshold(featuresdir,
                          outfile,
                          magbins=np.arange(8.0,16.25,0.25),
                          maxobjects=None,
                          timecols=None,
                          magcols=None,
                          errcols=None,
                          lcformat='hat-sql',
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

    outfile is a pickle file that will contain all the info.

    min_lcmad_stdev, min_stetj_stdev, min_iqr_stdev, min_inveta_stdev are all
    stdev multipliers to use for selecting variable objects. These are either
    scalar floats to apply the same sigma cut for each magbin or np.ndarrays of
    size = magbins.size - 1 to apply different sigma cuts for each magbin.

    FIXME: implement a voting classifier here. this will choose variables based
    on the thresholds in IQR, stetson, and inveta based on weighting carried
    over from the variability recovery sims.

    '''
    #
    # handle the lcformat
    #
    formatspec = get_lcformat_spec(lcformat)

    # first, import the reader module
    readermod = importlib.import_module(formatspec['lcreader_module'])

    # then, get the function we need to read the lightcurve
    readerfunc = getattr(readermod, formatspec['lcreader_func'])

    # get any default kwargs we've set for this LC format
    if formatspec['lcreader_kwargs'] is not None:
        readerkwargs = formatspec['lcreader_kwargs']
    else:
        readerkwargs = None

    # get the default timecols, magcols, errcols
    if timecols is None:
        timecols = formatspec['timecols']
    if magcols is None:
        magcols = formatspec['magcols']
    if errcols is None:
        errcols = formatspec['errcols']

    # get the normalization module and function if provided
    if formatspec['lcnorm_module'] is not None:
        normmod = importlib.import_module(formatspec['lcnorm_module'])
        normfunc = getattr(normmod, formatspec['lcnorm_func'])
    else:
        normfunc = None

    # get any normalization function kwargs we've set
    if formatspec['lcnorm_kwargs'] is not None:
        normkwargs = formatspec['lcnorm_kwargs']
    else:
        normkwargs = None

    # finally, get the magsarefluxes key and fileglob key
    magsarefluxes = formatspec['magsarefluxes']
    fileglob = formatspec['fileglob']


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
        allobjects[magcol]['objectids_all_thresh_all_magbins'] = np.unique(
            np.concatenate(allobjects[magcol]['binned_objectids_thresh_all'])
        )
        allobjects[magcol]['objectids_stetsonj_thresh_all_magbins'] = np.unique(
            np.concatenate(allobjects[magcol]['binned_objectids_thresh_stetsonj'])
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
                                magcols=None):
    '''
    This makes plots for the variability threshold distributions.

    '''
    import matplotlib.pyplot as plt

    #
    # handle the lcformat
    #
    formatspec = get_lcformat_spec(lcformat)

    # first, import the reader module
    readermod = importlib.import_module(formatspec['lcreader_module'])

    # then, get the function we need to read the lightcurve
    readerfunc = getattr(readermod, formatspec['lcreader_func'])

    # get any default kwargs we've set for this LC format
    if formatspec['lcreader_kwargs'] is not None:
        readerkwargs = formatspec['lcreader_kwargs']
    else:
        readerkwargs = None

    # get the default timecols, magcols, errcols
    if timecols is None:
        timecols = formatspec['timecols']
    if magcols is None:
        magcols = formatspec['magcols']
    if errcols is None:
        errcols = formatspec['errcols']

    # get the normalization module and function if provided
    if formatspec['lcnorm_module'] is not None:
        normmod = importlib.import_module(formatspec['lcnorm_module'])
        normfunc = getattr(normmod, formatspec['lcnorm_func'])
    else:
        normfunc = None

    # get any normalization function kwargs we've set
    if formatspec['lcnorm_kwargs'] is not None:
        normkwargs = formatspec['lcnorm_kwargs']
    else:
        normkwargs = None

    # finally, get the magsarefluxes key and fileglob key
    magsarefluxes = formatspec['magsarefluxes']
    fileglob = formatspec['fileglob']



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



#############################
## RUNNING PERIOD SEARCHES ##
#############################

def runpf(lcfile,
          outdir,
          timecols=None,
          magcols=None,
          errcols=None,
          lcformat='hat-sql',
          pfmethods=['gls','pdm','mav','win'],
          pfkwargs=[{},{},{},{}],
          sigclip=10.0,
          getblssnr=False,
          nworkers=None,
          excludeprocessed=False):
    '''This runs the period-finding for a single LC.

    pfmethods is a list of period finding methods to run. Each element is a
    string matching the keys of the PFMETHODS dict above. By default, this runs
    GLS, PDM, AoVMH, and the spectral window Lomb-Scargle periodogram.

    pfkwargs are any special kwargs to pass along to each period-finding method
    function.

    If excludeprocessing is True, light curves that have existing periodfinding
    result pickles in outdir will not be processed.

    FIXME: currently, this uses a dumb method of excluding already-processed
    files. A smarter way to do this is to (i) generate a SHA512 cachekey based
    on a repr of {'lcfile', 'timecols', 'magcols', 'errcols', 'lcformat',
    'pfmethods', 'sigclip', 'getblssnr', 'pfkwargs'}, (ii) make sure all list
    kwargs in the dict are sorted, (iii) check if the output file has the same
    cachekey in its filename (last 8 chars of cachekey should work), so the
    result was processed in exactly the same way as specifed in the input to
    this function, and can therefore be ignored. Will implement this later.

    '''

    from astrobase import periodbase
    from astrobase.periodbase.kbls import bls_snr

    # used to figure out which period finder to run given a list of methods
    PFMETHODS = {'bls':periodbase.bls_parallel_pfind,
                 'gls':periodbase.pgen_lsp,
                 'aov':periodbase.aov_periodfind,
                 'mav':periodbase.aovhm_periodfind,
                 'pdm':periodbase.stellingwerf_pdm,
                 'acf':periodbase.macf_period_find,
                 'win':periodbase.specwindow_lsp}

    #
    # handle the lcformat
    #
    formatspec = get_lcformat_spec(lcformat)

    # first, import the reader module
    readermod = importlib.import_module(formatspec['lcreader_module'])

    # then, get the function we need to read the lightcurve
    readerfunc = getattr(readermod, formatspec['lcreader_func'])

    # get any default kwargs we've set for this LC format
    if formatspec['lcreader_kwargs'] is not None:
        readerkwargs = formatspec['lcreader_kwargs']
    else:
        readerkwargs = None

    # get the default timecols, magcols, errcols
    if timecols is None:
        timecols = formatspec['timecols']
    if magcols is None:
        magcols = formatspec['magcols']
    if errcols is None:
        errcols = formatspec['errcols']

    # get the normalization module and function if provided
    if formatspec['lcnorm_module'] is not None:
        normmod = importlib.import_module(formatspec['lcnorm_module'])
        normfunc = getattr(normmod, formatspec['lcnorm_func'])
    else:
        normfunc = None

    # get any normalization function kwargs we've set
    if formatspec['lcnorm_kwargs'] is not None:
        normkwargs = formatspec['lcnorm_kwargs']
    else:
        normkwargs = None

    # finally, get the magsarefluxes key and fileglob key
    magsarefluxes = formatspec['magsarefluxes']
    fileglob = formatspec['fileglob']

    try:

        # get the LC into a dict
        if readerkwargs is not None:
            lcdict = readerfunc(lcfile, **readerkwargs)
        else:
            lcdict = readerfunc(lcfile)

        if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
            lcdict = lcdict[0]

        outfile = os.path.join(outdir, 'periodfinding-%s.pkl' %
                               lcdict['objectid'])

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
        resultdict = {'objectid':lcdict['objectid'],
                      'lcfbasename':os.path.basename(lcfile)}

        # normalize using the special function if specified
        if normfunc is not None:
            if normkwargs is not None:
                lcdict = normfunc(lcdict, **normkwargs)
            else:
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

            # run each of the requested period-finder functions
            resultdict[mcolget[-1]] = {}
            for pfm, pfkw in zip(pfmethods, pfkwargs):

                pf_func = PFMETHODS[pfm]

                # get any optional kwargs for this function
                pf_kwargs = pfkw
                pf_kwargs.update({'verbose':False,
                                  'nworkers':nworkers,
                                  'magsarefluxes':magsarefluxes,
                                  'sigclip':sigclip})

                # run this period-finder and save its results to the output dict
                resultdict[mcolget[-1]][pfm] = pf_func(
                    times, mags, errs,
                    **pf_kwargs
                )


            #
            # done with running the period finders
            #

            # check if we need to get the SNR for BLS
            if 'bls' in pfmethods and getblssnr:

                try:

                    bls = resultdict[mcolget[-1]]['bls']

                    # calculate the SNR for the BLS as well
                    blssnr = bls_snr(bls, times, mags, errs,
                                     magsarefluxes=magsarefluxes,
                                     verbose=False)

                    # add the SNR results to the BLS result dict
                    resultdict[mcolget[-1]]['bls'].update({
                        'snr':blssnr['snr'],
                        'altsnr':blssnr['altsnr'],
                        'transitdepth':blssnr['transitdepth'],
                        'transitduration':blssnr['transitduration'],
                    })

                except Exception as e:

                    LOGEXCEPTION('could not calculate BLS SNR for %s' %
                                 lcfile)
                    # add the SNR null results to the BLS result dict
                    resultdict[mcolget[-1]]['bls'].update({
                        'snr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                        'altsnr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                        'transitdepth':[np.nan,np.nan,np.nan,np.nan,np.nan],
                        'transitduration':[np.nan,np.nan,np.nan,np.nan,np.nan],
                    })

            elif 'bls' in pfmethods:

                # add the SNR null results to the BLS result dict
                resultdict[mcolget[-1]]['bls'].update({
                    'snr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                    'altsnr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                    'transitdepth':[np.nan,np.nan,np.nan,np.nan,np.nan],
                    'transitduration':[np.nan,np.nan,np.nan,np.nan,np.nan],
                })


        # once all mag cols have been processed, write out the pickle
        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to run for %s, because: %s' % (lcfile, e))
        return None



###################################
## CHECKPLOT NEIGHBOR OPERATIONS ##
###################################

# for the neighbors tab in checkplotserver: show a 5 row per neighbor x 3 col
# panel. Each col will have in order: best phased LC of target, phased LC of
# neighbor with same period and epoch, unphased LC of neighbor

def update_checkplotdict_nbrlcs(
        checkplotdict,
        timecol, magcol, errcol,
        lcformat='hat-sql',
        verbose=True,
):
    '''For all neighbors in checkplotdict, make LCs and phased LCs.

    Here, we specify the timecol, magcol, errcol explicitly because we're doing
    this per checkplot, which is for a single timecol-magcol-errcol combination.

    '''
    from astrobase.checkplot import _pkl_magseries_plot, \
        _pkl_phased_magseries_plot

    #
    # handle the lcformat
    #
    formatspec = get_lcformat_spec(lcformat)

    # first, import the reader module
    readermod = importlib.import_module(formatspec['lcreader_module'])

    # then, get the function we need to read the lightcurve
    readerfunc = getattr(readermod, formatspec['lcreader_func'])

    # get any default kwargs we've set for this LC format
    if formatspec['lcreader_kwargs'] is not None:
        readerkwargs = formatspec['lcreader_kwargs']
    else:
        readerkwargs = None

    # get the normalization module and function if provided
    if formatspec['lcnorm_module'] is not None:
        normmod = importlib.import_module(formatspec['lcnorm_module'])
        normfunc = getattr(normmod, formatspec['lcnorm_func'])
    else:
        normfunc = None

    # get any normalization function kwargs we've set
    if formatspec['lcnorm_kwargs'] is not None:
        normkwargs = formatspec['lcnorm_kwargs']
    else:
        normkwargs = None

    # finally, get the magsarefluxes key and fileglob key
    magsarefluxes = formatspec['magsarefluxes']
    fileglob = formatspec['fileglob']


    if not ('neighbors' in checkplotdict and
            checkplotdict['neighbors'] and
            len(checkplotdict['neighbors']) > 0):

        LOGWARNING('no neighbors for %s, not updating...' %
                   (checkplotdict['objectid']))
        return checkplotdict


    # if there are actually neighbors, go through them in order
    for nbr in checkplotdict['neighbors']:

        objectid, ra, decl, dist, lcfpath = (nbr['objectid'],
                                             nbr['ra'],
                                             nbr['decl'],
                                             nbr['dist'],
                                             nbr['lcfpath'])

        # get the light curve
        if readerkwargs is not None:
            lcdict = readerfunc(lcfpath, **readerkwargs)
        else:
            lcdict = readerfunc(lcfpath)
        if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
            lcdict = lcdict[0]

        # normalize using the special function if specified
        if normfunc is not None:
            if normkwargs is not None:
                lcdict = normfunc(lcdict,**normkwargs)
            else:
                lcdict = normfunc(lcdict)

        # get the times, mags, and errs
        # dereference the columns and get them from the lcdict
        if '.' in timecol:
            timecolget = timecol.split('.')
        else:
            timecolget = [timecol]
        times = dict_get(lcdict, timecolget)

        if '.' in magcol:
            magcolget = magcol.split('.')
        else:
            magcolget = [magcol]
        mags = dict_get(lcdict, magcolget)

        if '.' in errcol:
            errcolget = errcol.split('.')
        else:
            errcolget = [errcol]
        errs = dict_get(lcdict, errcolget)


        # filter the input times, mags, errs; do sigclipping and normalization
        stimes, smags, serrs = sigclip_magseries(times,
                                                 mags,
                                                 errs,
                                                 magsarefluxes=magsarefluxes,
                                                 sigclip=4.0)

        # normalize here if not using special normalization
        if normfunc is None:
            ntimes, nmags = normalize_magseries(
                stimes, smags,
                magsarefluxes=magsarefluxes
            )
            xtimes, xmags, xerrs = ntimes, nmags, serrs
        else:
            xtimes, xmags, xerrs = stimes, smags, serrs

        #
        # now we can start doing stuff
        #

        # 1. make an unphased mag-series plot
        nbrdict = _pkl_magseries_plot(xtimes,
                                      xmags,
                                      xerrs,
                                      magsarefluxes=magsarefluxes)
        # update the nbr
        nbr.update(nbrdict)

        # for each lspmethod in the checkplot, make a corresponding plot for
        # this neighbor
        for lspt in PFMETHODS:

            if lspt in checkplotdict:

                # initialize this lspmethod entry
                nbr[lspt] = {}

                # we only care about the best period and its options
                operiod, oepoch = (checkplotdict[lspt][0]['period'],
                                   checkplotdict[lspt][0]['epoch'])
                (ophasewrap, ophasesort, ophasebin,
                 ominbinelems, oplotxlim) = (
                     checkplotdict[lspt][0]['phasewrap'],
                     checkplotdict[lspt][0]['phasesort'],
                     checkplotdict[lspt][0]['phasebin'],
                     checkplotdict[lspt][0]['minbinelems'],
                     checkplotdict[lspt][0]['plotxlim'],
                 )

                # make the phasedlc plot for this period
                nbr = _pkl_phased_magseries_plot(
                    nbr,
                    lspt,
                    0,
                    xtimes, xmags, xerrs,
                    operiod, oepoch,
                    phasewrap=ophasewrap,
                    phasesort=ophasesort,
                    phasebin=ophasebin,
                    minbinelems=ominbinelems,
                    plotxlim=oplotxlim,
                    magsarefluxes=magsarefluxes,
                    verbose=verbose
                )

    # at this point, this neighbor's dict should be up to date with all
    # info, magseries plot, and all phased LC plots
    # return the updated checkplotdict
    return checkplotdict



########################
## RUNNING CHECKPLOTS ##
########################

def runcp(pfpickle,
          outdir,
          lcbasedir,
          lclistpkl=None,
          nbrradiusarcsec=30.0,
          lcformat='hat-sql',
          timecols=None,
          magcols=None,
          errcols=None,
          mindet=1000):
    '''This runs a checkplot for the given period-finding result pickle
    produced by runpf.

    '''
    from astrobase import checkplot

    PFMETHODS = ['bls',
                 'gls',
                 'aov',
                 'mav',
                 'pdm',
                 'acf',
                 'win']

    #
    # handle the lcformat
    #
    formatspec = get_lcformat_spec(lcformat)

    # first, import the reader module
    readermod = importlib.import_module(formatspec['lcreader_module'])

    # then, get the function we need to read the lightcurve
    readerfunc = getattr(readermod, formatspec['lcreader_func'])

    # get any default kwargs we've set for this LC format
    if formatspec['lcreader_kwargs'] is not None:
        readerkwargs = formatspec['lcreader_kwargs']
    else:
        readerkwargs = None

    # get the default timecols, magcols, errcols
    if timecols is None:
        timecols = formatspec['timecols']
    if magcols is None:
        magcols = formatspec['magcols']
    if errcols is None:
        errcols = formatspec['errcols']

    # get the normalization module and function if provided
    if formatspec['lcnorm_module'] is not None:
        normmod = importlib.import_module(formatspec['lcnorm_module'])
        normfunc = getattr(normmod, formatspec['lcnorm_func'])
    else:
        normfunc = None

    # get any normalization function kwargs we've set
    if formatspec['lcnorm_kwargs'] is not None:
        normkwargs = formatspec['lcnorm_kwargs']
    else:
        normkwargs = None

    # finally, get the magsarefluxes key and fileglob key
    magsarefluxes = formatspec['magsarefluxes']
    fileglob = formatspec['fileglob']


    if pfpickle.endswith('.gz'):
        infd = gzip.open(pfpickle,'rb')
    else:
        infd = open(pfpickle,'rb')
    pfresults = pickle.load(infd)
    infd.close()


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

    if readerkwargs is not None:
        lcdict = readerfunc(lcfpath,**readerkwargs)
    else:
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


        pflist = []

        # pick up all of the period-finding methods in this pfresults pkl
        for pfmethod in PFMETHODS:
            if pfmethod in pfresults[mcolget[-1]]:
                pflist.append(pfresults[mcolget[-1]][pfmethod])


        # generate the output filename
        outfile = os.path.join(outdir,
                               'checkplot-%s-%s.pkl' % (objectid, mcol))

        # make sure the checkplot has a valid objectid
        if 'objectid' not in lcdict['objectinfo']:
            lcdict['objectinfo']['objectid'] = objectid

        # generate the checkplotdict
        cpd = checkplot.checkplot_dict(
            pflist,
            times, mags, errs,
            objectinfo=lcdict['objectinfo'],
            lclistpkl=lclistpkl,
            nbrradiusarcsec=nbrradiusarcsec,
            verbose=False,
            mindet=mindet
        )

        # include any neighbor information as well
        cpdupdated = update_checkplotdict_nbrlcs(
            cpd,
            tcol, mcol, ecol,
            lcformat=lcformat,
            verbose=False
        )

        # write the update checkplot dict to disk
        cpf = checkplot._write_checkplot_picklefile(
            cpdupdated,
            outfile=outfile,
            protocol=pickle.HIGHEST_PROTOCOL,
            outgzip=False
        )

        cpfs.append(cpf)

    LOGINFO('done with %s -> %s' % (objectid, repr(cpfs)))
    return cpfs



def cp2png(checkplotpickle):
    '''
    This runs cp2png from checkplot.py.

    '''

    from astrobase.checkplot import checkplot_pickle_to_png

    if checkplotpickle.endswith('.gz'):
        outfile = checkplotpickle.replace('.pkl.gz','.png')
    else:
        outfile = checkplotpickle.replace('.pkl','.png')

    return checkplot_pickle_to_png(checkplotpickle, outfile)



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
    #
    # handle the lcformat
    #
    formatspec = get_lcformat_spec(lcformat)

    # first, import the reader module
    readermod = importlib.import_module(formatspec['lcreader_module'])

    # then, get the function we need to read the lightcurve
    readerfunc = getattr(readermod, formatspec['lcreader_func'])

    # get any default kwargs we've set for this LC format
    if formatspec['lcreader_kwargs'] is not None:
        readerkwargs = formatspec['lcreader_kwargs']
    else:
        readerkwargs = None

    # get the default timecols, magcols, errcols
    if timecols is None:
        timecols = formatspec['timecols']
    if magcols is None:
        magcols = formatspec['magcols']
    if errcols is None:
        errcols = formatspec['errcols']

    # get the normalization module and function if provided
    if formatspec['lcnorm_module'] is not None:
        normmod = importlib.import_module(formatspec['lcnorm_module'])
        normfunc = getattr(normmod, formatspec['lcnorm_func'])
    else:
        normfunc = None

    # get any normalization function kwargs we've set
    if formatspec['lcnorm_kwargs'] is not None:
        normkwargs = formatspec['lcnorm_kwargs']
    else:
        normkwargs = None

    # finally, get the magsarefluxes key and fileglob key
    magsarefluxes = formatspec['magsarefluxes']
    fileglob = formatspec['fileglob']


    # get the LC into a dict
    if readerkwargs is not None:
        lcdict = readerfunc(lcfile, **readerkwargs)
    else:
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

        # we use mcolget[-1] here so we can deal with dereferenced magcols like
        # sap.sap_flux or pdc.pdc_sapflux
        if 'binned' not in lcdict:
            lcdict['binned'] = {mcolget[-1]: {'times':binned['binnedtimes'],
                                              'mags':binned['binnedmags'],
                                              'errs':binned['binnederrs'],
                                              'nbins':binned['nbins'],
                                              'timebins':binned['jdbins'],
                                              'binsizesec':binsizesec}}

        else:
            lcdict['binned'][mcolget[-1]] = {'times':binned['binnedtimes'],
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
                           (lcdict['objectid'], binsizesec, lcformat))

    with open(outfile, 'wb') as outfd:
        pickle.dump(lcdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    return outfile



#####################################
## SUPPORT FOR EXECUTION AS SCRIPT ##
#####################################

PROGEPILOG = '''\
Specifying LC formats
---------------------

A custom LC format can be permanently registered using:

lcpbatch register-lcformat [specification options]

where specification options are strings surrounded by single quotes to prevent
the shell from expanding them. An example for HAT sqlitecurves:

--formatkey 'hat-sql'
--fileglob '*-hatlc.sqlite*'
--timecols '["rjd", "rjd"]'
--magcols '["aep_000", "atf_000"]'
--errcols '["aie_000", "aie_000"]'
--readermodule 'astrobase.hatlc'
--readerfunc 'read_and_filter_sqlitecurves'
--readerkwargs '{"verbose": false, "raiseonfail": false}'
--normmodule 'astrobase.hatlc'
--normfunc 'normalize_lcdict_byinst'
--normkwargs '{"normto": "sdssr", "magcols": "all"}'

This will generate a lcformat JSON with filename <formatkey>.json that will
allow light curve files matching the fileglob to be automatically recognized and
use the specified functions to read and normalize them. Any options left out
will be set to None. The generated JSON looks like the following (example for
HAT sqlitecurves):

{
  "fileglob": "*-hatlc.sqlite*",
  "timecols": ["rjd", "rjd"],
  "magcols": ["aep_000", "atf_000"],
  "errcols": ["aie_000", "aie_000"],
  "magsarefluxes": false,
  "lcreader_module": "astrobase.hatlc",
  "lcreader_func": "read_and_filter_sqlitecurves",
  "lcreader_kwargs": {"returnarrays": true, "raiseonfail": false},
  "lcnorm_module": "astrobase.hatlc",
  "lcnorm_func": "normalize_lcdict_byinst",
  "lcnorm_kwargs": {"normto": "sdssr", "magcols": "all"}
}

LC format specification requires modules defined in lcreader_module and
lcnorm_module containing the reader and normalization functions
respectively. These can be the same module; it won't be imported
twice. Astrobase built-in modules look like "astrobase.<module names>".
Your own modules must be specified as either a Python module path
(e.g. "yourpackage.yourmodule"), or a file path on disk
(e.g. "/path/to/yourpackage/yourmodule.py").

The lcreader_func function must have the following signature:

def lcreader_func(lc_filepath_string, **lcreader_kwargs)

and return a dict that contains the timecols, magcols, and errcols as key:val
elems: the lcdict.

The lcnorm_func function must have the following signature:

def lcnorm_func(lcdict, **lcnorm_kwargs)

and return the same lcdict with the columns normalized however as specified by
lcnorm_kwargs, etc.


To override automatic LC format recognition based on the fileglob and force the
use of a specific known lcformat, use:

--lcformat <formatkey>


To use a unregistered LC format for which you already have a lcformat JSON
elsewhere, use:

--lcformat-json /path/to/lcformat-name.json
'''


##########
## MAIN ##
##########

def main():
    '''
    This is the main function.

    '''

    ################
    ## PARSE ARGS ##
    ################

    aparser = argparse.ArgumentParser(
        epilog=PROGEPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )



if __name__ == '__main__':
    main()
