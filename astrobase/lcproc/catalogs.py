#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''catalogs.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

This contains functions to generate light curve catalogs from collections of
light curves.

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
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.random as npr
npr.seed(0xc0ffee)

import scipy.spatial as sps

import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, LinearStretch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    TQDM = True
except Exception as e:
    TQDM = False
    pass

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

# these translate filter operators given as strings to Python operators
FILTEROPS = {'eq':'==',
             'gt':'>',
             'ge':'>=',
             'lt':'<',
             'le':'<=',
             'ne':'!='}



###################
## LOCAL IMPORTS ##
###################

from astrobase.plotbase import fits_finder_chart
from astrobase.cpserver.checkplotlist import checkplot_infokey_worker
from astrobase.lcproc import get_lcformat



#####################################################
## FUNCTIONS TO GENERATE OBJECT CATALOGS (LCLISTS) ##
#####################################################

def lclist_parallel_worker(task):
    '''This is a parallel worker for makelclist.

    Parameters
    ----------

    task : tuple
        This is a tuple containing the following items:

        task[0] = lcf
        task[1] = columns
        task[2] = lcformat
        task[3] = lcformatdir
        task[4] = lcndetkey

    Returns
    -------

    dict or None
        This contains all of the info for the object processed in this LC read
        operation. If this fails, returns None

    '''

    lcf, columns, lcformat, lcformatdir, lcndetkey = task

    # get the bits needed for lcformat handling
    # NOTE: we re-import things in this worker function because sometimes
    # functions can't be pickled correctly for passing them to worker functions
    # in a processing pool
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

    # we store the full path of the light curve
    lcobjdict = {'lcfname':os.path.abspath(lcf)}

    try:

        # read the light curve in
        lcdict = readerfunc(lcf)

        # this should handle lists/tuples being returned by readerfunc
        # we assume that the first element is the actual lcdict
        # FIXME: figure out how to not need this assumption
        if ( (isinstance(lcdict, (list, tuple))) and
             (isinstance(lcdict[0], dict)) ):
            lcdict = lcdict[0]

        # insert all of the columns
        for colkey in columns:

            if '.' in colkey:
                getkey = colkey.split('.')
            else:
                getkey = [colkey]

            try:
                thiscolval = dict_get(lcdict, getkey)
            except Exception as e:
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
            lcobjdict['%s.ndet' % getdk[-1]] = actualndets

        except Exception as e:
            lcobjdict['%s.ndet' % getdk[-1]] = np.nan


    return lcobjdict



def make_lclist(basedir,
                outfile,
                use_list_of_filenames=None,
                lcformat='hat-sql',
                lcformatdir=None,
                fileglob=None,
                recursive=True,
                columns=['objectid',
                         'objectinfo.ra',
                         'objectinfo.decl',
                         'objectinfo.ndet'],
                makecoordindex=('objectinfo.ra','objectinfo.decl'),
                field_fitsfile=None,
                field_wcsfrom=None,
                field_scale=ZScaleInterval(),
                field_stretch=LinearStretch(),
                field_colormap=plt.cm.gray_r,
                field_findersize=None,
                field_pltopts={'marker':'o',
                               'markersize':10.0,
                               'markerfacecolor':'none',
                               'markeredgewidth':2.0,
                               'markeredgecolor':'red'},
                field_grid=False,
                field_gridcolor='k',
                field_zoomcontain=True,
                maxlcs=None,
                nworkers=NCPUS):

    '''This generates a list file compatible with filter_lclist below.

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
    used to make a kdtree for fast look-up by position later by filter_lclist.

    field_fitsfile if not None, is the path to a FITS image containing the
    objects these light curves are for. If this is provided, make_lclist will
    use the WCS information in the FITS itself if field_wcsfrom is None (or from
    a WCS header file pointed to by field_wcsfrom) to obtain x and y pixel
    coordinates for all of the objects in the field. A finder chart will also be
    made using astrobase.plotbase.fits_finder_chart using the corresponding
    field_scale, _stretch, _colormap, _findersize, _pltopts, _grid, and
    _gridcolors keyword arguments for that function.

    maxlcs sets how many light curves to process in the input LC list generated
    by searching for LCs in `basedir`.

    nworkers sets the number of parallel workers to launch to collect
    information from the light curves.

    This returns a pickle file.

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

    # this is to get the actual ndet
    # set to the magnitudes column
    lcndetkey = dmagcols

    if isinstance(use_list_of_filenames, list):

        matching = use_list_of_filenames

    else:

        # handle the case where basedir is a list of directories
        if isinstance(basedir, list):

            matching = []

            for bdir in basedir:

                # now find the files
                LOGINFO('searching for %s light curves in %s ...' % (lcformat,
                                                                     bdir))

                if recursive is False:
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

                        for root, dirs, _files in walker:
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
            LOGINFO('searching for %s light curves in %s ...' %
                    (lcformat, basedir))

            if recursive is False:
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

                    for root, dirs, _files in walker:
                        for sdir in dirs:
                            searchpath = os.path.join(root,
                                                      sdir,
                                                      fileglob)
                            foundfiles = glob.glob(searchpath)

                            if foundfiles:
                                matching.extend(foundfiles)

    #
    # now that we have all the files, process them
    #
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
        derefcols.extend(['%s.ndet' % x.split('.')[-1] for x in lcndetkey])

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

        tasks = [(x, columns, lcformat, lcformatdir, lcndetkey)
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

        # handle duplicate objectids with different light curves

        uniques, counts = np.unique(lclistdict['objects']['objectid'],
                                    return_counts=True)

        duplicated_objectids = uniques[counts > 1]

        if duplicated_objectids.size > 0:

            # redo the objectid array so it has a bit larger dtype so the extra
            # tag can fit into the field
            dt = lclistdict['objects']['objectid'].dtype.str
            dt = '<U%s' % (
                int(dt.replace('<','').replace('U','').replace('S','')) + 3
            )
            lclistdict['objects']['objectid'] = np.array(
                lclistdict['objects']['objectid'],
                dtype=dt
            )

            for objid in duplicated_objectids:

                objid_inds = np.where(
                    lclistdict['objects']['objectid'] == objid
                )

                # mark the duplicates, assume the first instance is the actual
                # one
                for ncounter, nind in enumerate(objid_inds[0][1:]):
                    lclistdict['objects']['objectid'][nind] = '%s-%s' % (
                        lclistdict['objects']['objectid'][nind],
                        ncounter+2
                    )
                    LOGWARNING(
                        'tagging duplicated instance %s of objectid: '
                        '%s as %s-%s, lightcurve: %s' %
                        (ncounter+2, objid, objid, ncounter+2,
                         lclistdict['objects']['lcfname'][nind])
                    )

        # if we're supposed to make a spatial index, do so
        if (makecoordindex and
            isinstance(makecoordindex, (list, tuple)) and
            len(makecoordindex) == 2):

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

                LOGINFO('kdtree generated for (ra, decl): (%s, %s)' %
                        (makecoordindex[0], makecoordindex[1]))

            except Exception as e:
                LOGEXCEPTION('could not make kdtree for (ra, decl): (%s, %s)' %
                             (makecoordindex[0], makecoordindex[1]))
                raise

        # generate the xy pairs if fieldfits is not None
        if field_fitsfile and os.path.exists(field_fitsfile):

            # read in the FITS file
            if field_wcsfrom is None:

                hdulist = pyfits.open(field_fitsfile)
                hdr = hdulist[0].header
                hdulist.close()

                w = WCS(hdr)
                wcsok = True

            elif os.path.exists(field_wcsfrom):

                w = WCS(field_wcsfrom)
                wcsok = True

            else:

                LOGERROR('could not determine WCS info for input FITS: %s' %
                         field_fitsfile)
                wcsok = False

            if wcsok:

                # first, transform the ra/decl to x/y and put these in the
                # lclist output dict
                radecl = np.column_stack((objra, objdecl))
                lclistdict['objects']['framexy'] = w.all_world2pix(
                    radecl,
                    1
                )

                # next, we'll make a PNG plot for the finder
                finder_outfile = os.path.join(
                    os.path.dirname(outfile),
                    os.path.splitext(os.path.basename(outfile))[0] + '.png'
                )

                finder_png = fits_finder_chart(
                    field_fitsfile,
                    finder_outfile,
                    wcsfrom=field_wcsfrom,
                    scale=field_scale,
                    stretch=field_stretch,
                    colormap=field_colormap,
                    findersize=field_findersize,
                    overlay_ra=objra,
                    overlay_decl=objdecl,
                    overlay_pltopts=field_pltopts,
                    overlay_zoomcontain=field_zoomcontain,
                    grid=field_grid,
                    gridcolor=field_gridcolor
                )

                if finder_png is not None:
                    LOGINFO('generated a finder PNG '
                            'with an object position overlay '
                            'for this LC list: %s' % finder_png)


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
                  racol='ra',
                  declcol='decl',
                  xmatchexternal=None,
                  xmatchdistarcsec=3.0,
                  externalcolnums=(0,1,2),
                  externalcolnames=['objectid','ra','decl'],
                  externalcoldtypes='U20,f8,f8',
                  externalcolsep=None,
                  externalcommentchar='#',
                  conesearch=None,
                  columnfilters=None,
                  conesearchworkers=1,
                  field_fitsfile=None,
                  field_wcsfrom=None,
                  field_scale=ZScaleInterval(),
                  field_stretch=LinearStretch(),
                  field_colormap=plt.cm.gray_r,
                  field_findersize=None,
                  field_pltopts={'marker':'o',
                                 'markersize':10.0,
                                 'markerfacecolor':'none',
                                 'markeredgewidth':2.0,
                                 'markeredgecolor':'red'},
                  field_grid=False,
                  field_gridcolor='k',
                  field_zoomcontain=True,
                  copylcsto=None):

    '''This is used to collect light curves based on selection criteria.

    Uses the output of make_lclist above. This function returns a list of light
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
    that are in the specified region. conesearchworkers specifies the number of
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


    field_fitsfile if not None, is the path to a FITS image containing the
    objects these light curves are for. If this is provided, filter_lclist will
    use the WCS information in the FITS itself if field_wcsfrom is None (or from
    a WCS header file pointed to by field_wcsfrom) to obtain x and y pixel
    coordinates for all of the objects in the field. A finder chart will also be
    made for the objects matching all the filters. This will use
    astrobase.plotbase.fits_finder_chart using the corresponding field_scale,
    _stretch, _colormap, _findersize, _pltopts, _grid, and _gridcolors keyword
    arguments for that function.

    If copylcsto is not None, it is interpreted as a directory target to copy
    all the light curves that match the specified conditions.

    '''

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
                                   dtype=externalcoldtypes,
                                   comments=externalcommentchar)

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

                    # get the whole matching row for the ext objects recarray
                    ext_matching_objects.append(extcat[extind])

            ext_matches = np.array(ext_matches)

            if ext_matches.size > 0:

                # update the xmatch_matching_index
                xmatch_matching_index[ext_matches] = True

                LOGINFO('xmatch: objects matched to %s within %.1f arcsec: %s' %
                        (xmatchexternal, xmatchdistarcsec, ext_matches.size))

            else:

                LOGERROR("xmatch: no objects were cross-matched to external "
                         "catalog spec: %s, can't continue" % xmatchexternal)
                return None, None, None


        except Exception as e:

            LOGEXCEPTION('could not match to external catalog spec: %s' %
                         repr(xmatchexternal))
            raise


    # do the cone search next
    if (conesearch and
        isinstance(conesearch, (list, tuple)) and
        len(conesearch) == 3):

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


    # if we're told to make a finder chart with the selected objects
    if field_fitsfile is not None and os.path.exists(field_fitsfile):

        # get the RA and DEC of the matching objects
        matching_ra = lclist['objects'][racol][finalfilterind]
        matching_decl = lclist['objects'][declcol][finalfilterind]

        matching_postfix = []

        if xmatchexternal is not None:
            matching_postfix.append(
                'xmatch_%s' %
                os.path.splitext(os.path.basename(xmatchexternal))[0]
            )
        if conesearch is not None:
            matching_postfix.append('conesearch_RA%.3f_DEC%.3f_RAD%.5f' %
                                    tuple(conesearch))

        if columnfilters is not None:
            for cfi, cf in enumerate(columnfilters):
                if cfi == 0:
                    matching_postfix.append('filter_%s_%s_%s' %
                                            tuple(cf.split('|')))
                else:
                    matching_postfix.append('_and_%s_%s_%s' %
                                            tuple(cf.split('|')))

        if len(matching_postfix) > 0:
            matching_postfix = '-%s' % '_'.join(matching_postfix)
        else:
            matching_postfix = ''

        # next, we'll make a PNG plot for the finder
        finder_outfile = os.path.join(
            os.path.dirname(listpickle),
            '%s%s.png' %
            (os.path.splitext(os.path.basename(listpickle))[0],
             matching_postfix)
        )

        finder_png = fits_finder_chart(
            field_fitsfile,
            finder_outfile,
            wcsfrom=field_wcsfrom,
            scale=field_scale,
            stretch=field_stretch,
            colormap=field_colormap,
            findersize=field_findersize,
            overlay_ra=matching_ra,
            overlay_decl=matching_decl,
            overlay_pltopts=field_pltopts,
            field_zoomcontain=field_zoomcontain,
            grid=field_grid,
            gridcolor=field_gridcolor
        )

        if finder_png is not None:
            LOGINFO('generated a finder PNG '
                    'with an object position overlay '
                    'for this filtered LC list: %s' % finder_png)



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
        return filteredlcfnames, filteredobjectids, ext_matching_objects
    else:
        return filteredlcfnames, filteredobjectids



############################################################
## ADDING CHECKPLOT INFO BACK TO THE LIGHT CURVE CATALOGS ##
############################################################

def cpinfo_key_worker(task):
    '''This wraps checkplotlist.checkplot_infokey_worker.

    This is used to get the correct dtype for each element in retrieved results.

    task[0] = cpfile
    task[1] = keyspeclist (infokeys kwarg from add_cpinfo_to_lclist)

    '''

    cpfile, keyspeclist = task

    keystoget = [x[0] for x in keyspeclist]
    nonesubs = [x[-2] for x in keyspeclist]
    nansubs = [x[-1] for x in keyspeclist]

    # reform the keystoget into a list of lists
    for i, k in enumerate(keystoget):

        thisk = k.split('.')
        if sys.version_info[:2] < (3,4):
            thisk = [(int(x) if x.isdigit() else x) for x in thisk]
        else:
            thisk = [(int(x) if x.isdecimal() else x) for x in thisk]

        keystoget[i] = thisk

    # add in the objectid as well to match to the object catalog later
    keystoget.insert(0,['objectid'])
    nonesubs.insert(0, '')
    nansubs.insert(0,'')

    # get all the keys we need
    vals = checkplot_infokey_worker((cpfile, keystoget))

    # if they have some Nones, nans, etc., reform them as expected
    for val, nonesub, nansub, valind in zip(vals, nonesubs,
                                            nansubs, range(len(vals))):

        if val is None:
            outval = nonesub
        elif isinstance(val, float) and not np.isfinite(val):
            outval = nansub
        elif isinstance(val, (list, tuple)):
            outval = ', '.join(val)
        else:
            outval = val

        vals[valind] = outval

    return vals



CPINFO_DEFAULTKEYS = [
    # key, dtype, first level, overwrite=T|append=F, None sub, nan sub
    ('comments',
     np.unicode_, False, True, '', ''),
    ('objectinfo.objecttags',
     np.unicode_, True, True, '', ''),
    ('objectinfo.twomassid',
     np.unicode_, True, True, '', ''),
    ('objectinfo.bmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.vmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.rmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.imag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.jmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.hmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.kmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.sdssu',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.sdssg',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.sdssr',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.sdssi',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.sdssz',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_bmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_vmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_rmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_imag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_jmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_hmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_kmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_sdssu',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_sdssg',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_sdssr',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_sdssi',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.dered_sdssz',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_bmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_vmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_rmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_imag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_jmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_hmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_kmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_sdssu',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_sdssg',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_sdssr',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_sdssi',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.extinction_sdssz',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.color_classes',
     np.unicode_, True, True, '', ''),
    ('objectinfo.pmra',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.pmdecl',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.propermotion',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.rpmj',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.gl',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.gb',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.gaia_status',
     np.unicode_, True, True, '', ''),
    ('objectinfo.gaia_ids.0',
     np.unicode_, True, True, '', ''),
    ('objectinfo.gaiamag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.gaia_parallax',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.gaia_parallax_err',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.gaia_absmag',
     np.float_, True, True, np.nan, np.nan),
    ('objectinfo.simbad_best_mainid',
     np.unicode_, True, True, '', ''),
    ('objectinfo.simbad_best_objtype',
     np.unicode_, True, True, '', ''),
    ('objectinfo.simbad_best_allids',
     np.unicode_, True, True, '', ''),
    ('objectinfo.simbad_best_distarcsec',
     np.float_, True, True, np.nan, np.nan),
    #
    # TIC info
    #
    ('objectinfo.ticid',
     np.unicode_, True, True, '', ''),
    ('objectinfo.tic_version',
     np.unicode_, True, True, '', ''),
    ('objectinfo.tessmag',
     np.float_, True, True, np.nan, np.nan),
    #
    # variability info
    #
    ('varinfo.vartags',
     np.unicode_, False, True, '', ''),
    ('varinfo.varperiod',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.varepoch',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.varisperiodic',
     np.int_, False, True, 0, 0),
    ('varinfo.objectisvar',
     np.int_, False, True, 0, 0),
    ('varinfo.features.median',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.mad',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.stdev',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.mag_iqr',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.skew',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.kurtosis',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.stetsonj',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.stetsonk',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.eta_normal',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.linear_fit_slope',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.magnitude_ratio',
     np.float_, False, True, np.nan, np.nan),
    ('varinfo.features.beyond1std',
     np.float_, False, True, np.nan, np.nan)
]



def add_cpinfo_to_lclist(
        checkplots,  # list or a directory path
        lclistpkl,
        magcol,  # to indicate checkplot magcol
        outfile,
        checkplotglob='checkplot*.pkl*',
        infokeys=CPINFO_DEFAULTKEYS,
        nworkers=NCPUS
):
    '''This adds checkplot info to the light curve catalogs from make_lclist.

    lclistpkl is the pickle made by make_lclist.

    magcol is the LC magnitude column being used in the checkplots' feature
    keys. This will be added as a prefix to the infokeys.

    checkplots is either a list of checkplot pickles to process or a string
    indicating a checkplot directory path to process.

    outfile is the pickle filename to write the augmented lclist pickle to.

    infokeys is a list of keys to extract from each checkplot.

    '''

    # get the checkplots from the directory if one is provided
    if not isinstance(checkplots, list) and os.path.exists(checkplots):
        checkplots = sorted(glob.glob(os.path.join(checkplots, checkplotglob)))

    tasklist = [(cpf, infokeys) for cpf in checkplots]

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(cpinfo_key_worker, tasklist)

    results = [x for x in resultfutures]
    executor.shutdown()

    # now that we have all the checkplot info, we need to match to the
    # objectlist in the lclist

    # open the lclist
    with open(lclistpkl,'rb') as infd:
        objectcatalog = pickle.load(infd)

    catalog_objectids = np.array(objectcatalog['objects']['objectid'])
    checkplot_objectids = np.array([x[0] for x in results])

    # add the extra key arrays in the lclist dict
    extrainfokeys = []
    actualkeys = []

    # set up the extrainfokeys list
    for keyspec in infokeys:

        key, dtype, firstlevel, overwrite_append, nonesub, nansub = keyspec

        if firstlevel:
            eik = key
        else:
            eik = '%s.%s' % (magcol, key)

        extrainfokeys.append(eik)

        # now handle the output dicts and column list
        eactual = eik.split('.')

        # this handles dereferenced list indices
        if not eactual[-1].isdigit():

            if not firstlevel:
                eactual = '.'.join([eactual[0], eactual[-1]])
            else:
                eactual = eactual[-1]

        else:
            elastkey = eactual[-2]

            # for list columns, this converts stuff like errs -> err,
            # and parallaxes -> parallax
            if elastkey.endswith('es'):
                elastkey = elastkey[:-2]
            elif elastkey.endswith('s'):
                elastkey = elastkey[:-1]

            if not firstlevel:
                eactual = '.'.join([eactual[0], elastkey])
            else:
                eactual = elastkey

        actualkeys.append(eactual)

        # add a new column only if required
        if eactual not in objectcatalog['columns']:
            objectcatalog['columns'].append(eactual)

        # we'll overwrite earlier existing columns in any case
        objectcatalog['objects'][eactual] = []


    # now go through each objectid in the catalog and add the extra keys to
    # their respective arrays
    for catobj in tqdm(catalog_objectids):

        cp_objind = np.where(checkplot_objectids == catobj)

        if len(cp_objind[0]) > 0:

            # get the info line for this checkplot
            thiscpinfo = results[cp_objind[0][0]]

            # the first element is the objectid which we remove
            thiscpinfo = thiscpinfo[1:]

            # update the object catalog entries for this object
            for ekind, ek in enumerate(actualkeys):

                # add the actual thing to the output list
                objectcatalog['objects'][ek].append(
                    thiscpinfo[ekind]
                )

        else:

            # update the object catalog entries for this object
            for ekind, ek in enumerate(actualkeys):

                thiskeyspec = infokeys[ekind]
                nonesub = thiskeyspec[-2]

                objectcatalog['objects'][ek].append(
                    nonesub
                )

    # now we should have all the new keys in the object catalog
    # turn them into arrays
    for ek in actualkeys:

        objectcatalog['objects'][ek] = np.array(
            objectcatalog['objects'][ek]
        )

    # add the magcol to the objectcatalog
    if 'magcols' in objectcatalog:
        if magcol not in objectcatalog['magcols']:
            objectcatalog['magcols'].append(magcol)
    else:
        objectcatalog['magcols'] = [magcol]

    # write back the new object catalog
    with open(outfile, 'wb') as outfd:
        pickle.dump(objectcatalog, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    return outfile
