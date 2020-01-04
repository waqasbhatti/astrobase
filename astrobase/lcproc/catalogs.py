#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# catalogs.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019

'''

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

import pickle
import os
import os.path
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
except Exception:
    TQDM = False
    pass

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

def _lclist_parallel_worker(task):
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
    except Exception:
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
                thiscolval = _dict_get(lcdict, getkey)
            except Exception:
                LOGWARNING('column %s does not exist for %s' %
                           (colkey, lcf))
                thiscolval = np.nan

            # update the lcobjdict with this value
            lcobjdict[getkey[-1]] = thiscolval

    except Exception:

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

            ndetcol = _dict_get(lcdict, getdk)
            actualndets = ndetcol[np.isfinite(ndetcol)].size
            lcobjdict['%s.ndet' % getdk[-1]] = actualndets

        except Exception:
            lcobjdict['%s.ndet' % getdk[-1]] = np.nan

    return lcobjdict


def make_lclist(basedir,
                outfile,
                use_list_of_filenames=None,
                lcformat='hat-sql',
                lcformatdir=None,
                fileglob=None,
                recursive=True,
                columns=('objectid',
                         'objectinfo.ra',
                         'objectinfo.decl',
                         'objectinfo.ndet'),
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

    '''This generates a light curve catalog for all light curves in a directory.

    Given a base directory where all the files are, and a light curve format,
    this will find all light curves, pull out the keys in each lcdict requested
    in the `columns` kwarg for each object, and write them to the requested
    output pickle file. These keys should be pointers to scalar values
    (i.e. something like `objectinfo.ra` is OK, but something like 'times' won't
    work because it's a vector).

    Generally, this works with light curve reading functions that produce
    lcdicts as detailed in the docstring for `lcproc.register_lcformat`. Once
    you've registered your light curve reader functions using the
    `lcproc.register_lcformat` function, pass in the `formatkey` associated with
    your light curve format, and this function will be able to read all light
    curves in that format as well as the object information stored in their
    `objectinfo` dict.

    Parameters
    ----------

    basedir : str or list of str
        If this is a str, points to a single directory to search for light
        curves. If this is a list of str, it must be a list of directories to
        search for light curves. All of these will be searched to find light
        curve files matching either your light curve format's default fileglob
        (when you registered your LC format), or a specific fileglob that you
        can pass in using the `fileglob` kwargh here. If the `recursive` kwarg
        is set, the provided directories will be searched recursively.

        If `use_list_of_filenames` is not None, it will override this argument
        and the function will take those light curves as the list of files it
        must process instead of whatever is specified in `basedir`.

    outfile : str
        This is the name of the output file to write. This will be a pickle
        file, so a good convention to use for this name is something like
        'my-lightcurve-catalog.pkl'.

    use_list_of_filenames : list of str or None
        Use this kwarg to override whatever is provided in `basedir` and
        directly pass in a list of light curve files to process. This can speed
        up this function by a lot because no searches on disk will be performed
        to find light curve files matching `basedir` and `fileglob`.

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

    fileglob : str or None
        If provided, is a string that is a valid UNIX filename glob. Used to
        override the default fileglob for this LC format when searching for
        light curve files in `basedir`.

    recursive : bool
        If True, the directories specified in `basedir` will be searched
        recursively for all light curve files that match the default fileglob
        for this LC format or a specific one provided in `fileglob`.

    columns : list of str
        This is a list of keys in the lcdict produced by your light curve reader
        function that contain object information, which will be extracted and
        put into the output light curve catalog. It's highly recommended that
        your LC reader function produce a lcdict that contains at least the
        default keys shown here.

        The lcdict keys to extract are specified by using an address scheme:

        - First level dict keys can be specified directly:
          e.g., 'objectid' will extract lcdict['objectid']
        - Keys at other levels can be specified by using a period to indicate
          the level:

          - e.g., 'objectinfo.ra' will extract lcdict['objectinfo']['ra']
          - e.g., 'objectinfo.varinfo.features.stetsonj' will extract
            lcdict['objectinfo']['varinfo']['features']['stetsonj']

    makecoordindex : list of two str or None
        This is used to specify which lcdict keys contain the right ascension
        and declination coordinates for this object. If these are provided, the
        output light curve catalog will have a kdtree built on all object
        coordinates, which enables fast spatial searches and cross-matching to
        external catalogs by `checkplot` and `lcproc` functions.

    field_fitsfile : str or None
        If this is not None, it should be the path to a FITS image containing
        the objects these light curves are for. If this is provided,
        `make_lclist` will use the WCS information in the FITS itself if
        `field_wcsfrom` is None (or from a WCS header file pointed to by
        `field_wcsfrom`) to obtain x and y pixel coordinates for all of the
        objects in the field. A finder chart will also be made using
        `astrobase.plotbase.fits_finder_chart` using the corresponding
        `field_scale`, `_stretch`, `_colormap`, `_findersize`, `_pltopts`,
        `_grid`, and `_gridcolors` kwargs for that function, reproduced here to
        enable customization of the finder chart plot.

    field_wcsfrom : str or None
        If `wcsfrom` is None, the WCS to transform the RA/Dec to pixel x/y will
        be taken from the FITS header of `fitsfile`. If this is not None, it
        must be a FITS or similar file that contains a WCS header in its first
        extension.

    field_scale : astropy.visualization.Interval object
        `scale` sets the normalization for the FITS pixel values. This is an
        astropy.visualization Interval object.
        See http://docs.astropy.org/en/stable/visualization/normalization.html
        for details on `scale` and `stretch` objects.

    field_stretch : astropy.visualization.Stretch object
        `stretch` sets the stretch function for mapping FITS pixel values to
        output pixel values. This is an astropy.visualization Stretch object.
        See http://docs.astropy.org/en/stable/visualization/normalization.html
        for details on `scale` and `stretch` objects.

    field_colormap : matplotlib Colormap object
        `colormap` is a matplotlib color map object to use for the output image.

    field_findersize : None or tuple of two ints
        If `findersize` is None, the output image size will be set by the NAXIS1
        and NAXIS2 keywords in the input `fitsfile` FITS header. Otherwise,
        `findersize` must be a tuple with the intended x and y size of the image
        in inches (all output images will use a DPI = 100).

    field_pltopts : dict
        `field_pltopts` controls how the overlay points will be plotted. This
        a dict with standard matplotlib marker, etc. kwargs as key-val pairs,
        e.g. 'markersize', 'markerfacecolor', etc. The default options make red
        outline circles at the location of each object in the overlay.

    field_grid : bool
        `grid` sets if a grid will be made on the output image.

    field_gridcolor : str
        `gridcolor` sets the color of the grid lines. This is a usual matplotib
        color spec string.

    field_zoomcontain : bool
        `field_zoomcontain` controls if the finder chart will be zoomed to
        just contain the overlayed points. Everything outside the footprint of
        these points will be discarded.

    maxlcs : int or None
        This sets how many light curves to process in the input LC list
        generated by searching for LCs in `basedir` or in the list provided as
        `use_list_of_filenames`.

    nworkers : int
        This sets the number of parallel workers to launch to collect
        information from the light curves.

    Returns
    -------

    str
        Returns the path to the generated light curve catalog pickle file.

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
                    matching.extend(glob.glob(os.path.join(bdir,
                                                           '**',
                                                           fileglob),
                                              recursive=True))

        # otherwise, handle the usual case of one basedir to search in
        else:

            # now find the files
            LOGINFO('searching for %s light curves in %s ...' %
                    (lcformat, basedir))

            if recursive is False:
                matching = glob.glob(os.path.join(basedir, fileglob))

            else:
                matching = glob.glob(os.path.join(basedir,
                                                  '**',
                                                  fileglob),recursive=True)

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
            results = executor.map(_lclist_parallel_worker, tasks)

        results = list(results)

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

            except Exception:
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


def filter_lclist(lc_catalog,
                  objectidcol='objectid',
                  racol='ra',
                  declcol='decl',
                  xmatchexternal=None,
                  xmatchdistarcsec=3.0,
                  externalcolnums=(0,1,2),
                  externalcolnames=('objectid','ra','decl'),
                  externalcoldtypes='U20,f8,f8',
                  externalcolsep=None,
                  externalcommentchar='#',
                  conesearch=None,
                  conesearchworkers=1,
                  columnfilters=None,
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

    '''This is used to perform cone-search, cross-match, and column-filter
    operations on a light curve catalog generated by `make_lclist`.

    Uses the output of `make_lclist` above. This function returns a list of
    light curves matching various criteria specified by the `xmatchexternal`,
    `conesearch`, and `columnfilters kwargs`. Use this function to generate
    input lists for other lcproc functions,
    e.g. `lcproc.lcvfeatures.parallel_varfeatures`,
    `lcproc.periodfinding.parallel_pf`, and `lcproc.lcbin.parallel_timebin`,
    among others.

    The operations are applied in this order if more than one is specified:
    `xmatchexternal` -> `conesearch` -> `columnfilters`. All results from these
    operations are joined using a logical AND operation.

    Parameters
    ----------

    objectidcol : str
        This is the name of the object ID column in the light curve catalog.

    racol : str
        This is the name of the RA column in the light curve catalog.

    declcol : str
        This is the name of the Dec column in the light curve catalog.

    xmatchexternal : str or None
        If provided, this is the filename of a text file containing objectids,
        ras and decs to match the objects in the light curve catalog to by their
        positions.

    xmatchdistarcsec : float
        This is the distance in arcseconds to use when cross-matching to the
        external catalog in `xmatchexternal`.

    externalcolnums : sequence of int
        This a list of the zero-indexed column numbers of columns to extract
        from the external catalog file.

    externalcolnames : sequence of str
        This is a list of names of columns that will be extracted from the
        external catalog file. This is the same length as
        `externalcolnums`. These must contain the names provided as the
        `objectid`, `ra`, and `decl` column names so this function knows which
        column numbers correspond to those columns and can use them to set up
        the cross-match.

    externalcoldtypes : str
        This is a CSV string containing numpy dtype definitions for all columns
        listed to extract from the external catalog file. The number of dtype
        definitions should be equal to the number of columns to extract.

    externalcolsep : str or None
        The column separator to use when extracting columns from the external
        catalog file. If None, any whitespace between columns is used as the
        separator.

    externalcommentchar : str
        The character indicating that a line in the external catalog file is to
        be ignored.

    conesearch : list of float

        This is used to specify cone-search parameters. It should be a three
        element list:

        [center_ra_deg, center_decl_deg, search_radius_deg]

    conesearchworkers : int
        The number of parallel workers to launch for the cone-search operation.

    columnfilters : list of str

        This is a list of strings indicating any filters to apply on each column
        in the light curve catalog. All column filters are applied in the
        specified sequence and are combined with a logical AND operator. The
        format of each filter string should be:

        '<lc_catalog column>|<operator>|<operand>'

        where:

        - <lc_catalog column> is a column in the lc_catalog pickle file
        - <operator> is one of: 'lt', 'gt', 'le', 'ge', 'eq', 'ne', which
          correspond to the usual operators: <, >, <=, >=, ==, != respectively.
        - <operand> is a float, int, or string.

    field_fitsfile : str or None
        If this is not None, it should be the path to a FITS image containing
        the objects these light curves are for. If this is provided,
        `make_lclist` will use the WCS information in the FITS itself if
        `field_wcsfrom` is None (or from a WCS header file pointed to by
        `field_wcsfrom`) to obtain x and y pixel coordinates for all of the
        objects in the field. A finder chart will also be made using
        `astrobase.plotbase.fits_finder_chart` using the corresponding
        `field_scale`, `_stretch`, `_colormap`, `_findersize`, `_pltopts`,
        `_grid`, and `_gridcolors` kwargs for that function, reproduced here to
        enable customization of the finder chart plot.

    field_wcsfrom : str or None
        If `wcsfrom` is None, the WCS to transform the RA/Dec to pixel x/y will
        be taken from the FITS header of `fitsfile`. If this is not None, it
        must be a FITS or similar file that contains a WCS header in its first
        extension.

    field_scale : astropy.visualization.Interval object
        `scale` sets the normalization for the FITS pixel values. This is an
        astropy.visualization Interval object.
        See http://docs.astropy.org/en/stable/visualization/normalization.html
        for details on `scale` and `stretch` objects.

    field_stretch : astropy.visualization.Stretch object
        `stretch` sets the stretch function for mapping FITS pixel values to
        output pixel values. This is an astropy.visualization Stretch object.
        See http://docs.astropy.org/en/stable/visualization/normalization.html
        for details on `scale` and `stretch` objects.

    field_colormap : matplotlib Colormap object
        `colormap` is a matplotlib color map object to use for the output image.

    field_findersize : None or tuple of two ints
        If `findersize` is None, the output image size will be set by the NAXIS1
        and NAXIS2 keywords in the input `fitsfile` FITS header. Otherwise,
        `findersize` must be a tuple with the intended x and y size of the image
        in inches (all output images will use a DPI = 100).

    field_pltopts : dict
        `field_pltopts` controls how the overlay points will be plotted. This
        a dict with standard matplotlib marker, etc. kwargs as key-val pairs,
        e.g. 'markersize', 'markerfacecolor', etc. The default options make red
        outline circles at the location of each object in the overlay.

    field_grid : bool
        `grid` sets if a grid will be made on the output image.

    field_gridcolor : str
        `gridcolor` sets the color of the grid lines. This is a usual matplotib
        color spec string.

    field_zoomcontain : bool
        `field_zoomcontain` controls if the finder chart will be zoomed to
        just contain the overlayed points. Everything outside the footprint of
        these points will be discarded.

    copylcsto : str
        If this is provided, it is interpreted as a directory target to copy
        all the light curves that match the specified conditions.

    Returns
    -------

    tuple
        Returns a two elem tuple: (matching_object_lcfiles, matching_objectids)
        if conesearch and/or column filters are used. If `xmatchexternal` is
        also used, a three-elem tuple is returned: (matching_object_lcfiles,
        matching_objectids, extcat_matched_objectids).

    '''

    with open(lc_catalog,'rb') as infd:
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

        except Exception:

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

        except Exception:

            LOGEXCEPTION('cone-search: could not run a cone-search, '
                         'is there a kdtree present in %s?' % lc_catalog)
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

            except Exception:

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
            os.path.dirname(lc_catalog),
            '%s%s.png' %
            (os.path.splitext(os.path.basename(lc_catalog))[0],
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

def _cpinfo_key_worker(task):
    '''This wraps `checkplotlist.checkplot_infokey_worker`.

    This is used to get the correct dtype for each element in retrieved results.

    Parameters
    ----------

    task : tuple
        task[0] = cpfile
        task[1] = keyspeclist (infokeys kwarg from `add_cpinfo_to_lclist`)

    Returns
    -------

    dict
        All of the requested keys from the checkplot are returned along with
        their values in a dict.

    '''

    cpfile, keyspeclist = task

    keystoget = [x[0] for x in keyspeclist]
    nonesubs = [x[-2] for x in keyspeclist]
    nansubs = [x[-1] for x in keyspeclist]

    # reform the keystoget into a list of lists
    for i, k in enumerate(keystoget):

        thisk = k.split('.')
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
        initial_lc_catalog,
        magcol,  # to indicate checkplot magcol
        outfile,
        checkplotglob='checkplot*.pkl*',
        infokeys=CPINFO_DEFAULTKEYS,
        nworkers=NCPUS
):
    '''This adds checkplot info to the initial light curve catalogs generated by
    `make_lclist`.

    This is used to incorporate all the extra info checkplots can have for
    objects back into columns in the light curve catalog produced by
    `make_lclist`. Objects are matched between the checkplots and the light
    curve catalog using their `objectid`. This then allows one to search this
    'augmented' light curve catalog by these extra columns. The 'augmented'
    light curve catalog also forms the basis for search interface provided by
    the LCC-Server.

    The default list of keys that will be extracted from a checkplot and added
    as columns in the initial light curve catalog is listed above in the
    `CPINFO_DEFAULTKEYS` list.

    Parameters
    ----------

    checkplots : str or list
        If this is a str, is interpreted as a directory which will be searched
        for checkplot pickle files using `checkplotglob`. If this is a list, it
        will be interpreted as a list of checkplot pickle files to process.

    initial_lc_catalog : str
        This is the path to the light curve catalog pickle made by
        `make_lclist`.

    magcol : str
        This is used to indicate the light curve magnitude column to extract
        magnitude column specific information. For example, Stetson variability
        indices can be generated using magnitude measurements in separate
        photometric apertures, which appear in separate `magcols` in the
        checkplot. To associate each such feature of the object with its
        specific `magcol`, pass that `magcol` in here. This `magcol` will then
        be added as a prefix to the resulting column in the 'augmented' LC
        catalog, e.g. Stetson J will appear as `magcol1_stetsonj` and
        `magcol2_stetsonj` for two separate magcols.

    outfile : str
        This is the file name of the output 'augmented' light curve catalog
        pickle file that will be written.

    infokeys : list of tuples

        This is a list of keys to extract from the checkplot and some info on
        how this extraction is to be done. Each key entry is a six-element
        tuple of the following form:

        - key name in the checkplot
        - numpy dtype of the value of this key
        - False if key is associated with a magcol or True otherwise
        - False if subsequent updates to the same column name will append to
          existing key values in the output augmented light curve catalog or
          True if these will overwrite the existing key value
        - character to use to substitute a None value of the key in the
          checkplot in the output light curve catalog column
        - character to use to substitute a nan value of the key in the
          checkplot in the output light curve catalog column

        See the `CPFINFO_DEFAULTKEYS` list above for examples.

    nworkers : int
        The number of parallel workers to launch to extract checkplot
        information.

    Returns
    -------

    str
        Returns the path to the generated 'augmented' light curve catalog pickle
        file.

    '''

    # get the checkplots from the directory if one is provided
    if not isinstance(checkplots, list) and os.path.exists(checkplots):
        checkplots = sorted(glob.glob(os.path.join(checkplots, checkplotglob)))

    tasklist = [(cpf, infokeys) for cpf in checkplots]

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(_cpinfo_key_worker, tasklist)

    results = list(resultfutures)
    executor.shutdown()

    # now that we have all the checkplot info, we need to match to the
    # objectlist in the lclist

    # open the lclist
    with open(initial_lc_catalog,'rb') as infd:
        lc_catalog = pickle.load(infd)

    catalog_objectids = np.array(lc_catalog['objects']['objectid'])
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
        if eactual not in lc_catalog['columns']:
            lc_catalog['columns'].append(eactual)

        # we'll overwrite earlier existing columns in any case
        lc_catalog['objects'][eactual] = []

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
                lc_catalog['objects'][ek].append(
                    thiscpinfo[ekind]
                )

        else:

            # update the object catalog entries for this object
            for ekind, ek in enumerate(actualkeys):

                thiskeyspec = infokeys[ekind]
                nonesub = thiskeyspec[-2]

                lc_catalog['objects'][ek].append(
                    nonesub
                )

    # now we should have all the new keys in the object catalog
    # turn them into arrays
    for ek in actualkeys:

        lc_catalog['objects'][ek] = np.array(
            lc_catalog['objects'][ek]
        )

    # add the magcol to the lc_catalog
    if 'magcols' in lc_catalog:
        if magcol not in lc_catalog['magcols']:
            lc_catalog['magcols'].append(magcol)
    else:
        lc_catalog['magcols'] = [magcol]

    # write back the new object catalog
    with open(outfile, 'wb') as outfd:
        pickle.dump(lc_catalog, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    return outfile
