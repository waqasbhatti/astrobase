#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''lcproc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - May 2017

This contains functions that serve as examples for running large batch jobs
processing HAT light curves.

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
    from cStringIO import StringIO as strio
except Exception as e:
    import pickle
    from io import BytesIO as strio
import gzip
import glob
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import base64
import uuid

import numpy as np
import numpy.random as npr
npr.seed(0xc0ffee)

import scipy.spatial as sps
import scipy.interpolate as spi
from scipy import linalg as spla

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

from tornado.escape import squeeze

# to turn a list of keys into a dict address
# from https://stackoverflow.com/a/14692747
from functools import reduce, partial
from operator import getitem
def dict_get(datadict, keylist):
    return reduce(getitem, keylist, datadict)



###################
## LOCAL IMPORTS ##
###################

# LC reading functions
from astrobase.hatsurveys.hatlc import read_and_filter_sqlitecurve, \
    read_csvlc, normalize_lcdict_byinst
from astrobase.hatsurveys.hplc import read_hatpi_textlc, read_hatpi_pklc
from astrobase.astrokep import read_kepler_fitslc, read_kepler_pklc, \
    filter_kepler_lcdict
from astrobase.astrotess import read_tess_fitslc, read_tess_pklc, \
    filter_tess_lcdict

from astrobase import periodbase, checkplot
from astrobase.varclass import varfeatures, starfeatures, periodicfeatures
from astrobase.lcmath import normalize_magseries, \
    time_bin_magseries_with_errs, sigclip_magseries
from astrobase.periodbase.kbls import bls_snr
from astrobase.plotbase import fits_finder_chart

from astrobase.checkplot import _pkl_magseries_plot, \
    _pkl_phased_magseries_plot, xmatch_external_catalogs, \
    _read_checkplot_picklefile, _write_checkplot_picklefile

from astrobase.magnitudes import jhk_to_sdssr

from astrobase.varbase.trends import epd_magseries, smooth_magseries_savgol

from astrobase.cpserver.checkplotlist import checkplot_infokey_worker

############
## CONFIG ##
############

NCPUS = mp.cpu_count()


#############################################
## MAPS FOR LCFORMAT TO LCREADER FUNCTIONS ##
#############################################

def read_pklc(lcfile):
    '''
    This just reads a pickle.

    '''

    if lcfile.endswith('.gz'):

        try:
            with gzip.open(lcfile,'rb') as infd:
                lcdict = pickle.load(infd)
        except UnicodeDecodeError:
            with gzip.open(lcfile,'rb') as infd:
                lcdict = pickle.load(infd, encoding='latin1')

    else:

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



# used to figure out which period finder to run given a list of methods
PFMETHODS = {'bls':periodbase.bls_parallel_pfind,
             'gls':periodbase.pgen_lsp,
             'aov':periodbase.aov_periodfind,
             'mav':periodbase.aovhm_periodfind,
             'pdm':periodbase.stellingwerf_pdm,
             'acf':periodbase.macf_period_find,
             'win':periodbase.specwindow_lsp}



# This is the lcproc dictionary to store registered light curve formats and the
# means to read and normalize light curve files associated with each format. The
# format spec for a light curve format is a list with the elements outlined
# below. To register a new light curve format, use the register_custom_lcformat
# function below.
LCFORM = {
    'hat-sql':[
        '*-hatlc.sqlite*',             # default fileglob
        read_and_filter_sqlitecurve,   # function to read this LC
        ['rjd','rjd'],                 # default timecols to use for period/var
        ['aep_000','atf_000'],         # default magcols to use for period/var
        ['aie_000','aie_000'],         # default errcols to use for period/var
        False,                         # default magsarefluxes = False
        normalize_lcdict_byinst,       # default special normalize function
    ],
    'hat-csv':[
        '*-hatlc.csv*',
        read_csvlc,
        ['rjd','rjd'],
        ['aep_000','atf_000'],
        ['aie_000','aie_000'],
        False,
        normalize_lcdict_byinst,
    ],
    'hp-txt':[
        'HAT-*tfalc.TF1*',
        read_hatpi_textlc,
        ['rjd','rjd'],
        ['iep1','itf1'],
        ['ire1','ire1'],
        False,
        None,
    ],
    'hp-pkl':[
        '*-pklc.pkl*',
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
        filter_kepler_lcdict,
    ],
    'kep-pkl':[
        '-keplc.pkl',
        read_kepler_pklc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdc_sapflux'],
        ['sap.sap_flux_err','pdc.pdc_sapflux_err'],
        True,
        filter_kepler_lcdict,
    ],
    'tess-fits':[
        '*_lc.fits',
        read_tess_fitslc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdc_sapflux'],
        ['sap.sap_flux_err','pdc.pdc_sapflux_err'],
        True,
        filter_tess_lcdict,
    ],
    'tess-pkl':[
        '-tesslc.pkl',
        read_tess_pklc,
        ['time','time'],
        ['sap.sap_flux','pdc.pdc_sapflux'],
        ['sap.sap_flux_err','pdc.pdc_sapflux_err'],
        True,
        filter_tess_lcdict,
    ],
    # binned light curve format
    'binned-hat':[
        '*binned*hat*.pkl',
        read_pklc,
        ['binned.aep_000.times','binned.atf_000.times'],
        ['binned.aep_000.mags','binned.atf_000.mags'],
        ['binned.aep_000.errs','binned.atf_000.errs'],
        False,
        None,
    ],
    'binned-hp':[
        '*binned*hp*.pkl',
        read_pklc,
        ['binned.iep1.times','binned.itf1.times'],
        ['binned.iep1.mags','binned.itf1.mags'],
        ['binned.iep1.errs','binned.itf1.errs'],
        False,
        None,
    ],
    'binned-kep':[
        '*binned*kep*.pkl',
        read_pklc,
        ['binned.sap_flux.times','binned.pdc_sapflux.times'],
        ['binned.sap_flux.mags','binned.pdc_sapflux.mags'],
        ['binned.sap_flux.errs','binned.pdc_sapflux.errs'],
        True,
        None,
    ],
}



def register_custom_lcformat(formatkey,
                             fileglob,
                             readerfunc,
                             timecols,
                             magcols,
                             errcols,
                             readerfunc_kwargs=None,
                             specialnormfunc=None,
                             normfunc_kwargs=None,
                             magsarefluxes=False):
    '''This adds a custom format LC to the dict above.

    Allows handling of custom format light curves for astrobase lcproc
    drivers. Once the format is successfully registered, light curves should
    work transparently with all of the functions in this module, by simply
    calling them with the formatkey in the lcformat keyword argument.

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


    readerfunc_kwargs is a dictionary containing any kwargs to pass through to
    the light curve reader function.


    specialnormfunc: <function>: if you intend to use a special normalization
    function for your lightcurves, indicate it here. If None, the default
    normalization method used by lcproc is to find gaps in the time-series,
    normalize measurements grouped by these gaps to zero, then normalize the
    entire magnitude time series to global time series median using the
    astrobase.lcmath.normalize_magseries function. The function should take and
    return an lcdict of the same form as that produced by readerfunc above. For
    an example of a special normalization function, see normalize_lcdict_by_inst
    in the astrobase.hatlc module.


    normfunc_kwargs is a dictionary containing any kwargs to pass through to
    the special light curve normalization function.


    magsarefluxes: <boolean>: if this is True, then all functions will treat the
    magnitude columns as flux instead, so things like default normalization and
    sigma-clipping will be done correctly. If this is False, magnitudes will be
    treated as magnitudes.

    '''

    #
    # generate the partials
    #

    if isinstance(readerfunc_kwargs, dict):
        lcrfunc = partial(readerfunc, **readerfunc_kwargs)
    else:
        lcrfunc = readerfunc

    if specialnormfunc is not None and isinstance(normfunc_kwargs, dict):
        lcnfunc = partial(specialnormfunc, **normfunc_kwargs)
    else:
        lcnfunc = specialnormfunc


    globals()['LCFORM'][formatkey] = [
        fileglob,
        lcrfunc,
        timecols,
        magcols,
        errcols,
        magsarefluxes,
        lcnfunc
    ]

    LOGINFO('added %s to registry' % formatkey)



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

    '''

    lcf, columns, readerfunc, lcndetkey = task

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

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR("can't figure out the light curve format")
        return

    if not fileglob:
        fileglob = LCFORM[lcformat][0]

    readerfunc = LCFORM[lcformat][1]

    # this is to get the actual ndet
    # set to the magnitudes column
    lcndetkey = LCFORM[lcformat][3]

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

        tasks = [(x, columns, readerfunc, lcndetkey) for x in matching]

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



##########################
## VARIABILITY FEATURES ##
##########################

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

                # get the features for this magcol
                lcfeatures = varfeatures.all_nonperiodic_features(
                    times, mags, errs
                )
                resultdict[mcol] = lcfeatures

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

                magmads[mind] = resultdict[mcol]['mad']

            # smallest MAD index
            bestmagcolind = np.where(magmads == np.min(magmads))[0]
            resultdict['bestmagcol'] = magcols[bestmagcolind]

        except Exception as e:
            resultdict['bestmagcol'] = None

        outfile = os.path.join(outdir,
                               'varfeatures-%s.pkl' %
                               squeeze(resultdict['objectid']).replace(' ','-'))

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

    try:
        lcfile, outdir, timecols, magcols, errcols, mindet, lcformat = task
        return get_varfeatures(lcfile, outdir,
                               timecols=timecols,
                               magcols=magcols,
                               errcols=errcols,
                               mindet=mindet,
                               lcformat=lcformat)

    except Exception as e:
        return None


def serial_varfeatures(lclist,
                       outdir,
                       maxobjects=None,
                       timecols=None,
                       magcols=None,
                       errcols=None,
                       mindet=1000,
                       lcformat='hat-sql',
                       nworkers=NCPUS):

    if maxobjects:
        lclist = lclist[:maxobjects]

    tasks = [(x, outdir, timecols, magcols, errcols, mindet, lcformat)
             for x in lclist]

    for task in tqdm(tasks):
        result = varfeatures_worker(task)

    return result



def parallel_varfeatures(lclist,
                         outdir,
                         maxobjects=None,
                         timecols=None,
                         magcols=None,
                         errcols=None,
                         mindet=1000,
                         lcformat='hat-sql',
                         nworkers=NCPUS):
    '''
    This runs varfeatures in parallel for all light curves in lclist.

    '''
    # make sure to make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maxobjects:
        lclist = lclist[:maxobjects]

    tasks = [(x, outdir, timecols, magcols, errcols, mindet, lcformat)
             for x in lclist]

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(varfeatures_worker, tasks)

    results = [x for x in resultfutures]
    resdict = {os.path.basename(x):y for (x,y) in zip(lclist, results)}

    return resdict



def parallel_varfeatures_lcdir(lcdir,
                               outdir,
                               maxobjects=None,
                               timecols=None,
                               magcols=None,
                               errcols=None,
                               recursive=True,
                               mindet=1000,
                               lcformat='hat-sql',
                               nworkers=NCPUS):
    '''
    This runs parallel variable feature extraction for a directory of LCs.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    fileglob = LCFORM[lcformat][0]

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

        LOGINFO('found %s light curves, getting varfeatures...' %
                len(matching))

        return parallel_varfeatures(matching,
                                    outdir,
                                    maxobjects=maxobjects,
                                    timecols=timecols,
                                    magcols=magcols,
                                    errcols=errcols,
                                    mindet=mindet,
                                    lcformat=lcformat,
                                    nworkers=nworkers)

    else:

        LOGERROR('no light curve files in %s format found in %s' % (lcformat,
                                                                    lcdir))
        return None



#######################
## PERIODIC FEATURES ##
#######################

def get_periodicfeatures(pfpickle,
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
                         sigclip=10.0,
                         magsarefluxes=False,
                         verbose=True,
                         raiseonfail=False):
    '''This gets all periodic features for the object.

    If starfeatures is not None, it should be the filename of the
    starfeatures-<objectid>.pkl created by get_starfeatures for this
    object. This is used to get the neighbor's light curve and phase it with
    this object's period to see if this object is blended.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

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
            times = dict_get(lcdict, tcolget)

            if nbrlcf:
                nbrtimes = dict_get(nbrlcdict, tcolget)
            else:
                nbrtimes = None


            if '.' in mcol:
                mcolget = mcol.split('.')
            else:
                mcolget = [mcol]

            mags = dict_get(lcdict, mcolget)

            if nbrlcf:
                nbrmags = dict_get(nbrlcdict, mcolget)
            else:
                nbrmags = None


            if '.' in ecol:
                ecolget = ecol.split('.')
            else:
                ecolget = [ecol]

            errs = dict_get(lcdict, ecolget)

            if nbrlcf:
                nbrerrs = dict_get(nbrlcdict, ecolget)
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
                            sigclip=10.0,
                            magsarefluxes=False,
                            verbose=False,
                            maxobjects=None,
                            nworkers=NCPUS):
    '''This drives the periodicfeatures collection for a list of periodfinding
    pickles.

    '''
    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
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
              'sigclip':sigclip,
              'magsarefluxes':magsarefluxes,
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
                              sigclip=10.0,
                              magsarefluxes=False,
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
              'sigclip':sigclip,
              'magsarefluxes':magsarefluxes,
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
        sigclip=10.0,
        magsarefluxes=False,
        verbose=False,
        maxobjects=None,
        nworkers=NCPUS,
        recursive=True,
):
    '''This runs parallel periodicfeature extraction for a directory of
    periodfinding result pickles.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
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
            sigclip=sigclip,
            magsarefluxes=magsarefluxes,
            verbose=verbose,
            maxobjects=maxobjects,
            nworkers=nworkers,
        )

    else:

        LOGERROR('no periodfinding pickles found in %s' % (pfpkl_dir))
        return None



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
                     lcformat='hat-sql'):
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

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

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
        nbrfeat = starfeatures.neighbor_features(lcdict['objectinfo'],
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
         deredden, custom_bandpasses, lcformat) = task

        return get_starfeatures(lcfile, outdir,
                                kdtree, objlist, lcflist,
                                neighbor_radius_arcsec,
                                deredden=deredden,
                                custom_bandpasses=custom_bandpasses,
                                lcformat=lcformat)
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
              deredden, custom_bandpasses, lcformat) for x in lclist]

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
                          nworkers=NCPUS):
    '''
    This runs starfeatures in parallel for all light curves in lclist.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
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
                                maxobjects=None,
                                deredden=True,
                                custom_bandpasses=None,
                                lcformat='hat-sql',
                                nworkers=NCPUS,
                                recursive=True):
    '''
    This runs parallel star feature extraction for a directory of LCs.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    fileglob = LCFORM[lcformat][0]

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
                                     nworkers=nworkers)

    else:

        LOGERROR('no light curve files in %s format found in %s' % (lcformat,
                                                                    lcdir))
        return None



###########################
## VARIABILITY THRESHOLD ##
###########################

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
                                magcols=None):
    '''
    This makes plots for the variability threshold distributions.

    '''
    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

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



#############################
## RUNNING PERIOD SEARCHES ##
#############################

def runpf(lcfile,
          outdir,
          timecols=None,
          magcols=None,
          errcols=None,
          lcformat='hat-sql',
          pfmethods=('gls','pdm','mav','win'),
          pfkwargs=({},{},{},{}),
          sigclip=10.0,
          getblssnr=False,
          nworkers=NCPUS,
          minobservations=500,
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

    try:

        # get the LC into a dict
        lcdict = readerfunc(lcfile)

        # this should handle lists/tuples being returned by readerfunc
        # we assume that the first element is the actual lcdict
        # FIXME: figure out how to not need this assumption
        if ( (isinstance(lcdict, (list, tuple))) and
             (isinstance(lcdict[0], dict)) ):
            lcdict = lcdict[0]

        outfile = os.path.join(outdir, 'periodfinding-%s.pkl' %
                               squeeze(lcdict['objectid']).replace(' ', '-'))

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
        resultdict = {
            'objectid':lcdict['objectid'],
            'lcfbasename':os.path.basename(lcfile),
            'kwargs':{'timecols':timecols,
                      'magcols':magcols,
                      'errcols':errcols,
                      'lcformat':lcformat,
                      'pfmethods':pfmethods,
                      'pfkwargs':pfkwargs,
                      'sigclip':sigclip,
                      'getblssnr':getblssnr}
        }

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


            # run each of the requested period-finder functions
            resultdict[mcol] = {}

            # check if we have enough non-nan observations to proceed
            finmags = mags[np.isfinite(mags)]

            if finmags.size < minobservations:

                LOGERROR('not enough non-nan observations for '
                         'this LC. have: %s, required: %s, '
                         'magcol: %s, skipping...' %
                         (finmags.size, minobservations, mcol))
                continue

            pfmkeys = []

            for pfmind, pfm, pfkw in zip(range(len(pfmethods)),
                                         pfmethods,
                                         pfkwargs):

                pf_func = PFMETHODS[pfm]

                # get any optional kwargs for this function
                pf_kwargs = pfkw
                pf_kwargs.update({'verbose':False,
                                  'nworkers':nworkers,
                                  'magsarefluxes':magsarefluxes,
                                  'sigclip':sigclip})

                # we'll always prefix things with their index to allow multiple
                # invocations and results from the same period-finder (for
                # different period ranges, for example).
                pfmkey = '%s-%s' % (pfmind, pfm)
                pfmkeys.append(pfmkey)

                # run this period-finder and save its results to the output dict
                resultdict[mcol][pfmkey] = pf_func(
                    times, mags, errs,
                    **pf_kwargs
                )


            #
            # done with running the period finders
            #
            # append the pfmkeys list to the magcol dict
            resultdict[mcol]['pfmethods'] = pfmkeys

            # check if we need to get the SNR from any BLS pfresults
            if 'bls' in pfmethods and getblssnr:

                # we need to scan thru the pfmethods to get to any BLS pfresults
                for pfmk in resultdict[mcol]['pfmethods']:

                    if 'bls' in pfmk:

                        try:

                            bls = resultdict[mcol][pfmk]

                            # calculate the SNR for the BLS as well
                            blssnr = bls_snr(bls, times, mags, errs,
                                             magsarefluxes=magsarefluxes,
                                             verbose=False)

                            # add the SNR results to the BLS result dict
                            resultdict[mcol][pfmk].update({
                                'snr':blssnr['snr'],
                                'transitdepth':blssnr['transitdepth'],
                                'transitduration':blssnr['transitduration'],
                            })

                        except Exception as e:

                            LOGEXCEPTION('could not calculate BLS SNR for %s' %
                                         lcfile)
                            # add the SNR null results to the BLS result dict
                            resultdict[mcol][pfmk].update({
                                'snr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                                'transitdepth':[np.nan,np.nan,np.nan,
                                                np.nan,np.nan],
                                'transitduration':[np.nan,np.nan,np.nan,
                                                   np.nan,np.nan],
                            })

            elif 'bls' in pfmethods:

                # we need to scan thru the pfmethods to get to any BLS pfresults
                for pfmk in resultdict[mcol]['pfmethods']:

                    if 'bls' in pfmk:

                        # add the SNR null results to the BLS result dict
                        resultdict[mcol][pfmk].update({
                            'snr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                            'transitdepth':[np.nan,np.nan,np.nan,
                                            np.nan,np.nan],
                            'transitduration':[np.nan,np.nan,np.nan,
                                               np.nan,np.nan],
                        })


        # once all mag cols have been processed, write out the pickle
        with open(outfile, 'wb') as outfd:
            pickle.dump(resultdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

        return outfile

    except Exception as e:

        LOGEXCEPTION('failed to run for %s, because: %s' % (lcfile, e))
        return None



def runpf_worker(task):
    '''
    This runs the runpf function.

    '''

    (lcfile, outdir, timecols, magcols, errcols, lcformat,
     pfmethods, pfkwargs, getblssnr, sigclip, nworkers, minobservations,
     excludeprocessed) = task

    if os.path.exists(lcfile):
        pfresult = runpf(lcfile,
                         outdir,
                         timecols=timecols,
                         magcols=magcols,
                         errcols=errcols,
                         lcformat=lcformat,
                         pfmethods=pfmethods,
                         pfkwargs=pfkwargs,
                         getblssnr=getblssnr,
                         sigclip=sigclip,
                         nworkers=nworkers,
                         minobservations=minobservations,
                         excludeprocessed=excludeprocessed)
        return pfresult
    else:
        LOGERROR('LC does not exist for requested file %s' % lcfile)
        return None



def parallel_pf(lclist,
                outdir,
                timecols=None,
                magcols=None,
                errcols=None,
                lcformat='hat-sql',
                pfmethods=('gls','pdm','mav','win'),
                pfkwargs=({},{},{},{}),
                getblssnr=False,
                sigclip=10.0,
                nperiodworkers=NCPUS,
                ncontrolworkers=1,
                liststartindex=None,
                listmaxobjects=None,
                minobservations=500,
                excludeprocessed=True):
    '''This drives the overall parallel period processing.

    Use pfmethods to specify which periodfinders to run. These must be in
    lcproc.PFMETHODS.

    Use pfkwargs to provide optional kwargs to the periodfinders.

    If getblssnr is True, will run BLS SNR calculations for each object and
    magcol. This takes a while to run, so it's disabled (False) by default.

    sigclip sets the sigma-clip to use for the light curves before putting them
    through each of the periodfinders.

    nperiodworkers is the number of period-finder workers to launch.

    ncontrolworkers is the number of controlling processes to launch.

    liststartindex sets the index from where to start in lclist. listmaxobjects
    sets the maximum number of objects in lclist to run periodfinding for in
    this invocation. Together, these can be used to distribute processing over
    several independent machines if the number of light curves is very large.

    If excludeprocessed is True, light curves that have been processed already
    and have existing corresponding periodfinding-<objectid-suffix>.pkl[.gz]
    files in outdir will be ignored.

    As a rough benchmark, 25000 HATNet light curves with up to 50000 points per
    LC take about 26 days in total for an invocation of this function using
    GLS+PDM+BLS, 10 periodworkers, and 4 controlworkers (so all 40 'cores') on a
    2 x Xeon E5-2660v3 machine.

    '''

    # make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if (liststartindex is not None) and (listmaxobjects is None):
        lclist = lclist[liststartindex:]

    elif (liststartindex is None) and (listmaxobjects is not None):
        lclist = lclist[:listmaxobjects]

    elif (liststartindex is not None) and (listmaxobjects is not None):
        lclist = lclist[liststartindex:liststartindex+listmaxobjects]

    tasklist = [(x, outdir, timecols, magcols, errcols, lcformat,
                 pfmethods, pfkwargs, getblssnr, sigclip, nperiodworkers,
                 minobservations,
                 excludeprocessed)
                for x in lclist]

    with ProcessPoolExecutor(max_workers=ncontrolworkers) as executor:
        resultfutures = executor.map(runpf_worker, tasklist)

    results = [x for x in resultfutures]
    return results



def parallel_pf_lcdir(lcdir,
                      outdir,
                      recursive=True,
                      timecols=None,
                      magcols=None,
                      errcols=None,
                      lcformat='hat-sql',
                      pfmethods=('gls','pdm','mav','win'),
                      pfkwargs=({},{},{},{}),
                      getblssnr=False,
                      sigclip=10.0,
                      nperiodworkers=NCPUS,
                      ncontrolworkers=1,
                      liststartindex=None,
                      listmaxobjects=None,
                      minobservations=500,
                      excludeprocessed=True):
    '''
    This runs parallel light curve period finding for directory of LCs.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    fileglob = LCFORM[lcformat][0]

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

        # this helps us process things in deterministic order when we distribute
        # processing over several machines
        matching = sorted(matching)

        LOGINFO('found %s light curves, running pf...' % len(matching))

        return parallel_pf(matching,
                           outdir,
                           timecols=timecols,
                           magcols=magcols,
                           errcols=errcols,
                           lcformat=lcformat,
                           pfmethods=pfmethods,
                           pfkwargs=pfkwargs,
                           getblssnr=getblssnr,
                           sigclip=sigclip,
                           nperiodworkers=nperiodworkers,
                           ncontrolworkers=ncontrolworkers,
                           liststartindex=liststartindex,
                           listmaxobjects=listmaxobjects,
                           minobservations=minobservations,
                           excludeprocessed=excludeprocessed)

    else:

        LOGERROR('no light curve files in %s format found in %s' % (lcformat,
                                                                    lcdir))
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

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return checkplotdict

    if not ('neighbors' in checkplotdict and
            checkplotdict['neighbors'] and
            len(checkplotdict['neighbors']) > 0):

        LOGERROR('no neighbors for %s, not updating...' %
                 (checkplotdict['objectid']))
        return checkplotdict

    # get the lcformat specific info
    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

    # get our object's magkeys to compare to the neighbor
    objmagkeys = {}

    # handle diff generations of checkplots
    if 'available_bands' in checkplotdict['objectinfo']:
        mclist = checkplotdict['objectinfo']['available_bands']
    else:
        mclist = ('bmag','vmag','rmag','imag','jmag','hmag','kmag',
                  'sdssu','sdssg','sdssr','sdssi','sdssz')

    for mc in mclist:
        if (mc in checkplotdict['objectinfo'] and
            checkplotdict['objectinfo'][mc] is not None and
            np.isfinite(checkplotdict['objectinfo'][mc])):

            objmagkeys[mc] = checkplotdict['objectinfo'][mc]


    # if there are actually neighbors, go through them in order
    for nbr in checkplotdict['neighbors']:

        objectid, lcfpath = (nbr['objectid'],
                             nbr['lcfpath'])

        # get the light curve
        if not os.path.exists(lcfpath):
            LOGERROR('objectid: %s, neighbor: %s, '
                     'lightcurve: %s not found, skipping...' %
                     (checkplotdict['objectid'], objectid, lcfpath))
            continue

        lcdict = readerfunc(lcfpath)

        # this should handle lists/tuples being returned by readerfunc
        # we assume that the first element is the actual lcdict
        # FIXME: figure out how to not need this assumption
        if ( (isinstance(lcdict, (list, tuple))) and
             (isinstance(lcdict[0], dict)) ):
            lcdict = lcdict[0]


        # 0. get this neighbor's magcols and get the magdiff and colordiff
        # between it and the object

        nbrmagkeys = {}

        for mc in objmagkeys:

            if (('objectinfo' in lcdict) and
                (isinstance(lcdict['objectinfo'], dict)) and
                (mc in lcdict['objectinfo']) and
                (lcdict['objectinfo'][mc] is not None) and
                (np.isfinite(lcdict['objectinfo'][mc]))):

                nbrmagkeys[mc] = lcdict['objectinfo'][mc]

        # now calculate the magdiffs
        magdiffs = {}
        for omc in objmagkeys:
            if omc in nbrmagkeys:
                magdiffs[omc] = objmagkeys[omc] - nbrmagkeys[omc]

        # calculate colors and colordiffs
        colordiffs = {}

        # generate the list of colors to get
        # NOTE: here, we don't really bother with new/old gen checkplots
        # maybe change this later to handle arbitrary colors

        for ctrio in (['bmag','vmag','bvcolor'],
                      ['vmag','kmag','vkcolor'],
                      ['jmag','kmag','jkcolor'],
                      ['sdssi','jmag','ijcolor'],
                      ['sdssg','kmag','gkcolor'],
                      ['sdssg','sdssr','grcolor']):
            m1, m2, color = ctrio

            if (m1 in objmagkeys and
                m2 in objmagkeys and
                m1 in nbrmagkeys and
                m2 in nbrmagkeys):

                objcolor = objmagkeys[m1] - objmagkeys[m2]
                nbrcolor = nbrmagkeys[m1] - nbrmagkeys[m2]
                colordiffs[color] = objcolor - nbrcolor

        # finally, add all the color and magdiff info to the nbr dict
        nbr.update({'magdiffs':magdiffs,
                    'colordiffs':colordiffs})

        #
        # process magcols
        #

        # normalize using the special function if specified
        if normfunc is not None:
            lcdict = normfunc(lcdict)

        try:

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

        except KeyError:

            LOGERROR('LC for neighbor: %s (target object: %s) does not '
                     'have one or more of the required columns: %s, '
                     'skipping...' %
                     (objectid, checkplotdict['objectid'],
                      ', '.join([timecol, magcol, errcol])))
            continue

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


        # check if this neighbor has enough finite points in its LC
        # fail early if not enough light curve points
        if ((xtimes is None) or (xmags is None) or (xerrs is None) or
            (xtimes.size < 49) or (xmags.size < 49) or (xerrs.size < 49)):

            LOGERROR("one or more of times, mags, errs appear to be None "
                     "after sig-clipping. are the measurements all nan? "
                     "can't make neighbor light curve plots "
                     "for target: %s, neighbor: %s, neighbor LC: %s" %
                     (checkplotdict['objectid'],
                      nbr['objectid'],
                      nbr['lcfpath']))
            continue

        #
        # now we can start doing stuff if everything checks out
        #

        # make an unphased mag-series plot
        nbrdict = _pkl_magseries_plot(xtimes,
                                      xmags,
                                      xerrs,
                                      magsarefluxes=magsarefluxes)
        # update the nbr
        nbr.update(nbrdict)

        # for each lspmethod in the checkplot, make a corresponding plot for
        # this neighbor

        # figure out the period finder methods present
        if 'pfmethods' in checkplotdict:
            pfmethods = checkplotdict['pfmethods']
        else:
            pfmethods = []
            for cpkey in checkplotdict:
                for pfkey in PFMETHODS:
                    if pfkey in cpkey:
                        pfmethods.append(pfkey)

        for lspt in pfmethods:

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
                lspt.split('-')[1],  # this splits '<pfindex>-<pfmethod>'
                0,
                xtimes, xmags, xerrs,
                operiod, oepoch,
                phasewrap=ophasewrap,
                phasesort=ophasesort,
                phasebin=ophasebin,
                minbinelems=ominbinelems,
                plotxlim=oplotxlim,
                magsarefluxes=magsarefluxes,
                verbose=verbose,
                override_pfmethod=lspt
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
          fast_mode=False,
          lcfname=None,
          cprenorm=False,
          lclistpkl=None,
          nbrradiusarcsec=60.0,
          maxnumneighbors=5,
          makeneighborlcs=True,
          gaia_max_timeout=60.0,
          gaia_mirror='cds',
          xmatchinfo=None,
          xmatchradiusarcsec=3.0,
          minobservations=99,
          sigclip=10.0,
          lcformat='hat-sql',
          timecols=None,
          magcols=None,
          errcols=None,
          skipdone=False,
          done_callback=None,
          done_callback_args=None,
          done_callback_kwargs=None):
    '''This runs a checkplot for the given period-finding result pickle
    produced by runpf.

    Args
    ----

    `pfpickle` is the filename of the pickle created by lcproc.runpf. If this is
    None, the checkplot will be made anyway, but no phased LC information will
    be collected into the output checkplot pickle. This can be useful for just
    collecting GAIA and other external information and making LC plots for an
    object.

    `outdir` is the directory to which the output checkplot pickle will be
    written.

    `lcbasedir` is the base directory where the light curves are located.

    `fast_mode` tries to speed up hits to external services. If this is True,
    the following kwargs will be set for calls to checkplot.checkplot_pickle:

    skyview_timeout = 10.0
    skyview_retry_failed = False
    simbad_search = False
    dust_timeout = 10.0
    gaia_submit_timeout = 5.0
    gaia_max_timeout = 5.0
    gaia_submit_tries = 1
    complete_query_later = False

    `lcfname` is usually None because we get the LC filename from the
    pfpickle. If pfpickle is None, however, lcfname is used instead. It will
    also be used as an override if it's provided instead of whatever the lcfname
    in pfpickle is.

    `cprenorm` is True if the light curves should be renormalized by
    checkplot.checkplot_pickle. This is set to False by default because we do
    our own normalization in this function using the light curve's registered
    normalization function and pass the normalized times, mags, errs to the
    checkplot.checkplot_pickle function.

    `lclistpkl` is the name of a pickle or the actual dict produced by
    lcproc.make_lclist. This is used to gather neighbor information.

    `nbrradiusarcsec` is the maximum radius in arcsec around the object which
    will be searched for any neighbors in lclistpkl.

    `maxnumneighbors` is the maximum number of neighbors that will be processed.

    `xmatchinfo` is the pickle or the actual dict containing external catalog
    information for cross-matching.

    `xmatchradiusarcsec` is the maximum match distance in arcseconds for
    cross-matching.

    `minobservations` is the minimum number of observations required to process
    the light curve.

    `sigclip` is the sigma-clip to apply to the light curve.

    `lcformat` is a key from the LCFORM dict to use when reading the light
    curves.

    `timecols` is a list of time columns from the light curve to process.

    `magcols` is a list of mag columns from the light curve to process.

    `errcols` is a list of err columns from the light curve to process.

    `skipdone` indicates if this function will skip creating checkplots that
    already exist corresponding to the current objectid and magcol. If
    `skipdone` is set to True, this will be done.

    `done_callback` is used to provide a function to execute after the checkplot
    pickles are generated. This is useful if you want to stream the results of
    checkplot making to some other process, e.g. directly running an ingestion
    into an LCC-Server collection. The function will always get the list of the
    generated checkplot pickles as its first arg, and all of the kwargs for
    runcp in the kwargs dict. Additional args and kwargs can be provided by
    giving a list in the `done_callbacks_args` kwarg and a dict in the
    `done_callbacks_kwargs` kwarg.

    NOTE: the function you pass in here should be pickleable by normal Python if
    you want to use it with the parallel_cp and parallel_cp_lcdir functions
    below.

    Returns
    -------

    a list of checkplot pickle filenames with one element for each (timecol,
    magcol, errcol) combination provided in the default lcformat config or in
    the timecols, magcols, errcols kwargs.

    '''

    if lcformat not in LCFORM or lcformat is None:
        LOGERROR('unknown light curve format specified: %s' % lcformat)
        return None

    if pfpickle is not None:

        if pfpickle.endswith('.gz'):
            infd = gzip.open(pfpickle,'rb')
        else:
            infd = open(pfpickle,'rb')

        pfresults = pickle.load(infd)

        infd.close()

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

    if ((lcfname is not None or pfpickle is None) and os.path.exists(lcfname)):

        lcfpath = lcfname
        objectid = None

    else:

        if pfpickle is not None:

            objectid = pfresults['objectid']
            lcfbasename = pfresults['lcfbasename']
            lcfsearchpath = os.path.join(lcbasedir, lcfbasename)

            if os.path.exists(lcfsearchpath):
                lcfpath = lcfsearchpath

            elif lcfname is not None and os.path.exists(lcfname):
                lcfpath = lcfname

            else:
                LOGERROR('could not find light curve for '
                         'pfresult %s, objectid %s, '
                         'used search path: %s, lcfname kwarg: %s' %
                         (pfpickle, objectid, lcfsearchpath, lcfname))
                return None

        else:

            LOGERROR("no light curve provided and pfpickle is None, "
                     "can't continue")
            return None

    lcdict = readerfunc(lcfpath)

    # this should handle lists/tuples being returned by readerfunc
    # we assume that the first element is the actual lcdict
    # FIXME: figure out how to not need this assumption
    if ( (isinstance(lcdict, (list, tuple))) and
         (isinstance(lcdict[0], dict)) ):
        lcdict = lcdict[0]

    # get the object ID from the light curve if pfpickle is None or we used
    # lcfname directly
    if objectid is None:

        if 'objectid' in lcdict:
            objectid = lcdict['objectid']
        elif ('objectid' in lcdict['objectinfo'] and
              lcdict['objectinfo']['objectid']):
            objectid = lcdict['objectinfo']['objectid']
        elif 'hatid' in lcdict['objectinfo'] and lcdict['objectinfo']['hatid']:
            objectid = lcdict['objectinfo']['hatid']
        else:
            objectid = uuid.uuid4().hex[:5]
            LOGWARNING('no objectid found for this object, '
                       'generated a random one: %s' % objectid)

    # normalize using the special function if specified
    if normfunc is not None:
        lcdict = normfunc(lcdict)

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

        # get all the period-finder results from this magcol
        if pfpickle is not None:

            if 'pfmethods' in pfresults[mcol]:
                pflist = [
                    pfresults[mcol][x] for x in
                    pfresults[mcol]['pfmethods'] if
                    len(pfresults[mcol][x].keys()) > 0
                ]
            else:
                pflist = []
                for pfm in PFMETHODS:
                    if (pfm in pfresults[mcol] and
                        len(pfresults[mcol][pfm].keys()) > 0):
                        pflist.append(pfresults[mcol][pfm])

        # special case of generating a checkplot with no phased LCs
        else:
            pflist = []

        # generate the output filename
        outfile = os.path.join(outdir,
                               'checkplot-%s-%s.pkl' % (
                                   squeeze(objectid).replace(' ','-'),
                                   mcol
                               ))

        if skipdone and os.path.exists(outfile):
            LOGWARNING('skipdone = True and '
                       'checkplot for this objectid/magcol combination '
                       'exists already: %s, skipping...' % outfile)
            return outfile

        # make sure the checkplot has a valid objectid
        if 'objectid' not in lcdict['objectinfo']:
            lcdict['objectinfo']['objectid'] = objectid

        # normalize here if not using special normalization
        if normfunc is None:
            ntimes, nmags = normalize_magseries(
                times, mags,
                magsarefluxes=magsarefluxes
            )
            xtimes, xmags, xerrs = ntimes, nmags, errs
        else:
            xtimes, xmags, xerrs = times, mags, errs

        # generate the checkplotdict
        cpd = checkplot.checkplot_dict(
            pflist,
            xtimes, xmags, xerrs,
            objectinfo=lcdict['objectinfo'],
            gaia_max_timeout=gaia_max_timeout,
            gaia_mirror=gaia_mirror,
            lclistpkl=lclistpkl,
            nbrradiusarcsec=nbrradiusarcsec,
            maxnumneighbors=maxnumneighbors,
            xmatchinfo=xmatchinfo,
            xmatchradiusarcsec=xmatchradiusarcsec,
            sigclip=sigclip,
            mindet=minobservations,
            verbose=False,
            fast_mode=fast_mode,
            magsarefluxes=magsarefluxes,
            normto=cprenorm  # we've done the renormalization already, so this
                             # should be False by default. just messes up the
                             # plots otherwise, destroying LPVs in particular
        )

        if makeneighborlcs:

            # include any neighbor information as well
            cpdupdated = update_checkplotdict_nbrlcs(
                cpd,
                tcol, mcol, ecol,
                lcformat=lcformat,
                verbose=False
            )

        else:

            cpdupdated = cpd

        # write the update checkplot dict to disk
        cpf = checkplot._write_checkplot_picklefile(
            cpdupdated,
            outfile=outfile,
            protocol=pickle.HIGHEST_PROTOCOL,
            outgzip=False
        )

        cpfs.append(cpf)

    #
    # done with checkplot making
    #

    LOGINFO('done with %s -> %s' % (objectid, repr(cpfs)))
    if done_callback is not None:

        if (done_callback_args is not None and
            isinstance(done_callback_args,list)):
            done_callback_args = tuple([cpfs] + done_callback_args)

        else:
            done_callback_args = (cpfs,)

        if (done_callback_kwargs is not None and
            isinstance(done_callback_kwargs, dict)):
            done_callback_kwargs.update(dict(
                fast_mode=fast_mode,
                lcfname=lcfname,
                cprenorm=cprenorm,
                lclistpkl=lclistpkl,
                nbrradiusarcsec=nbrradiusarcsec,
                maxnumneighbors=maxnumneighbors,
                gaia_max_timeout=gaia_max_timeout,
                gaia_mirror=gaia_mirror,
                xmatchinfo=xmatchinfo,
                xmatchradiusarcsec=xmatchradiusarcsec,
                minobservations=minobservations,
                sigclip=sigclip,
                lcformat=lcformat,
                fileglob=fileglob,
                readerfunc=readerfunc,
                normfunc=normfunc,
                magsarefluxes=magsarefluxes,
                timecols=timecols,
                magcols=magcols,
                errcols=errcols,
                skipdone=skipdone,
            ))

        else:
            done_callback_kwargs = dict(
                fast_mode=fast_mode,
                lcfname=lcfname,
                cprenorm=cprenorm,
                lclistpkl=lclistpkl,
                nbrradiusarcsec=nbrradiusarcsec,
                maxnumneighbors=maxnumneighbors,
                gaia_max_timeout=gaia_max_timeout,
                gaia_mirror=gaia_mirror,
                xmatchinfo=xmatchinfo,
                xmatchradiusarcsec=xmatchradiusarcsec,
                minobservations=minobservations,
                sigclip=sigclip,
                lcformat=lcformat,
                fileglob=fileglob,
                readerfunc=readerfunc,
                normfunc=normfunc,
                magsarefluxes=magsarefluxes,
                timecols=timecols,
                magcols=magcols,
                errcols=errcols,
                skipdone=skipdone,
            )

        # fire the callback
        try:
            done_callback(*done_callback_args, **done_callback_kwargs)
            LOGINFO('callback fired successfully for %r' % cpfs)
        except Exception as e:
            LOGEXCEPTION('callback function failed for %r' % cpfs)

    # at the end, return the list of checkplot files generated
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



def parallel_cp(pfpicklelist,
                outdir,
                lcbasedir,
                fast_mode=False,
                lcfnamelist=None,
                cprenorm=False,
                lclistpkl=None,
                gaia_max_timeout=60.0,
                gaia_mirror='cds',
                nbrradiusarcsec=60.0,
                maxnumneighbors=5,
                makeneighborlcs=True,
                xmatchinfo=None,
                xmatchradiusarcsec=3.0,
                sigclip=10.0,
                minobservations=99,
                liststartindex=None,
                maxobjects=None,
                lcformat='hat-sql',
                timecols=None,
                magcols=None,
                errcols=None,
                skipdone=False,
                nworkers=NCPUS,
                done_callback=None,
                done_callback_args=None,
                done_callback_kwargs=None):
    '''This drives the parallel execution of runcp for a list of periodfinding
    result pickles.

    '''

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # handle the start and end indices
    if (liststartindex is not None) and (maxobjects is None):
        pfpicklelist = pfpicklelist[liststartindex:]
        if lcfnamelist is not None:
            lcfnamelist = lcfnamelist[liststartindex:]

    elif (liststartindex is None) and (maxobjects is not None):
        pfpicklelist = pfpicklelist[:maxobjects]
        if lcfnamelist is not None:
            lcfnamelist = lcfnamelist[:maxobjects]

    elif (liststartindex is not None) and (maxobjects is not None):
        pfpicklelist = (
            pfpicklelist[liststartindex:liststartindex+maxobjects]
        )
        if lcfnamelist is not None:
            lcfnamelist = lcfnamelist[liststartindex:liststartindex+maxobjects]

    # if the lcfnamelist is not provided, create a dummy
    if lcfnamelist is None:
        lcfnamelist = [None]*len(pfpicklelist)

    tasklist = [(x, outdir, lcbasedir,
                 {'lcformat':lcformat,
                  'lcfname':y,
                  'timecols':timecols,
                  'magcols':magcols,
                  'errcols':errcols,
                  'lclistpkl':lclistpkl,
                  'gaia_max_timeout':gaia_max_timeout,
                  'gaia_mirror':gaia_mirror,
                  'nbrradiusarcsec':nbrradiusarcsec,
                  'maxnumneighbors':maxnumneighbors,
                  'makeneighborlcs':makeneighborlcs,
                  'xmatchinfo':xmatchinfo,
                  'xmatchradiusarcsec':xmatchradiusarcsec,
                  'sigclip':sigclip,
                  'minobservations':minobservations,
                  'skipdone':skipdone,
                  'cprenorm':cprenorm,
                  'fast_mode':fast_mode,
                  'done_callback':done_callback,
                  'done_callback_args':done_callback_args,
                  'done_callback_kwargs':done_callback_kwargs}) for
                x,y in zip(pfpicklelist, lcfnamelist)]

    resultfutures = []
    results = []

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(runcp_worker, tasklist)

    results = [x for x in resultfutures]

    executor.shutdown()
    return results



def parallel_cp_pfdir(pfpickledir,
                      outdir,
                      lcbasedir,
                      fast_mode=False,
                      cprenorm=False,
                      lclistpkl=None,
                      gaia_max_timeout=60.0,
                      gaia_mirror='cds',
                      nbrradiusarcsec=60.0,
                      maxnumneighbors=5,
                      makeneighborlcs=True,
                      xmatchinfo=None,
                      xmatchradiusarcsec=3.0,
                      sigclip=10.0,
                      minobservations=99,
                      maxobjects=None,
                      pfpickleglob='periodfinding-*.pkl*',
                      lcformat='hat-sql',
                      timecols=None,
                      magcols=None,
                      errcols=None,
                      skipdone=False,
                      nworkers=32,
                      done_callback=None,
                      done_callback_args=None,
                      done_callback_kwargs=None):

    '''This drives the parallel execution of runcp for a directory of
    periodfinding pickles.

    '''

    pfpicklelist = sorted(glob.glob(os.path.join(pfpickledir, pfpickleglob)))

    LOGINFO('found %s period-finding pickles, running cp...' %
            len(pfpicklelist))

    return parallel_cp(pfpicklelist,
                       outdir,
                       lcbasedir,
                       fast_mode=fast_mode,
                       lclistpkl=lclistpkl,
                       nbrradiusarcsec=nbrradiusarcsec,
                       gaia_max_timeout=gaia_max_timeout,
                       gaia_mirror=gaia_mirror,
                       maxnumneighbors=maxnumneighbors,
                       makeneighborlcs=makeneighborlcs,
                       xmatchinfo=xmatchinfo,
                       xmatchradiusarcsec=xmatchradiusarcsec,
                       sigclip=sigclip,
                       minobservations=minobservations,
                       cprenorm=cprenorm,
                       maxobjects=maxobjects,
                       lcformat=lcformat,
                       timecols=timecols,
                       magcols=magcols,
                       errcols=errcols,
                       skipdone=skipdone,
                       nworkers=nworkers,
                       done_callback=done_callback,
                       done_callback_args=done_callback_args,
                       done_callback_kwargs=done_callback_kwargs)



###############################
## ADDING INFO TO CHECKPLOTS ##
###############################

def xmatch_cplist_external_catalogs(cplist,
                                    xmatchpkl,
                                    xmatchradiusarcsec=2.0,
                                    updateexisting=True,
                                    resultstodir=None):
    '''This xmatches external catalogs to a collection of checkplots in cpdir.

    cplist is a list of checkplot files to process.

    xmatchpkl is a pickle prepared with the
    checkplot.load_xmatch_external_catalogs function.

    xmatchradiusarcsec is the match radius to use in arcseconds.

    updateexisting if True, will only update the xmatch dict in each checkplot
    pickle. If False, will overwrite the xmatch dict with results from the
    current run.

    If resultstodir is not None, then it must be a directory to write the
    resulting checkplots after xmatch is done to. This can be used to keep the
    original checkplots in pristine condition for some reason.

    '''

    # load the external catalog
    with open(xmatchpkl,'rb') as infd:
        xmd = pickle.load(infd)

    # match each object. this is fairly fast, so this is not parallelized at the
    # moment

    status_dict = {}

    for cpf in cplist:

        cpd = _read_checkplot_picklefile(cpf)

        try:

            # match in place
            xmatch_external_catalogs(cpd, xmd,
                                     xmatchradiusarcsec=xmatchradiusarcsec,
                                     updatexmatch=updateexisting)

            for xmi in cpd['xmatch']:

                if cpd['xmatch'][xmi]['found']:
                    LOGINFO('checkplot %s: %s matched to %s, '
                            'match dist: %s arcsec' %
                            (os.path.basename(cpf),
                             cpd['objectid'],
                             cpd['xmatch'][xmi]['name'],
                             cpd['xmatch'][xmi]['distarcsec']))

                if not resultstodir:
                    outcpf = checkplot._write_checkplot_picklefile(cpd,
                                                                   outfile=cpf)
                else:
                    xcpf = os.path.join(resultstodir, os.path.basename(cpf))
                    outcpf = checkplot._write_checkplot_picklefile(cpd,
                                                                   outfile=xcpf)

            status_dict[cpf] = outcpf

        except Exception as e:

            LOGEXCEPTION('failed to match objects for %s' % cpf)
            status_dict[cpf] = None

    return status_dict



def xmatch_cpdir_external_catalogs(cpdir,
                                   xmatchpkl,
                                   cpfileglob='checkplot-*.pkl*',
                                   xmatchradiusarcsec=2.0,
                                   updateexisting=True,
                                   resultstodir=None):
    '''This xmatches external catalogs to all checkplots in cpdir.

    All arguments are the same as for xmatch_cplist_external_catalogs, except
    for:

    cpdir is the directory to search in for checkplots.

    cpfileglob is the fileglob to use in searching for checkplots.

    '''

    cplist = glob.glob(os.path.join(cpdir, cpfileglob))

    return xmatch_cplist_external_catalogs(
        cplist,
        xmatchpkl,
        xmatchradiusarcsec=xmatchradiusarcsec,
        updateexisting=updateexisting,
        resultstodir=resultstodir
    )



CMD_LABELS = {
    'umag':'U',
    'bmag':'B',
    'vmag':'V',
    'rmag':'R',
    'imag':'I',
    'jmag':'J',
    'hmag':'H',
    'kmag':'K_s',
    'sdssu':'u',
    'sdssg':'g',
    'sdssr':'r',
    'sdssi':'i',
    'sdssz':'z',
    'gaiamag':'G',
    'gaia_absmag':'M_G',
    'rpmj':'\mathrm{RPM}_{J}',
}



def colormagdiagram_cplist(cplist,
                           outpkl,
                           color_mag1=['gaiamag','sdssg'],
                           color_mag2=['kmag','kmag'],
                           yaxis_mag=['gaia_absmag','rpmj']):
    '''This makes a CMD for all objects in cplist.

    cplist is a list of checkplot pickles to process.

    color_mag1 and color_mag2 are lists of keys in each checkplot's objectinfo
    dict to use for the color x-axes: color = color_mag1 - color_mag2

    yaxis_mag is a list of keys in each checkplot's objectinfo dict to use as
    the (absolute) magnitude y axes.

    '''
    # first, we'll collect all of the info
    cplist_objectids = []
    cplist_mags = []
    cplist_colors = []

    for cpf in cplist:

        cpd = _read_checkplot_picklefile(cpf)
        cplist_objectids.append(cpd['objectid'])

        thiscp_mags = []
        thiscp_colors = []

        for cm1, cm2, ym in zip(color_mag1, color_mag2, yaxis_mag):

            if (ym in cpd['objectinfo'] and
                cpd['objectinfo'][ym] is not None):
                thiscp_mags.append(cpd['objectinfo'][ym])
            else:
                thiscp_mags.append(np.nan)

            if (cm1 in cpd['objectinfo'] and
                cpd['objectinfo'][cm1] is not None and
                cm2 in cpd['objectinfo'] and
                cpd['objectinfo'][cm2] is not None):
                thiscp_colors.append(cpd['objectinfo'][cm1] -
                                     cpd['objectinfo'][cm2])
            else:
                thiscp_colors.append(np.nan)

        cplist_mags.append(thiscp_mags)
        cplist_colors.append(thiscp_colors)


    # convert these to arrays
    cplist_objectids = np.array(cplist_objectids)
    cplist_mags = np.array(cplist_mags)
    cplist_colors = np.array(cplist_colors)

    # prepare the outdict
    cmddict = {'objectids':cplist_objectids,
               'mags':cplist_mags,
               'colors':cplist_colors,
               'color_mag1':color_mag1,
               'color_mag2':color_mag2,
               'yaxis_mag':yaxis_mag}

    # save the pickled figure and dict for fast retrieval later
    with open(outpkl,'wb') as outfd:
        pickle.dump(cmddict, outfd, pickle.HIGHEST_PROTOCOL)

    plt.close('all')

    return cmddict



def colormagdiagram_cpdir(cpdir,
                          outpkl,
                          cpfileglob='checkplot*.pkl*',
                          color_mag1=['gaiamag','sdssg'],
                          color_mag2=['kmag','kmag'],
                          yaxis_mag=['gaia_absmag','rpmj']):
    '''This makes a CMD for all objects in cpdir.

    All params are the same as for colormagdiagram_cplist, except for:

    cpfileglob: the fileglob to use to find the checkplot pickles in cpdir.

    '''

    cplist = glob.glob(os.path.join(cpdir, cpfileglob))

    return colormagdiagram_cplist(cplist,
                                  outpkl,
                                  color_mag1=color_mag1,
                                  color_mag2=color_mag2,
                                  yaxis_mag=yaxis_mag)



def add_cmd_to_checkplot(cpx, cmdpkl,
                         require_cmd_magcolor=True,
                         save_cmd_pngs=False):
    '''This adds CMD figures to the checkplot dict or pickle cpx.

    Looks up the CMDs in the cmdpkl, adds the object in the checkplot as a
    gold(-ish) star in the plot, and then saves the figure to a base64 encoded
    PNG, which can then be read and used by the checkplotserver.

    If require_cmd_magcolor is True, a plot will not be made if the color and
    mag keys required by the CMD are not present or are nan in this checkplot's
    objectinfo dict.

    If save_cmd_png = True, then will save the CMD plots made as PNGs to the
    same directory as cpx. If cpx is a dict, will save them to the current
    directory.

    '''

    # get the checkplot
    if isinstance(cpx, str) and os.path.exists(cpx):
        cpdict = _read_checkplot_picklefile(cpx)
    elif isinstance(cpx, dict):
        cpdict = cpx
    else:
        LOGERROR('unknown type of checkplot provided as the cpx arg')
        return None

    # get the CMD
    if isinstance(cmdpkl, str) and os.path.exists(cmdpkl):
        with open(cmdpkl, 'rb') as infd:
            cmd = pickle.load(infd)
    elif isinstance(cmdpkl, dict):
        cmd = cmdpkl


    cpdict['colormagdiagram'] = {}

    # get the mags and colors from the CMD dict
    cplist_mags = cmd['mags']
    cplist_colors = cmd['colors']

    # now make the CMD plots for each color-mag combination in the CMD
    for c1, c2, ym, ind in zip(cmd['color_mag1'],
                               cmd['color_mag2'],
                               cmd['yaxis_mag'],
                               range(len(cmd['color_mag1']))):

        # get these from the checkplot for this object
        if (c1 in cpdict['objectinfo'] and
            cpdict['objectinfo'][c1] is not None):
            c1mag = cpdict['objectinfo'][c1]
        else:
            c1mag = np.nan

        if (c2 in cpdict['objectinfo'] and
            cpdict['objectinfo'][c2] is not None):
            c2mag = cpdict['objectinfo'][c2]
        else:
            c2mag = np.nan

        if (ym in cpdict['objectinfo'] and
            cpdict['objectinfo'][ym] is not None):
            ymmag = cpdict['objectinfo'][ym]
        else:
            ymmag = np.nan

        if (require_cmd_magcolor and
            not (np.isfinite(c1mag) and
                 np.isfinite(c2mag) and
                 np.isfinite(ymmag))):

            LOGWARNING("required color: %s-%s or mag: %s are not "
                       "in this checkplot's objectinfo dict "
                       "(objectid: %s), skipping CMD..." %
                       (c1, c2, ym, cpdict['objectid']))
            continue

        # make the CMD for this color-mag combination
        try:

            thiscmd_title = r'%s-%s/%s' % (CMD_LABELS[c1],
                                           CMD_LABELS[c2],
                                           CMD_LABELS[ym])

            # make the scatter plot
            fig = plt.figure(figsize=(10,8))
            plt.plot(cplist_colors[:,ind],
                     cplist_mags[:,ind],
                     rasterized=True,
                     marker='o',
                     linestyle='none',
                     mew=0,
                     ms=3)

            # put this object on the plot
            plt.plot([c1mag - c2mag], [ymmag],
                     ms=20,
                     color='#b0ff05',
                     marker='*',
                     mew=0)

            plt.xlabel(r'$%s - %s$' % (CMD_LABELS[c1], CMD_LABELS[c2]))
            plt.ylabel(r'$%s$' % CMD_LABELS[ym])
            plt.title('%s - $%s$ CMD' % (cpdict['objectid'], thiscmd_title))
            plt.gca().invert_yaxis()

            # now save the figure to strio and put it back in the checkplot
            cmdpng = strio()
            plt.savefig(cmdpng, bbox_inches='tight',
                        pad_inches=0.0, format='png')
            cmdpng.seek(0)
            cmdb64 = base64.b64encode(cmdpng.read())
            cmdpng.close()

            plt.close('all')
            plt.gcf().clear()

            cpdict['colormagdiagram']['%s-%s/%s' % (c1,c2,ym)] = cmdb64

            # if we're supposed to export to PNG, do so
            if save_cmd_pngs:

                if isinstance(cpx, str):
                    outpng = os.path.join(os.path.dirname(cpx),
                                          'cmd-%s-%s-%s.%s.png' %
                                          (cpdict['objectid'],
                                           c1,c2,ym))
                else:
                    outpng = 'cmd-%s-%s-%s.%s.png' % (cpdict['objectid'],
                                                      c1,c2,ym)

                checkplot._base64_to_file(cmdb64, outpng)

        except Exception as e:
            LOGEXCEPTION('CMD for %s-%s/%s does not exist in %s, skipping...' %
                         (c1, c2, ym, cmdpkl))
            continue


    #
    # end of making CMDs
    #

    if isinstance(cpx, str):
        cpf = _write_checkplot_picklefile(cpdict, outfile=cpx, protocol=4)
        return cpf
    elif isinstance(cpx, dict):
        return cpdict



def add_cmds_cplist(cplist, cmdpkl,
                    require_cmd_magcolor=True,
                    save_cmd_pngs=False):
    '''This adds CMDs for each object in cplist.

    NOTE: If the object doesn't have the color and mag keys required in its
    objectinfo dict for the CMD plot, it won't appear on the CMD.

    '''

    # load the CMD first to save on IO
    with open(cmdpkl,'rb') as infd:
        cmd = pickle.load(infd)

    for cpf in cplist:

        add_cmd_to_checkplot(cpf, cmd,
                             require_cmd_magcolor=require_cmd_magcolor,
                             save_cmd_pngs=save_cmd_pngs)



def add_cmds_cpdir(cpdir, cmdpkl,
                   cpfileglob='checkplot*.pkl*',
                   require_cmd_magcolor=True,
                   save_cmd_pngs=False):
    '''This adds CMDs for each object in cpdir.

    All params are the same as for add_cmds_cplist, except for:

    cpfileglob: the fileglob to use to find the checkplot pickles in cpdir.

    '''

    cplist = glob.glob(os.path.join(cpdir, cpfileglob))

    return add_cmds_cplist(cplist,
                           cmdpkl,
                           require_cmd_magcolor=require_cmd_magcolor,
                           save_cmd_pngs=save_cmd_pngs)



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



##################################
## LIGHT CURVE DETRENDING - EPD ##
##################################

def apply_epd_magseries(lcfile,
                        timecol,
                        magcol,
                        errcol,
                        externalparams,
                        lcformat='hat-sql',
                        magsarefluxes=False,
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

    readerfunc = LCFORM[lcformat][1]
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
    task[6] = magsarefluxes
    task[7] = epdsmooth_sigclip
    task[8] = epdsmooth_windowsize
    task[9] = epdsmooth_func
    task[10] = epdsmooth_extraparams

    '''

    (lcfile, timecol, magcol, errcol,
     externalparams, lcformat, magsarefluxes,
     epdsmooth_sigclip, epdsmooth_windowsize,
     epdsmooth_func, epdsmooth_extraparams) = task

    try:

        epd = apply_epd_magseries(lcfile,
                                  timecol,
                                  magcol,
                                  errcol,
                                  externalparams,
                                  lcformat=lcformat,
                                  magsarefluxes=magsarefluxes,
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

    # get the default time, mag, err cols if not provided
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

    outdict = {}

    # run by magcol
    for t, m, e in zip(timecols, magcols, errcols):

        tasks = [(x, t, m, e, externalparams, lcformat,
                  magsarefluxes, epdsmooth_sigclip, epdsmooth_windowsize,
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

    # get the default time, mag, err cols if not provided
    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

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



##################################
## LIGHT CURVE DETRENDING - TFA ##
##################################

def collect_tfa_stats(task):
    '''
    This is a parallel worker to gather LC stats.

    task[0] = lcfile
    task[1] = lcformat
    task[2] = timecols
    task[3] = magcols
    task[4] = errcols
    task[5] = custom_bandpasses

    '''

    try:

        lcfile, lcformat, timecols, magcols, errcols, custom_bandpasses = task

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

        #
        # collect the necessary stats for this light curve
        #

        # 1. number of observations
        # 2. median mag
        # 3. eta_normal
        # 4. MAD
        # 5. objectid
        # 6. get mags and colors from objectinfo if there's one in lcdict

        if 'objectid' in lcdict:
            objectid = lcdict['objectid']
        elif 'objectinfo' in lcdict and 'objectid' in lcdict['objectinfo']:
            objectid = lcdict['objectinfo']['objectid']
        elif 'objectinfo' in lcdict and 'hatid' in lcdict['objectinfo']:
            objectid = lcdict['objectinfo']['hatid']
        else:
            LOGERROR('no objectid present in lcdict for LC %s, '
                     'using filename prefix as objectid' % lcfile)
            objectid = os.path.splitext(os.path.basename(lcfile))[0]

        if 'objectinfo' in lcdict:

            colorfeat = starfeatures.color_features(
                lcdict['objectinfo'],
                deredden=False,
                custom_bandpasses=custom_bandpasses
            )

        else:
            LOGERROR('no objectinfo dict in lcdict, '
                     'could not get magnitudes for LC %s, '
                     'cannot use for TFA template ensemble' %
                     lcfile)
            return None


        # this is the initial dict
        resultdict = {'objectid':objectid,
                      'ra':lcdict['objectinfo']['ra'],
                      'decl':lcdict['objectinfo']['decl'],
                      'colorfeat':colorfeat,
                      'lcfpath':os.path.abspath(lcfile),
                      'lcformat':lcformat,
                      'timecols':timecols,
                      'magcols':magcols,
                      'errcols':errcols}

        for tcol, mcol, ecol in zip(timecols, magcols, errcols):

            try:

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

                # get the variability features for this object
                varfeat = varfeatures.all_nonperiodic_features(
                    times, mags, errs
                )

                resultdict[mcol] = varfeat

            except Exception as e:

                LOGEXCEPTION('%s, magcol: %s, probably ran into all-nans' %
                             (lcfile, mcol))
                resultdict[mcol] = {'ndet':0,
                                    'mad':np.nan,
                                    'eta_normal':np.nan}


        return resultdict

    except Exception as e:

        LOGEXCEPTION('could not execute get_tfa_stats for task: %s' %
                     repr(task))
        return None



def reform_templatelc_for_tfa(task):
    '''
    This is a parallel worker that reforms light curves for TFA.

    task[0] = lcfile
    task[1] = lcformat
    task[2] = timecol
    task[3] = magcol
    task[4] = errcol
    task[5] = timebase
    task[6] = interpolate_type
    task[7] = sigclip

    '''

    try:

        (lcfile, lcformat,
         tcol, mcol, ecol,
         timebase, interpolate_type, sigclip) = task

        if lcformat not in LCFORM or lcformat is None:
            LOGERROR('unknown light curve format specified: %s' % lcformat)
            return None

        (fileglob, readerfunc, dtimecols, dmagcols,
         derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

        # get the LC into a dict
        lcdict = readerfunc(lcfile)

        # this should handle lists/tuples being returned by readerfunc
        # we assume that the first element is the actual lcdict
        # FIXME: figure out how to not need this assumption
        if ( (isinstance(lcdict, (list, tuple))) and
             (isinstance(lcdict[0], dict)) ):
            lcdict = lcdict[0]

        outdict = {}

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

        #
        # now we'll do: 1. sigclip, 2. reform to timebase, 3. renorm to zero
        #

        # 1. sigclip as requested
        stimes, smags, serrs = sigclip_magseries(times,
                                                 mags,
                                                 errs,
                                                 sigclip=sigclip)

        # 2. now, we'll renorm to the timebase
        mags_interpolator = spi.interp1d(stimes, smags,
                                         kind=interpolate_type,
                                         fill_value='extrapolate')
        errs_interpolator = spi.interp1d(stimes, serrs,
                                         kind=interpolate_type,
                                         fill_value='extrapolate')

        interpolated_mags = mags_interpolator(timebase)
        interpolated_errs = errs_interpolator(timebase)

        # 3. renorm to zero
        magmedian = np.median(interpolated_mags)

        renormed_mags = interpolated_mags - magmedian

        # update the dict
        outdict = {'mags':renormed_mags,
                   'errs':interpolated_errs,
                   'origmags':interpolated_mags}

        #
        # done with this magcol
        #
        return outdict

    except Exception as e:

        LOGEXCEPTION('reform LC task failed: %s' % repr(task))
        return None



def tfa_templates_lclist(
        lclist,
        outfile=None,
        target_template_frac=0.1,
        max_target_frac_obs=0.25,
        min_template_number=10,
        max_template_number=1000,
        max_rms=0.15,
        max_mult_above_magmad=1.5,
        max_mult_above_mageta=1.5,
        mag_bandpass='sdssr',
        custom_bandpasses=None,
        mag_bright_limit=10.0,
        mag_faint_limit=12.0,
        template_sigclip=5.0,
        template_interpolate='nearest',
        lcformat='hat-sql',
        timecols=None,
        magcols=None,
        errcols=None,
        nworkers=NCPUS,
        maxworkertasks=1000,
):
    '''This selects template objects for TFA.

    lclist is a list of light curves to use as input to generate the template
    set.

    outfile is a pickle filename to which the TFA template list will be written
    to.

    target_template_frac is the fraction of total objects in lclist to use for
    the number of templates.

    max_target_frac_obs sets the number of templates to generate if the number
    of observations for the light curves is smaller than the number of objects
    in the collection. The number of templates will be set to this fraction of
    the number of observations if this is the case.

    min_template_number is the minimum number of templates to generate.

    max_template_number is the maximum number of templates to generate. If
    target_template_frac times the number of objects is greater than
    max_template_number, only max_template_number templates will be used.

    max_rms is the maximum light curve RMS for an object to consider it as a
    possible template ensemble member.

    max_mult_above_magmad is the maximum multiplier above the mag-RMS fit to
    consider an object as variable and thus not part of the template ensemble.

    max_mult_above_mageta is the maximum multiplier above the mag-eta (variable
    index) fit to consider an object as variable and thus not part of the
    template ensemble.

    mag_bandpass sets the key in the light curve dict's objectinfo dict to use
    as the canonical magnitude for the object and apply any magnitude limits to.

    custom_bandpasses can be used to provide any custom band name keys to the
    star feature collection function.

    mag_bright_limit sets the brightest mag for a potential member of the TFA
    template ensemble.

    mag_faint_limit sets the faintest mag for a potential member of the TFA
    template ensemble.

    template_sigclip sets the sigma-clip to be applied to the template light
    curves.

    template_interpolate sets the kwarg to pass to scipy.interpolate.interp1d to
    set the kind of interpolation to use when reforming light curves to the TFA
    template timebase.

    lcformat sets the key in LCFORM to use to read the light curves. Use the
    lcproc.register_custom_lcformat function to register a custom light curve
    format in the lcproc.LCFORM dict.

    timecols, magcols, errcols are lists of lcdict keys to use to generate the
    TFA template ensemble. These will be the light curve magnitude columns that
    TFA will be ultimately applied to by apply_tfa_magseries below.

    nworkers and maxworkertasks control the number of parallel workers and tasks
    per worker used by this function to collect light curve information and to
    reform light curves to the TFA template's timebase.

    Selection criteria for TFA template ensemble objects:

    - not variable: use a poly fit to the mag-MAD relation and eta-normal
      variability index to get nonvar objects
    - not more than 10% of the total number of objects in the field or
      maxtfatemplates at most
    - allow shuffling of the templates if the target ends up in them
    - nothing with less than the median number of observations in the field
    - sigma-clip the input time series observations
    - TODO: uniform sampling in tangent plane coordinates (we'll need ra and
      decl)

    This also determines the effective cadence that all TFA LCs will be binned
    to as the template LC with the largest number of non-nan observations will
    be used. All template LCs will be renormed to zero.

    This function returns a dict that can be passed directly to
    apply_tfa_magseries below. It can optionally produce a pickle with the same
    dict, which can also be passed to that function.

    '''

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

    LOGINFO('collecting light curve information for %s in list...' %
            len(lclist))

    # first, we'll collect the light curve info
    tasks = [(x, lcformat, timecols, magcols, errcols, custom_bandpasses) for x
             in lclist]

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
    results = pool.map(collect_tfa_stats, tasks)
    pool.close()
    pool.join()

    # now, go through the light curves

    outdict = {
        'timecols':[],
        'magcols':[],
        'errcols':[],
    }

    # for each magcol, we'll generate a separate template list
    for tcol, mcol, ecol in zip(timecols, magcols, errcols):

        if '.' in tcol:
            tcolget = tcol.split('.')
        else:
            tcolget = [tcol]

        if '.' in mcol:
            mcolget = mcol.split('.')
        else:
            mcolget = [mcol]

        # these are the containers for possible template collection LC info
        (lcmag, lcmad, lceta,
         lcndet, lcobj, lcfpaths,
         lcra, lcdecl) = [], [], [], [], [], [], [], []

        outdict['timecols'].append(tcol)
        outdict['magcols'].append(mcol)
        outdict['errcols'].append(ecol)

        # add to the collection of all light curves
        outdict[mcol] = {'collection':{'mag':[],
                                       'mad':[],
                                       'eta':[],
                                       'ndet':[],
                                       'obj':[],
                                       'lcf':[],
                                       'ra':[],
                                       'decl':[]}}

        LOGINFO('magcol: %s, collecting prospective template LC info...' %
                mcol)


        # collect the template LCs for this magcol
        for result in results:

            # we'll only append objects that have all of these elements
            try:

                thismag = result['colorfeat'][mag_bandpass]
                thismad = result[mcol]['mad']
                thiseta = result[mcol]['eta_normal']
                thisndet = result[mcol]['ndet']
                thisobj = result['objectid']
                thislcf = result['lcfpath']
                thisra = result['ra']
                thisdecl = result['decl']

                outdict[mcol]['collection']['mag'].append(thismag)
                outdict[mcol]['collection']['mad'].append(thismad)
                outdict[mcol]['collection']['eta'].append(thiseta)
                outdict[mcol]['collection']['ndet'].append(thisndet)
                outdict[mcol]['collection']['obj'].append(thisobj)
                outdict[mcol]['collection']['lcf'].append(thislcf)
                outdict[mcol]['collection']['ra'].append(thisra)
                outdict[mcol]['collection']['decl'].append(thisdecl)

                # make sure the object lies in the mag limits and RMS limits we
                # set before to try to accept it into the TFA ensemble
                if ((mag_bright_limit < thismag < mag_faint_limit) and
                    (1.4826*thismad < max_rms)):

                    lcmag.append(thismag)
                    lcmad.append(thismad)
                    lceta.append(thiseta)
                    lcndet.append(thisndet)
                    lcobj.append(thisobj)
                    lcfpaths.append(thislcf)
                    lcra.append(thisra)
                    lcdecl.append(thisdecl)

            except Exception as e:
                pass

        # make sure we have enough LCs to work on
        if len(lcobj) >= min_template_number:

            LOGINFO('magcol: %s, %s objects eligible for '
                    'template selection after filtering on mag '
                    'limits (%s, %s) and max RMS (%s)' %
                    (mcol, len(lcobj),
                     mag_bright_limit, mag_faint_limit, max_rms))

            lcmag = np.array(lcmag)
            lcmad = np.array(lcmad)
            lceta = np.array(lceta)
            lcndet = np.array(lcndet)
            lcobj = np.array(lcobj)
            lcfpaths = np.array(lcfpaths)
            lcra = np.array(lcra)
            lcdecl = np.array(lcdecl)

            sortind = np.argsort(lcmag)
            lcmag = lcmag[sortind]
            lcmad = lcmad[sortind]
            lceta = lceta[sortind]
            lcndet = lcndet[sortind]
            lcobj = lcobj[sortind]
            lcfpaths = lcfpaths[sortind]
            lcra = lcra[sortind]
            lcdecl = lcdecl[sortind]

            # 1. get the mag-MAD relation

            # this is needed for spline fitting
            # should take care of the pesky 'x must be strictly increasing' bit
            splfit_ind = np.diff(lcmag) > 0.0
            splfit_ind = np.concatenate((np.array([True]), splfit_ind))

            fit_lcmag = lcmag[splfit_ind]
            fit_lcmad = lcmad[splfit_ind]
            fit_lceta = lceta[splfit_ind]

            magmadfit = np.poly1d(np.polyfit(
                fit_lcmag,
                fit_lcmad,
                2
            ))
            magmadind = lcmad/magmadfit(lcmag) < max_mult_above_magmad

            # 2. get the mag-eta relation
            magetafit = np.poly1d(np.polyfit(
                fit_lcmag,
                fit_lceta,
                2
            ))
            magetaind = magetafit(lcmag)/lceta < max_mult_above_mageta

            # 3. get the median ndet
            median_ndet = np.median(lcndet)
            ndetind = lcndet >= median_ndet

            # form the final template ensemble
            templateind = magmadind & magetaind & ndetind

            # check again if we have enough LCs in the template
            if templateind.sum() >= min_template_number:

                LOGINFO('magcol: %s, %s objects selectable for TFA templates' %
                        (mcol, templateind.sum()))

                templatemag = lcmag[templateind]
                templatemad = lcmad[templateind]
                templateeta = lceta[templateind]
                templatendet = lcndet[templateind]
                templateobj = lcobj[templateind]
                templatelcf = lcfpaths[templateind]
                templatera = lcra[templateind]
                templatedecl = lcdecl[templateind]

                # now, check if we have no more than the required fraction of
                # TFA templates
                target_number_templates = int(target_template_frac*len(lclist))

                if target_number_templates > max_template_number:
                    target_number_templates = max_template_number

                LOGINFO('magcol: %s, selecting %s TFA templates randomly' %
                        (mcol, target_number_templates))

                # FIXME: how do we select uniformly in xi-eta?

                # select random uniform objects from the template candidates
                targetind = npr.choice(templateobj.size,
                                       target_number_templates,
                                       replace=False)

                templatemag = templatemag[targetind]
                templatemad = templatemad[targetind]
                templateeta = templateeta[targetind]
                templatendet = templatendet[targetind]
                templateobj = templateobj[targetind]
                templatelcf = templatelcf[targetind]
                templatera = templatera[targetind]
                templatedecl = templatedecl[targetind]

                # get the max ndet so far to use that LC as the timebase
                maxndetind = templatendet == templatendet.max()
                timebaselcf = templatelcf[maxndetind][0]
                timebasendet = templatendet[maxndetind][0]
                LOGINFO('magcol: %s, selected %s as template time '
                        'base LC with %s observations' %
                        (mcol, timebaselcf, timebasendet))

                timebaselcdict = readerfunc(timebaselcf)

                if ( (isinstance(timebaselcdict, (list, tuple))) and
                     (isinstance(timebaselcdict[0], dict)) ):
                    timebaselcdict = timebaselcdict[0]

                # this is the timebase to use for all of the templates
                timebase = dict_get(timebaselcdict, tcolget)

                # also check if the number of templates is longer than the
                # actual timebase of the observations. this will cause issues
                # with overcorrections and will probably break TFA
                if target_number_templates > timebasendet:

                    LOGWARNING('the number of TFA templates (%s) is '
                               'larger than the number of observations '
                               'of the time base (%s). This will likely '
                               'overcorrect all light curves to a '
                               'constant level. '
                               'Will use up to %s x timebase ndet '
                               'templates instead' %
                               (target_number_templates,
                                timebasendet,
                                max_target_frac_obs))

                    # regen the templates based on the new number
                    newmaxtemplates = int(max_target_frac_obs*timebasendet)

                    # choose this number out of the already chosen templates
                    # randomly

                    LOGWARNING('magcol: %s, re-selecting %s TFA '
                               'templates randomly' %
                               (mcol, newmaxtemplates))

                    # select random uniform objects from the template candidates
                    targetind = npr.choice(templateobj.size,
                                           newmaxtemplates,
                                           replace=False)

                    templatemag = templatemag[targetind]
                    templatemad = templatemad[targetind]
                    templateeta = templateeta[targetind]
                    templatendet = templatendet[targetind]
                    templateobj = templateobj[targetind]
                    templatelcf = templatelcf[targetind]
                    templatera = templatera[targetind]
                    templatedecl = templatedecl[targetind]

                    # get the max ndet so far to use that LC as the timebase
                    maxndetind = templatendet == templatendet.max()
                    timebaselcf = templatelcf[maxndetind][0]
                    timebasendet = templatendet[maxndetind][0]
                    LOGWARNING('magcol: %s, re-selected %s as template time '
                               'base LC with %s observations' %
                               (mcol, timebaselcf, timebasendet))

                    timebaselcdict = readerfunc(timebaselcf)

                    if ( (isinstance(timebaselcdict, (list, tuple))) and
                         (isinstance(timebaselcdict[0], dict)) ):
                        timebaselcdict = timebaselcdict[0]

                    # this is the timebase to use for all of the templates
                    timebase = dict_get(timebaselcdict, tcolget)

                LOGINFO('magcol: %s, reforming TFA template LCs to '
                        ' chosen timebase...' % mcol)

                # reform all template LCs to this time base, normalize to
                # zero, and sigclip as requested. this is a parallel op
                # first, we'll collect the light curve info
                tasks = [(x, lcformat,
                          tcol, mcol, ecol,
                          timebase, template_interpolate,
                          template_sigclip) for x
                         in templatelcf]

                pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
                results = pool.map(reform_templatelc_for_tfa, tasks)
                pool.close()
                pool.join()

                # generate a 2D array for the template magseries with dimensions
                # = (n_objects, n_lcpoints)
                template_magseries = np.array([x['mags'] for x in results])
                template_errseries = np.array([x['errs'] for x in results])

                # put everything into a templateinfo dict for this magcol
                outdict[mcol].update({
                    'timebaselcf':timebaselcf,
                    'timebase':timebase,
                    'trendfits':{'mag-mad':magmadfit,
                                 'mag-eta':magetafit},
                    'template_objects':templateobj,
                    'template_ra':templatera,
                    'template_decl':templatedecl,
                    'template_mag':templatemag,
                    'template_mad':templatemad,
                    'template_eta':templateeta,
                    'template_ndet':templatendet,
                    'template_magseries':template_magseries,
                    'template_errseries':template_errseries
                })

            # if we don't have enough, return nothing for this magcol
            else:
                LOGERROR('not enough objects meeting requested '
                         'MAD, eta, ndet conditions to '
                         'select templates for magcol: %s' % mcol)
                continue

        else:

            LOGERROR('nobjects: %s, not enough in requested mag range to '
                     'select templates for magcol: %s' % (len(lcobj),mcol))
            continue

        # make the plots for mag-MAD/mag-eta relation and fits used
        plt.plot(lcmag, lcmad, marker='o', linestyle='none', ms=1.0)
        modelmags = np.linspace(lcmag.min(), lcmag.max(), num=1000)
        plt.plot(modelmags, outdict[mcol]['trendfits']['mag-mad'](modelmags))
        plt.yscale('log')
        plt.xlabel('catalog magnitude')
        plt.ylabel('light curve MAD')
        plt.title('catalog mag vs. light curve MAD and fit')
        plt.savefig('catmag-lcmad-fit.png',bbox_inches='tight')
        plt.close('all')

        plt.plot(lcmag, lceta, marker='o', linestyle='none', ms=1.0)
        modelmags = np.linspace(lcmag.min(), lcmag.max(), num=1000)
        plt.plot(modelmags, outdict[mcol]['trendfits']['mag-eta'](modelmags))
        plt.yscale('log')
        plt.xlabel('catalog magnitude')
        plt.ylabel('light curve eta variable index')
        plt.title('catalog mag vs. light curve eta and fit')
        plt.savefig('catmag-lceta-fit.png',bbox_inches='tight')
        plt.close('all')


    #
    # end of operating on each magcol
    #

    # save the templateinfo dict to a pickle if requested
    if outfile:

        if outfile.endswith('.gz'):
            outfd = gzip.open(outfile,'wb')
        else:
            outfd = open(outfile,'wb')

        with outfd:
            pickle.dump(outdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    # return the templateinfo dict
    return outdict



def apply_tfa_magseries(lcfile,
                        timecol,
                        magcol,
                        errcol,
                        templateinfo,
                        mintemplatedist_arcmin=1.0,
                        lcformat='hat-sql',
                        interp='nearest',
                        sigclip=5.0):
    '''This applies the TFA correction to an LC given TFA template information.

    lcfile is the light curve file to apply the TFA correction to.

    timecol, magcol, errcol are the column keys in the lcdict for the LC file to
    apply the TFA correction to.

    templateinfo is either the dict produced by tfa_templates_lclist or the
    pickle produced by the same function.

    TODO: mintemplatedist_arcmin sets the minimum distance required from the
    target object for objects in the TFA template ensemble. Objects closer than
    this distance will be removed from the ensemble.

    lcformat is the LCFORM dict key for the light curve format of lcfile.

    interp is passed to scipy.interpolate.interp1d as the kind of interpolation
    to use when reforming this light curve to the timebase of the TFA templates.

    sigclip is the sigma clip to apply to this light curve before running TFA on
    it.

    This returns the filename of the light curve file generated after TFA
    applications. This is a pickle (that can be read by lcproc.read_pklc) in the
    same directory as lcfile. The magcol will be encoded in the filename, so
    each magcol in lcfile gets its own output file.

    '''

    # get the templateinfo from a pickle if necessary
    if isinstance(templateinfo,str) and os.path.exists(templateinfo):
        with open(templateinfo,'rb') as infd:
            templateinfo = pickle.load(infd)

    readerfunc = LCFORM[lcformat][1]
    lcdict = readerfunc(lcfile)

    if ((isinstance(lcdict, (tuple, list))) and
        isinstance(lcdict[0], dict)):
        lcdict = lcdict[0]

    objectid = lcdict['objectid']

    # if the object itself is in the template ensemble, remove it

    # TODO: also remove objects from the template that lie within some radius of
    # the target object (let's make this 1 arcminute by default)

    if objectid in templateinfo[magcol]['template_objects']:

        LOGWARNING('object %s found in the TFA template ensemble, removing...' %
                   objectid)

        templateind = templateinfo[magcol]['template_objects'] == objectid

        # we need to copy over this template instance
        tmagseries = templateinfo[magcol][
            'template_magseries'
        ][~templateind,:][::]

    # otherwise, get the full ensemble
    else:

        tmagseries = templateinfo[magcol][
            'template_magseries'
        ][::]

    # this is the normal matrix
    normal_matrix = np.dot(tmagseries, tmagseries.T)

    # get the inverse of the matrix
    normal_matrix_inverse = spla.pinv2(normal_matrix)

    # get the timebase from the template
    timebase = templateinfo[magcol]['timebase']

    # use this to reform the target lc in the same manner as that for a TFA
    # template LC
    reformed_targetlc = reform_templatelc_for_tfa((
        lcfile,
        lcformat,
        timecol,
        magcol,
        errcol,
        timebase,
        interp,
        sigclip
    ))

    # calculate the scalar products of the target and template magseries
    scalar_products = np.dot(tmagseries, reformed_targetlc['mags'])

    # calculate the corrections
    corrections = np.dot(normal_matrix_inverse, scalar_products)

    # finally, get the corrected time series for the target object
    corrected_magseries = (
        reformed_targetlc['origmags'] -
        np.dot(tmagseries.T, corrections)
    )

    outdict = {
        'times':timebase,
        'mags':corrected_magseries,
        'errs':reformed_targetlc['errs'],
        'mags_median':np.median(corrected_magseries),
        'mags_mad': np.median(np.abs(corrected_magseries -
                                     np.median(corrected_magseries))),
        'work':{'tmagseries':tmagseries,
                'normal_matrix':normal_matrix,
                'normal_matrix_inverse':normal_matrix_inverse,
                'scalar_products':scalar_products,
                'corrections':corrections,
                'reformed_targetlc':reformed_targetlc},
    }


    # we'll write back the tfa times and mags to the lcdict
    lcdict['tfa'] = outdict
    outfile = os.path.join(
        os.path.dirname(lcfile),
        '%s-tfa-%s-pklc.pkl' % (
            squeeze(objectid).replace(' ','-'),
            magcol
        )
    )
    with open(outfile,'wb') as outfd:
        pickle.dump(lcdict, outfd, pickle.HIGHEST_PROTOCOL)

    return outfile



def parallel_tfa_worker(task):
    '''
    This is a parallel worker for the function below.

    task[0] = lcfile
    task[1] = timecol
    task[2] = magcol
    task[3] = errcol
    task[4] = templateinfo
    task[5] = lcformat
    task[6] = interp
    task[7] = sigclip

    '''

    (lcfile, timecol, magcol, errcol,
     templateinfo, lcformat, interp, sigclip) = task

    try:

        res = apply_tfa_magseries(lcfile, timecol, magcol, errcol,
                                  templateinfo,
                                  lcformat=lcformat,
                                  interp=interp,
                                  sigclip=sigclip)
        if res:
            LOGINFO('%s -> %s TFA OK' % (lcfile, res))

    except Exception as e:

        LOGEXCEPTION('TFA failed for %s' % lcfile)
        return None



def parallel_tfa_lclist(lclist,
                        templateinfo,
                        timecols=None,
                        magcols=None,
                        errcols=None,
                        lcformat='hat-sql',
                        interp='nearest',
                        sigclip=5.0,
                        nworkers=NCPUS,
                        maxworkertasks=1000):
    '''This applies TFA in parallel to all LCs in lclist.

    lclist is a list of light curve files to apply the TFA correction to.

    templateinfo is either the dict produced by tfa_templates_lclist or the
    pickle produced by the same function.

    timecols, magcols, errcols are lists of column keys in the lcdict for each
    LC file to apply the TFA correction to. each magcol will get their own
    output TFA light curve file. If these are None, then magcols used for the
    TFA template will be re-used for TFA application.

    lcformat is the LCFORM dict key for the light curve format of lcfile.

    interp is passed to scipy.interpolate.interp1d as the kind of interpolation
    to use when reforming this light curve to the timebase of the TFA templates.

    sigclip is the sigma clip to apply to this light curve before running TFA on
    it.

    nworkers and maxworkertasks set the number of parallel workers and max tasks
    per worker used to run TFA in parallel.

    '''

    # open the templateinfo first
    if isinstance(templateinfo,str) and os.path.exists(templateinfo):
        with open(templateinfo,'rb') as infd:
            templateinfo = pickle.load(infd)

    # get the default time, mag, err cols if not provided
    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

    # override the default timecols, magcols, and errcols
    # using the ones provided to the function
    # we'll get the defaults from the templateinfo object
    if timecols is None:
        timecols = templateinfo['timecols']
    if magcols is None:
        magcols = templateinfo['magcols']
    if errcols is None:
        errcols = templateinfo['errcols']

    outdict = {}

    # run by magcol
    for t, m, e in zip(timecols, magcols, errcols):

        tasks = [(x, t, m, e, templateinfo, lcformat, interp, sigclip) for
                 x in lclist]

        pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
        results = pool.map(parallel_tfa_worker, tasks)
        pool.close()
        pool.join()

        outdict[m] = results

    return outdict



def parallel_tfa_lcdir(lcdir,
                       templateinfo,
                       lcfileglob=None,
                       timecols=None,
                       magcols=None,
                       errcols=None,
                       lcformat='hat-sql',
                       interp='nearest',
                       sigclip=5.0,
                       nworkers=NCPUS,
                       maxworkertasks=1000):
    '''This applies TFA in parallel to all LCs in lcdir.

    lcfileglob is the glob to use to find the target light curves in lcdir. If
    this is None, the default fileglob provided in the LC format registration in
    lcproc.LCFORM will be used instead.

    '''

    # open the templateinfo first
    if isinstance(templateinfo,str) and os.path.exists(templateinfo):
        with open(templateinfo,'rb') as infd:
            templateinfo = pickle.load(infd)

    # get the default time, mag, err cols if not provided
    (fileglob, readerfunc, dtimecols, dmagcols,
     derrcols, magsarefluxes, normfunc) = LCFORM[lcformat]

    # find all the files matching the lcglob in lcdir
    if lcfileglob is None:
        lcfileglob = fileglob

    lclist = sorted(glob.glob(os.path.join(lcdir, lcfileglob)))

    return parallel_tfa_lclist(
        lclist,
        templateinfo,
        timecols=timecols,
        magcols=magcols,
        errcols=errcols,
        lcformat=lcformat,
        interp=interp,
        sigclip=sigclip,
        nworkers=nworkers,
        maxworkertasks=maxworkertasks
    )
