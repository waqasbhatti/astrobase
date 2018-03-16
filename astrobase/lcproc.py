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
except:
    import pickle
    from io import BytesIO as strio
import gzip
import glob
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import base64

import numpy as np
import scipy.spatial as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# LC reading functions
from astrobase.hatsurveys.hatlc import read_and_filter_sqlitecurve, \
    read_csvlc, normalize_lcdict_byinst
from astrobase.hatsurveys.hplc import read_hatpi_textlc, read_hatpi_pklc
from astrobase.astrokep import read_kepler_fitslc, read_kepler_pklc

from astrobase import periodbase, checkplot
from astrobase.varclass import varfeatures, starfeatures, periodicfeatures
from astrobase.lcmath import normalize_magseries, \
    time_bin_magseries_with_errs, sigclip_magseries
from astrobase.periodbase.kbls import bls_snr

from astrobase.checkplot import _pkl_magseries_plot, \
    _pkl_phased_magseries_plot, xmatch_external_catalogs, \
    _read_checkplot_picklefile, _write_checkplot_picklefile

from astrobase.magnitudes import jhk_to_sdssr

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



# LC format -> [default fileglob,  function to read LC format]
LCFORM = {
    'hat-sql':[
        '*-hatlc.sqlite*',           # default fileglob
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
                             magsarefluxes=False,
                             specialnormfunc=None):
    '''This adds a custom format LC to the dict above.

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

    globals()['LCFORM'][formatkey] = [
        fileglob,
        readerfunc,
        timecols,
        magcols,
        errcols,
        magsarefluxes,
        specialnormfunc
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
    lcobjdict = {'lcfname':lcf}

    try:

        # read the light curve in
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
                fieldfits=None,
                fitswcsfrom=None,
                maxlcs=None,
                nworkers=20):

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

    fieldfits if not None, is the path to a FITS image containing the objects
    these light curves are for. If this is provided, make_lclist will use the
    WCS information in the FITS itself if fitswcsfrom is None (or from a WCS
    header file pointed to by fitswcsfrom) to obtain x and y pixel coordinates
    for all of the objects in the field. This can be later visualized easily.

    TODO: implement fieldfits and fitswcsfrom
    TODO: implement a make_lclist_finder function to generate a PNG with overlay

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

        # if we're supposed to make a spatial index, do so
        if (makecoordindex and
            isinstance(makecoordindex, list) and
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
                  externalcommentchar='#',
                  conesearch=None,
                  columnfilters=None,
                  conesearchworkers=1,
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



def parallel_timebin(lclist,
                     binsizesec,
                     maxobjects=None,
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
                resultdict[mcolget[-1]] = None

            else:

                # get the features for this magcol
                lcfeatures = varfeatures.all_nonperiodic_features(
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

    except:
        return None


def serial_varfeatures(lclist,
                       outdir,
                       maxobjects=None,
                       timecols=None,
                       magcols=None,
                       errcols=None,
                       mindet=1000,
                       lcformat='hat-sql',
                       nworkers=None):

    if maxobjects:
        lclist = lclist[:maxobjects]

    tasks = [(x, outdir, timecols, magcols, errcols, mindet, lcformat)
             for x in lclist]

    for task in tqdm(tasks):
        result = varfeatures_worker(task)



def parallel_varfeatures(lclist,
                         outdir,
                         maxobjects=None,
                         timecols=None,
                         magcols=None,
                         errcols=None,
                         mindet=1000,
                         lcformat='hat-sql',
                         nworkers=None):
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
        if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
            lcdict = lcdict[0]

        # get the nbr object LC into a dict if there is one
        if nbrlcf is not None:

            nbrlcdict = readerfunc(nbrlcf)
            if isinstance(nbrlcdict, tuple) and isinstance(nbrlcdict[0],dict):
                nbrlcdict = nbrlcdict[0]

        # this will be the output file
        outfile = os.path.join(outdir, 'periodicfeatures-%s.pkl' % objectid)

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

                for k in pf[mcolget[-1]].keys():

                    if k in PFMETHODS:

                        available_pgrams.append(pf[mcolget[-1]][k])

                        if k != 'win':
                            available_pfmethods.append(
                                pf[mcolget[-1]][k]['method']
                            )
                            available_bestperiods.append(
                                pf[mcolget[-1]][k]['bestperiod']
                            )

                #
                # process periodic features for this magcol
                #
                featkey = 'periodicfeatures-%s' % mcolget[-1]
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
                for ind, pfm, bp in zip(range(len(available_bestperiods)),
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
                featkey = 'periodicfeatures-%s' % mcolget[-1]
                resultdict[featkey] = None

        #
        # end of per magcol processing
        #
        # write resultdict to pickle
        outfile = os.path.join(outdir, 'periodicfeatures-%s.pkl' % objectid)
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
                            nworkers=None):
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
        result = periodicfeatures_worker(task)



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
                              nworkers=None):
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
        nworkers=None,
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

    if recursive == False:
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
        if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
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
                               'starfeatures-%s.pkl' % resultdict['objectid'])

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
    except:
        return None


def serial_starfeatures(lclist,
                        outdir,
                        lclistpickle,
                        neighbor_radius_arcsec,
                        maxobjects=None,
                        deredden=True,
                        custom_bandpasses=None,
                        lcformat='hat-sql',
                        nworkers=None):
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



def parallel_starfeatures(lclist,
                          outdir,
                          lclistpickle,
                          neighbor_radius_arcsec,
                          maxobjects=None,
                          deredden=True,
                          custom_bandpasses=None,
                          lcformat='hat-sql',
                          nworkers=None):
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
                                nworkers=None,
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
          pfmethods=['gls','pdm','mav','win'],
          pfkwargs=[{},{},{},{}],
          sigclip=10.0,
          getblssnr=False,
          nworkers=10,
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
            resultdict[mcolget[-1]] = {}

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
                resultdict[mcolget[-1]][pfmkey] = pf_func(
                    times, mags, errs,
                    **pf_kwargs
                )


            #
            # done with running the period finders
            #
            # append the pfmkeys list to the magcol dict
            resultdict[mcolget[-1]]['pfmethods'] = pfmkeys

            # check if we need to get the SNR from any BLS pfresults
            if 'bls' in pfmethods and getblssnr:

                # we need to scan thru the pfmethods to get to any BLS pfresults
                for pfmk in resultdict[mcolget[-1]]['pfmethods']:

                    if 'bls' in pfmk:

                        try:

                            bls = resultdict[mcolget[-1]][pfmk]

                            # calculate the SNR for the BLS as well
                            blssnr = bls_snr(bls, times, mags, errs,
                                             magsarefluxes=magsarefluxes,
                                             verbose=False)

                            # add the SNR results to the BLS result dict
                            resultdict[mcolget[-1]][pfmk].update({
                                'snr':blssnr['snr'],
                                'altsnr':blssnr['altsnr'],
                                'transitdepth':blssnr['transitdepth'],
                                'transitduration':blssnr['transitduration'],
                            })

                        except Exception as e:

                            LOGEXCEPTION('could not calculate BLS SNR for %s' %
                                         lcfile)
                            # add the SNR null results to the BLS result dict
                            resultdict[mcolget[-1]][pfmk].update({
                                'snr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                                'altsnr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                                'transitdepth':[np.nan,np.nan,np.nan,
                                                np.nan,np.nan],
                                'transitduration':[np.nan,np.nan,np.nan,
                                                   np.nan,np.nan],
                            })

            elif 'bls' in pfmethods:

                # we need to scan thru the pfmethods to get to any BLS pfresults
                for pfmk in resultdict[mcolget[-1]]['pfmethods']:

                    if 'bls' in pfmk:

                        # add the SNR null results to the BLS result dict
                        resultdict[mcolget[-1]][pfmk].update({
                            'snr':[np.nan,np.nan,np.nan,np.nan,np.nan],
                            'altsnr':[np.nan,np.nan,np.nan,np.nan,np.nan],
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
     pfmethods, pfkwargs, getblssnr, sigclip, nworkers, excludeprocessed) = task

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
                pfmethods=['gls','pdm','mav','win'],
                pfkwargs=[{},{},{},{}],
                getblssnr=False,
                sigclip=10.0,
                nperiodworkers=10,
                ncontrolworkers=4,
                liststartindex=None,
                listmaxobjects=None,
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
        os.mkdirs(outdir)

    if (liststartindex is not None) and (listmaxobjects is None):
        lclist = lclist[liststartindex:]

    elif (liststartindex is None) and (listmaxobjects is not None):
        lclist = lclist[:listmaxobjects]

    elif (liststartindex is not None) and (listmaxobjects is not None):
        lclist = lclist[liststartindex:liststartindex+listmaxobjects]

    tasklist = [(x, outdir, timecols, magcols, errcols, lcformat,
                 pfmethods, pfkwargs, getblssnr, sigclip, nperiodworkers,
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
                      pfmethods=['gls','pdm','mav','win'],
                      pfkwargs=[{},{},{},{}],
                      getblssnr=False,
                      sigclip=10.0,
                      nperiodworkers=10,
                      ncontrolworkers=4,
                      liststartindex=None,
                      listmaxobjects=None,
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
        mclist =  ('bmag','vmag','rmag','imag','jmag','hmag','kmag',
                   'sdssu','sdssg','sdssr','sdssi','sdssz')

    for mc in mclist:
        if (mc in checkplotdict['objectinfo'] and
            checkplotdict['objectinfo'][mc] is not None and
            np.isfinite(checkplotdict['objectinfo'][mc])):

            objmagkeys[mc] = checkplotdict['objectinfo'][mc]


    # if there are actually neighbors, go through them in order
    for nbr in checkplotdict['neighbors']:

        objectid, ra, decl, dist, lcfpath = (nbr['objectid'],
                                             nbr['ra'],
                                             nbr['decl'],
                                             nbr['dist'],
                                             nbr['lcfpath'])

        # get the light curve
        if not os.path.exists(lcfpath):
            LOGERROR('objectid: %s, neighbor: %s, '
                     'lightcurve: %s not found, skipping...' %
                     (checkplotdict['objectid'], objectid, lcfpath))
            continue

        lcdict = readerfunc(lcfpath)
        if isinstance(lcdict, tuple) and isinstance(lcdict[0],dict):
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
        for lspt in checkplotdict['pfmethods']:

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
                lspt.split('-')[1], # this splits '<pfindex>-<pfmethod>'
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
          cprenorm=False,
          lclistpkl=None,
          nbrradiusarcsec=60.0,
          xmatchinfo=None,
          xmatchradiusarcsec=3.0,
          sigclip=10.0,
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

    objectid = pfresults['objectid']
    lcfbasename = pfresults['lcfbasename']

    lcfsearchpath = os.path.join(lcbasedir, lcfbasename)

    if os.path.exists(lcfsearchpath):
        lcfpath = lcfsearchpath
    else:
        LOGERROR('could not find light curve for '
                 'pfresult %s, objectid %s, '
                 'used search path: %s' %
                 (pfpickle, objectid, lcfsearchpath))
        return None


    lcdict = readerfunc(lcfpath)
    if isinstance(lcdict, tuple) and isinstance(lcdict[0], dict):
        lcdict = lcdict[0]

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
        pflist = [pfresults[mcolget[-1]][x]
                  for x in pfresults[mcolget[-1]]['pfmethods']]

        # generate the output filename
        outfile = os.path.join(outdir,
                               'checkplot-%s-%s.pkl' % (objectid, mcol))

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
            lclistpkl=lclistpkl,
            nbrradiusarcsec=nbrradiusarcsec,
            xmatchinfo=xmatchinfo,
            xmatchradiusarcsec=xmatchradiusarcsec,
            sigclip=sigclip,
            verbose=False,
            normto=cprenorm # we've done the renormalization already, so this
                            # should be False by default. just messes up the
                            # plots otherwise, destroying LPVs in particular
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
                cprenorm=False,
                lclistpkl=None,
                nbrradiusarcsec=60.0,
                xmatchinfo=None,
                xmatchradiusarcsec=3.0,
                sigclip=10.0,
                maxobjects=None,
                lcformat='hat-sql',
                timecols=None,
                magcols=None,
                errcols=None,
                nworkers=32):
    '''This drives the parallel execution of runcp for a list of periodfinding
    result pickles.

    '''

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if maxobjects:
        pfpicklelist = pfpicklelist[:maxobjects]


    tasklist = [(x, outdir, lcbasedir,
                 {'lcformat':lcformat,
                  'timecols':timecols,
                  'magcols':magcols,
                  'errcols':errcols,
                  'lclistpkl':lclistpkl,
                  'nbrradiusarcsec':nbrradiusarcsec,
                  'xmatchinfo':xmatchinfo,
                  'xmatchradiusarcsec':xmatchradiusarcsec,
                  'sigclip':sigclip,
                  'cprenorm':cprenorm}) for
                x in pfpicklelist]

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
                      cprenorm=False,
                      lclistpkl=None,
                      nbrradiusarcsec=60.0,
                      xmatchinfo=None,
                      xmatchradiusarcsec=3.0,
                      sigclip=10.0,
                      maxobjects=None,
                      pfpickleglob='periodfinding-*.pkl*',
                      lcformat='hat-sql',
                      timecols=None,
                      magcols=None,
                      errcols=None,
                      nworkers=32):

    '''This drives the parallel execution of runcp for a directory of
    periodfinding pickles.

    '''

    pfpicklelist = sorted(glob.glob(os.path.join(pfpickledir, pfpickleglob)))

    LOGINFO('found %s period-finding pickles, running cp...' %
            len(pfpicklelist))

    return parallel_cp(pfpicklelist,
                       outdir,
                       lcbasedir,
                       lclistpkl=lclistpkl,
                       nbrradiusarcsec=nbrradiusarcsec,
                       xmatchinfo=xmatchinfo,
                       xmatchradiusarcsec=xmatchradiusarcsec,
                       sigclip=sigclip,
                       cprenorm=cprenorm,
                       maxobjects=maxobjects,
                       lcformat=lcformat,
                       timecols=timecols,
                       magcols=magcols,
                       errcols=errcols,
                       nworkers=nworkers)



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

            thiscmd_label = '%s-%s/%s' % (c1,
                                          c2,
                                          ym)
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

                pngf = checkplot._base64_to_file(cmdb64, outpng)

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
