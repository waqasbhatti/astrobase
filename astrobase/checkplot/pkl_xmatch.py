#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''pkl_xmatch.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
License: MIT.

This contains utility functions that support the checkplot.pkl xmatch
functionality.

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

import os
import os.path
import gzip
import sys
import json

try:
    import cPickle as pickle
except Exception as e:
    import pickle

import numpy as np

# import sps.cKDTree for external catalog xmatches
from scipy.spatial import cKDTree



###################
## LOCAL IMPORTS ##
###################

from .pkl_utils import _xyzdist_to_distarcsec
from .pkl_io import _write_checkplot_picklefile



#########################################
## XMATCHING AGAINST EXTERNAL CATALOGS ##
#########################################

def _parse_xmatch_catalog_header(xc, xk):
    '''
    This parses the header for a catalog file.

    '''

    catdef = []

    # read in this catalog and transparently handle gzipped files
    if xc.endswith('.gz'):
        infd = gzip.open(xc,'rb')
    else:
        infd = open(xc,'rb')

    # read in the defs
    for line in infd:
        if line.decode().startswith('#'):
            catdef.append(
                line.decode().replace('#','').strip().rstrip('\n')
            )
        if not line.decode().startswith('#'):
            break

    if not len(catdef) > 0:
        LOGERROR("catalog definition not parseable "
                 "for catalog: %s, skipping..." % xc)
        return None

    catdef = ' '.join(catdef)
    catdefdict = json.loads(catdef)

    catdefkeys = [x['key'] for x in catdefdict['columns']]
    catdefdtypes = [x['dtype'] for x in catdefdict['columns']]
    catdefnames = [x['name'] for x in catdefdict['columns']]
    catdefunits = [x['unit'] for x in catdefdict['columns']]

    # get the correct column indices and dtypes for the requested columns
    # from the catdefdict

    catcolinds = []
    catcoldtypes = []
    catcolnames = []
    catcolunits = []

    for xkcol in xk:

        if xkcol in catdefkeys:

            xkcolind = catdefkeys.index(xkcol)

            catcolinds.append(xkcolind)
            catcoldtypes.append(catdefdtypes[xkcolind])
            catcolnames.append(catdefnames[xkcolind])
            catcolunits.append(catdefunits[xkcolind])


    return (infd, catdefdict,
            catcolinds, catcoldtypes, catcolnames, catcolunits)



def load_xmatch_external_catalogs(xmatchto, xmatchkeys, outfile=None):
    '''This loads the external xmatch catalogs into a dict for use here.

    xmatchto is a list of text files that contain each catalog.

    the text files must be 'CSVs' that use the '|' character as the separator
    betwen columns. These files should all begin with a header in JSON format on
    lines starting with the '#' character. this header will define the catalog
    and contains the name of the catalog and the column definitions. Column
    definitions must have the column name and the numpy dtype of the columns (in
    the same format as that expected for the numpy.genfromtxt function). Any
    line that does not begin with '#' is assumed to be part of the columns in
    the catalog. An example is shown below.

    # {"name":"NSVS catalog of variable stars",
    #  "columns":[
    #    {"key":"objectid", "dtype":"U20", "name":"Object ID", "unit": null},
    #    {"key":"ra", "dtype":"f8", "name":"RA", "unit":"deg"},
    #    {"key":"decl","dtype":"f8", "name": "Declination", "unit":"deg"},
    #    {"key":"sdssr","dtype":"f8","name":"SDSS r", "unit":"mag"},
    #    {"key":"vartype","dtype":"U20","name":"Variable type", "unit":null}
    #  ],
    #  "colra":"ra",
    #  "coldec":"decl",
    #  "description":"Contains variable stars from the NSVS catalog"}
    objectid1 | 45.0  | -20.0 | 12.0 | detached EB
    objectid2 | 145.0 | 23.0  | 10.0 | RRab
    objectid3 | 12.0  | 11.0  | 14.0 | Cepheid
    .
    .
    .

    xmatchkeys is the list of lists of columns to get out of each xmatchto
    catalog. this should be the same length as xmatchto and each element here
    will apply to the respective file in xmatchto.

    if outfile is not None, set this to the name of the pickle to write the
    collect xmatch catalogs to. this pickle can then be loaded transparently by
    the checkplot_dict, checkplot_pickle functions to provide xmatch info the
    _xmatch_external_catalog function below.

    '''

    outdict = {}

    for xc, xk in zip(xmatchto, xmatchkeys):

        parsed_catdef = _parse_xmatch_catalog_header(xc, xk)

        if not parsed_catdef:
            continue

        (infd, catdefdict,
         catcolinds, catcoldtypes,
         catcolnames, catcolunits) = parsed_catdef

        # get the specified columns out of the catalog
        catarr = np.genfromtxt(infd,
                               usecols=catcolinds,
                               names=xk,
                               dtype=','.join(catcoldtypes),
                               comments='#',
                               delimiter='|',
                               autostrip=True)
        infd.close()

        catshortname = os.path.splitext(os.path.basename(xc))[0]
        catshortname = catshortname.replace('.csv','')

        #
        # make a kdtree for this catalog
        #

        # get the ra and decl columns
        objra, objdecl = (catarr[catdefdict['colra']],
                          catarr[catdefdict['coldec']])

        # get the xyz unit vectors from ra,decl
        cosdecl = np.cos(np.radians(objdecl))
        sindecl = np.sin(np.radians(objdecl))
        cosra = np.cos(np.radians(objra))
        sinra = np.sin(np.radians(objra))
        xyz = np.column_stack((cosra*cosdecl,sinra*cosdecl, sindecl))

        # generate the kdtree
        kdt = cKDTree(xyz,copy_data=True)

        # generate the outdict element for this catalog
        catoutdict = {'kdtree':kdt,
                      'data':catarr,
                      'columns':xk,
                      'colnames':catcolnames,
                      'colunits':catcolunits,
                      'name':catdefdict['name'],
                      'desc':catdefdict['description']}

        outdict[catshortname] = catoutdict

    if outfile is not None:

        # if we're on OSX, we apparently need to save the file in chunks smaller
        # than 2 GB to make it work right. can't load pickles larger than 4 GB
        # either, but 3 GB < total size < 4 GB appears to be OK when loading.
        # also see: https://bugs.python.org/issue24658.
        # fix adopted from: https://stackoverflow.com/a/38003910
        if sys.platform == 'darwin':

            dumpbytes = pickle.dumps(outdict, protocol=pickle.HIGHEST_PROTOCOL)
            max_bytes = 2**31 - 1

            with open(outfile, 'wb') as outfd:
                for idx in range(0, len(dumpbytes), max_bytes):
                    outfd.write(dumpbytes[idx:idx+max_bytes])

        else:
            with open(outfile, 'wb') as outfd:
                pickle.dump(outdict, outfd, pickle.HIGHEST_PROTOCOL)

        return outfile

    else:

        return outdict



def xmatch_external_catalogs(checkplotdict,
                             xmatchinfo,
                             xmatchradiusarcsec=2.0,
                             returndirect=False,
                             updatexmatch=True,
                             savepickle=None):
    '''This matches the current object to the external match catalogs in
    xmatchdict.

    checkplotdict is the usual checkplot dict. this must contain at least
    'objectid', and in the 'objectinfo' subdict: 'ra', and 'decl'. an 'xmatch'
    key will be added to this dict, with something like the following dict as
    the value:

    {'xmatchradiusarcsec':xmatchradiusarcsec,
     'catalog1':{'name':'Catalog of interesting things',
                 'found':True,
                 'distarcsec':0.7,
                 'info':{'objectid':...,'ra':...,'decl':...,'desc':...}},
     'catalog2':{'name':'Catalog of more interesting things',
                 'found':False,
                 'distarcsec':nan,
                 'info':None},
    .
    .
    .
    ....}

    xmatchinfo is the either a dict produced by load_xmatch_external_catalogs or
    the pickle produced by the same function.

    xmatchradiusarcsec is the xmatch radius in arcseconds.

    NOTE: this modifies checkplotdict IN PLACE if returndirect is False. If it
    is True, then just returns the xmatch results as a dict.

    If updatexmatch is True, any previous 'xmatch' elements in the checkplotdict
    will be added on to instead of being overwritten.

    If savepickle is not None, it should be the name of a checkplot pickle file
    to write the pickle back to.

    '''

    # load the xmatch info
    if isinstance(xmatchinfo, str) and os.path.exists(xmatchinfo):
        with open(xmatchinfo,'rb') as infd:
            xmatchdict = pickle.load(infd)
    elif isinstance(xmatchinfo, dict):
        xmatchdict = xmatchinfo
    else:
        LOGERROR("can't figure out xmatch info, can't xmatch, skipping...")
        return checkplotdict

    #
    # generate the xmatch spec
    #

    # get our ra, decl
    objra = checkplotdict['objectinfo']['ra']
    objdecl = checkplotdict['objectinfo']['decl']

    cosdecl = np.cos(np.radians(objdecl))
    sindecl = np.sin(np.radians(objdecl))
    cosra = np.cos(np.radians(objra))
    sinra = np.sin(np.radians(objra))

    objxyz = np.column_stack((cosra*cosdecl,
                              sinra*cosdecl,
                              sindecl))

    # this is the search distance in xyz unit vectors
    xyzdist = 2.0 * np.sin(np.radians(xmatchradiusarcsec/3600.0)/2.0)

    #
    # now search in each external catalog
    #

    xmatchresults = {}

    extcats = sorted(list(xmatchdict.keys()))

    for ecat in extcats:

        # get the kdtree
        kdt = xmatchdict[ecat]['kdtree']

        # look up the coordinates
        kdt_dist, kdt_ind = kdt.query(objxyz,
                                      k=1,
                                      distance_upper_bound=xyzdist)

        # sort by matchdist
        mdsorted = np.argsort(kdt_dist)
        matchdists = kdt_dist[mdsorted]
        matchinds = kdt_ind[mdsorted]

        if matchdists[np.isfinite(matchdists)].size == 0:

            xmatchresults[ecat] = {'name':xmatchdict[ecat]['name'],
                                   'desc':xmatchdict[ecat]['desc'],
                                   'found':False,
                                   'distarcsec':None,
                                   'info':None}

        else:

            for md, mi in zip(matchdists, matchinds):

                if np.isfinite(md) and md < xyzdist:

                    infodict = {}

                    distarcsec = _xyzdist_to_distarcsec(md)

                    for col in xmatchdict[ecat]['columns']:

                        coldata = xmatchdict[ecat]['data'][col][mi]

                        if isinstance(coldata, str):
                            coldata = coldata.strip()

                        infodict[col] = coldata

                    xmatchresults[ecat] = {
                        'name':xmatchdict[ecat]['name'],
                        'desc':xmatchdict[ecat]['desc'],
                        'found':True,
                        'distarcsec':distarcsec,
                        'info':infodict,
                        'colkeys':xmatchdict[ecat]['columns'],
                        'colnames':xmatchdict[ecat]['colnames'],
                        'colunit':xmatchdict[ecat]['colunits'],
                    }
                    break

    #
    # should now have match results for all external catalogs
    #

    if returndirect:

        return xmatchresults

    else:

        if updatexmatch and 'xmatch' in checkplotdict:
            checkplotdict['xmatch'].update(xmatchresults)
        else:
            checkplotdict['xmatch'] = xmatchresults


        if savepickle:

            cpf = _write_checkplot_picklefile(checkplotdict,
                                              outfile=savepickle,
                                              protocol=4)
            return cpf

        else:
            return checkplotdict
