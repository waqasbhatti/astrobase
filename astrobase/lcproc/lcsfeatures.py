#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# starfeatures.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
'''
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

import pickle
import os
import os.path
import glob
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
    '''This runs the functions from :py:func:`astrobase.varclass.starfeatures`
    on a single light curve file.

    Parameters
    ----------

    lcfile : str
        This is the LC file to extract star features for.

    outdir : str
        This is the directory to write the output pickle to.

    kdtree: scipy.spatial.cKDTree
        This is a `scipy.spatial.KDTree` or `cKDTree` used to calculate neighbor
        proximity features. This is for the light curve catalog this object is
        in.

    objlist : np.array
        This is a Numpy array of object IDs in the same order as the
        `kdtree.data` np.array. This is for the light curve catalog this object
        is in.

    lcflist : np.array
        This is a Numpy array of light curve filenames in the same order as
        `kdtree.data`. This is for the light curve catalog this object is in.

    neighbor_radius_arcsec : float
        This indicates the radius in arcsec to search for neighbors for this
        object using the light curve catalog's `kdtree`, `objlist`, `lcflist`,
        and in GAIA.

    deredden : bool
        This controls if the colors and any color classifications will be
        dereddened using 2MASS DUST.

    custom_bandpasses : dict or None
        This is a dict used to define any custom bandpasses in the
        `in_objectinfo` dict you want to make this function aware of and
        generate colors for. Use the format below for this dict::

            {
            '<bandpass_key_1>':{'dustkey':'<twomass_dust_key_1>',
                                'label':'<band_label_1>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            .
            ...
            .
            '<bandpass_key_N>':{'dustkey':'<twomass_dust_key_N>',
                                'label':'<band_label_N>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            }

        Where:

        `bandpass_key` is a key to use to refer to this bandpass in the
        `objectinfo` dict, e.g. 'sdssg' for SDSS g band

        `twomass_dust_key` is the key to use in the 2MASS DUST result table for
        reddening per band-pass. For example, given the following DUST result
        table (using http://irsa.ipac.caltech.edu/applications/DUST/)::

            |Filter_name|LamEff |A_over_E_B_V_SandF|A_SandF|A_over_E_B_V_SFD|A_SFD|
            |char       |float  |float             |float  |float           |float|
            |           |microns|                  |mags   |                |mags |
             CTIO U       0.3734              4.107   0.209            4.968 0.253
             CTIO B       0.4309              3.641   0.186            4.325 0.221
             CTIO V       0.5517              2.682   0.137            3.240 0.165
            .
            .
            ...

        The `twomass_dust_key` for 'vmag' would be 'CTIO V'. If you want to
        skip DUST lookup and want to pass in a specific reddening magnitude
        for your bandpass, use a float for the value of
        `twomass_dust_key`. If you want to skip DUST lookup entirely for
        this bandpass, use None for the value of `twomass_dust_key`.

        `band_label` is the label to use for this bandpass, e.g. 'W1' for
        WISE-1 band, 'u' for SDSS u, etc.

        The 'colors' list contains color definitions for all colors you want
        to generate using this bandpass. this list contains elements of the
        form::

            ['<bandkey1>-<bandkey2>','<BAND1> - <BAND2>']

        where the the first item is the bandpass keys making up this color,
        and the second item is the label for this color to be used by the
        frontends. An example::

            ['sdssu-sdssg','u - g']

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

    Returns
    -------

    str
        Path to the output pickle containing all of the star features for this
        object.

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


def _starfeatures_worker(task):
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
    except Exception:
        return None


def serial_starfeatures(lclist,
                        outdir,
                        lc_catalog_pickle,
                        neighbor_radius_arcsec,
                        maxobjects=None,
                        deredden=True,
                        custom_bandpasses=None,
                        lcformat='hat-sql',
                        lcformatdir=None):
    '''This drives the `get_starfeatures` function for a collection of LCs.

    Parameters
    ----------

    lclist : list of str
        The list of light curve file names to process.

    outdir : str
        The output directory where the results will be placed.

    lc_catalog_pickle : str
        The path to a catalog containing at a dict with least:

        - an object ID array accessible with `dict['objects']['objectid']`

        - an LC filename array accessible with `dict['objects']['lcfname']`

        - a `scipy.spatial.KDTree` or `cKDTree` object to use for finding
          neighbors for each object accessible with `dict['kdtree']`

        A catalog pickle of the form needed can be produced using
        :py:func:`astrobase.lcproc.catalogs.make_lclist` or
        :py:func:`astrobase.lcproc.catalogs.filter_lclist`.

    neighbor_radius_arcsec : float
        This indicates the radius in arcsec to search for neighbors for this
        object using the light curve catalog's `kdtree`, `objlist`, `lcflist`,
        and in GAIA.

    maxobjects : int
        The number of objects to process from `lclist`.

    deredden : bool
        This controls if the colors and any color classifications will be
        dereddened using 2MASS DUST.

    custom_bandpasses : dict or None
        This is a dict used to define any custom bandpasses in the
        `in_objectinfo` dict you want to make this function aware of and
        generate colors for. Use the format below for this dict::

            {
            '<bandpass_key_1>':{'dustkey':'<twomass_dust_key_1>',
                                'label':'<band_label_1>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            .
            ...
            .
            '<bandpass_key_N>':{'dustkey':'<twomass_dust_key_N>',
                                'label':'<band_label_N>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            }

        Where:

        `bandpass_key` is a key to use to refer to this bandpass in the
        `objectinfo` dict, e.g. 'sdssg' for SDSS g band

        `twomass_dust_key` is the key to use in the 2MASS DUST result table for
        reddening per band-pass. For example, given the following DUST result
        table (using http://irsa.ipac.caltech.edu/applications/DUST/)::

            |Filter_name|LamEff |A_over_E_B_V_SandF|A_SandF|A_over_E_B_V_SFD|A_SFD|
            |char       |float  |float             |float  |float           |float|
            |           |microns|                  |mags   |                |mags |
             CTIO U       0.3734              4.107   0.209            4.968 0.253
             CTIO B       0.4309              3.641   0.186            4.325 0.221
             CTIO V       0.5517              2.682   0.137            3.240 0.165
            .
            .
            ...

        The `twomass_dust_key` for 'vmag' would be 'CTIO V'. If you want to
        skip DUST lookup and want to pass in a specific reddening magnitude
        for your bandpass, use a float for the value of
        `twomass_dust_key`. If you want to skip DUST lookup entirely for
        this bandpass, use None for the value of `twomass_dust_key`.

        `band_label` is the label to use for this bandpass, e.g. 'W1' for
        WISE-1 band, 'u' for SDSS u, etc.

        The 'colors' list contains color definitions for all colors you want
        to generate using this bandpass. this list contains elements of the
        form::

            ['<bandkey1>-<bandkey2>','<BAND1> - <BAND2>']

        where the the first item is the bandpass keys making up this color,
        and the second item is the label for this color to be used by the
        frontends. An example::

            ['sdssu-sdssg','u - g']

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

    Returns
    -------

    list of str
        A list of all star features pickles produced.

    '''
    # make sure to make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maxobjects:
        lclist = lclist[:maxobjects]

    # read in the kdtree pickle
    with open(lc_catalog_pickle, 'rb') as infd:
        kdt_dict = pickle.load(infd)

    kdt = kdt_dict['kdtree']
    objlist = kdt_dict['objects']['objectid']
    objlcfl = kdt_dict['objects']['lcfname']

    tasks = [(x, outdir, kdt, objlist, objlcfl,
              neighbor_radius_arcsec,
              deredden, custom_bandpasses,
              lcformat, lcformatdir) for x in lclist]

    for task in tqdm(tasks):
        result = _starfeatures_worker(task)

    return result


def parallel_starfeatures(lclist,
                          outdir,
                          lc_catalog_pickle,
                          neighbor_radius_arcsec,
                          maxobjects=None,
                          deredden=True,
                          custom_bandpasses=None,
                          lcformat='hat-sql',
                          lcformatdir=None,
                          nworkers=NCPUS):
    '''This runs `get_starfeatures` in parallel for all light curves in `lclist`.

    Parameters
    ----------

    lclist : list of str
        The list of light curve file names to process.

    outdir : str
        The output directory where the results will be placed.

    lc_catalog_pickle : str
        The path to a catalog containing at a dict with least:

        - an object ID array accessible with `dict['objects']['objectid']`

        - an LC filename array accessible with `dict['objects']['lcfname']`

        - a `scipy.spatial.KDTree` or `cKDTree` object to use for finding
          neighbors for each object accessible with `dict['kdtree']`

        A catalog pickle of the form needed can be produced using
        :py:func:`astrobase.lcproc.catalogs.make_lclist` or
        :py:func:`astrobase.lcproc.catalogs.filter_lclist`.

    neighbor_radius_arcsec : float
        This indicates the radius in arcsec to search for neighbors for this
        object using the light curve catalog's `kdtree`, `objlist`, `lcflist`,
        and in GAIA.

    maxobjects : int
        The number of objects to process from `lclist`.

    deredden : bool
        This controls if the colors and any color classifications will be
        dereddened using 2MASS DUST.

    custom_bandpasses : dict or None
        This is a dict used to define any custom bandpasses in the
        `in_objectinfo` dict you want to make this function aware of and
        generate colors for. Use the format below for this dict::

            {
            '<bandpass_key_1>':{'dustkey':'<twomass_dust_key_1>',
                                'label':'<band_label_1>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            .
            ...
            .
            '<bandpass_key_N>':{'dustkey':'<twomass_dust_key_N>',
                                'label':'<band_label_N>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            }

        Where:

        `bandpass_key` is a key to use to refer to this bandpass in the
        `objectinfo` dict, e.g. 'sdssg' for SDSS g band

        `twomass_dust_key` is the key to use in the 2MASS DUST result table for
        reddening per band-pass. For example, given the following DUST result
        table (using http://irsa.ipac.caltech.edu/applications/DUST/)::

            |Filter_name|LamEff |A_over_E_B_V_SandF|A_SandF|A_over_E_B_V_SFD|A_SFD|
            |char       |float  |float             |float  |float           |float|
            |           |microns|                  |mags   |                |mags |
             CTIO U       0.3734              4.107   0.209            4.968 0.253
             CTIO B       0.4309              3.641   0.186            4.325 0.221
             CTIO V       0.5517              2.682   0.137            3.240 0.165
            .
            .
            ...

        The `twomass_dust_key` for 'vmag' would be 'CTIO V'. If you want to
        skip DUST lookup and want to pass in a specific reddening magnitude
        for your bandpass, use a float for the value of
        `twomass_dust_key`. If you want to skip DUST lookup entirely for
        this bandpass, use None for the value of `twomass_dust_key`.

        `band_label` is the label to use for this bandpass, e.g. 'W1' for
        WISE-1 band, 'u' for SDSS u, etc.

        The 'colors' list contains color definitions for all colors you want
        to generate using this bandpass. this list contains elements of the
        form::

            ['<bandkey1>-<bandkey2>','<BAND1> - <BAND2>']

        where the the first item is the bandpass keys making up this color,
        and the second item is the label for this color to be used by the
        frontends. An example::

            ['sdssu-sdssg','u - g']

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

    nworkers : int
        The number of parallel workers to launch.

    Returns
    -------

    dict
        A dict with key:val pairs of the input light curve filename and the
        output star features pickle for each LC processed.

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

    # make sure to make the output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if maxobjects:
        lclist = lclist[:maxobjects]

    # read in the kdtree pickle
    with open(lc_catalog_pickle, 'rb') as infd:
        kdt_dict = pickle.load(infd)

    kdt = kdt_dict['kdtree']
    objlist = kdt_dict['objects']['objectid']
    objlcfl = kdt_dict['objects']['lcfname']

    tasks = [(x, outdir, kdt, objlist, objlcfl,
              neighbor_radius_arcsec,
              deredden, custom_bandpasses, lcformat) for x in lclist]

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(_starfeatures_worker, tasks)

    results = list(resultfutures)
    resdict = {os.path.basename(x):y for (x,y) in zip(lclist, results)}

    return resdict


def parallel_starfeatures_lcdir(lcdir,
                                outdir,
                                lc_catalog_pickle,
                                neighbor_radius_arcsec,
                                fileglob=None,
                                maxobjects=None,
                                deredden=True,
                                custom_bandpasses=None,
                                lcformat='hat-sql',
                                lcformatdir=None,
                                nworkers=NCPUS,
                                recursive=True):
    '''This runs parallel star feature extraction for a directory of LCs.

    Parameters
    ----------

    lcdir : list of str
        The directory to search for light curves.

    outdir : str
        The output directory where the results will be placed.

    lc_catalog_pickle : str
        The path to a catalog containing at a dict with least:

        - an object ID array accessible with `dict['objects']['objectid']`

        - an LC filename array accessible with `dict['objects']['lcfname']`

        - a `scipy.spatial.KDTree` or `cKDTree` object to use for finding
          neighbors for each object accessible with `dict['kdtree']`

        A catalog pickle of the form needed can be produced using
        :py:func:`astrobase.lcproc.catalogs.make_lclist` or
        :py:func:`astrobase.lcproc.catalogs.filter_lclist`.

    neighbor_radius_arcsec : float
        This indicates the radius in arcsec to search for neighbors for this
        object using the light curve catalog's `kdtree`, `objlist`, `lcflist`,
        and in GAIA.

    fileglob : str
        The UNIX file glob to use to search for the light curves in `lcdir`. If
        None, the default value for the light curve format specified will be
        used.

    maxobjects : int
        The number of objects to process from `lclist`.

    deredden : bool
        This controls if the colors and any color classifications will be
        dereddened using 2MASS DUST.

    custom_bandpasses : dict or None
        This is a dict used to define any custom bandpasses in the
        `in_objectinfo` dict you want to make this function aware of and
        generate colors for. Use the format below for this dict::

            {
            '<bandpass_key_1>':{'dustkey':'<twomass_dust_key_1>',
                                'label':'<band_label_1>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            .
            ...
            .
            '<bandpass_key_N>':{'dustkey':'<twomass_dust_key_N>',
                                'label':'<band_label_N>'
                                'colors':[['<bandkey1>-<bandkey2>',
                                           '<BAND1> - <BAND2>'],
                                          ['<bandkey3>-<bandkey4>',
                                           '<BAND3> - <BAND4>']]},
            }

        Where:

        `bandpass_key` is a key to use to refer to this bandpass in the
        `objectinfo` dict, e.g. 'sdssg' for SDSS g band

        `twomass_dust_key` is the key to use in the 2MASS DUST result table for
        reddening per band-pass. For example, given the following DUST result
        table (using http://irsa.ipac.caltech.edu/applications/DUST/)::

            |Filter_name|LamEff |A_over_E_B_V_SandF|A_SandF|A_over_E_B_V_SFD|A_SFD|
            |char       |float  |float             |float  |float           |float|
            |           |microns|                  |mags   |                |mags |
             CTIO U       0.3734              4.107   0.209            4.968 0.253
             CTIO B       0.4309              3.641   0.186            4.325 0.221
             CTIO V       0.5517              2.682   0.137            3.240 0.165
            .
            .
            ...

        The `twomass_dust_key` for 'vmag' would be 'CTIO V'. If you want to
        skip DUST lookup and want to pass in a specific reddening magnitude
        for your bandpass, use a float for the value of
        `twomass_dust_key`. If you want to skip DUST lookup entirely for
        this bandpass, use None for the value of `twomass_dust_key`.

        `band_label` is the label to use for this bandpass, e.g. 'W1' for
        WISE-1 band, 'u' for SDSS u, etc.

        The 'colors' list contains color definitions for all colors you want
        to generate using this bandpass. this list contains elements of the
        form::

            ['<bandkey1>-<bandkey2>','<BAND1> - <BAND2>']

        where the the first item is the bandpass keys making up this color,
        and the second item is the label for this color to be used by the
        frontends. An example::

            ['sdssu-sdssg','u - g']

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

    nworkers : int
        The number of parallel workers to launch.

    Returns
    -------

    dict
        A dict with key:val pairs of the input light curve filename and the
        output star features pickle for each LC processed.

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

    # now find the files
    LOGINFO('searching for %s light curves in %s ...' % (lcformat, lcdir))

    if recursive is False:
        matching = glob.glob(os.path.join(lcdir, fileglob))

    else:
        matching = glob.glob(os.path.join(lcdir,
                                          '**',
                                          fileglob),
                             recursive=True)

    # now that we have all the files, process them
    if matching and len(matching) > 0:

        LOGINFO('found %s light curves, getting starfeatures...' %
                len(matching))

        return parallel_starfeatures(matching,
                                     outdir,
                                     lc_catalog_pickle,
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
