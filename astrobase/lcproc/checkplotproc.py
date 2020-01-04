#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# checkplotproc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
'''
This contains functions to post-process checkplot pickles generated from a
collection of light curves beforehand (perhaps using `lcproc.checkplotgen`).

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
from io import BytesIO as StrIO
import sys
import os
import os.path
import glob
import base64
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

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

############
## CONFIG ##
############

NCPUS = mp.cpu_count()


###################
## LOCAL IMPORTS ##
###################

from astrobase.checkplot.pkl_io import (
    _read_checkplot_picklefile,
    _write_checkplot_picklefile,
    _base64_to_file
)
from astrobase.checkplot.pkl_xmatch import xmatch_external_catalogs
from astrobase.checkplot.pkl_postproc import update_checkplot_objectinfo


###############################
## ADDING INFO TO CHECKPLOTS ##
###############################

def xmatch_cplist_external_catalogs(cplist,
                                    xmatchpkl,
                                    xmatchradiusarcsec=2.0,
                                    updateexisting=True,
                                    resultstodir=None):
    '''This xmatches external catalogs to a collection of checkplots.

    Parameters
    ----------

    cplist : list of str
        This is the list of checkplot pickle files to process.

    xmatchpkl : str
        The filename of a pickle prepared beforehand with the
        `checkplot.pkl_xmatch.load_xmatch_external_catalogs` function,
        containing collected external catalogs to cross-match the objects in the
        input `cplist` against.

    xmatchradiusarcsec : float
        The match radius to use for the cross-match in arcseconds.

    updateexisting : bool
        If this is True, will only update the `xmatch` dict in each checkplot
        pickle with any new cross-matches to the external catalogs. If False,
        will overwrite the `xmatch` dict with results from the current run.

    resultstodir : str or None
        If this is provided, then it must be a directory to write the resulting
        checkplots to after xmatch is done. This can be used to keep the
        original checkplots in pristine condition for some reason.

    Returns
    -------

    dict
        Returns a dict with keys = input checkplot pickle filenames and vals =
        xmatch status dict for each checkplot pickle.

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
                    outcpf = _write_checkplot_picklefile(cpd,
                                                         outfile=cpf)
                else:
                    xcpf = os.path.join(resultstodir, os.path.basename(cpf))
                    outcpf = _write_checkplot_picklefile(cpd,
                                                         outfile=xcpf)

            status_dict[cpf] = outcpf

        except Exception:

            LOGEXCEPTION('failed to match objects for %s' % cpf)
            status_dict[cpf] = None

    return status_dict


def xmatch_cpdir_external_catalogs(cpdir,
                                   xmatchpkl,
                                   cpfileglob='checkplot-*.pkl*',
                                   xmatchradiusarcsec=2.0,
                                   updateexisting=True,
                                   resultstodir=None):
    '''This xmatches external catalogs to all checkplots in a directory.

    Parameters
    -----------

    cpdir : str
        This is the directory to search in for checkplots.

    xmatchpkl : str
        The filename of a pickle prepared beforehand with the
        `checkplot.pkl_xmatch.load_xmatch_external_catalogs` function,
        containing collected external catalogs to cross-match the objects in the
        input `cplist` against.

    cpfileglob : str
        This is the UNIX fileglob to use in searching for checkplots.

    xmatchradiusarcsec : float
        The match radius to use for the cross-match in arcseconds.

    updateexisting : bool
        If this is True, will only update the `xmatch` dict in each checkplot
        pickle with any new cross-matches to the external catalogs. If False,
        will overwrite the `xmatch` dict with results from the current run.

    resultstodir : str or None
        If this is provided, then it must be a directory to write the resulting
        checkplots to after xmatch is done. This can be used to keep the
        original checkplots in pristine condition for some reason.

    Returns
    -------

    dict
        Returns a dict with keys = input checkplot pickle filenames and vals =
        xmatch status dict for each checkplot pickle.

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
    'rpmj':r'\mathrm{RPM}_{J}',
}


def colormagdiagram_cplist(cplist,
                           outpkl,
                           color_mag1=('gaiamag','sdssg'),
                           color_mag2=('kmag','kmag'),
                           yaxis_mag=('gaia_absmag','rpmj')):
    '''This makes color-mag diagrams for all checkplot pickles in the provided
    list.

    Can make an arbitrary number of CMDs given lists of x-axis colors and y-axis
    mags to use.

    Parameters
    ----------

    cplist : list of str
        This is the list of checkplot pickles to process.

    outpkl : str
        The filename of the output pickle that will contain the color-mag
        information for all objects in the checkplots specified in `cplist`.

    color_mag1 : list of str
        This a list of the keys in each checkplot's `objectinfo` dict that will
        be used as color_1 in the equation::

                x-axis color = color_mag1 - color_mag2

    color_mag2 : list of str
        This a list of the keys in each checkplot's `objectinfo` dict that will
        be used as color_2 in the equation::

                x-axis color = color_mag1 - color_mag2

    yaxis_mag : list of str
        This is a list of the keys in each checkplot's `objectinfo` dict that
        will be used as the (absolute) magnitude y-axis of the color-mag
        diagrams.

    Returns
    -------

    str
        The path to the generated CMD pickle file for the collection of objects
        in the input checkplot list.

    Notes
    -----

    This can make many CMDs in one go. For example, the default kwargs for
    `color_mag`, `color_mag2`, and `yaxis_mag` result in two CMDs generated and
    written to the output pickle file:

    - CMD1 -> gaiamag - kmag on the x-axis vs gaia_absmag on the y-axis
    - CMD2 -> sdssg - kmag on the x-axis vs rpmj (J reduced PM) on the y-axis

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


def colormagdiagram_cpdir(
        cpdir,
        outpkl,
        cpfileglob='checkplot*.pkl*',
        color_mag1=('gaiamag','sdssg'),
        color_mag2=('kmag','kmag'),
        yaxis_mag=('gaia_absmag','rpmj')
):
    '''This makes CMDs for all checkplot pickles in the provided directory.

    Can make an arbitrary number of CMDs given lists of x-axis colors and y-axis
    mags to use.

    Parameters
    ----------

    cpdir : list of str
        This is the directory to get the list of input checkplot pickles from.

    outpkl : str
        The filename of the output pickle that will contain the color-mag
        information for all objects in the checkplots specified in `cplist`.

    cpfileglob : str
        The UNIX fileglob to use to search for checkplot pickle files.

    color_mag1 : list of str
        This a list of the keys in each checkplot's `objectinfo` dict that will
        be used as color_1 in the equation::

                x-axis color = color_mag1 - color_mag2

    color_mag2 : list of str
        This a list of the keys in each checkplot's `objectinfo` dict that will
        be used as color_2 in the equation::

                x-axis color = color_mag1 - color_mag2

    yaxis_mag : list of str
        This is a list of the keys in each checkplot's `objectinfo` dict that
        will be used as the (absolute) magnitude y-axis of the color-mag
        diagrams.

    Returns
    -------

    str
        The path to the generated CMD pickle file for the collection of objects
        in the input checkplot directory.

    Notes
    -----

    This can make many CMDs in one go. For example, the default kwargs for
    `color_mag`, `color_mag2`, and `yaxis_mag` result in two CMDs generated and
    written to the output pickle file:

    - CMD1 -> gaiamag - kmag on the x-axis vs gaia_absmag on the y-axis
    - CMD2 -> sdssg - kmag on the x-axis vs rpmj (J reduced PM) on the y-axis

    '''

    cplist = glob.glob(os.path.join(cpdir, cpfileglob))

    return colormagdiagram_cplist(cplist,
                                  outpkl,
                                  color_mag1=color_mag1,
                                  color_mag2=color_mag2,
                                  yaxis_mag=yaxis_mag)


def add_cmd_to_checkplot(
        cpx,
        cmdpkl,
        require_cmd_magcolor=True,
        save_cmd_pngs=False
):
    '''This adds CMD figures to a checkplot dict or pickle.

    Looks up the CMDs in `cmdpkl`, adds the object from `cpx` as a gold(-ish)
    star in the plot, and then saves the figure to a base64 encoded PNG, which
    can then be read and used by the `checkplotserver`.

    Parameters
    ----------

    cpx : str or dict
        This is the input checkplot pickle or dict to add the CMD to.

    cmdpkl : str or dict
        The CMD pickle generated by the `colormagdiagram_cplist` or
        `colormagdiagram_cpdir` functions above, or the dict produced by reading
        this pickle in.

    require_cmd_magcolor : bool
        If this is True, a CMD plot will not be made if the color and mag keys
        required by the CMD are not present or are nan in this checkplot's
        objectinfo dict.

    save_cmd_png : bool
        If this is True, then will save the CMD plots that were generated and
        added back to the checkplotdict as PNGs to the same directory as
        `cpx`. If `cpx` is a dict, will save them to the current working
        directory.

    Returns
    -------

    str or dict
        If `cpx` was a str filename of checkplot pickle, this will return that
        filename to indicate that the CMD was added to the file. If `cpx` was a
        checkplotdict, this will return the checkplotdict with a new key called
        'colormagdiagram' containing the base64 encoded PNG binary streams of
        all CMDs generated.

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

            # now save the figure to StrIO and put it back in the checkplot
            cmdpng = StrIO()
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

                _base64_to_file(cmdb64, outpng)

        except Exception:
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

    Parameters
    ----------

    cplist : list of str
        This is the input list of checkplot pickles to add the CMDs to.

    cmdpkl : str
        This is the filename of the CMD pickle created previously.

    require_cmd_magcolor : bool
        If this is True, a CMD plot will not be made if the color and mag keys
        required by the CMD are not present or are nan in each checkplot's
        objectinfo dict.

    save_cmd_pngs : bool
        If this is True, then will save the CMD plots that were generated and
        added back to the checkplotdict as PNGs to the same directory as
        `cpx`.

    Returns
    -------

    Nothing.


    '''

    # load the CMD first to save on IO
    with open(cmdpkl,'rb') as infd:
        cmd = pickle.load(infd)

    for cpf in cplist:

        add_cmd_to_checkplot(cpf, cmd,
                             require_cmd_magcolor=require_cmd_magcolor,
                             save_cmd_pngs=save_cmd_pngs)


def add_cmds_cpdir(cpdir,
                   cmdpkl,
                   cpfileglob='checkplot*.pkl*',
                   require_cmd_magcolor=True,
                   save_cmd_pngs=False):
    '''This adds CMDs for each object in cpdir.

    Parameters
    ----------

    cpdir : list of str
        This is the directory to search for checkplot pickles.

    cmdpkl : str
        This is the filename of the CMD pickle created previously.

    cpfileglob : str
        The UNIX fileglob to use when searching for checkplot pickles to operate
        on.

    require_cmd_magcolor : bool
        If this is True, a CMD plot will not be made if the color and mag keys
        required by the CMD are not present or are nan in each checkplot's
        objectinfo dict.

    save_cmd_pngs : bool
        If this is True, then will save the CMD plots that were generated and
        added back to the checkplotdict as PNGs to the same directory as
        `cpx`.

    Returns
    -------

    Nothing.

    '''

    cplist = glob.glob(os.path.join(cpdir, cpfileglob))

    return add_cmds_cplist(cplist,
                           cmdpkl,
                           require_cmd_magcolor=require_cmd_magcolor,
                           save_cmd_pngs=save_cmd_pngs)


#######################################
## UPDATING OBJECTINFO IN CHECKPLOTS ##
#######################################

def cp_objectinfo_worker(task):
    '''This is a parallel worker for `parallel_update_cp_objectinfo`.

    Parameters
    ----------

    task : tuple
        - task[0] = checkplot pickle file
        - task[1] = kwargs

    Returns
    -------

    str
        The name of the checkplot file that was updated. None if the update
        fails for some reason.

    '''

    cpf, cpkwargs = task

    try:

        newcpf = update_checkplot_objectinfo(cpf, **cpkwargs)
        return newcpf

    except Exception:
        LOGEXCEPTION('failed to update objectinfo for %s' % cpf)
        return None


def parallel_update_objectinfo_cplist(
        cplist,
        liststartindex=None,
        maxobjects=None,
        nworkers=NCPUS,
        fast_mode=False,
        findercmap='gray_r',
        finderconvolve=None,
        deredden_object=True,
        custom_bandpasses=None,
        gaia_submit_timeout=10.0,
        gaia_submit_tries=3,
        gaia_max_timeout=180.0,
        gaia_mirror=None,
        complete_query_later=True,
        lclistpkl=None,
        nbrradiusarcsec=60.0,
        maxnumneighbors=5,
        plotdpi=100,
        findercachedir='~/.astrobase/stamp-cache',
        verbose=True
):
    '''
    This updates objectinfo for a list of checkplots.

    Useful in cases where a previous round of GAIA/finderchart/external catalog
    acquisition failed. This will preserve the following keys in the checkplots
    if they exist:

    comments
    varinfo
    objectinfo.objecttags

    Parameters
    ----------

    cplist : list of str
        A list of checkplot pickle file names to update.

    liststartindex : int
        The index of the input list to start working at.

    maxobjects : int
        The maximum number of objects to process in this run. Use this with
        `liststartindex` to effectively distribute working on a large list of
        input checkplot pickles over several sessions or machines.

    nworkers : int
        The number of parallel workers that will work on the checkplot
        update process.

    fast_mode : bool or float
        This runs the external catalog operations in a "fast" mode, with short
        timeouts and not trying to hit external catalogs that take a long time
        to respond. See the docstring for
        `checkplot.pkl_utils._pkl_finder_objectinfo` for details on how this
        works. If this is True, will run in "fast" mode with default timeouts (5
        seconds in most cases). If this is a float, will run in "fast" mode with
        the provided timeout value in seconds.

    findercmap : str or matplotlib.cm.Colormap object

    findercmap : str or matplotlib.cm.ColorMap object
        The Colormap object to use for the finder chart image.

    finderconvolve : astropy.convolution.Kernel object or None
        If not None, the Kernel object to use for convolving the finder image.

    deredden_objects : bool
        If this is True, will use the 2MASS DUST service to get extinction
        coefficients in various bands, and then try to deredden the magnitudes
        and colors of the object already present in the checkplot's objectinfo
        dict.

    custom_bandpasses : dict
        This is a dict used to provide custom bandpass definitions for any
        magnitude measurements in the objectinfo dict that are not automatically
        recognized by the `varclass.starfeatures.color_features` function. See
        its docstring for details on the required format.

    gaia_submit_timeout : float
        Sets the timeout in seconds to use when submitting a request to look up
        the object's information to the GAIA service. Note that if `fast_mode`
        is set, this is ignored.

    gaia_submit_tries : int
        Sets the maximum number of times the GAIA services will be contacted to
        obtain this object's information. If `fast_mode` is set, this is
        ignored, and the services will be contacted only once (meaning that a
        failure to respond will be silently ignored and no GAIA data will be
        added to the checkplot's objectinfo dict).

    gaia_max_timeout : float
        Sets the timeout in seconds to use when waiting for the GAIA service to
        respond to our request for the object's information. Note that if
        `fast_mode` is set, this is ignored.

    gaia_mirror : str
        This sets the GAIA mirror to use. This is a key in the
        `services.gaia.GAIA_URLS` dict which defines the URLs to hit for each
        mirror.

    complete_query_later : bool
        If this is True, saves the state of GAIA queries that are not yet
        complete when `gaia_max_timeout` is reached while waiting for the GAIA
        service to respond to our request. A later call for GAIA info on the
        same object will attempt to pick up the results from the existing query
        if it's completed. If `fast_mode` is True, this is ignored.

    lclistpkl : dict or str
        If this is provided, must be a dict resulting from reading a catalog
        produced by the `lcproc.catalogs.make_lclist` function or a str path
        pointing to the pickle file produced by that function. This catalog is
        used to find neighbors of the current object in the current light curve
        collection. Looking at neighbors of the object within the radius
        specified by `nbrradiusarcsec` is useful for light curves produced by
        instruments that have a large pixel scale, so are susceptible to
        blending of variability and potential confusion of neighbor variability
        with that of the actual object being looked at. If this is None, no
        neighbor lookups will be performed.

    nbrradiusarcsec : float
        The radius in arcseconds to use for a search conducted around the
        coordinates of this object to look for any potential confusion and
        blending of variability amplitude caused by their proximity.

    maxnumneighbors : int
        The maximum number of neighbors that will have their light curves and
        magnitudes noted in this checkplot as potential blends with the target
        object.

    plotdpi : int
        The resolution in DPI of the plots to generate in this function
        (e.g. the finder chart, etc.)

    findercachedir : str
        The path to the astrobase cache directory for finder chart downloads
        from the NASA SkyView service.

    verbose : bool
        If True, will indicate progress and warn about potential problems.

    Returns
    -------

    list of str
        Paths to the updated checkplot pickle file.

    '''

    # work around the Darwin segfault after fork if no network activity in
    # main thread bug: https://bugs.python.org/issue30385#msg293958
    if sys.platform == 'darwin':
        import requests
        requests.get('http://captive.apple.com/hotspot-detect.html')

    # handle the start and end indices
    if (liststartindex is not None) and (maxobjects is None):
        cplist = cplist[liststartindex:]

    elif (liststartindex is None) and (maxobjects is not None):
        cplist = cplist[:maxobjects]

    elif (liststartindex is not None) and (maxobjects is not None):
        cplist = (
            cplist[liststartindex:liststartindex+maxobjects]
        )

    tasks = [(x, {'fast_mode':fast_mode,
                  'findercmap':findercmap,
                  'finderconvolve':finderconvolve,
                  'deredden_object':deredden_object,
                  'custom_bandpasses':custom_bandpasses,
                  'gaia_submit_timeout':gaia_submit_timeout,
                  'gaia_submit_tries':gaia_submit_tries,
                  'gaia_max_timeout':gaia_max_timeout,
                  'gaia_mirror':gaia_mirror,
                  'complete_query_later':complete_query_later,
                  'lclistpkl':lclistpkl,
                  'nbrradiusarcsec':nbrradiusarcsec,
                  'maxnumneighbors':maxnumneighbors,
                  'plotdpi':plotdpi,
                  'findercachedir':findercachedir,
                  'verbose':verbose}) for x in cplist]

    resultfutures = []
    results = []

    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        resultfutures = executor.map(cp_objectinfo_worker, tasks)

    results = list(resultfutures)

    executor.shutdown()
    return results


def parallel_update_objectinfo_cpdir(cpdir,
                                     cpglob='checkplot-*.pkl*',
                                     liststartindex=None,
                                     maxobjects=None,
                                     nworkers=NCPUS,
                                     fast_mode=False,
                                     findercmap='gray_r',
                                     finderconvolve=None,
                                     deredden_object=True,
                                     custom_bandpasses=None,
                                     gaia_submit_timeout=10.0,
                                     gaia_submit_tries=3,
                                     gaia_max_timeout=180.0,
                                     gaia_mirror=None,
                                     complete_query_later=True,
                                     lclistpkl=None,
                                     nbrradiusarcsec=60.0,
                                     maxnumneighbors=5,
                                     plotdpi=100,
                                     findercachedir='~/.astrobase/stamp-cache',
                                     verbose=True):
    '''This updates the objectinfo for a directory of checkplot pickles.

    Useful in cases where a previous round of GAIA/finderchart/external catalog
    acquisition failed. This will preserve the following keys in the checkplots
    if they exist:

    comments
    varinfo
    objectinfo.objecttags

    Parameters
    ----------

    cpdir : str
        The directory to look for checkplot pickles in.

    cpglob : str
        The UNIX fileglob to use when searching for checkplot pickle files.

    liststartindex : int
        The index of the input list to start working at.

    maxobjects : int
        The maximum number of objects to process in this run. Use this with
        `liststartindex` to effectively distribute working on a large list of
        input checkplot pickles over several sessions or machines.

    nworkers : int
        The number of parallel workers that will work on the checkplot
        update process.

    fast_mode : bool or float
        This runs the external catalog operations in a "fast" mode, with short
        timeouts and not trying to hit external catalogs that take a long time
        to respond. See the docstring for
        `checkplot.pkl_utils._pkl_finder_objectinfo` for details on how this
        works. If this is True, will run in "fast" mode with default timeouts (5
        seconds in most cases). If this is a float, will run in "fast" mode with
        the provided timeout value in seconds.

    findercmap : str or matplotlib.cm.Colormap object

    findercmap : str or matplotlib.cm.ColorMap object
        The Colormap object to use for the finder chart image.

    finderconvolve : astropy.convolution.Kernel object or None
        If not None, the Kernel object to use for convolving the finder image.

    deredden_objects : bool
        If this is True, will use the 2MASS DUST service to get extinction
        coefficients in various bands, and then try to deredden the magnitudes
        and colors of the object already present in the checkplot's objectinfo
        dict.

    custom_bandpasses : dict
        This is a dict used to provide custom bandpass definitions for any
        magnitude measurements in the objectinfo dict that are not automatically
        recognized by the `varclass.starfeatures.color_features` function. See
        its docstring for details on the required format.

    gaia_submit_timeout : float
        Sets the timeout in seconds to use when submitting a request to look up
        the object's information to the GAIA service. Note that if `fast_mode`
        is set, this is ignored.

    gaia_submit_tries : int
        Sets the maximum number of times the GAIA services will be contacted to
        obtain this object's information. If `fast_mode` is set, this is
        ignored, and the services will be contacted only once (meaning that a
        failure to respond will be silently ignored and no GAIA data will be
        added to the checkplot's objectinfo dict).

    gaia_max_timeout : float
        Sets the timeout in seconds to use when waiting for the GAIA service to
        respond to our request for the object's information. Note that if
        `fast_mode` is set, this is ignored.

    gaia_mirror : str
        This sets the GAIA mirror to use. This is a key in the
        `services.gaia.GAIA_URLS` dict which defines the URLs to hit for each
        mirror.

    complete_query_later : bool
        If this is True, saves the state of GAIA queries that are not yet
        complete when `gaia_max_timeout` is reached while waiting for the GAIA
        service to respond to our request. A later call for GAIA info on the
        same object will attempt to pick up the results from the existing query
        if it's completed. If `fast_mode` is True, this is ignored.

    lclistpkl : dict or str
        If this is provided, must be a dict resulting from reading a catalog
        produced by the `lcproc.catalogs.make_lclist` function or a str path
        pointing to the pickle file produced by that function. This catalog is
        used to find neighbors of the current object in the current light curve
        collection. Looking at neighbors of the object within the radius
        specified by `nbrradiusarcsec` is useful for light curves produced by
        instruments that have a large pixel scale, so are susceptible to
        blending of variability and potential confusion of neighbor variability
        with that of the actual object being looked at. If this is None, no
        neighbor lookups will be performed.

    nbrradiusarcsec : float
        The radius in arcseconds to use for a search conducted around the
        coordinates of this object to look for any potential confusion and
        blending of variability amplitude caused by their proximity.

    maxnumneighbors : int
        The maximum number of neighbors that will have their light curves and
        magnitudes noted in this checkplot as potential blends with the target
        object.

    plotdpi : int
        The resolution in DPI of the plots to generate in this function
        (e.g. the finder chart, etc.)

    findercachedir : str
        The path to the astrobase cache directory for finder chart downloads
        from the NASA SkyView service.

    verbose : bool
        If True, will indicate progress and warn about potential problems.

    Returns
    -------

    list of str
        Paths to the updated checkplot pickle file.

    '''

    cplist = sorted(glob.glob(os.path.join(cpdir, cpglob)))

    return parallel_update_objectinfo_cplist(
        cplist,
        liststartindex=liststartindex,
        maxobjects=maxobjects,
        nworkers=nworkers,
        fast_mode=fast_mode,
        findercmap=findercmap,
        finderconvolve=finderconvolve,
        deredden_object=deredden_object,
        custom_bandpasses=custom_bandpasses,
        gaia_submit_timeout=gaia_submit_timeout,
        gaia_submit_tries=gaia_submit_tries,
        gaia_max_timeout=gaia_max_timeout,
        gaia_mirror=gaia_mirror,
        complete_query_later=complete_query_later,
        lclistpkl=lclistpkl,
        nbrradiusarcsec=nbrradiusarcsec,
        maxnumneighbors=maxnumneighbors,
        plotdpi=plotdpi,
        findercachedir=findercachedir,
        verbose=verbose
    )
