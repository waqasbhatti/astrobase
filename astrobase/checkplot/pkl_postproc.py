#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pkl_postproc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
# License: MIT.

'''
This contains utility functions that support the checkplot pickle
post-processing functionality.

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

from copy import deepcopy
import numpy as np


###################
## LOCAL IMPORTS ##
###################

from .pkl_io import _read_checkplot_picklefile, _write_checkplot_picklefile
from .pkl_utils import _pkl_finder_objectinfo


################################
## POST-PROCESSING CHECKPLOTS ##
################################

def update_checkplot_objectinfo(cpf,
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
    '''This updates a checkplot objectinfo dict.

    Useful in cases where a previous round of GAIA/finderchart/external catalog
    acquisition failed. This will preserve the following keys in the checkplot
    if they exist::

        comments
        varinfo
        objectinfo.objecttags

    Parameters
    ----------

    cpf : str
        The path to the checkplot pickle to update.

    fast_mode : bool or float
        This runs the external catalog operations in a "fast" mode, with short
        timeouts and not trying to hit external catalogs that take a long time
        to respond. See the docstring for
        :py:func:`astrobase.checkplot.pkl_utils._pkl_finder_objectinfo` for
        details on how this works. If this is True, will run in "fast" mode with
        default timeouts (5 seconds in most cases). If this is a float, will run
        in "fast" mode with the provided timeout value in seconds.

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
        :py:data:`astrobase.services.gaia.GAIA_URLS` dict which defines the URLs
        to hit for each mirror.

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

    str
        Path to the updated checkplot pickle file.

    '''

    cpd = _read_checkplot_picklefile(cpf)

    if cpd['objectinfo']['objecttags'] is not None:
        objecttags = cpd['objectinfo']['objecttags'][::]
    else:
        objecttags = None

    varinfo = deepcopy(cpd['varinfo'])

    if 'comments' in cpd and cpd['comments'] is not None:
        comments = cpd['comments'][::]
    else:
        comments = None

    newcpd = _pkl_finder_objectinfo(cpd['objectinfo'],
                                    varinfo,
                                    findercmap,
                                    finderconvolve,
                                    cpd['sigclip'],
                                    cpd['normto'],
                                    cpd['normmingap'],
                                    fast_mode=fast_mode,
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
                                    verbose=verbose)

    #
    # don't update neighbors or finder chart if the new one is bad
    #
    if (newcpd['finderchart'] is None and
        cpd['finderchart'] is not None):
        newcpd['finderchart'] = deepcopy(
            cpd['finderchart']
        )

    if (newcpd['neighbors'] is None and
        cpd['neighbors'] is not None):
        newcpd['neighbors'] = deepcopy(
            cpd['neighbors']
        )

    #
    # if there's existing GAIA info, don't overwrite if the new objectinfo dict
    # doesn't have any
    #
    if (('failed' in newcpd['objectinfo']['gaia_status'] or
         ('gaiaid' in newcpd['objectinfo'] and
          newcpd['objectinfo']['gaiaid'] is None)) and
        'ok' in cpd['objectinfo']['gaia_status']):

        newcpd['objectinfo']['gaia_status'] = deepcopy(
            cpd['objectinfo']['gaia_status']
        )
        if 'gaiaid' in cpd['objectinfo']:
            newcpd['objectinfo']['gaiaid'] = deepcopy(
                cpd['objectinfo']['gaiaid']
            )
        newcpd['objectinfo']['gaiamag'] = deepcopy(
            cpd['objectinfo']['gaiamag']
        )
        newcpd['objectinfo']['gaia_absmag'] = deepcopy(
            cpd['objectinfo']['gaia_absmag']
        )
        newcpd['objectinfo']['gaia_parallax'] = deepcopy(
            cpd['objectinfo']['gaia_parallax']
        )
        newcpd['objectinfo']['gaia_parallax_err'] = deepcopy(
            cpd['objectinfo']['gaia_parallax_err']
        )
        newcpd['objectinfo']['gaia_pmra'] = deepcopy(
            cpd['objectinfo']['gaia_pmra']
        )
        newcpd['objectinfo']['gaia_pmra_err'] = deepcopy(
            cpd['objectinfo']['gaia_pmra_err']
        )
        newcpd['objectinfo']['gaia_pmdecl'] = deepcopy(
            cpd['objectinfo']['gaia_pmdecl']
        )
        newcpd['objectinfo']['gaia_pmdecl_err'] = deepcopy(
            cpd['objectinfo']['gaia_pmdecl_err']
        )

    if (not np.isfinite(newcpd['objectinfo']['gaia_neighbors']) and
        np.isfinite(cpd['objectinfo']['gaia_neighbors'])):
        newcpd['objectinfo']['gaia_neighbors'] = deepcopy(
            cpd['objectinfo']['gaia_neighbors']
        )
    if (not np.isfinite(newcpd['objectinfo']['gaia_closest_distarcsec']) and
        np.isfinite(cpd['objectinfo']['gaia_closest_distarcsec'])):
        newcpd['objectinfo']['gaia_closest_distarcsec'] = deepcopy(
            cpd['objectinfo']['gaia_closest_gmagdiff']
        )
    if (not np.isfinite(newcpd['objectinfo']['gaia_closest_gmagdiff']) and
        np.isfinite(cpd['objectinfo']['gaia_closest_gmagdiff'])):
        newcpd['objectinfo']['gaia_closest_gmagdiff'] = deepcopy(
            cpd['objectinfo']['gaia_closest_gmagdiff']
        )

    if (newcpd['objectinfo']['gaia_ids'] is None and
        cpd['objectinfo']['gaia_ids'] is not None):
        newcpd['objectinfo']['gaia_ids'] = deepcopy(
            cpd['objectinfo']['gaia_ids']
        )
    if (newcpd['objectinfo']['gaia_xypos'] is None and
        cpd['objectinfo']['gaia_xypos'] is not None):
        newcpd['objectinfo']['gaia_xypos'] = deepcopy(
            cpd['objectinfo']['gaia_xypos']
        )
    if (newcpd['objectinfo']['gaia_mags'] is None and
        cpd['objectinfo']['gaia_mags'] is not None):
        newcpd['objectinfo']['gaia_mags'] = deepcopy(
            cpd['objectinfo']['gaia_mags']
        )
    if (newcpd['objectinfo']['gaia_parallaxes'] is None and
        cpd['objectinfo']['gaia_parallaxes'] is not None):
        newcpd['objectinfo']['gaia_parallaxes'] = deepcopy(
            cpd['objectinfo']['gaia_parallaxes']
        )
    if (newcpd['objectinfo']['gaia_parallax_errs'] is None and
        cpd['objectinfo']['gaia_parallax_errs'] is not None):
        newcpd['objectinfo']['gaia_parallax_errs'] = deepcopy(
            cpd['objectinfo']['gaia_parallax_errs']
        )
    if (newcpd['objectinfo']['gaia_pmras'] is None and
        cpd['objectinfo']['gaia_pmras'] is not None):
        newcpd['objectinfo']['gaia_pmras'] = deepcopy(
            cpd['objectinfo']['gaia_pmras']
        )
    if (newcpd['objectinfo']['gaia_pmra_errs'] is None and
        cpd['objectinfo']['gaia_pmra_errs'] is not None):
        newcpd['objectinfo']['gaia_pmra_errs'] = deepcopy(
            cpd['objectinfo']['gaia_pmra_errs']
        )
    if (newcpd['objectinfo']['gaia_pmdecls'] is None and
        cpd['objectinfo']['gaia_pmdecls'] is not None):
        newcpd['objectinfo']['gaia_pmdecls'] = deepcopy(
            cpd['objectinfo']['gaia_pmdecls']
        )
    if (newcpd['objectinfo']['gaia_pmdecl_errs'] is None and
        cpd['objectinfo']['gaia_pmdecl_errs'] is not None):
        newcpd['objectinfo']['gaia_pmdecl_errs'] = deepcopy(
            cpd['objectinfo']['gaia_pmdecl_errs']
        )
    if (newcpd['objectinfo']['gaia_absolute_mags'] is None and
        cpd['objectinfo']['gaia_absolute_mags'] is not None):
        newcpd['objectinfo']['gaia_absolute_mags'] = deepcopy(
            cpd['objectinfo']['gaia_absolute_mags']
        )
    if (newcpd['objectinfo']['gaiak_colors'] is None and
        cpd['objectinfo']['gaiak_colors'] is not None):
        newcpd['objectinfo']['gaiak_colors'] = deepcopy(
            cpd['objectinfo']['gaiak_colors']
        )
    if (newcpd['objectinfo']['gaia_dists'] is None and
        cpd['objectinfo']['gaia_dists'] is not None):
        newcpd['objectinfo']['gaia_dists'] = deepcopy(
            cpd['objectinfo']['gaia_dists']
        )

    #
    # don't overwrite good SIMBAD info with bad
    #
    if ('failed' in newcpd['objectinfo']['simbad_status'] and
        'ok' in cpd['objectinfo']['simbad_status']):
        newcpd['objectinfo']['simbad_status'] = deepcopy(
            cpd['objectinfo']['simbad_status']
        )

    if (newcpd['objectinfo']['simbad_nmatches'] is None and
        cpd['objectinfo']['simbad_nmatches'] is not None):
        newcpd['objectinfo']['simbad_nmatches'] = deepcopy(
            cpd['objectinfo']['simbad_nmatches']
        )
    if (newcpd['objectinfo']['simbad_mainid'] is None and
        cpd['objectinfo']['simbad_mainid'] is not None):
        newcpd['objectinfo']['simbad_mainid'] = deepcopy(
            cpd['objectinfo']['simbad_mainid']
        )
    if (newcpd['objectinfo']['simbad_objtype'] is None and
        cpd['objectinfo']['simbad_objtype'] is not None):
        newcpd['objectinfo']['simbad_objtype'] = deepcopy(
            cpd['objectinfo']['simbad_objtype']
        )
    if (newcpd['objectinfo']['simbad_allids'] is None and
        cpd['objectinfo']['simbad_allids'] is not None):
        newcpd['objectinfo']['simbad_allids'] = deepcopy(
            cpd['objectinfo']['simbad_allids']
        )
    if (newcpd['objectinfo']['simbad_distarcsec'] is None and
        cpd['objectinfo']['simbad_distarcsec'] is not None):
        newcpd['objectinfo']['simbad_distarcsec'] = deepcopy(
            cpd['objectinfo']['simbad_distarcsec']
        )
    if (newcpd['objectinfo']['simbad_best_mainid'] is None and
        cpd['objectinfo']['simbad_best_mainid'] is not None):
        newcpd['objectinfo']['simbad_best_mainid'] = deepcopy(
            cpd['objectinfo']['simbad_best_mainid']
        )
    if (newcpd['objectinfo']['simbad_best_objtype'] is None and
        cpd['objectinfo']['simbad_best_objtype'] is not None):
        newcpd['objectinfo']['simbad_best_objtype'] = deepcopy(
            cpd['objectinfo']['simbad_best_objtype']
        )
    if (newcpd['objectinfo']['simbad_best_allids'] is None and
        cpd['objectinfo']['simbad_best_allids'] is not None):
        newcpd['objectinfo']['simbad_best_allids'] = deepcopy(
            cpd['objectinfo']['simbad_best_allids']
        )
    if (newcpd['objectinfo']['simbad_best_distarcsec'] is None and
        cpd['objectinfo']['simbad_best_distarcsec'] is not None):
        newcpd['objectinfo']['simbad_best_distarcsec'] = deepcopy(
            cpd['objectinfo']['simbad_best_distarcsec']
        )

    #
    # update the objectinfo dict
    #
    cpd.update(newcpd)
    cpd['objectinfo']['objecttags'] = objecttags
    cpd['comments'] = comments

    newcpf = _write_checkplot_picklefile(cpd, outfile=cpf)

    return newcpf


########################################################
## FINALIZING CHECKPLOTS AFTER ALL PROCESSING IS DONE ##
########################################################

def finalize_checkplot(cpx,
                       outdir,
                       all_lclistpkl,
                       objfits=None):
    '''This is used to prevent any further changes to the checkplot.

    TODO: finish this.

    Use this function after all variable classification, period-finding, and
    object xmatches are done. This function will add a 'final' key to the
    checkplot, which will contain:

    - a phased LC plot with the period and epoch set after review using the
      times, mags, errs after any appropriate filtering and sigclip was done in
      the checkplotserver UI

    - The unphased LC using the times, mags, errs after any appropriate
      filtering and sigclip was done in the checkplotserver UI

    - the same plots for any LC collection neighbors

    - the survey cutout for the object if objfits is provided and checks out

    - a redone neighbor search using GAIA and all light curves in the collection
      even if they don't have at least 1000 observations.

    These items will be shown in a special 'Final' tab in the `checkplotserver`
    webapp (this should be run in `readonly` mode as well). The final tab will
    also contain downloadable links for the checkplot pickle in pkl and PNG
    format, as well as the final times, mags, errs as a gzipped CSV with a
    header containing all of this info.

    Parameters
    ----------

    cpx : dict or str
        This is the path to the checkplot dict or pickle file to process.

    outdir : str
        This is the directory to where the final pickle will be written. If this
        is set to the same dir as `cpx` and `cpx` is a pickle, the function will
        return a failure. This is meant to keep the in-process checkplots
        separate from the finalized versions.

    all_lclistpkl : str or dict
        This is the path to the pickle or the dict created by
        :py:func:`astrobase.lcproc.catalogs.make_lclist` with no restrictions on
        the number of observations (so ALL light curves in the collection). This
        is used to make sure all neighbors of this object in the light curve
        collection have had their proximity to this object noted.

    objfits : str or None
        If this is not None, should be a file path to a FITS file containing a
        WCS header and this object from the instrument that was used to observe
        it. This will be used to make a stamp cutout of the object using the
        actual image it was detected on. This will be a useful comparison to the
        usual DSS POSS-RED2 image used by the checkplots.

    Returns
    -------

    str
        The path to the updated checkplot pickle file with a 'final' key added
        to it , as described above.

    '''


def parallel_finalize_cplist(cplist,
                             outdir,
                             objfits=None):
    '''This is a parallel driver for `finalize_checkplot`, operating on a list
    of checkplots.

    Parameters
    ----------

    cplist : list of str
        This is a list of paths of all checkplot pickles to process and run
        through the finalization process.

    outdir : str
        The directory to where the finalized checkplot pickles will be written.

    objfits : str
        Path to the FITS file containing a WCS header and detections of all
        objects observed by the actual instrument that obtained the light
        curves. This should generally be a high quality 'reference' frame so
        that all of the objects whose checkplots we're finalizing (in `cplist`)
        can be seen on the frame.

    Returns
    -------

    dict
        Dict indicating the success/failure (as True/False) of the checkplot
        finalize operations for each checkplot pickle provided in `cplist`.

    '''


def parallel_finalize_cpdir(cpdir,
                            outdir,
                            cpfileglob='checkplot-*.pkl*',
                            objfits=None):
    '''This is a parallel driver for `finalize_checkplot`, operating on a
    directory of checkplots.

    Parameters
    ----------

    cpdir : str
        This is the path to the directory containing all the checkplot pickles
        to process and run through the finalization process.

    outdir : str
        The directory to where the finalized checkplot pickles will be written.

    objfits : str
        Path to the FITS file containing a WCS header and detections of all
        objects observed by the actual instrument that obtained the light
        curves. This should generally be a high quality 'reference' frame so
        that all of the objects whose checkplots we're finalizing (in `cplist`)
        can be seen on the frame.

    Returns
    -------

    dict
        Dict indicating the success/failure (as True/False) of the checkplot
        finalize operations for each checkplot pickle provided in `cplist`.

    '''
