#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''pkl_postproc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
License: MIT.

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
    '''This just updates the checkplot objectinfo dict.

    Useful in cases where a previous round of GAIA/finderchart acquisition
    failed. This will preserve the following keys in the checkplot if they
    exist:

    comments
    varinfo
    objectinfo.objecttags

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

    cpx is the checkplot dict or pickle to process.

    outdir is the directory to where the final pickle will be written. If this
    is set to the same dir as cpx and cpx is a pickle, the function will return
    a failure. This is meant to keep the in-process checkplots separate from the
    finalized versions.

    all_lclistpkl is a pickle created by lcproc.make_lclist above with no
    restrictions on the number of observations (so ALL light curves in the
    collection).

    objfits if not None should be a file path to a FITS file containing a WCS
    header and this object. This will be used to make a stamp cutout of the
    object using the actual image it was detected on. This will be a useful
    comparison to the usual DSS POSS-RED2 image used by the checkplots.

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

    These items will be shown in a special 'Final' tab in the checkplotserver
    webapp (this should be run in readonly mode as well). The final tab will
    also contain downloadable links for the checkplot pickle in pkl and PNG
    format, as well as the final times, mags, errs as a gzipped CSV with a
    header containing all of this info (will be readable by the usual
    astrobase.hatsurveys.hatlc module).

    '''



def parallel_finalize_cplist(cplist,
                             outdir,
                             objfits=None):
    '''This is a parallel driver for the function above, operating on list of
    checkplots.

    '''



def parallel_finalize_cpdir(cpdir,
                            outdir,
                            cpfileglob='checkplot-*.pkl*',
                            objfits=None):
    '''This is a parallel driver for the function above, operating on a
    directory of checkplots.

    '''
