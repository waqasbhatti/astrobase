#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''checkplot.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017
License: MIT.

The `checkplot_pickle` function takes, for a single object, an arbitrary number
of results from independent period-finding functions (e.g. BLS, PDM, AoV, GLS,
etc.) in periodbase, and generates a pickle file that contains object and
variability information, finder chart, mag series plot, and for each
period-finding result: a periodogram and phased mag series plots for an
arbitrary number of 'best periods'.

Checkplot pickles are intended for use with an external checkplot viewer: the
Tornado webapp `astrobase.cpserver.checkplotserver.py`, but you can also use the
`checkplot.pkl_png.checkplot_pickle_to_png` function to render checkplot pickles
to PNGs that will look something like:

    [    finder    ] [  objectinfo  ] [ variableinfo ] [ unphased LC  ]
    [ periodogram1 ] [ phased LC P1 ] [ phased LC P2 ] [ phased LC P3 ]
    [ periodogram2 ] [ phased LC P1 ] [ phased LC P2 ] [ phased LC P3 ]
                                     .
                                     .
    [ periodogramN ] [ phased LC P1 ] [ phased LC P2 ] [ phased LC P3 ]

    for N independent period-finding methods producing:

    - periodogram1,2,3...N: the periodograms from each method
    - phased LC P1,P2,P3: the phased lightcurves using the best 3 peaks in each
                          periodogram

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
import hashlib

try:
    import cPickle as pickle
except Exception as e:
    import pickle

# we're going to plot using Agg only
import matplotlib
MPLVERSION = tuple([int(x) for x in matplotlib.__version__.split('.')])
matplotlib.use('Agg')

# import this to check if stimes, smags, serrs are Column objects
from astropy.table import Column as AstColumn



###################
## LOCAL IMPORTS ##
###################

from ..lcmath import normalize_magseries, sigclip_magseries
from ..varclass.varfeatures import all_nonperiodic_features


#
# import the checkplot pickle helper functions
#

from .pkl_io import (
    _read_checkplot_picklefile,
    _write_checkplot_picklefile
)

from .pkl_utils import (
    _pkl_finder_objectinfo,
    _pkl_periodogram,
    _pkl_magseries_plot,
    _pkl_phased_magseries_plot
)

from .pkl_xmatch import xmatch_external_catalogs



#############################
## CHECKPLOT DICT FUNCTION ##
#############################

def checkplot_dict(lspinfolist,
                   times,
                   mags,
                   errs,
                   fast_mode=False,
                   magsarefluxes=False,
                   nperiodstouse=3,
                   objectinfo=None,
                   deredden_object=True,
                   custom_bandpasses=None,
                   gaia_submit_timeout=10.0,
                   gaia_submit_tries=3,
                   gaia_max_timeout=180.0,
                   gaia_mirror=None,
                   complete_query_later=True,
                   varinfo=None,
                   getvarfeatures=True,
                   lclistpkl=None,
                   nbrradiusarcsec=60.0,
                   maxnumneighbors=5,
                   xmatchinfo=None,
                   xmatchradiusarcsec=3.0,
                   lcfitfunc=None,
                   lcfitparams=None,
                   externalplots=None,
                   findercmap='gray_r',
                   finderconvolve=None,
                   findercachedir='~/.astrobase/stamp-cache',
                   normto='globalmedian',
                   normmingap=4.0,
                   sigclip=4.0,
                   varepoch='min',
                   phasewrap=True,
                   phasesort=True,
                   phasebin=0.002,
                   minbinelems=7,
                   plotxlim=(-0.8,0.8),
                   xliminsetmode=False,
                   plotdpi=100,
                   bestperiodhighlight=None,
                   xgridlines=None,
                   mindet=99,
                   verbose=True):

    '''This writes a multiple lspinfo checkplot to a dict.

    This function can take input from multiple lspinfo dicts (e.g. a list of
    output dicts or gzipped pickles of dicts from independent runs of BLS, PDM,
    AoV, or GLS period-finders in periodbase).

    NOTE: if lspinfolist contains more than one lspinfo object with the same
    lspmethod ('pdm','gls','sls','aov','bls'), the latest one in the list will
    overwrite the earlier ones.

    The output dict contains all the plots (magseries and phased
    magseries), periodograms, object information, variability information, light
    curves, and phased light curves. This can be written to:

    - a pickle with checkplot.checkplot_pickle below
    - a PNG with checkplot.checkplot_dict_png below

    All kwargs are the same as for checkplot_png, except for the following:

    If fast_mode is True, the following kwargs will be set to try to speed up
    hits to external services:

    skyview_timeout = 10.0
    dust_timeout = 10.0
    gaia_submit_timeout = 5.0
    gaia_max_timeout = 10.0
    gaia_submit_tries = 2
    complete_query_later = False

    If fast_mode = True, no calls will be made to SkyView or SIMBAD.

    If fast_mode is a positive integer or float, timeouts will be set to
    fast_mode and the gaia_submit_timeout will be set to
    0.66*fast_mode. gaia_submit_timeout and gaia_max_timeout are re-used for
    SIMBAD as well. No calls will be made to SIMBAD, but SkyView will still be
    queried for the finder-chart.

    nperiodstouse controls how many 'best' periods to make phased LC plots
    for. By default, this is the 3 best. If this is set to None, all 'best'
    periods present in each lspinfo dict's 'nbestperiods' key will be plotted
    (this is 5 according to periodbase functions' defaults).

    varinfo is a dictionary with the following keys:

      {'objectisvar': True if object is time-variable,
       'vartags': list of variable type tags (strings),
       'varisperiodic': True if object is a periodic variable,
       'varperiod': variability period of the object,
       'varepoch': epoch of variability in JD}

    if varinfo is None, an initial empty dictionary of this form will be created
    and written to the output pickle. This can be later updated using
    checkplotviewer.py, etc.

    If getvarfeatures is True, will use the function
    varbase.features.all_nonperiodic_features to calculate several light curve
    features such as the median, MAD, Stetson J index, CDPP, percentiles, etc.

    maxnumneighbors is the maximum number of neighbors within
    nbrradiusarcsec to include as neighbors in the checkplot for checking
    for blends.

    lcfitfunc is a Python function that is used to fit a model to the light
    curve. This is then overplotted for each phased light curve in the
    checkplot. This function should have the following signature:

    def lcfitfunc(times, mags, errs, period, **lcfitparams)

    where lcfitparams encapsulates all external parameters (i.e. number of knots
    for a spline function, the degree of a Legendre polynomial fit, etc.)  This
    function should return a Python dict with the following structure (similar
    to the functions in astrobase.varbase.lcfit) and at least the keys below:

    {'fittype':<str: name of fit method>,
     'fitchisq':<float: the chi-squared value of the fit>,
     'fitredchisq':<float: the reduced chi-squared value of the fit>,
     'fitinfo':{'fitmags':<ndarray: model mags or fluxes from fit function>},
     'magseries':{'times':<ndarray: times at which the fitmags are evaluated>}}

    additional keys can include ['fitinfo']['finalparams'] for the final model
    fit parameters (this will be used by the checkplotserver if present),
    ['fitinfo']['fitepoch'] for the minimum light epoch returned by the model
    fit, among others. in any case, the output dict of lcfitfunc will be copied
    to the output checkplot pickle's ['lcfit'][<fittype>] key:val dict for each
    phased light curve.

    externalplots is a list of 4-element tuples containing:

    1. path to PNG of periodogram from a external period-finding method
    2. path to PNG of best period phased light curve from external period-finder
    3. path to PNG of 2nd-best phased light curve from external period-finder
    4. path to PNG of 3rd-best phased light curve from external period-finder

    This can be used to incorporate external period-finding method results into
    the output checkplot pickle or exported PNG to allow for comparison with
    astrobase results.

    example of externalplots:

    extrarows = [('/path/to/external/bls-periodogram.png',
                  '/path/to/external/bls-phasedlc-plot-bestpeak.png',
                  '/path/to/external/bls-phasedlc-plot-peak2.png',
                  '/path/to/external/bls-phasedlc-plot-peak3.png'),
                 ('/path/to/external/pdm-periodogram.png',
                  '/path/to/external/pdm-phasedlc-plot-bestpeak.png',
                  '/path/to/external/pdm-phasedlc-plot-peak2.png',
                  '/path/to/external/pdm-phasedlc-plot-peak3.png'),
                  ...]

    If externalplots is provided, the checkplot_pickle_to_png function below
    will automatically retrieve these plot PNGs and put them into the exported
    checkplot PNG.

    varepoch sets the time of minimum light finding strategy for the checkplot:

                                               the epoch used for all phased
    if varepoch is None                     -> light curve plots will be
                                               min(times)

    if varepoch is a single string == 'min' -> automatic epoch finding for all
                                               periods using light curve fits

    if varepoch is a single float           -> this epoch will be used for all
                                               phased light curve plots

    if varepoch is a list of lists             each epoch will be applied for
    each of which has floats with           -> the phased light curve for each
    list length == nperiodstouse               period for each period-finder
    from period-finder results                 method specifically


    sigclip is either a single float or a list of two floats. in the first case,
    the sigclip is applied symmetrically. in the second case, the first sigclip
    in the list is applied to +ve magnitude deviations (fainter) and the second
    sigclip in the list is applied to -ve magnitude deviations (brighter).
    An example list would be `[10.,3.]` (for 10 sigma dimmings, 3 sigma
    brightenings).

    bestperiodhighlight sets whether user wants a background on the phased light
    curve from each periodogram type to distinguish them from others. this is an
    HTML hex color specification. If this is None, no highlight will be added.

    xgridlines (default None) can be a list, e.g., [-0.5,0.,0.5] that sets the
    x-axis grid lines on plotted phased LCs for easy visual identification of
    important features.

    xliminsetmode = True sets up the phased mag series plot to show a zoomed-in
    portion (set by plotxlim) as the main plot and an inset version of the full
    phased light curve from phase 0.0 to 1.0. This can be useful if searching
    for small dips near phase 0.0 caused by planetary transits for example.

    '''

    # if an objectinfo dict is absent, we'll generate a fake objectid based on
    # the second five time and mag array values. this should be OK to ID the
    # object across repeated runs of this function with the same times, mags,
    # errs, but should provide enough uniqueness otherwise (across different
    # times/mags array inputs). this is all done so we can still save checkplots
    # correctly to pickles after reviewing them using checkplotserver
    try:
        objuuid = hashlib.sha512(times[5:10].tostring() +
                                 mags[5:10].tostring()).hexdigest()[:5]
    except Exception as e:
        if verbose:
            LOGWARNING('times, mags, and errs may have too few items')
            objuuid = hashlib.sha512(times.tostring() +
                                     mags.tostring()).hexdigest()[:5]

    if (objectinfo is None):
        if verbose:
            LOGWARNING('no objectinfo dict provided as kwarg, '
                       'adding a randomly generated objectid')
        objectinfo = {'objectid':objuuid}

    # special for HAT stuff, eventually we should add objectid to
    # lcd['objectinfo'] there as well
    elif (isinstance(objectinfo, dict) and 'hatid' in objectinfo):
        objectinfo['objectid'] = objectinfo['hatid']

    elif ((isinstance(objectinfo, dict) and 'objectid' not in objectinfo) or
          (isinstance(objectinfo, dict) and 'objectid' in objectinfo and
           (objectinfo['objectid'] is None or objectinfo['objectid'] == ''))):
        if verbose:
            LOGWARNING('adding a randomly generated objectid '
                       'since none was provided in objectinfo dict')
        objectinfo['objectid'] = objuuid



    # 0. get the objectinfo and finder chart and initialize the checkplotdict
    checkplotdict = _pkl_finder_objectinfo(
        objectinfo,
        varinfo,
        findercmap,
        finderconvolve,
        sigclip,
        normto,
        normmingap,
        deredden_object=deredden_object,
        custom_bandpasses=custom_bandpasses,
        lclistpkl=lclistpkl,
        nbrradiusarcsec=nbrradiusarcsec,
        maxnumneighbors=maxnumneighbors,
        plotdpi=plotdpi,
        verbose=verbose,
        findercachedir=findercachedir,
        gaia_submit_timeout=gaia_submit_timeout,
        gaia_submit_tries=gaia_submit_tries,
        gaia_max_timeout=gaia_max_timeout,
        gaia_mirror=gaia_mirror,
        complete_query_later=complete_query_later,
        fast_mode=fast_mode
    )

    # try again to get the right objectid
    if (objectinfo and isinstance(objectinfo, dict) and
        'objectid' in objectinfo and objectinfo['objectid']):
        checkplotdict['objectid'] = objectinfo['objectid']

    # filter the input times, mags, errs; do sigclipping and normalization
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # fail early if not enough light curve points
    if ((stimes is None) or (smags is None) or (serrs is None) or
        (stimes.size < 49) or (smags.size < 49) or (serrs.size < 49)):

        LOGERROR("one or more of times, mags, errs appear to be None "
                 "after sig-clipping. are the measurements all nan? "
                 "can't make a checkplot for this objectid: %s" %
                 checkplotdict['objectid'])
        checkplotdict['magseries'] = None
        checkplotdict['status'] = 'failed: LC points appear to be all nan'
        return checkplotdict


    # this may fix some unpickling issues for astropy.table.Column objects
    # we convert them back to ndarrays
    if isinstance(stimes, AstColumn):
        stimes = stimes.data
        LOGWARNING('times is an astropy.table.Column object, '
                   'changing to numpy array because of '
                   'potential unpickling issues')
    if isinstance(smags, AstColumn):
        smags = smags.data
        LOGWARNING('mags is an astropy.table.Column object, '
                   'changing to numpy array because of '
                   'potential unpickling issues')
    if isinstance(serrs, AstColumn):
        serrs = serrs.data
        LOGWARNING('errs is an astropy.table.Column object, '
                   'changing to numpy array because of '
                   'potential unpickling issues')


    # report on how sigclip went
    if verbose:
        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))


    # take care of the normalization
    if normto is not False:
        stimes, smags = normalize_magseries(stimes, smags,
                                            normto=normto,
                                            magsarefluxes=magsarefluxes,
                                            mingap=normmingap)

    # make sure we have some lightcurve points to plot after sigclip
    if len(stimes) > mindet:

        # 1. get the mag series plot using these filtered stimes, smags, serrs
        magseriesdict = _pkl_magseries_plot(stimes, smags, serrs,
                                            plotdpi=plotdpi,
                                            magsarefluxes=magsarefluxes)

        # update the checkplotdict
        checkplotdict.update(magseriesdict)

        # 2. for each lspinfo in lspinfolist, read it in (from pkl or pkl.gz
        # if necessary), make the periodogram, make the phased mag series plots
        # for each of the nbestperiods in each lspinfo dict
        checkplot_pfmethods = []

        for lspind, lspinfo in enumerate(lspinfolist):

            # get the LSP from a pickle file transparently
            if isinstance(lspinfo,str) and os.path.exists(lspinfo):
                LOGINFO('loading LSP info from pickle %s' % lspinfo)

                if '.gz' in lspinfo:
                    with gzip.open(lspinfo,'rb') as infd:
                        lspinfo = pickle.load(infd)
                else:
                    with open(lspinfo,'rb') as infd:
                        lspinfo = pickle.load(infd)

            # make the periodogram first

            # we'll prepend the lspmethod index to allow for multiple same
            # lspmethods
            override_pfmethod = '%s-%s' % (lspind, lspinfo['method'])

            periodogramdict = _pkl_periodogram(
                lspinfo,
                plotdpi=plotdpi,
                override_pfmethod=override_pfmethod
            )

            # update the checkplotdict.
            checkplotdict.update(periodogramdict)

            # now, make the phased light curve plots for each of the
            # nbestperiods from this periodogram
            for nbpind, nbperiod in enumerate(
                    lspinfo['nbestperiods'][:nperiodstouse]
            ):

                # if there's a function to use for fitting, do the fit
                if lcfitfunc:
                    try:
                        if lcfitparams is None:
                            lcfitparams = {}
                        overplotfit = lcfitfunc(stimes,
                                                smags,
                                                serrs,
                                                nbperiod,
                                                **lcfitparams)
                    except Exception as e:
                        LOGEXCEPTION('the light curve fitting function '
                                     'failed, not plotting a fit over the '
                                     'phased light curve')
                        overplotfit = None
                else:
                    overplotfit = None


                # get the varepoch from a run of bls_snr if available. this
                # allows us to use the correct transit center epochs if
                # calculated using bls_snr and added back to the kbls function
                # result dicts
                if (lspinfo is not None and
                    'bls' in lspinfo['method'] and
                    'epochs' in lspinfo):
                    thisvarepoch = lspinfo['epochs'][nbpind]
                    if verbose:
                        LOGINFO(
                            'using pre-calculated transit-center epoch value: '
                            '%.6f from kbls.bls_snr() for period: %.5f'
                            % (thisvarepoch, nbperiod)
                        )
                else:
                    thisvarepoch = varepoch

                # this updates things as it runs
                checkplotdict = _pkl_phased_magseries_plot(
                    checkplotdict,
                    lspinfo['method'],
                    nbpind,
                    stimes, smags, serrs,
                    nbperiod, thisvarepoch,
                    lspmethodind=lspind,
                    phasewrap=phasewrap,
                    phasesort=phasesort,
                    phasebin=phasebin,
                    minbinelems=minbinelems,
                    plotxlim=plotxlim,
                    overplotfit=overplotfit,
                    plotdpi=plotdpi,
                    bestperiodhighlight=bestperiodhighlight,
                    magsarefluxes=magsarefluxes,
                    xliminsetmode=xliminsetmode,
                    xgridlines=xgridlines,
                    verbose=verbose,
                    override_pfmethod=override_pfmethod,
                )

            # if there's an snr key for this lspmethod, add the info in it to
            # the checkplotdict as well
            if 'snr' in lspinfo:
                if override_pfmethod in checkplotdict:
                    checkplotdict[override_pfmethod]['snr'] = (
                        lspinfo['snr']
                    )
            if 'transitdepth' in lspinfo:
                if override_pfmethod in checkplotdict:
                    checkplotdict[override_pfmethod]['transitdepth'] = (
                        lspinfo['transitdepth']
                    )
            if 'transitduration' in lspinfo:
                if override_pfmethod in checkplotdict:
                    checkplotdict[override_pfmethod]['transitduration'] = (
                        lspinfo['transitduration']
                    )

            checkplot_pfmethods.append(override_pfmethod)

        #
        # end of processing each pfmethod
        #

        ## update the checkplot dict with some other stuff that's needed by
        ## checkplotserver

        # 3. add a comments key:val
        checkplotdict['comments'] = None

        # 4. calculate some variability features
        if getvarfeatures is True:
            checkplotdict['varinfo']['features'] = all_nonperiodic_features(
                stimes,
                smags,
                serrs,
                magsarefluxes=magsarefluxes,
            )

        # 5. add a signals key:val. this will be used by checkplotserver's
        # pre-whitening and masking functions. these will write to
        # checkplotdict['signals']['whiten'] and
        # checkplotdict['signals']['mask'] respectively.
        checkplotdict['signals'] = {}

        # 6. add any externalplots if we have them
        checkplotdict['externalplots'] = []

        if (externalplots and
            isinstance(externalplots, list) and
            len(externalplots) > 0):

            for externalrow in externalplots:

                if all(os.path.exists(erowfile) for erowfile in externalrow):
                    if verbose:
                        LOGINFO('adding external plots: %s to checkplot dict' %
                                repr(externalrow))
                    checkplotdict['externalplots'].append(externalrow)
                else:
                    LOGWARNING('could not add some external '
                               'plots in: %s to checkplot dict'
                               % repr(externalrow))

        # 7. do any xmatches required
        if xmatchinfo is not None:
            checkplotdict = xmatch_external_catalogs(
                checkplotdict,
                xmatchinfo,
                xmatchradiusarcsec=xmatchradiusarcsec
            )

        # the checkplotdict now contains everything we need
        contents = sorted(list(checkplotdict.keys()))
        checkplotdict['status'] = 'ok: contents are %s' % contents

        if verbose:
            LOGINFO('checkplot dict complete for %s' %
                    checkplotdict['objectid'])
            LOGINFO('checkplot dict contents: %s' % contents)


        # 8. update the pfmethods key
        checkplotdict['pfmethods'] = checkplot_pfmethods

    # otherwise, we don't have enough LC points, return nothing
    else:

        LOGERROR('not enough light curve points for %s, have %s, need %s' %
                 (checkplotdict['objectid'],len(stimes),mindet))
        checkplotdict['magseries'] = None
        checkplotdict['status'] = 'failed: not enough LC points'

    # at the end, return the dict
    return checkplotdict



################################
## CHECKPLOT PICKLE FUNCTIONS ##
################################

def checkplot_pickle(lspinfolist,
                     times,
                     mags,
                     errs,
                     fast_mode=False,
                     magsarefluxes=False,
                     nperiodstouse=3,
                     objectinfo=None,
                     deredden_object=True,
                     custom_bandpasses=None,
                     gaia_submit_timeout=10.0,
                     gaia_submit_tries=3,
                     gaia_max_timeout=180.0,
                     gaia_mirror=None,
                     complete_query_later=True,
                     lcfitfunc=None,
                     lcfitparams=None,
                     varinfo=None,
                     getvarfeatures=True,
                     lclistpkl=None,
                     nbrradiusarcsec=60.0,
                     maxnumneighbors=5,
                     xmatchinfo=None,
                     xmatchradiusarcsec=3.0,
                     externalplots=None,
                     findercmap='gray_r',
                     finderconvolve=None,
                     findercachedir='~/.astrobase/stamp-cache',
                     normto='globalmedian',
                     normmingap=4.0,
                     outfile=None,
                     outgzip=False,
                     sigclip=4.0,
                     varepoch='min',
                     phasewrap=True,
                     phasesort=True,
                     phasebin=0.002,
                     minbinelems=7,
                     plotxlim=(-0.8,0.8),
                     xliminsetmode=False,
                     plotdpi=100,
                     returndict=False,
                     pickleprotocol=None,
                     bestperiodhighlight=None,
                     xgridlines=None,
                     mindet=99,
                     verbose=True):

    '''This writes a multiple lspinfo checkplot to a (gzipped) pickle file.

    This function can take input from multiple lspinfo dicts (e.g. a list of
    output dicts or gzipped pickles of dicts from independent runs of BLS, PDM,
    AoV, or GLS period-finders in periodbase).

    NOTE: if lspinfolist contains more than one lspinfo object with the same
    lspmethod ('pdm','gls','sls','aov','bls'), the latest one in the list will
    overwrite the earlier ones.

    The output pickle contains all the plots (magseries and phased magseries),
    periodograms, object information, variability information, light curves, and
    phased light curves. The pickle produced by this function can be used with
    an external viewer app (e.g. checkplotserver.py), or by using the
    checkplot_pickle_to_png function below.

    All kwargs are the same as for checkplot_png, except for the following:

    If fast_mode is True, the following kwargs will be set to try to speed up
    hits to external services:

    skyview_timeout = 10.0
    dust_timeout = 10.0
    gaia_submit_timeout = 5.0
    gaia_max_timeout = 10.0
    gaia_submit_tries = 2
    complete_query_later = False

    If fast_mode = True, no calls will be made to SkyView or SIMBAD.

    If fast_mode is a positive integer or float, timeouts will be set to
    fast_mode and the gaia_submit_timeout will be set to
    0.66*fast_mode. gaia_submit_timeout and gaia_max_timeout are re-used for
    SIMBAD as well. No calls will be made to SIMBAD, but SkyView will still be
    queried for the finder-chart.

    nperiodstouse controls how many 'best' periods to make phased LC plots
    for. By default, this is the 3 best. If this is set to None, all 'best'
    periods present in each lspinfo dict's 'nbestperiods' key will be plotted
    (this is 5 according to periodbase functions' defaults).

    varinfo is a dictionary with the following keys:

      {'objectisvar': True if object is time-variable,
       'vartags': list of variable type tags (strings),
       'varisperiodic': True if object is a periodic variable,
       'varperiod': variability period of the object,
       'varepoch': epoch of variability in JD}

    if varinfo is None, an initial empty dictionary of this form will be created
    and written to the output pickle. This can be later updated using
    checkplotviewer.py, etc.

    If getvarfeatures is True, will use the function
    varbase.features.all_nonperiodic_features to calculate several light curve
    features such as the median, MAD, Stetson J index, CDPP, percentiles, etc.

    lcfitfunc is a Python function that is used to fit a model to the light
    curve. This is then overplotted for each phased light curve in the
    checkplot. This function should have the following signature:

    def lcfitfunc(times, mags, errs, period, **lcfitparams)

    where lcfitparams encapsulates all external parameters (i.e. number of knots
    for a spline function, the degree of a Legendre polynomial fit, etc.)  This
    function should return a Python dict with the following structure (similar
    to the functions in astrobase.varbase.lcfit) and at least the keys below:

    {'fittype':<str: name of fit method>,
     'fitchisq':<float: the chi-squared value of the fit>,
     'fitredchisq':<float: the reduced chi-squared value of the fit>,
     'fitinfo':{'fitmags':<ndarray: model mags or fluxes from fit function>},
     'magseries':{'times':<ndarray: times at which the fitmags are evaluated>}}

    additional keys can include ['fitinfo']['finalparams'] for the final model
    fit parameters, ['fitinfo']['fitepoch'] for the minimum light epoch returned
    by the model fit, among others. the output dict of lcfitfunc will be copied
    to the output checkplot dict's ['fitinfo'][<fittype>] key:val dict.

    externalplots is a list of 4-element tuples containing:

    1. path to PNG of periodogram from a external period-finding method
    2. path to PNG of best period phased light curve from external period-finder
    3. path to PNG of 2nd-best phased light curve from external period-finder
    4. path to PNG of 3rd-best phased light curve from external period-finder

    This can be used to incorporate external period-finding method results into
    the output checkplot pickle or exported PNG to allow for comparison with
    astrobase results.

    example of externalplots:

    extrarows = [('/path/to/external/bls-periodogram.png',
                  '/path/to/external/bls-phasedlc-plot-bestpeak.png',
                  '/path/to/external/bls-phasedlc-plot-peak2.png',
                  '/path/to/external/bls-phasedlc-plot-peak3.png'),
                 ('/path/to/external/pdm-periodogram.png',
                  '/path/to/external/pdm-phasedlc-plot-bestpeak.png',
                  '/path/to/external/pdm-phasedlc-plot-peak2.png',
                  '/path/to/external/pdm-phasedlc-plot-peak3.png'),
                  ...]

    If externalplots is provided, the checkplot_pickle_to_png function below
    will automatically retrieve these plot PNGs and put them into the exported
    checkplot PNG.

    sigclip is either a single float or a list of two floats. in the first case,
    the sigclip is applied symmetrically. in the second case, the first sigclip
    in the list is applied to +ve magnitude deviations (fainter) and the second
    sigclip in the list is applied to -ve magnitude deviations (brighter).
    An example list would be `[10.,3.]` (for 10 sigma dimmings, 3 sigma
    brightenings).

    bestperiodhighlight sets whether user wants a background on the phased light
    curve from each periodogram type to distinguish them from others. this is an
    HTML hex color specification. If this is None, no highlight will be added.

    xgridlines (default None) can be a list, e.g., [-0.5,0.,0.5] that sets the
    x-axis grid lines on plotted phased LCs for easy visual identification of
    important features.

    xliminsetmode = True sets up the phased mag series plot to show a zoomed-in
    portion (set by plotxlim) as the main plot and an inset version of the full
    phased light curve from phase 0.0 to 1.0. This can be useful if searching
    for small dips near phase 0.0 caused by planetary transits for example.

    outgzip controls whether to gzip the output pickle. it turns out that this
    is the slowest bit in the output process, so if you're after speed, best not
    to use this. this is False by default since it turns out that gzip actually
    doesn't save that much space (29 MB vs. 35 MB for the average checkplot
    pickle).

    '''

    # call checkplot_dict for most of the work
    checkplotdict = checkplot_dict(
        lspinfolist,
        times,
        mags,
        errs,
        magsarefluxes=magsarefluxes,
        nperiodstouse=nperiodstouse,
        objectinfo=objectinfo,
        deredden_object=deredden_object,
        custom_bandpasses=custom_bandpasses,
        gaia_submit_timeout=gaia_submit_timeout,
        gaia_submit_tries=gaia_submit_tries,
        gaia_max_timeout=gaia_max_timeout,
        gaia_mirror=gaia_mirror,
        complete_query_later=complete_query_later,
        varinfo=varinfo,
        getvarfeatures=getvarfeatures,
        lclistpkl=lclistpkl,
        nbrradiusarcsec=nbrradiusarcsec,
        maxnumneighbors=maxnumneighbors,
        xmatchinfo=xmatchinfo,
        xmatchradiusarcsec=xmatchradiusarcsec,
        lcfitfunc=lcfitfunc,
        lcfitparams=lcfitparams,
        externalplots=externalplots,
        findercmap=findercmap,
        finderconvolve=finderconvolve,
        findercachedir=findercachedir,
        normto=normto,
        normmingap=normmingap,
        sigclip=sigclip,
        varepoch=varepoch,
        phasewrap=phasewrap,
        phasesort=phasesort,
        phasebin=phasebin,
        minbinelems=minbinelems,
        plotxlim=plotxlim,
        xliminsetmode=xliminsetmode,
        plotdpi=plotdpi,
        bestperiodhighlight=bestperiodhighlight,
        xgridlines=xgridlines,
        mindet=mindet,
        verbose=verbose,
        fast_mode=fast_mode
    )

    # for Python >= 3.4, use v4
    if ((sys.version_info[0:2] >= (3,4) and not pickleprotocol) or
        (pickleprotocol > 2)):
        pickleprotocol = 4

    elif ((sys.version_info[0:2] >= (3,0) and not pickleprotocol) or
          (pickleprotocol > 2)):
        pickleprotocol = 3

    # for Python == 2.7; use v2
    elif sys.version_info[0:2] == (2,7) and not pickleprotocol:
        pickleprotocol = 2

    # otherwise, if left unspecified, use the slowest but most compatible
    # protocol. this will be readable by all (most?) Pythons
    elif not pickleprotocol:
        pickleprotocol = 0


    # generate the output file path
    if outgzip:

        # generate the outfile filename
        if (not outfile and
            len(lspinfolist) > 0 and
            isinstance(lspinfolist[0], str)):
            plotfpath = os.path.join(os.path.dirname(lspinfolist[0]),
                                     'checkplot-%s.pkl.gz' %
                                     checkplotdict['objectid'])
        elif outfile:
            plotfpath = outfile
        else:
            plotfpath = 'checkplot.pkl.gz'

    else:

        # generate the outfile filename
        if (not outfile and
            len(lspinfolist) > 0 and
            isinstance(lspinfolist[0], str)):
            plotfpath = os.path.join(os.path.dirname(lspinfolist[0]),
                                     'checkplot-%s.pkl' %
                                     checkplotdict['objectid'])
        elif outfile:
            plotfpath = outfile
        else:
            plotfpath = 'checkplot.pkl'


    # write the completed checkplotdict to a gzipped pickle
    picklefname = _write_checkplot_picklefile(checkplotdict,
                                              outfile=plotfpath,
                                              protocol=pickleprotocol,
                                              outgzip=outgzip)

    # at the end, return the dict and filename if asked for
    if returndict:
        if verbose:
            LOGINFO('checkplot done -> %s' % picklefname)
        return checkplotdict, picklefname

    # otherwise, just return the filename
    else:
        # just to make sure: free up space
        del checkplotdict
        if verbose:
            LOGINFO('checkplot done -> %s' % picklefname)
        return picklefname



def checkplot_pickle_update(currentcp, updatedcp,
                            outfile=None,
                            outgzip=False,
                            pickleprotocol=None,
                            verbose=True):
    '''This updates the current checkplot dict with updated values provided.

    current is either a checkplot dict produced by checkplot_pickle above or a
    gzipped pickle file produced by the same function. updated is a dict or
    pickle file with the same format as current.

    Writes out the new checkplot gzipped pickle file to outfile. If current is a
    file, updates it in place if outfile is None. Mostly only useful for
    checkplotserver.py.

    '''

    # break out python 2.7 and > 3 nonsense
    if sys.version_info[:2] > (3,2):

        # generate the outfile filename
        if not outfile and isinstance(currentcp,str):
            plotfpath = currentcp
        elif outfile:
            plotfpath = outfile
        elif isinstance(currentcp, dict) and currentcp['objectid']:
            if outgzip:
                plotfpath = 'checkplot-%s.pkl.gz' % currentcp['objectid']
            else:
                plotfpath = 'checkplot-%s.pkl' % currentcp['objectid']
        else:
            # we'll get this later below
            plotfpath = None

        if (isinstance(currentcp, str) and os.path.exists(currentcp)):
            cp_current = _read_checkplot_picklefile(currentcp)
        elif isinstance(currentcp, dict):
            cp_current = currentcp
        else:
            LOGERROR('currentcp: %s of type %s is not a '
                     'valid checkplot filename (or does not exist), or a dict' %
                     (os.path.abspath(currentcp), type(currentcp)))
            return None

        if (isinstance(updatedcp, str) and os.path.exists(updatedcp)):
            cp_updated = _read_checkplot_picklefile(updatedcp)
        elif isinstance(updatedcp, dict):
            cp_updated = updatedcp
        else:
            LOGERROR('updatedcp: %s of type %s is not a '
                     'valid checkplot filename (or does not exist), or a dict' %
                     (os.path.abspath(updatedcp), type(updatedcp)))
            return None

    # check for unicode in python 2.7
    else:

        # generate the outfile filename
        if (not outfile and
            (isinstance(currentcp, str) or isinstance(currentcp, unicode))):
            plotfpath = currentcp
        elif outfile:
            plotfpath = outfile
        elif isinstance(currentcp, dict) and currentcp['objectid']:
            if outgzip:
                plotfpath = 'checkplot-%s.pkl.gz' % currentcp['objectid']
            else:
                plotfpath = 'checkplot-%s.pkl' % currentcp['objectid']
        else:
            # we'll get this later below
            plotfpath = None

        # get the current checkplotdict
        if ((isinstance(currentcp, str) or isinstance(currentcp, unicode)) and
            os.path.exists(currentcp)):
            cp_current = _read_checkplot_picklefile(currentcp)
        elif isinstance(currentcp,dict):
            cp_current = currentcp
        else:
            LOGERROR('currentcp: %s of type %s is not a '
                     'valid checkplot filename (or does not exist), or a dict' %
                     (os.path.abspath(currentcp), type(currentcp)))
            return None

        # get the updated checkplotdict
        if ((isinstance(updatedcp, str) or isinstance(updatedcp, unicode)) and
            os.path.exists(updatedcp)):
            cp_updated = _read_checkplot_picklefile(updatedcp)
        elif isinstance(updatedcp, dict):
            cp_updated = updatedcp
        else:
            LOGERROR('updatedcp: %s of type %s is not a '
                     'valid checkplot filename (or does not exist), or a dict' %
                     (os.path.abspath(updatedcp), type(updatedcp)))
            return None

    # do the update using python's dict update mechanism
    # this requires updated to be in the same checkplotdict format as current
    # all keys in current will now be from updated
    cp_current.update(cp_updated)

    # figure out the plotfpath if we haven't by now
    if not plotfpath and outgzip:
        plotfpath = 'checkplot-%s.pkl.gz' % cp_current['objectid']
    elif (not plotfpath) and (not outgzip):
        plotfpath = 'checkplot-%s.pkl' % cp_current['objectid']

    # make sure we write the correct postfix
    if plotfpath.endswith('.gz'):
        outgzip = True

    # write the new checkplotdict
    return _write_checkplot_picklefile(cp_current,
                                       outfile=plotfpath,
                                       outgzip=outgzip,
                                       protocol=pickleprotocol)
