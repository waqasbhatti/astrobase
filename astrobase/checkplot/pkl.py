#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# checkplot.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017
# License: MIT.

'''
The `checkplot_pickle` function takes, for a single object, an arbitrary number
of results from independent period-finding functions (e.g. BLS, PDM, AoV, GLS,
etc.) in periodbase, and generates a pickle file that contains object and
variability information, finder chart, mag series plot, and for each
period-finding result: a periodogram and phased mag series plots for an
arbitrary number of 'best periods'.

Checkplot pickles are intended for use with an external checkplot viewer: the
Tornado webapp `astrobase.cpserver.checkplotserver.py`, but you can also use the
`checkplot.pkl_png.checkplot_pickle_to_png` function to render checkplot pickles
to PNGs that will look something like::

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
import hashlib
import pickle

# we're going to plot using Agg only
import matplotlib
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

def checkplot_dict(
        lspinfolist,
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
        verbose=True
):

    '''This writes a multiple lspinfo checkplot to a dict.

    This function can take input from multiple lspinfo dicts (e.g. a list of
    output dicts or gzipped pickles of dicts from independent runs of BLS, PDM,
    AoV, or GLS period-finders in periodbase).

    NOTE: if `lspinfolist` contains more than one lspinfo object with the same
    lspmethod ('pdm','gls','sls','aov','bls'), the latest one in the list will
    overwrite the earlier ones.

    The output dict contains all the plots (magseries and phased magseries),
    periodograms, object information, variability information, light curves, and
    phased light curves. This can be written to:

    - a pickle with `checkplot_pickle` below
    - a PNG with `checkplot.pkl_png.checkplot_pickle_to_png`

    Parameters
    ----------

    lspinfolist : list of dicts
        This is a list of dicts containing period-finder results ('lspinfo'
        dicts). These can be from any of the period-finder methods in
        astrobase.periodbase. To incorporate external period-finder results into
        checkplots, these dicts must be of the form below, including at least
        the keys indicated here::

            {'periods': np.array of all periods searched by the period-finder,
             'lspvals': np.array of periodogram power value for each period,
             'bestperiod': a float value that is the period with the highest
                           peak in the periodogram, i.e. the most-likely actual
                           period,
             'method': a three-letter code naming the period-finder used; must
                       be one of the keys in the
                       `astrobase.periodbase.METHODLABELS` dict,
             'nbestperiods': a list of the periods corresponding to periodogram
                             peaks (`nbestlspvals` below) to annotate on the
                             periodogram plot so they can be called out
                             visually,
             'nbestlspvals': a list of the power values associated with
                             periodogram peaks to annotate on the periodogram
                             plot so they can be called out visually; should be
                             the same length as `nbestperiods` above}

        `nbestperiods` and `nbestlspvals` in each lspinfo dict must have at
        least as many elements as the `nperiodstouse` kwarg to this function.

    times,mags,errs : np.arrays
        The magnitude/flux time-series to process for this checkplot along with
        their associated measurement errors.

    fast_mode : bool or float
        This runs the external catalog operations in a "fast" mode, with short
        timeouts and not trying to hit external catalogs that take a long time
        to respond.

        If this is set to True, the default settings for the external requests
        will then become::

                skyview_lookup = False
                skyview_timeout = 10.0
                skyview_retry_failed = False
                dust_timeout = 10.0
                gaia_submit_timeout = 7.0
                gaia_max_timeout = 10.0
                gaia_submit_tries = 2
                complete_query_later = False
                search_simbad = False

        If this is a float, will run in "fast" mode with the provided timeout
        value in seconds and the following settings::

                skyview_lookup = True
                skyview_timeout = fast_mode
                skyview_retry_failed = False
                dust_timeout = fast_mode
                gaia_submit_timeout = 0.66*fast_mode
                gaia_max_timeout = fast_mode
                gaia_submit_tries = 2
                complete_query_later = False
                search_simbad = False

    magsarefluxes : bool
        If True, indicates the input time-series is fluxes and not mags so the
        plot y-axis direction and range can be set appropriately.

    nperiodstouse : int
        This controls how many 'best' periods to make phased LC plots for. By
        default, this is the 3 best. If this is set to None, all 'best' periods
        present in each lspinfo dict's 'nbestperiods' key will be processed for
        this checkplot.

    objectinfo : dict or None
        This is a dict containing information on the object whose light
        curve is being processed. This function will then be able to
        look up and download a finder chart for this object and write
        that to the output checkplotdict. External services such as
        GAIA, SIMBAD, TIC, etc. will also be used to look up this object
        by its coordinates, and will add in information available from
        those services.

        This dict must be of the form and contain at least the keys described
        below::

            {'objectid': the name of the object,
             'ra': the right ascension of the object in decimal degrees,
             'decl': the declination of the object in decimal degrees,
             'ndet': the number of observations of this object}

        You can also provide magnitudes and proper motions of the object using
        the following keys and the appropriate values in the `objectinfo`
        dict. These will be used to calculate colors, total and reduced proper
        motion, etc. and display these in the output checkplot PNG::

            'pmra'   -> the proper motion in mas/yr in right ascension,
            'pmdecl' -> the proper motion in mas/yr in declination,
            'umag'  -> U mag		 -> colors: U-B, U-V, U-g
            'bmag'  -> B mag		 -> colors: U-B, B-V
            'vmag'  -> V mag		 -> colors: U-V, B-V, V-R, V-I, V-K
            'rmag'  -> R mag		 -> colors: V-R, R-I
            'imag'  -> I mag		 -> colors: g-I, V-I, R-I, B-I
            'jmag'  -> 2MASS J mag	 -> colors: J-H, J-K, g-J, i-J
            'hmag'  -> 2MASS H mag	 -> colors: J-H, H-K
            'kmag'  -> 2MASS Ks mag	 -> colors: g-Ks, H-Ks, J-Ks, V-Ks
            'sdssu' -> SDSS u mag	 -> colors: u-g, u-V
            'sdssg' -> SDSS g mag	 -> colors: g-r, g-i, g-K, u-g, U-g, g-J
            'sdssr' -> SDSS r mag	 -> colors: r-i, g-r
            'sdssi' -> SDSS i mag	 -> colors: r-i, i-z, g-i, i-J, i-W1
            'sdssz' -> SDSS z mag	 -> colors: i-z, z-W2, g-z
            'ujmag' -> UKIRT J mag	 -> colors: J-H, H-K, J-K, g-J, i-J
            'uhmag' -> UKIRT H mag	 -> colors: J-H, H-K
            'ukmag' -> UKIRT K mag	 -> colors: g-K, H-K, J-K, V-K
            'irac1' -> Spitzer IRAC1 mag -> colors: i-I1, I1-I2
            'irac2' -> Spitzer IRAC2 mag -> colors: I1-I2, I2-I3
            'irac3' -> Spitzer IRAC3 mag -> colors: I2-I3
            'irac4' -> Spitzer IRAC4 mag -> colors: I3-I4
            'wise1' -> WISE W1 mag	 -> colors: i-W1, W1-W2
            'wise2' -> WISE W2 mag	 -> colors: W1-W2, W2-W3
            'wise3' -> WISE W3 mag	 -> colors: W2-W3
            'wise4' -> WISE W4 mag	 -> colors: W3-W4

        If you have magnitude measurements in other bands, use the
        `custom_bandpasses` kwarg to pass these in.

        If this is None, no object information will be incorporated into the
        checkplot (kind of making it effectively useless for anything other than
        glancing at the phased light curves at various 'best' periods from the
        period-finder results).

    deredden_object : bool
        If this is True, will use the 2MASS DUST service to get extinction
        coefficients in various bands, and then try to deredden the magnitudes
        and colors of the object already present in the checkplot's objectinfo
        dict.

    custom_bandpasses : dict
        This is a dict used to provide custom bandpass definitions for any
        magnitude measurements in the objectinfo dict that are not automatically
        recognized by :py:func:`astrobase.varclass.starfeatures.color_features`.

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

    gaia_mirror : str or None
        This sets the GAIA mirror to use. This is a key in the
        `services.gaia.GAIA_URLS` dict which defines the URLs to hit for each
        mirror.

    complete_query_later : bool
        If this is True, saves the state of GAIA queries that are not yet
        complete when `gaia_max_timeout` is reached while waiting for the GAIA
        service to respond to our request. A later call for GAIA info on the
        same object will attempt to pick up the results from the existing query
        if it's completed. If `fast_mode` is True, this is ignored.

    varinfo : dict
        If this is None, a blank dict of the form below will be added to the
        checkplotdict::

            {'objectisvar': None -> variability flag (None indicates unset),
             'vartags': CSV str containing variability type tags from review,
             'varisperiodic': None -> periodic variability flag (None -> unset),
             'varperiod': the period associated with the periodic variability,
             'varepoch': the epoch associated with the periodic variability}

        If you provide a dict matching this format in this kwarg, this will be
        passed unchanged to the output checkplotdict produced.

    getvarfeatures : bool
        If this is True, several light curve variability features for this
        object will be calculated and added to the output checkpotdict as
        checkplotdict['varinfo']['features']. This uses the function
        `varclass.varfeatures.all_nonperiodic_features` so see its docstring for
        the measures that are calculated (e.g. Stetson J indices, dispersion
        measures, etc.)

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

    nbrradiusarcsec : flaot
        The radius in arcseconds to use for a search conducted around the
        coordinates of this object to look for any potential confusion and
        blending of variability amplitude caused by their proximity.

    maxnumneighbors : int
        The maximum number of neighbors that will have their light curves and
        magnitudes noted in this checkplot as potential blends with the target
        object.

    xmatchinfo : str or dict
        This is either the xmatch dict produced by the function
        `load_xmatch_external_catalogs` above, or the path to the xmatch info
        pickle file produced by that function.

    xmatchradiusarcsec : float
        This is the cross-matching radius to use in arcseconds.

    lcfitfunc : Python function or None
        If provided, this should be a Python function that is used to fit a
        model to the light curve. This fit is then overplotted for each phased
        light curve in the checkplot. This function should have the following
        signature:

        `def lcfitfunc(times, mags, errs, period, **lcfitparams)`

        where `lcfitparams` encapsulates all external parameters (i.e. number of
        knots for a spline function, the degree of a Legendre polynomial fit,
        etc., planet transit parameters) This function should return a Python
        dict with the following structure (similar to the functions in
        `astrobase.lcfit`) and at least the keys below::

            {'fittype':<str: name of fit method>,
             'fitchisq':<float: the chi-squared value of the fit>,
             'fitredchisq':<float: the reduced chi-squared value of the fit>,
             'fitinfo':{'fitmags':<ndarray: model mags/fluxes from fit func>},
             'magseries':{'times':<ndarray: times where fitmags are evaluated>}}

        Additional keys in the dict returned from this function can include
        `fitdict['fitinfo']['finalparams']` for the final model fit parameters
        (this will be used by the checkplotserver if present),
        `fitdict['fitinfo']['fitepoch']` for the minimum light epoch returned by
        the model fit, among others.

        In any case, the output dict of `lcfitfunc` will be copied to the output
        checkplotdict as::

            checkplotdict[lspmethod][periodind]['lcfit'][<fittype>]

        for each phased light curve.

    lcfitparams : dict
        A dict containing the LC fit parameters to use when calling the function
        provided in `lcfitfunc`. This contains key-val pairs corresponding to
        parameter names and their respective initial values to be used by the
        fit function.

    externalplots : list of tuples of str
        If provided, this is a list of 4-element tuples containing:

        1. path to PNG of periodogram from an external period-finding method
        2. path to PNG of best period phased LC from the external period-finder
        3. path to PNG of 2nd-best phased LC from the external period-finder
        4. path to PNG of 3rd-best phased LC from the external period-finder

        This can be used to incorporate external period-finding method results
        into the output checkplot pickle or exported PNG to allow for comparison
        with astrobase results.

        Example of externalplots::

                [('/path/to/external/bls-periodogram.png',
                 '/path/to/external/bls-phasedlc-plot-bestpeak.png',
                 '/path/to/external/bls-phasedlc-plot-peak2.png',
                 '/path/to/external/bls-phasedlc-plot-peak3.png'),
                 ('/path/to/external/pdm-periodogram.png',
                 '/path/to/external/pdm-phasedlc-plot-bestpeak.png',
                 '/path/to/external/pdm-phasedlc-plot-peak2.png',
                 '/path/to/external/pdm-phasedlc-plot-peak3.png'),
                 ...]

        If `externalplots` is provided here, these paths will be stored in the
        output checkplotdict. The `checkplot.pkl_png.checkplot_pickle_to_png`
        function can then automatically retrieve these plot PNGs and put
        them into the exported checkplot PNG.

    findercmap : str or matplotlib.cm.ColorMap object
        The Colormap object to use for the finder chart image.

    finderconvolve : astropy.convolution.Kernel object or None
        If not None, the Kernel object to use for convolving the finder image.

    findercachedir : str
        The path to the astrobase cache directory for finder chart downloads
        from the NASA SkyView service.

    normto : {'globalmedian', 'zero'} or a float
        These are specified as below:
        - 'globalmedian' -> norms each mag to the global median of the LC column
        - 'zero'         -> norms each mag to zero
        - a float        -> norms each mag to this specified float value.

    normmingap : float
        This defines how much the difference between consecutive measurements is
        allowed to be to consider them as parts of different timegroups. By
        default it is set to 4.0 days.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    varepoch : 'min' or float or list of lists or None
        The epoch to use for this phased light curve plot tile. If this is a
        float, will use the provided value directly. If this is 'min', will
        automatically figure out the time-of-minimum of the phased light
        curve. If this is None, will use the mimimum value of `stimes` as the
        epoch of the phased light curve plot. If this is a list of lists, will
        use the provided value of `lspmethodind` to look up the current
        period-finder method and the provided value of `periodind` to look up
        the epoch associated with that method and the current period. This is
        mostly only useful when `twolspmode` is True.

    phasewrap : bool
        If this is True, the phased time-series will be wrapped around
        phase 0.0.

    phasesort : bool
        If True, will sort the phased light curve in order of increasing phase.

    phasebin: float
        The bin size to use to group together measurements closer than this
        amount in phase. This is in units of phase. If this is a float, a
        phase-binned version of the phased light curve will be overplotted on
        top of the regular phased light curve.

    minbinelems : int
        The minimum number of elements required per phase bin to include it in
        the phased LC plot.

    plotxlim : sequence of two floats or None
        The x-range (min, max) of the phased light curve plot. If None, will be
        determined automatically.

    xliminsetmode : bool
        If this is True, the generated phased light curve plot will use the
        values of `plotxlim` as the main plot x-axis limits (i.e. zoomed-in if
        `plotxlim` is a range smaller than the full phase range), and will show
        the full phased light curve plot as an smaller inset. Useful for
        planetary transit light curves.

    plotdpi : int
        The resolution of the output plot PNGs in dots per inch.

    bestperiodhighlight : str or None
        If not None, this is a str with a matplotlib color specification to use
        as the background color to highlight the phased light curve plot of the
        'best' period and epoch combination. If None, no highlight will be
        applied.

    xgridlines : list of floats or None
        If this is provided, must be a list of floats corresponding to the phase
        values where to draw vertical dashed lines as a means of highlighting
        these.

    mindet : int
        The minimum of observations the input object's mag/flux time-series must
        have for this function to plot its light curve and phased light
        curve. If the object has less than this number, no light curves will be
        plotted, but the checkplotdict will still contain all of the other
        information.

    verbose : bool
        If True, will indicate progress and warn about problems.

    Returns
    -------

    dict
        Returns a checkplotdict.

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
    except Exception:
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
                    except Exception:
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
        contents = sorted(checkplotdict.keys())
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

def checkplot_pickle(
        lspinfolist,
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
        verbose=True,
        outfile=None,
        outgzip=False,
        pickleprotocol=None,
        returndict=False
):

    '''This writes a multiple lspinfo checkplot to a (gzipped) pickle file.

    This function can take input from multiple lspinfo dicts (e.g. a list of
    output dicts or gzipped pickles of dicts from independent runs of BLS, PDM,
    AoV, or GLS period-finders in periodbase).

    NOTE: if `lspinfolist` contains more than one lspinfo object with the same
    lspmethod ('pdm','gls','sls','aov','bls'), the latest one in the list will
    overwrite the earlier ones.

    The output dict contains all the plots (magseries and phased magseries),
    periodograms, object information, variability information, light curves, and
    phased light curves. This can be written to:

    - a pickle with `checkplot_pickle` below
    - a PNG with `checkplot.pkl_png.checkplot_pickle_to_png`

    Parameters
    ----------

    lspinfolist : list of dicts
        This is a list of dicts containing period-finder results ('lspinfo'
        dicts). These can be from any of the period-finder methods in
        astrobase.periodbase. To incorporate external period-finder results into
        checkplots, these dicts must be of the form below, including at least
        the keys indicated here::

            {'periods': np.array of all periods searched by the period-finder,
             'lspvals': np.array of periodogram power value for each period,
             'bestperiod': a float value that is the period with the highest
                           peak in the periodogram, i.e. the most-likely actual
                           period,
             'method': a three-letter code naming the period-finder used; must
                       be one of the keys in the
                       `astrobase.periodbase.METHODLABELS` dict,
             'nbestperiods': a list of the periods corresponding to periodogram
                             peaks (`nbestlspvals` below) to annotate on the
                             periodogram plot so they can be called out
                             visually,
             'nbestlspvals': a list of the power values associated with
                             periodogram peaks to annotate on the periodogram
                             plot so they can be called out visually; should be
                             the same length as `nbestperiods` above}

        `nbestperiods` and `nbestlspvals` in each lspinfo dict must have at
        least as many elements as the `nperiodstouse` kwarg to this function.

    times,mags,errs : np.arrays
        The magnitude/flux time-series to process for this checkplot along with
        their associated measurement errors.

    fast_mode : bool or float
        This runs the external catalog operations in a "fast" mode, with short
        timeouts and not trying to hit external catalogs that take a long time
        to respond.

        If this is set to True, the default settings for the external requests
        will then become::

            skyview_lookup = False
            skyview_timeout = 10.0
            skyview_retry_failed = False
            dust_timeout = 10.0
            gaia_submit_timeout = 7.0
            gaia_max_timeout = 10.0
            gaia_submit_tries = 2
            complete_query_later = False
            search_simbad = False

        If this is a float, will run in "fast" mode with the provided timeout
        value in seconds and the following settings::

            skyview_lookup = True
            skyview_timeout = fast_mode
            skyview_retry_failed = False
            dust_timeout = fast_mode
            gaia_submit_timeout = 0.66*fast_mode
            gaia_max_timeout = fast_mode
            gaia_submit_tries = 2
            complete_query_later = False
            search_simbad = False

    magsarefluxes : bool
        If True, indicates the input time-series is fluxes and not mags so the
        plot y-axis direction and range can be set appropriately.

    nperiodstouse : int
        This controls how many 'best' periods to make phased LC plots for. By
        default, this is the 3 best. If this is set to None, all 'best' periods
        present in each lspinfo dict's 'nbestperiods' key will be processed for
        this checkplot.

    objectinfo : dict or None
        If provided, this is a dict containing information on the object whose
        light curve is being processed. This function will then be able to look
        up and download a finder chart for this object and write that to the
        output checkplotdict. External services such as GAIA, SIMBAD, TIC@MAST,
        etc. will also be used to look up this object by its coordinates, and
        will add in information available from those services.

        The `objectinfo` dict must be of the form and contain at least the keys
        described below::

            {'objectid': the name of the object,
             'ra': the right ascension of the object in decimal degrees,
             'decl': the declination of the object in decimal degrees,
             'ndet': the number of observations of this object}

        You can also provide magnitudes and proper motions of the object using
        the following keys and the appropriate values in the `objectinfo`
        dict. These will be used to calculate colors, total and reduced proper
        motion, etc. and display these in the output checkplot PNG::

            'pmra' -> the proper motion in mas/yr in right ascension,
            'pmdecl' -> the proper motion in mas/yr in the declination,
            'umag'  -> U mag		 -> colors: U-B, U-V, U-g
            'bmag'  -> B mag		 -> colors: U-B, B-V
            'vmag'  -> V mag		 -> colors: U-V, B-V, V-R, V-I, V-K
            'rmag'  -> R mag		 -> colors: V-R, R-I
            'imag'  -> I mag		 -> colors: g-I, V-I, R-I, B-I
            'jmag'  -> 2MASS J mag	 -> colors: J-H, J-K, g-J, i-J
            'hmag'  -> 2MASS H mag	 -> colors: J-H, H-K
            'kmag'  -> 2MASS Ks mag	 -> colors: g-Ks, H-Ks, J-Ks, V-Ks
            'sdssu' -> SDSS u mag	 -> colors: u-g, u-V
            'sdssg' -> SDSS g mag	 -> colors: g-r, g-i, g-K, u-g, U-g, g-J
            'sdssr' -> SDSS r mag	 -> colors: r-i, g-r
            'sdssi' -> SDSS i mag	 -> colors: r-i, i-z, g-i, i-J, i-W1
            'sdssz' -> SDSS z mag	 -> colors: i-z, z-W2, g-z
            'ujmag' -> UKIRT J mag	 -> colors: J-H, H-K, J-K, g-J, i-J
            'uhmag' -> UKIRT H mag	 -> colors: J-H, H-K
            'ukmag' -> UKIRT K mag	 -> colors: g-K, H-K, J-K, V-K
            'irac1' -> Spitzer IRAC1 mag -> colors: i-I1, I1-I2
            'irac2' -> Spitzer IRAC2 mag -> colors: I1-I2, I2-I3
            'irac3' -> Spitzer IRAC3 mag -> colors: I2-I3
            'irac4' -> Spitzer IRAC4 mag -> colors: I3-I4
            'wise1' -> WISE W1 mag	 -> colors: i-W1, W1-W2
            'wise2' -> WISE W2 mag	 -> colors: W1-W2, W2-W3
            'wise3' -> WISE W3 mag	 -> colors: W2-W3
            'wise4' -> WISE W4 mag	 -> colors: W3-W4

        If you have magnitude measurements in other bands, use the
        `custom_bandpasses` kwarg to pass these in.

        If this is None, no object information will be incorporated into the
        checkplot (kind of making it effectively useless for anything other than
        glancing at the phased light curves at various 'best' periods from the
        period-finder results).

    deredden_object : bool
        If this is True, will use the 2MASS DUST service to get extinction
        coefficients in various bands, and then try to deredden the magnitudes
        and colors of the object already present in the checkplot's objectinfo
        dict.

    custom_bandpasses : dict
        This is a dict used to provide custom bandpass definitions for any
        magnitude measurements in the objectinfo dict that are not automatically
        recognized by :py:func:`astrobase.varclass.starfeatures.color_features`.

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

    gaia_mirror : str or None
        This sets the GAIA mirror to use. This is a key in the
        `services.gaia.GAIA_URLS` dict which defines the URLs to hit for each
        mirror.

    complete_query_later : bool
        If this is True, saves the state of GAIA queries that are not yet
        complete when `gaia_max_timeout` is reached while waiting for the GAIA
        service to respond to our request. A later call for GAIA info on the
        same object will attempt to pick up the results from the existing query
        if it's completed. If `fast_mode` is True, this is ignored.

    varinfo : dict
        If this is None, a blank dict of the form below will be added to the
        checkplotdict::

            {'objectisvar': None -> variability flag (None indicates unset),
             'vartags': CSV str containing variability type tags from review,
             'varisperiodic': None -> periodic variability flag (None -> unset),
             'varperiod': the period associated with the periodic variability,
             'varepoch': the epoch associated with the periodic variability}

        If you provide a dict matching this format in this kwarg, this will be
        passed unchanged to the output checkplotdict produced.

    getvarfeatures : bool
        If this is True, several light curve variability features for this
        object will be calculated and added to the output checkpotdict as
        checkplotdict['varinfo']['features']. This uses the function
        `varclass.varfeatures.all_nonperiodic_features` so see its docstring for
        the measures that are calculated (e.g. Stetson J indices, dispersion
        measures, etc.)

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

    nbrradiusarcsec : flaot
        The radius in arcseconds to use for a search conducted around the
        coordinates of this object to look for any potential confusion and
        blending of variability amplitude caused by their proximity.

    maxnumneighbors : int
        The maximum number of neighbors that will have their light curves and
        magnitudes noted in this checkplot as potential blends with the target
        object.

    xmatchinfo : str or dict
        This is either the xmatch dict produced by the function
        `load_xmatch_external_catalogs` above, or the path to the xmatch info
        pickle file produced by that function.

    xmatchradiusarcsec : float
        This is the cross-matching radius to use in arcseconds.

    lcfitfunc : Python function or None
        If provided, this should be a Python function that is used to fit a
        model to the light curve. This fit is then overplotted for each phased
        light curve in the checkplot. This function should have the following
        signature:

        `def lcfitfunc(times, mags, errs, period, **lcfitparams)`

        where `lcfitparams` encapsulates all external parameters (i.e. number of
        knots for a spline function, the degree of a Legendre polynomial fit,
        etc., planet transit parameters) This function should return a Python
        dict with the following structure (similar to the functions in
        `astrobase.lcfit`) and at least the keys below::

            {'fittype':<str: name of fit method>,
             'fitchisq':<float: the chi-squared value of the fit>,
             'fitredchisq':<float: the reduced chi-squared value of the fit>,
             'fitinfo':{'fitmags':<ndarray: model mags/fluxes from fit func>},
             'magseries':{'times':<ndarray: times where fitmags are evaluated>}}

        Additional keys in the dict returned from this function can include
        `fitdict['fitinfo']['finalparams']` for the final model fit parameters
        (this will be used by the checkplotserver if present),
        `fitdict['fitinfo']['fitepoch']` for the minimum light epoch returned by
        the model fit, among others.

        In any case, the output dict of `lcfitfunc` will be copied to the output
        checkplotdict as
        `checkplotdict[lspmethod][periodind]['lcfit'][<fittype>]` for each
        phased light curve.

    lcfitparams : dict
        A dict containing the LC fit parameters to use when calling the function
        provided in `lcfitfunc`. This contains key-val pairs corresponding to
        parameter names and their respective initial values to be used by the
        fit function.

    externalplots : list of tuples of str
        If provided, this is a list of 4-element tuples containing:

        1. path to PNG of periodogram from an external period-finding method
        2. path to PNG of best period phased LC from the external period-finder
        3. path to PNG of 2nd-best phased LC from the external period-finder
        4. path to PNG of 3rd-best phased LC from the external period-finder

        This can be used to incorporate external period-finding method results
        into the output checkplot pickle or exported PNG to allow for comparison
        with astrobase results. Example of `externalplots`::

            [('/path/to/external/bls-periodogram.png',
              '/path/to/external/bls-phasedlc-plot-bestpeak.png',
              '/path/to/external/bls-phasedlc-plot-peak2.png',
              '/path/to/external/bls-phasedlc-plot-peak3.png'),
             ('/path/to/external/pdm-periodogram.png',
              '/path/to/external/pdm-phasedlc-plot-bestpeak.png',
              '/path/to/external/pdm-phasedlc-plot-peak2.png',
              '/path/to/external/pdm-phasedlc-plot-peak3.png'),
             ...]

        If `externalplots` is provided here, these paths will be stored in the
        output checkplotdict. The `checkplot.pkl_png.checkplot_pickle_to_png`
        function can then automatically retrieve these plot PNGs and put
        them into the exported checkplot PNG.

    findercmap : str or matplotlib.cm.ColorMap object
        The Colormap object to use for the finder chart image.

    finderconvolve : astropy.convolution.Kernel object or None
        If not None, the Kernel object to use for convolving the finder image.

    findercachedir : str
        The path to the astrobase cache directory for finder chart downloads
        from the NASA SkyView service.

    normto : {'globalmedian', 'zero'} or a float
        This specifies the normalization target::

            'globalmedian' -> norms each mag to global median of the LC column
            'zero'         -> norms each mag to zero
            a float        -> norms each mag to this specified float value.

    normmingap : float
        This defines how much the difference between consecutive measurements is
        allowed to be to consider them as parts of different timegroups. By
        default it is set to 4.0 days.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    varepoch : 'min' or float or list of lists or None
        The epoch to use for this phased light curve plot tile. If this is a
        float, will use the provided value directly. If this is 'min', will
        automatically figure out the time-of-minimum of the phased light
        curve. If this is None, will use the mimimum value of `stimes` as the
        epoch of the phased light curve plot. If this is a list of lists, will
        use the provided value of `lspmethodind` to look up the current
        period-finder method and the provided value of `periodind` to look up
        the epoch associated with that method and the current period. This is
        mostly only useful when `twolspmode` is True.

    phasewrap : bool
        If this is True, the phased time-series will be wrapped around
        phase 0.0.

    phasesort : bool
        If True, will sort the phased light curve in order of increasing phase.

    phasebin: float
        The bin size to use to group together measurements closer than this
        amount in phase. This is in units of phase. If this is a float, a
        phase-binned version of the phased light curve will be overplotted on
        top of the regular phased light curve.

    minbinelems : int
        The minimum number of elements required per phase bin to include it in
        the phased LC plot.

    plotxlim : sequence of two floats or None
        The x-range (min, max) of the phased light curve plot. If None, will be
        determined automatically.

    xliminsetmode : bool
        If this is True, the generated phased light curve plot will use the
        values of `plotxlim` as the main plot x-axis limits (i.e. zoomed-in if
        `plotxlim` is a range smaller than the full phase range), and will show
        the full phased light curve plot as an smaller inset. Useful for
        planetary transit light curves.

    plotdpi : int
        The resolution of the output plot PNGs in dots per inch.

    bestperiodhighlight : str or None
        If not None, this is a str with a matplotlib color specification to use
        as the background color to highlight the phased light curve plot of the
        'best' period and epoch combination. If None, no highlight will be
        applied.

    xgridlines : list of floats or None
        If this is provided, must be a list of floats corresponding to the phase
        values where to draw vertical dashed lines as a means of highlighting
        these.

    mindet : int
        The minimum of observations the input object's mag/flux time-series must
        have for this function to plot its light curve and phased light
        curve. If the object has less than this number, no light curves will be
        plotted, but the checkplotdict will still contain all of the other
        information.

    verbose : bool
        If True, will indicate progress and warn about problems.

    outfile : str or None
        The name of the output checkplot pickle file. If this is None, will
        write the checkplot pickle to file called 'checkplot.pkl' in the current
        working directory.

    outgzip : bool
        This controls whether to gzip the output pickle. It turns out that this
        is the slowest bit in the output process, so if you're after speed, best
        not to use this. This is False by default since it turns out that gzip
        actually doesn't save that much space (29 MB vs. 35 MB for the average
        checkplot pickle).

    pickleprotocol : int or None
        This sets the pickle file protocol to use when writing the pickle:

        If None, will choose a protocol using the following rules:

        - 4 -> default in Python >= 3.4 - fast but incompatible with Python 2
        - 3 -> default in Python 3.0-3.3 - mildly fast
        - 2 -> default in Python 2 - very slow, but compatible with Python 2/3

        The default protocol kwarg is None, this will make an automatic choice
        for pickle protocol that's best suited for the version of Python in
        use. Note that this will make pickles generated by Py3 incompatible with
        Py2.

    returndict : bool
        If this is True, will return the checkplotdict instead of returning the
        filename of the output checkplot pickle.

    Returns
    -------

    dict or str
        If returndict is False, will return the path to the generated checkplot
        pickle file. If returndict is True, will return the checkplotdict
        instead.

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

    # if no pickle protocol is specified, use v4
    if not pickleprotocol:
        pickleprotocol = 4

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


def checkplot_pickle_update(
        currentcp,
        updatedcp,
        outfile=None,
        outgzip=False,
        pickleprotocol=None,
        verbose=True
):
    '''This updates the current checkplotdict with updated values provided.


    Parameters
    ----------

    currentcp : dict or str
        This is either a checkplotdict produced by `checkplot_pickle` above or a
        checkplot pickle file produced by the same function. This checkplot will
        be updated from the `updatedcp` checkplot.

    updatedcp : dict or str
        This is either a checkplotdict produced by `checkplot_pickle` above or a
        checkplot pickle file produced by the same function. This checkplot will
        be the source of the update to the  `currentcp` checkplot.

    outfile : str or None
        The name of the output checkplot pickle file. The function will output
        the new checkplot gzipped pickle file to `outfile` if outfile is a
        filename. If `currentcp` is a file and `outfile`, this will be set to
        that filename, so the function updates it in place.

    outgzip : bool
        This controls whether to gzip the output pickle. It turns out that this
        is the slowest bit in the output process, so if you're after speed, best
        not to use this. This is False by default since it turns out that gzip
        actually doesn't save that much space (29 MB vs. 35 MB for the average
        checkplot pickle).

    pickleprotocol : int or None
        This sets the pickle file protocol to use when writing the pickle:

        If None, will choose a protocol using the following rules:

        - 4 -> default in Python >= 3.4 - fast but incompatible with Python 2
        - 3 -> default in Python 3.0-3.3 - mildly fast
        - 2 -> default in Python 2 - very slow, but compatible with Python 2/3

        The default protocol kwarg is None, this will make an automatic choice
        for pickle protocol that's best suited for the version of Python in
        use. Note that this will make pickles generated by Py3 incompatible with
        Py2.

    verbose : bool
        If True, will indicate progress and warn about problems.

    Returns
    -------

    str
        The path to the updated checkplot pickle file. If `outfile` was None and
        `currentcp` was a filename, this will return `currentcp` to indicate
        that the checkplot pickle file was updated in place.

    '''

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
