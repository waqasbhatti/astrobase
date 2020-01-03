#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# png.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
# License: MIT - see the LICENSE file for the full text.

'''
This contains the implementation of checkplots that generate PNG files only.

The `checkplot_png` function takes a single period-finding result and makes the
following 3 x 3 grid and writes to a PNG::

    [LSP plot + objectinfo] [     unphased LC     ] [ period 1 phased LC ]
    [period 1 phased LC /2] [period 1 phased LC x2] [ period 2 phased LC ]
    [ period 3 phased LC  ] [period 4 phased LC   ] [ period 5 phased LC ]


The `twolsp_checkplot_png` function makes a similar plot for two independent
period-finding routines and writes to a PNG::

    [ pgram1 + objectinfo ] [        pgram2       ] [     unphased LC     ]
    [ pgram1 P1 phased LC ] [ pgram1 P2 phased LC ] [ pgram1 P3 phased LC ]
    [ pgram2 P1 phased LC ] [ pgram2 P2 phased LC ] [ pgram2 P3 phased LC ]

where:

- pgram1 is the plot for the periodogram in the lspinfo1 dict
- pgram1 P1, P2, and P3 are the best three periods from lspinfo1
- pgram2 is the plot for the periodogram in the lspinfo2 dict
- pgram2 P1, P2, and P3 are the best three periods from lspinfo2

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
import re
import gzip

import pickle

from numpy import (
    isfinite as npisfinite,
    min as npmin, max as npmax,
    abs as npabs, ravel as npravel, nan as npnan,
    percentile as nppercentile
)

# we're going to plot using Agg only
import matplotlib

mpl_regex = re.findall('rc[0-9]', matplotlib.__version__)

if len(mpl_regex) == 1:
    # some matplotlib versions are e.g., "3.1.0rc1", which we resolve to
    # "(3,1,0)".
    MPLVERSION = tuple(
        int(x) for x in
        matplotlib.__version__.replace(mpl_regex[0],'').split('.')
    )
else:
    MPLVERSION = tuple(int(x) for x in matplotlib.__version__.split('.'))

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


###################
## LOCAL IMPORTS ##
###################

from ..lcmath import (
    phase_magseries, phase_bin_magseries,
    normalize_magseries, sigclip_magseries
)
from ..lcfit.nonphysical import spline_fit_magseries, savgol_fit_magseries

from ..plotbase import (
    skyview_stamp, PLOTYLABELS, METHODLABELS, METHODSHORTLABELS
)
from ..coordutils import total_proper_motion, reduced_proper_motion


#######################
## UTILITY FUNCTIONS ##
#######################

def _make_periodogram(axes,
                      lspinfo,
                      objectinfo,
                      findercmap,
                      finderconvolve,
                      verbose=True,
                      circleoverlay=False,
                      findercachedir='~/.astrobase/stamp-cache'):
    '''Makes periodogram, objectinfo, and finder tile for `checkplot_png` and
    `twolsp_checkplot_png`.

    Parameters
    ----------

    axes : matplotlib.axes.Axes object
        The Axes object which will contain the plot being made.

    lspinfo : dict
        Dict containing results from a period-finder in `astrobase.periodbase`
        or a dict that corresponds to that format.

    objectinfo : dict
        Dict containing basic info about the object being processed.

    findercmap : matplotlib Colormap object or str
        The Colormap object to use for the finder chart image.

    finderconvolve : astropy.convolution.Kernel object or None
        If not None, the Kernel object to use for convolving the finder image.

    verbose : bool
        If True, indicates progress.

    findercachedir : str
        The directory where the FITS finder images are downloaded and cached.

    circleoverlay : False or float
        If float, give the radius in arcseconds of circle to overlay

    Returns
    -------

    Does not return anything, works on the input Axes object directly.

    '''

    # get the appropriate plot ylabel
    pgramylabel = PLOTYLABELS[lspinfo['method']]

    # get the periods and lspvals from lspinfo
    periods = lspinfo['periods']
    lspvals = lspinfo['lspvals']
    bestperiod = lspinfo['bestperiod']
    nbestperiods = lspinfo['nbestperiods']
    nbestlspvals = lspinfo['nbestlspvals']

    # make the LSP plot on the first subplot
    axes.plot(periods,lspvals)

    axes.set_xscale('log',basex=10)
    axes.set_xlabel('Period [days]')
    axes.set_ylabel(pgramylabel)
    plottitle = '%s - %.6f d' % (METHODLABELS[lspinfo['method']],
                                 bestperiod)
    axes.set_title(plottitle)

    # show the best five peaks on the plot
    for bestperiod, bestpeak in zip(nbestperiods,
                                    nbestlspvals):
        axes.annotate('%.6f' % bestperiod,
                      xy=(bestperiod, bestpeak), xycoords='data',
                      xytext=(0.0,25.0), textcoords='offset points',
                      arrowprops=dict(arrowstyle="->"),fontsize='14.0')

    # make a grid
    axes.grid(color='#a9a9a9',
              alpha=0.9,
              zorder=0,
              linewidth=1.0,
              linestyle=':')

    # if objectinfo is present, get things from it
    if (objectinfo and isinstance(objectinfo, dict) and
        ('objectid' in objectinfo or 'hatid' in objectinfo) and
        'ra' in objectinfo and 'decl' in objectinfo and
        objectinfo['ra'] and objectinfo['decl']):

        if 'objectid' not in objectinfo:
            objectid = objectinfo['hatid']
        else:
            objectid = objectinfo['objectid']

        if verbose:
            LOGINFO('adding in object information and '
                    'finder chart for %s at RA: %.3f, DEC: %.3f' %
                    (objectid, objectinfo['ra'], objectinfo['decl']))

        # calculate colors
        if ('bmag' in objectinfo and 'vmag' in objectinfo and
            'jmag' in objectinfo and 'kmag' in objectinfo and
            'sdssi' in objectinfo and
            objectinfo['bmag'] and objectinfo['vmag'] and
            objectinfo['jmag'] and objectinfo['kmag'] and
            objectinfo['sdssi']):
            bvcolor = objectinfo['bmag'] - objectinfo['vmag']
            jkcolor = objectinfo['jmag'] - objectinfo['kmag']
            ijcolor = objectinfo['sdssi'] - objectinfo['jmag']
        else:
            bvcolor = None
            jkcolor = None
            ijcolor = None

        if ('teff' in objectinfo and 'gmag' in objectinfo and
            objectinfo['teff'] and objectinfo['gmag']):
            # Gaia data input
            teff_val = objectinfo['teff']
            gmag = objectinfo['gmag']

        # bump the ylim of the LSP plot so that the overplotted finder and
        # objectinfo can fit in this axes plot
        lspylim = axes.get_ylim()
        axes.set_ylim(lspylim[0], lspylim[1]+0.75*(lspylim[1]-lspylim[0]))

        # get the stamp
        try:
            dss, dssheader = skyview_stamp(objectinfo['ra'],
                                           objectinfo['decl'],
                                           convolvewith=finderconvolve,
                                           flip=False,
                                           cachedir=findercachedir,
                                           verbose=verbose)
            stamp = dss

            # inset plot it on the current axes
            inset = inset_axes(axes, width="40%", height="40%", loc=1)
            inset.imshow(stamp, cmap=findercmap, origin='lower')
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_frame_on(False)

            # grid lines pointing to the center of the frame
            if not circleoverlay:
                inset.axvline(x=150,ymin=0.375,
                              ymax=0.45,linewidth=2.0,color='b')
                inset.axhline(y=150,xmin=0.375,
                              xmax=0.45,linewidth=2.0,color='b')
            else:
                # DSS is ~1 arcsecond per pixel
                radius_px = circleoverlay
                circle2 = plt.Circle((150, 150), radius_px,
                                     color='orange', fill=False)
                inset.add_artist(circle2)

        except OSError:

            LOGERROR('downloaded FITS appears to be corrupt, retrying...')

            dss, dssheader = skyview_stamp(objectinfo['ra'],
                                           objectinfo['decl'],
                                           convolvewith=finderconvolve,
                                           flip=False,
                                           forcefetch=True,
                                           cachedir=findercachedir,
                                           verbose=verbose)
            stamp = dss

            # inset plot it on the current axes
            inset = inset_axes(axes, width="40%", height="40%", loc=1)
            inset.imshow(stamp, cmap=findercmap, origin='lower')
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_frame_on(False)

            # grid lines pointing to the center of the frame
            inset.axvline(x=150,ymin=0.375,ymax=0.45,linewidth=2.0,color='b')
            inset.axhline(y=150,xmin=0.375,xmax=0.45,linewidth=2.0,color='b')

        except Exception:
            LOGEXCEPTION('could not fetch a DSS stamp for this '
                         'object %s using coords (%.3f,%.3f)' %
                         (objectid, objectinfo['ra'], objectinfo['decl']))

        # annotate with objectinfo
        axes.text(
            0.05,0.95,
            '%s' % objectid,
            ha='left',va='center',transform=axes.transAxes,
            fontsize=18.0
        )

        axes.text(
            0.05,0.91,
            'RA = %.3f, DEC = %.3f' % (objectinfo['ra'], objectinfo['decl']),
            ha='left',va='center',transform=axes.transAxes,
            fontsize=18.0
        )

        if bvcolor:
            axes.text(0.05,0.87,
                      '$B - V$ = %.3f, $V$ = %.3f' % (bvcolor,
                                                      objectinfo['vmag']),
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)
        elif 'vmag' in objectinfo and objectinfo['vmag']:
            axes.text(0.05,0.87,
                      '$V$ = %.3f' % (objectinfo['vmag'],),
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)

        if ijcolor:
            axes.text(0.05,0.83,
                      '$i - J$ = %.3f, $J$ = %.3f' % (ijcolor,
                                                      objectinfo['jmag']),
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)
        elif 'jmag' in objectinfo and objectinfo['jmag']:
            axes.text(0.05,0.83,
                      '$J$ = %.3f' % (objectinfo['jmag'],),
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)

        if jkcolor:
            axes.text(0.05,0.79,
                      '$J - K$ = %.3f, $K$ = %.3f' % (jkcolor,
                                                      objectinfo['kmag']),
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)
        elif 'kmag' in objectinfo and objectinfo['kmag']:
            axes.text(0.05,0.79,
                      '$K$ = %.3f' % (objectinfo['kmag'],),
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)

        if 'sdssr' in objectinfo and objectinfo['sdssr']:
            axes.text(0.05,0.75,'SDSS $r$ = %.3f' % objectinfo['sdssr'],
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)

        if ('teff' in objectinfo and 'gmag' in objectinfo and
            objectinfo['teff'] and objectinfo['gmag']):

            # gaia data available
            try:
                axes.text(0.05,0.87,
                          r'$G$ = %.1f, $T_\mathrm{eff}$ = %d' % (
                              gmag, int(teff_val)),
                          ha='left',va='center',transform=axes.transAxes,
                          fontsize=18.0)
            except Exception:
                axes.text(0.05,0.87,
                          'G and Teff failed',
                          ha='left',va='center',transform=axes.transAxes,
                          fontsize=18.0)

        # add in proper motion stuff if available in objectinfo
        if ('pmra' in objectinfo and objectinfo['pmra'] and
            'pmdecl' in objectinfo and objectinfo['pmdecl']):

            try:
                pm = total_proper_motion(objectinfo['pmra'],
                                         objectinfo['pmdecl'],
                                         objectinfo['decl'])
            except Exception:
                pm = npnan

            axes.text(0.05,0.67,r'$\mu$ = %.2f mas yr$^{-1}$' % pm,
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)

            if 'jmag' in objectinfo and objectinfo['jmag'] and npisfinite(pm):
                rpm = reduced_proper_motion(objectinfo['jmag'], pm)
            else:
                rpm = npnan

            axes.text(0.05,0.63,'$H_J$ = %.2f' % rpm,
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)


def _make_magseries_plot(axes,
                         stimes,
                         smags,
                         serrs,
                         magsarefluxes=False,
                         ms=2.0):
    '''Makes the mag-series plot tile for `checkplot_png` and
    `twolsp_checkplot_png`.

    axes : matplotlib.axes.Axes object
        The Axes object where the generated plot will go.

    stimes,smags,serrs : np.array
        The mag/flux time-series arrays along with associated errors. These
        should all have been run through nan-stripping and sigma-clipping
        beforehand.

    magsarefluxes : bool
        If True, indicates the input time-series is fluxes and not mags so the
        plot y-axis direction and range can be set appropriately.

    ms : float
        The `markersize` kwarg to use when making the mag-series plot.

    Returns
    -------

    Does not return anything, works on the input Axes object directly.

    '''

    scaledplottime = stimes - npmin(stimes)

    axes.plot(scaledplottime,
              smags,
              marker='o',
              ms=ms, ls='None',mew=0,
              color='green',
              rasterized=True)

    # flip y axis for mags
    if not magsarefluxes:
        plot_ylim = axes.get_ylim()
        axes.set_ylim((plot_ylim[1], plot_ylim[0]))

    # set the x axis limit
    axes.set_xlim((npmin(scaledplottime)-1.0,
                   npmax(scaledplottime)+1.0))

    # make a grid
    axes.grid(color='#a9a9a9',
              alpha=0.9,
              zorder=0,
              linewidth=1.0,
              linestyle=':')

    # make the x and y axis labels
    plot_xlabel = 'JD - %.3f' % npmin(stimes)
    if magsarefluxes:
        plot_ylabel = 'flux'
    else:
        plot_ylabel = 'magnitude'

    axes.set_xlabel(plot_xlabel)
    axes.set_ylabel(plot_ylabel)

    # fix the yaxis ticks (turns off offset and uses the full
    # value of the yaxis tick)
    axes.get_yaxis().get_major_formatter().set_useOffset(False)
    axes.get_xaxis().get_major_formatter().set_useOffset(False)


def _make_phased_magseries_plot(axes,
                                periodind,
                                stimes, smags, serrs,
                                varperiod, varepoch,
                                phasewrap, phasesort,
                                phasebin, minbinelems,
                                plotxlim,
                                lspmethod,
                                lspmethodind=0,
                                xliminsetmode=False,
                                twolspmode=False,
                                magsarefluxes=False,
                                verbose=True,
                                phasems=2.0,
                                phasebinms=4.0,
                                xticksize=None,
                                yticksize=None,
                                titlefontsize='medium',
                                makegrid=True,
                                lowerleftstr=None,
                                lowerleftfontsize=None):
    '''Makes the phased magseries plot tile for the `checkplot_png` and
    `twolsp_checkplot_png` functions.

    Parameters
    ----------

    axes : matplotlib.axes.Axes object
        The Axes object where the generated plot will be written.

    periodind : int
        The index of the current best period being processed in the lspinfo
        dict.

    stimes,smags,serrs : np.array
        The mag/flux time-series arrays along with associated errors. These
        should all have been run through nan-stripping and sigma-clipping
        beforehand.

    varperiod : float or None
        The period to use for this phased light curve plot tile.

    varepoch : 'min' or float or list of lists or None
        The epoch to use for this phased light curve plot tile.

        - If this is a float, will use the provided value directly.

        - If this is 'min', will automatically figure out the time-of-minimum of
          the phased light curve by fitting a spline or Savitsky-Golay smoothing
          curve to it.

        - If it is "t_fluxpercentile_N", for N an integer, it will be phased to
          that time at the specified percentile of the flux.

        - If this is None, will use the mimimum value of `stimes` as the epoch
          of the phased light curve plot.

        - If this is a list of lists, will use the provided value of
          `lspmethodind` to look up the current period-finder method and the
          provided value of `periodind` to look up the epoch associated with
          that method and the current period. This is mostly only useful when
          `twolspmode` is True.

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

    lspmethod : str
        One of the three-letter keys corresponding to period-finder method names
        in the `astrobase.plotbase.METHODSHORTLABELS` dict. Used to set the plot
        title correctly.

    lspmethodind : int
        If `twolspmode` is set, this will be used to look up the correct epoch
        associated with the current period-finder method and period.

    xliminsetmode : bool
        If this is True, the generated phased light curve plot will use the
        values of `plotxlim` as the main plot x-axis limits (i.e. zoomed-in if
        `plotxlim` is a range smaller than the full phase range), and will show
        the full phased light curve plot as an smaller inset. Useful for
        planetary transit light curves.

    twolspmode : bool
        If this is True, will use the `lspmethodind` and `periodind` to look up
        the correct values of epoch, etc. in the provided `varepoch` list of
        lists for plotting purposes.

    magsarefluxes : bool
        If True, indicates the input time-series is fluxes and not mags so the
        plot y-axis direction and range can be set appropriately.

    verbose : bool
        If True, indicates progress.

    phasems : float
        The marker size to use for the main phased light curve plot symbols.

    phasebinms : float
        The marker size to use for the binned phased light curve plot symbols.

    xticksize,yticksize : int or None
        Fontsize for x and y ticklabels

    titlefontsize: str or float
        Fontsize for the panel title. Default: 'medium'

    lowerleftstr : str or None
        Optional text to overplot in lower left of plot

    lowerleftfontsize : int or str or None
        Font size of optional text to overplot in lower left of plot

    Returns
    -------

    Does not return anything, works on the input Axes object directly.

    '''

    plotvarepoch = None

    # figure out the epoch, if it's None, use the min of the time
    if varepoch is None:
        plotvarepoch = npmin(stimes)

    # if the varepoch is 'min', then fit a spline to the light curve
    # phased using the min of the time, find the fit mag minimum and use
    # the time for that as the varepoch
    elif isinstance(varepoch, str) and varepoch == 'min':

        try:
            spfit = spline_fit_magseries(stimes,
                                         smags,
                                         serrs,
                                         varperiod,
                                         magsarefluxes=magsarefluxes,
                                         sigclip=None,
                                         verbose=verbose)
            plotvarepoch = spfit['fitinfo']['fitepoch']
            if len(plotvarepoch) != 1:
                plotvarepoch = varepoch[0]

        except Exception:

            LOGERROR('spline fit failed, trying SavGol fit')

            sgfit = savgol_fit_magseries(stimes,
                                         smags,
                                         serrs,
                                         varperiod,
                                         sigclip=None,
                                         magsarefluxes=magsarefluxes,
                                         verbose=verbose)
            plotvarepoch = sgfit['fitinfo']['fitepoch']
            if len(plotvarepoch) != 1:
                plotvarepoch = plotvarepoch[0]

        finally:

            if plotvarepoch is None:

                LOGERROR('could not find a min epoch time, '
                         'using min(times) as the epoch for '
                         'the phase-folded LC')

                plotvarepoch = npmin(stimes)

    elif isinstance(varepoch, str) and 't_fluxpercentile' in varepoch:

        # assume format of "percentile_N"
        percentile_int = int(varepoch.split('_')[-1])

        nearest_index = (
            abs(
                smags
                -
                nppercentile(smags, percentile_int, interpolation='nearest')
            ).argmin()
        )

        plotvarepoch = stimes[nearest_index]

    elif isinstance(varepoch, list):

        try:

            if twolspmode:

                thisvarepochlist = varepoch[lspmethodind]
                plotvarepoch = thisvarepochlist[periodind]

            else:
                plotvarepoch = varepoch[periodind]

        except Exception:
            LOGEXCEPTION(
                "varepoch provided in list form either doesn't match "
                "the length of nbestperiods from the period-finder "
                "result, or something else went wrong. using min(times) "
                "as the epoch instead"
            )
            plotvarepoch = npmin(stimes)

    # the final case is to use the provided varepoch directly
    else:
        plotvarepoch = varepoch

    if verbose:
        LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                (varperiod, plotvarepoch))

    # phase the magseries
    phasedlc = phase_magseries(stimes,
                               smags,
                               varperiod,
                               plotvarepoch,
                               wrap=phasewrap,
                               sort=phasesort)
    plotphase = phasedlc['phase']
    plotmags = phasedlc['mags']

    # if we're supposed to bin the phases, do so
    if phasebin:

        binphasedlc = phase_bin_magseries(plotphase,
                                          plotmags,
                                          binsize=phasebin,
                                          minbinelems=minbinelems)
        binplotphase = binphasedlc['binnedphases']
        binplotmags = binphasedlc['binnedmags']

    # finally, make the phased LC plot
    axes.plot(plotphase,
              plotmags,
              marker='o',
              ms=phasems, ls='None',mew=0,
              color='gray',
              rasterized=True)

    # overlay the binned phased LC plot if we're making one
    if phasebin:
        axes.plot(binplotphase,
                  binplotmags,
                  marker='o',
                  ms=phasebinms, ls='None',mew=0,
                  color='#1c1e57',
                  rasterized=True)

    # flip y axis for mags
    if not magsarefluxes:
        plot_ylim = axes.get_ylim()
        axes.set_ylim((plot_ylim[1], plot_ylim[0]))

    # set the x axis limit
    if not plotxlim:
        axes.set_xlim((npmin(plotphase)-0.1,
                       npmax(plotphase)+0.1))
    else:
        axes.set_xlim((plotxlim[0],plotxlim[1]))

    # make a grid
    if makegrid:
        axes.grid(color='#a9a9a9',
                  alpha=0.9,
                  zorder=0,
                  linewidth=1.0,
                  linestyle=':')

    # make the x and y axis labels
    plot_xlabel = 'phase'
    if magsarefluxes:
        plot_ylabel = 'flux'
    else:
        plot_ylabel = 'magnitude'

    axes.set_xlabel(plot_xlabel)
    axes.set_ylabel(plot_ylabel)

    # fix the yaxis ticks (turns off offset and uses the full
    # value of the yaxis tick)
    axes.get_yaxis().get_major_formatter().set_useOffset(False)
    axes.get_xaxis().get_major_formatter().set_useOffset(False)

    if isinstance(xticksize, (int, float)):
        axes.xaxis.set_tick_params(labelsize=xticksize)
    if isinstance(yticksize, (int, float)):
        axes.yaxis.set_tick_params(labelsize=yticksize)

    # make the plot title
    if periodind == 0:
        plottitle = '%s best period: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            varperiod,
            plotvarepoch
        )
    elif periodind == 1 and not twolspmode:
        plottitle = '%s best period x 0.5: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            varperiod,
            plotvarepoch
        )
    elif periodind == 2 and not twolspmode:
        plottitle = '%s best period x 2: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            varperiod,
            plotvarepoch
        )
    elif periodind > 2 and not twolspmode:
        plottitle = '%s peak %s: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            periodind-1,
            varperiod,
            plotvarepoch
        )
    elif periodind > 0:
        plottitle = '%s peak %s: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            periodind+1,
            varperiod,
            plotvarepoch
        )

    axes.set_title(plottitle, fontsize=titlefontsize)

    if isinstance(lowerleftstr, str):
        axes.text(
            0.02, 0.02, lowerleftstr, fontsize=lowerleftfontsize,
            transform=axes.transAxes
        )

    # if we're making an inset plot showing the full range
    if (plotxlim and isinstance(plotxlim, (list,tuple)) and
        len(plotxlim) == 2 and xliminsetmode is True):

        # bump the ylim of the plot so that the inset can fit in this axes plot
        axesylim = axes.get_ylim()

        if magsarefluxes:
            axes.set_ylim(axesylim[0],
                          axesylim[1] + 0.5*npabs(axesylim[1]-axesylim[0]))
        else:
            axes.set_ylim(axesylim[0],
                          axesylim[1] - 0.5*npabs(axesylim[1]-axesylim[0]))

        # put the inset axes in
        inset = inset_axes(axes, width="40%", height="40%", loc=1)

        # make the scatter plot for the phased LC plot
        inset.plot(plotphase,
                   plotmags,
                   marker='o',
                   ms=2.0, ls='None',mew=0,
                   color='gray',
                   rasterized=True)

        # overlay the binned phased LC plot if we're making one
        if phasebin:
            inset.plot(binplotphase,
                       binplotmags,
                       marker='o',
                       ms=4.0, ls='None',mew=0,
                       color='#1c1e57',
                       rasterized=True)

        # show the full phase coverage
        if phasewrap:
            inset.set_xlim(-0.2,0.8)
        else:
            inset.set_xlim(-0.1,1.1)

        # flip y axis for mags
        if not magsarefluxes:
            inset_ylim = inset.get_ylim()
            inset.set_ylim((inset_ylim[1], inset_ylim[0]))

        # set the plot title
        inset.text(0.5,0.1,'full phased light curve',
                   ha='center',va='center',transform=inset.transAxes)
        # don't show axes labels or ticks
        inset.set_xticks([])
        inset.set_yticks([])


############################################
## CHECKPLOT FUNCTIONS THAT WRITE TO PNGS ##
############################################

def checkplot_png(lspinfo,
                  times,
                  mags,
                  errs,
                  varepoch='min',
                  magsarefluxes=False,
                  objectinfo=None,
                  findercmap='gray_r',
                  finderconvolve=None,
                  findercachedir='~/.astrobase/stamp-cache',
                  normto='globalmedian',
                  normmingap=4.0,
                  sigclip=4.0,
                  phasewrap=True,
                  phasesort=True,
                  phasebin=0.002,
                  minbinelems=7,
                  plotxlim=(-0.8,0.8),
                  xliminsetmode=False,
                  bestperiodhighlight=None,
                  circleoverlay=False,
                  plotdpi=100,
                  outfile=None,
                  xticksize=None,
                  yticksize=None,
                  verbose=True):
    '''This makes a checkplot PNG using the output from a period-finder routine.

    A checkplot is a 3 x 3 grid of plots like so::

        [periodogram + objectinfo] [     unphased LC     ] [period 1 phased LC]
        [  period 1 phased LC /2 ] [period 1 phased LC x2] [period 2 phased LC]
        [   period 3 phased LC   ] [period 4 phased LC   ] [period 5 phased LC]

    This is used to sanity check the five best periods obtained from a
    period-finder function in `astrobase.periodbase` or from your own
    period-finder routines if their results can be turned into a dict with the
    format shown below.

    Parameters
    ----------

    lspinfo : dict or str
        If this is a dict, it must be a dict produced by an
        `astrobase.periodbase` period-finder function or a dict from your own
        period-finder function or routine that is of the form below with at
        least these keys::

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

        `nbestperiods` and `nbestlspvals` must have at least 5 elements each,
        e.g. describing the five 'best' (highest power) peaks in the
        periodogram.

        If lspinfo is a str, then it must be a path to a pickle file (ending
        with the extension '.pkl' or '.pkl.gz') that contains a dict of the form
        described above.

    times,mags,errs : np.array
        The mag/flux time-series arrays to process along with associated errors.

    varepoch : 'min' or float or None or list of lists
        This sets the time of minimum light finding strategy for the checkplot::

                                                   the epoch used for all phased
            If `varepoch` is None               -> light curve plots will be
                                                   `min(times)`.

            If `varepoch='min'`                 -> automatic epoch finding for all
                                                   periods using light curve fits.

            If varepoch is a single float       -> this epoch will be used for all
                                                   phased light curve plots

            If varepoch is a list of floats        each epoch will be applied to
            with length = `len(nbestperiods)+2` -> the phased light curve for each
            from period-finder results             period specifically

        If you use a list for varepoch, it must be of length
        `len(lspinfo['nbestperiods']) + 2`, because we insert half and twice the
        period into the best periods list to make those phased LC plots.

    magsarefluxes : bool
        If True, indicates the input time-series is fluxes and not mags so the
        plot y-axis direction and range can be set appropriately.

    objectinfo : dict or None
        If provided, this is a dict containing information on the object whose
        light curve is being processed. This function will then be able to look
        up and download a finder chart for this object and write that to the
        output checkplot PNG image.The `objectinfo` dict must be of the form and
        contain at least the keys described below::

            {'objectid': the name of the object,
             'ra': the right ascension of the object in decimal degrees,
             'decl': the declination of the object in decimal degrees,
             'ndet': the number of observations of this object}

        You can also provide magnitudes and proper motions of the object using
        the following keys and the appropriate values in the `objectinfo`
        dict. These will be used to calculate colors, total and reduced proper
        motion, etc. and display these in the output checkplot PNG.

        - SDSS mag keys: 'sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz'
        - 2MASS mag keys: 'jmag', 'hmag', 'kmag'
        - Cousins mag keys: 'bmag', 'vmag'
        - GAIA specific keys: 'gmag', 'teff'
        - proper motion keys: 'pmra', 'pmdecl'

    findercmap : str or matplotlib.cm.ColorMap object
        The Colormap object to use for the finder chart image.

    finderconvolve : astropy.convolution.Kernel object or None
        If not None, the Kernel object to use for convolving the finder image.

    findercachedir : str
        The directory where the FITS finder images are downloaded and cached.

    normto : {'globalmedian', 'zero'} or a float
        This sets the normalization target::

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

    phasewrap : bool
        If this is True, the phased time-series will be wrapped around phase
        0.0.

    phasesort : bool
        If this is True, the phased time-series will be sorted in phase.

    phasebin : float or None
        If this is provided, indicates the bin size to use to group together
        measurements closer than this amount in phase. This is in units of
        phase. The binned phased light curve will be overplotted on top of the
        phased light curve. Useful for when one has many measurement points and
        needs to pick out a small trend in an otherwise noisy phased light
        curve.

    minbinelems : int
        The minimum number of elements in each phase bin.

    plotxlim : sequence of two floats or None
        The x-axis limits to use when making the phased light curve plot. By
        default, this is (-0.8, 0.8), which places phase 0.0 at the center of
        the plot and covers approximately two cycles in phase to make any trends
        clear.

    xliminsetmode : bool
        If this is True, the generated phased light curve plot will use the
        values of `plotxlim` as the main plot x-axis limits (i.e. zoomed-in if
        `plotxlim` is a range smaller than the full phase range), and will show
        the full phased light curve plot as an smaller inset. Useful for
        planetary transit light curves.

    bestperiodhighlight : str or None
        If not None, this is a str with a matplotlib color specification to use
        as the background color to highlight the phased light curve plot of the
        'best' period and epoch combination. If None, no highlight will be
        applied.

    circleoverlay : False or float
        If float, give the radius in arcseconds of circle to overlay

    outfile : str or None
        The file name of the file to save the checkplot to. If this is None,
        will write to a file called 'checkplot.png' in the current working
        directory.

    plotdpi : int
        Sets the resolution in DPI for PNG plots (default = 100).

    verbose : bool
        If False, turns off many of the informational messages. Useful for
        when an external function is driving lots of `checkplot_png` calls.

    xticksize,yticksize : int or None
        Fontsize for x and y ticklabels

    Returns
    -------

    str
        The file path to the generated checkplot PNG file.

    '''

    if not outfile and isinstance(lspinfo,str):
        # generate the plot filename
        plotfpath = os.path.join(
            os.path.dirname(lspinfo),
            'checkplot-%s.png' % (
                os.path.basename(lspinfo),
            )
        )
    elif outfile:
        plotfpath = outfile
    else:
        plotfpath = 'checkplot.png'

    # get the lspinfo from a pickle file transparently
    if isinstance(lspinfo,str) and os.path.exists(lspinfo):
        if verbose:
            LOGINFO('loading LSP info from pickle %s' % lspinfo)

        if '.gz' in lspinfo:
            with gzip.open(lspinfo,'rb') as infd:
                lspinfo = pickle.load(infd)
        else:
            with open(lspinfo,'rb') as infd:
                lspinfo = pickle.load(infd)

    # get the things to plot out of the data
    if ('periods' in lspinfo and
        'lspvals' in lspinfo and
        'bestperiod' in lspinfo):

        bestperiod = lspinfo['bestperiod']
        nbestperiods = lspinfo['nbestperiods']
        lspmethod = lspinfo['method']

    else:

        LOGERROR('could not understand lspinfo for this object, skipping...')
        return None

    if not npisfinite(bestperiod):

        LOGWARNING('no best period found for this object, skipping...')
        return None

    # initialize the plot
    fig, axes = plt.subplots(3,3)
    axes = npravel(axes)

    # this is a full page plot
    fig.set_size_inches(30,24)

    #######################
    ## PLOT 1 is the LSP ##
    #######################

    _make_periodogram(axes[0],lspinfo,objectinfo,
                      findercmap, finderconvolve,
                      verbose=verbose,
                      findercachedir=findercachedir,
                      circleoverlay=circleoverlay)

    ######################################
    ## NOW MAKE THE PHASED LIGHT CURVES ##
    ######################################

    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # take care of the normalization
    if normto is not False:
        stimes, smags = normalize_magseries(stimes, smags,
                                            normto=normto,
                                            magsarefluxes=magsarefluxes,
                                            mingap=normmingap)

    # make sure we have some lightcurve points to plot after sigclip
    if len(stimes) >= 50:

        ##############################
        ## PLOT 2 is an unphased LC ##
        ##############################

        _make_magseries_plot(axes[1], stimes, smags, serrs,
                             magsarefluxes=magsarefluxes)

        ###########################
        ### NOW PLOT PHASED LCS ###
        ###########################

        # make the plot for each best period
        lspbestperiods = nbestperiods[::]

        lspperiodone = lspbestperiods[0]
        lspbestperiods.insert(1,lspperiodone*2.0)
        lspbestperiods.insert(1,lspperiodone*0.5)

        for periodind, varperiod in enumerate(lspbestperiods):

            # make sure the best period phased LC plot stands out
            if periodind == 0 and bestperiodhighlight:
                if MPLVERSION >= (2,0,0):
                    axes[periodind+2].set_facecolor(bestperiodhighlight)
                else:
                    axes[periodind+2].set_axis_bgcolor(bestperiodhighlight)

            _make_phased_magseries_plot(axes[periodind+2],
                                        periodind,
                                        stimes, smags, serrs,
                                        varperiod, varepoch,
                                        phasewrap, phasesort,
                                        phasebin, minbinelems,
                                        plotxlim, lspmethod,
                                        xliminsetmode=xliminsetmode,
                                        magsarefluxes=magsarefluxes,
                                        verbose=verbose,
                                        xticksize=xticksize,
                                        yticksize=yticksize)

        # end of plotting for each ax

        # save the plot to disk
        fig.set_tight_layout(True)
        if plotfpath.endswith('.png'):
            fig.savefig(plotfpath,dpi=plotdpi)
        else:
            fig.savefig(plotfpath)
        plt.close('all')

        if verbose:
            LOGINFO('checkplot done -> %s' % plotfpath)
        return plotfpath

    # otherwise, there's no valid data for this plot
    else:

        LOGWARNING('no good data')

        for periodind in range(5):

            axes[periodind+2].text(
                0.5,0.5,
                ('no best aperture light curve available'),
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[periodind+2].transAxes
            )

        fig.set_tight_layout(True)

        if plotfpath.endswith('.png'):
            fig.savefig(plotfpath, dpi=plotdpi)
        else:
            fig.savefig(plotfpath)

        plt.close('all')

        if verbose:
            LOGINFO('checkplot done -> %s' % plotfpath)
        return plotfpath


def twolsp_checkplot_png(lspinfo1,
                         lspinfo2,
                         times,
                         mags,
                         errs,
                         varepoch='min',
                         magsarefluxes=False,
                         objectinfo=None,
                         findercmap='gray_r',
                         finderconvolve=None,
                         findercachedir='~/.astrobase/stamp-cache',
                         normto='globalmedian',
                         normmingap=4.0,
                         sigclip=4.0,
                         phasewrap=True,
                         phasesort=True,
                         phasebin=0.002,
                         minbinelems=7,
                         plotxlim=(-0.8,0.8),
                         unphasedms=2.0,
                         phasems=2.0,
                         phasebinms=4.0,
                         xliminsetmode=False,
                         bestperiodhighlight=None,
                         circleoverlay=False,
                         plotdpi=100,
                         outfile=None,
                         figsize=(30,24),
                         returnfigure=False,
                         xticksize=None,
                         yticksize=None,
                         verbose=True):
    '''This makes a checkplot using results from two independent period-finders.

    Adapted from Luke Bouma's implementation of a similar function in his
    work. This makes a special checkplot that uses two lspinfo dictionaries,
    from two independent period-finding methods. For EBs, it's probably best to
    use Stellingwerf PDM or Schwarzenberg-Czerny AoV as one of these, and the
    Box Least-squared Search method as the other one.

    The checkplot layout in this case is::

        [ pgram1 + objectinfo ] [        pgram2       ] [     unphased LC     ]
        [ pgram1 P1 phased LC ] [ pgram1 P2 phased LC ] [ pgram1 P3 phased LC ]
        [ pgram2 P1 phased LC ] [ pgram2 P2 phased LC ] [ pgram2 P3 phased LC ]

    where:

    - pgram1 is the plot for the periodogram in the lspinfo1 dict
    - pgram1 P1, P2, and P3 are the best three periods from lspinfo1
    - pgram2 is the plot for the periodogram in the lspinfo2 dict
    - pgram2 P1, P2, and P3 are the best three periods from lspinfo2

    Note that we take the output file name from lspinfo1 if lspinfo1 is a string
    filename pointing to a (gzipped) pickle containing the results dict from a
    period-finding routine similar to those in periodbase.

    Parameters
    ----------

    lspinfo1,lspinfo2 : dict or str
        If this is a dict, it must be a dict produced by an
        `astrobase.periodbase` period-finder function or a dict from your own
        period-finder function or routine that is of the form below with at
        least these keys::

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

        `nbestperiods` and `nbestlspvals` must have at least 3 elements each,
        e.g. describing the three 'best' (highest power) peaks in the
        periodogram.

        If lspinfo is a str, then it must be a path to a pickle file (ending
        with the extension '.pkl' or '.pkl.gz') that contains a dict of the form
        described above.

    times,mags,errs : np.array
        The mag/flux time-series arrays to process along with associated errors.

    varepoch : 'min' or float or None or list of lists
        This sets the time of minimum light finding strategy for the checkplot::

                                                   the epoch used for all phased
            If `varepoch` is None               -> light curve plots will be
                                                   `min(times)`.

            If `varepoch='min'`                 -> automatic epoch finding for all
                                                   periods using light curve fits.

            If varepoch is a single float       -> this epoch will be used for all
                                                   phased light curve plots

            If varepoch is a list of floats        each epoch will be applied to
            with length = `len(nbestperiods)` ->   the phased light curve for each
            from period-finder results             period specifically

        If you use a list for varepoch, it must be of length
        `len(lspinfo['nbestperiods'])`.

    magsarefluxes : bool
        If True, indicates the input time-series is fluxes and not mags so the
        plot y-axis direction and range can be set appropriately/

    objectinfo : dict or None
        If provided, this is a dict containing information on the object whose
        light curve is being processed. This function will then be able to look
        up and download a finder chart for this object and write that to the
        output checkplot PNG image.The `objectinfo` dict must be of the form and
        contain at least the keys described below::

            {'objectid': the name of the object,
             'ra': the right ascension of the object in decimal degrees,
             'decl': the declination of the object in decimal degrees,
             'ndet': the number of observations of this object}

        You can also provide magnitudes and proper motions of the object using
        the following keys and the appropriate values in the `objectinfo`
        dict. These will be used to calculate colors, total and reduced proper
        motion, etc. and display these in the output checkplot PNG.

        - SDSS mag keys: 'sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz'
        - 2MASS mag keys: 'jmag', 'hmag', 'kmag'
        - Cousins mag keys: 'bmag', 'vmag'
        - GAIA specific keys: 'gmag', 'teff'
        - proper motion keys: 'pmra', 'pmdecl'

    findercmap : str or matplotlib.cm.ColorMap object
        The Colormap object to use for the finder chart image.

    finderconvolve : astropy.convolution.Kernel object or None
        If not None, the Kernel object to use for convolving the finder image.

    findercachedir : str
        The directory where the FITS finder images are downloaded and cached.

    normto : {'globalmedian', 'zero'} or a float
        This sets the LC normalization target::

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

    phasewrap : bool
        If this is True, the phased time-series will be wrapped around phase
        0.0.

    phasesort : bool
        If this is True, the phased time-series will be sorted in phase.

    phasebin : float or None
        If this is provided, indicates the bin size to use to group together
        measurements closer than this amount in phase. This is in units of
        phase. The binned phased light curve will be overplotted on top of the
        phased light curve. Useful for when one has many measurement points and
        needs to pick out a small trend in an otherwise noisy phased light
        curve.

    minbinelems : int
        The minimum number of elements in each phase bin.

    plotxlim : sequence of two floats or None
        The x-axis limits to use when making the phased light curve plot. By
        default, this is (-0.8, 0.8), which places phase 0.0 at the center of
        the plot and covers approximately two cycles in phase to make any trends
        clear.

    unphasedms : float
        The marker size to use for the main unphased light curve plot symbols.

    phasems : float
        The marker size to use for the main phased light curve plot symbols.

    phasebinms : float
        The marker size to use for the binned phased light curve plot symbols.

    xliminsetmode : bool
        If this is True, the generated phased light curve plot will use the
        values of `plotxlim` as the main plot x-axis limits (i.e. zoomed-in if
        `plotxlim` is a range smaller than the full phase range), and will show
        the full phased light curve plot as an smaller inset. Useful for
        planetary transit light curves.

    bestperiodhighlight : str or None
        If not None, this is a str with a matplotlib color specification to use
        as the background color to highlight the phased light curve plot of the
        'best' period and epoch combination. If None, no highlight will be
        applied.

    circleoverlay : False or float
        If float, give the radius in arcseconds of circle to overlay

    plotdpi : int
        Sets the resolution in DPI for PNG plots (default = 100).

    outfile : str or None
        The file name of the file to save the checkplot to. If this is None,
        will write to a file called 'checkplot.png' in the current working
        directory.

    figsize : tuple of two int
        The output figure size in inches.

    returnfigure : bool
        If True, will return the figure directly as a ``matplotlib.Figure``
        object.

    xticksize,yticksize : int or None
        Fontsize for x and y ticklabels

    verbose : bool
        If False, turns off many of the informational messages. Useful for
        when an external function is driving lots of `checkplot_png` calls.

    Returns
    -------

    figure : str or matplotlib.Figure
        The file path to the generated checkplot PNG file if ``returnfigure`` is
        False. A ``matplotlib.Figure`` if ``returnfigure`` is True.

    '''

    # generate the plot filename
    if not outfile and isinstance(lspinfo1,str):
        plotfpath = os.path.join(
            os.path.dirname(lspinfo1),
            'twolsp-checkplot-%s.png' % (
                os.path.basename(lspinfo1),
            )
        )
    elif outfile:
        plotfpath = outfile
    else:
        plotfpath = 'twolsp-checkplot.png'

    # get the first LSP from a pickle file transparently
    if isinstance(lspinfo1,str) and os.path.exists(lspinfo1):
        if verbose:
            LOGINFO('loading LSP info from pickle %s' % lspinfo1)

        if '.gz' in lspinfo1:
            with gzip.open(lspinfo1,'rb') as infd:
                lspinfo1 = pickle.load(infd)
        else:
            with open(lspinfo1,'rb') as infd:
                lspinfo1 = pickle.load(infd)

    # get the second LSP from a pickle file transparently
    if isinstance(lspinfo2,str) and os.path.exists(lspinfo2):
        if verbose:
            LOGINFO('loading LSP info from pickle %s' % lspinfo2)

        if '.gz' in lspinfo2:
            with gzip.open(lspinfo2,'rb') as infd:
                lspinfo2 = pickle.load(infd)
        else:
            with open(lspinfo2,'rb') as infd:
                lspinfo2 = pickle.load(infd)

    # get the things to plot out of the data
    if ('periods' in lspinfo1 and 'periods' in lspinfo2 and
        'lspvals' in lspinfo1 and 'lspvals' in lspinfo2 and
        'bestperiod' in lspinfo1 and 'bestperiod' in lspinfo2):

        bestperiod1 = lspinfo1['bestperiod']
        nbestperiods1 = lspinfo1['nbestperiods']
        lspmethod1 = lspinfo1['method']

        bestperiod2 = lspinfo2['bestperiod']
        nbestperiods2 = lspinfo2['nbestperiods']
        lspmethod2 = lspinfo2['method']

    else:

        LOGERROR('could not understand lspinfo1 or lspinfo2 '
                 'for this object, skipping...')
        return None

    if (not npisfinite(bestperiod1)) or (not npisfinite(bestperiod2)):

        LOGWARNING('no best period found for this object, skipping...')
        return None

    # initialize the plot
    fig, axes = plt.subplots(3,3)
    axes = npravel(axes)

    # this is a full page plot
    fig.set_size_inches(figsize)

    ######################################################################
    ## PLOT 1 is the LSP from lspinfo1, including objectinfo and finder ##
    ######################################################################

    _make_periodogram(axes[0], lspinfo1, objectinfo,
                      findercmap, finderconvolve,
                      verbose=verbose,
                      findercachedir=findercachedir,
                      circleoverlay=circleoverlay)

    #####################################
    ## PLOT 2 is the LSP from lspinfo2 ##
    #####################################

    _make_periodogram(axes[1], lspinfo2, None,
                      findercmap, finderconvolve)

    ##########################################
    ## FIX UP THE MAGS AND REMOVE BAD STUFF ##
    ##########################################

    # sigclip first
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # take care of the normalization
    if normto is not False:
        stimes, smags = normalize_magseries(stimes, smags,
                                            normto=normto,
                                            magsarefluxes=magsarefluxes,
                                            mingap=normmingap)

    # make sure we have some lightcurve points to plot after sigclip
    if len(stimes) >= 50:

        ##############################
        ## PLOT 3 is an unphased LC ##
        ##############################

        _make_magseries_plot(axes[2], stimes, smags, serrs,
                             magsarefluxes=magsarefluxes,
                             ms=unphasedms)

        # make the plot for each best period
        lspbestperiods1 = nbestperiods1[::]
        lspbestperiods2 = nbestperiods2[::]

        ##########################################################
        ### NOW PLOT PHASED LCS FOR 3 BEST PERIODS IN LSPINFO1 ###
        ##########################################################
        for periodind, varperiod, plotaxes in zip([0,1,2],
                                                  lspbestperiods1[:3],
                                                  [axes[3], axes[4], axes[5]]):

            # make sure the best period phased LC plot stands out
            if periodind == 0 and bestperiodhighlight:
                if MPLVERSION >= (2,0,0):
                    plotaxes.set_facecolor(bestperiodhighlight)
                else:
                    plotaxes.set_axis_bgcolor(bestperiodhighlight)

            _make_phased_magseries_plot(plotaxes,
                                        periodind,
                                        stimes, smags, serrs,
                                        varperiod, varepoch,
                                        phasewrap, phasesort,
                                        phasebin, minbinelems,
                                        plotxlim, lspmethod1,
                                        lspmethodind=0,
                                        twolspmode=True,
                                        magsarefluxes=magsarefluxes,
                                        xliminsetmode=xliminsetmode,
                                        verbose=verbose,
                                        phasems=phasems,
                                        phasebinms=phasebinms,
                                        xticksize=xticksize,
                                        yticksize=yticksize)

        ##########################################################
        ### NOW PLOT PHASED LCS FOR 3 BEST PERIODS IN LSPINFO2 ###
        ##########################################################
        for periodind, varperiod, plotaxes in zip([0,1,2],
                                                  lspbestperiods2[:3],
                                                  [axes[6], axes[7], axes[8]]):

            # make sure the best period phased LC plot stands out
            if periodind == 0 and bestperiodhighlight:
                if MPLVERSION >= (2,0,0):
                    plotaxes.set_facecolor(bestperiodhighlight)
                else:
                    plotaxes.set_axis_bgcolor(bestperiodhighlight)

            _make_phased_magseries_plot(plotaxes,
                                        periodind,
                                        stimes, smags, serrs,
                                        varperiod, varepoch,
                                        phasewrap, phasesort,
                                        phasebin, minbinelems,
                                        plotxlim, lspmethod2,
                                        lspmethodind=1,
                                        twolspmode=True,
                                        magsarefluxes=magsarefluxes,
                                        xliminsetmode=xliminsetmode,
                                        verbose=verbose,
                                        phasems=phasems,
                                        phasebinms=phasebinms,
                                        xticksize=xticksize,
                                        yticksize=yticksize)

        # end of plotting for each ax

        # save the plot to disk
        fig.set_tight_layout(True)
        if not returnfigure:
            if plotfpath.endswith('.png'):
                fig.savefig(plotfpath,dpi=plotdpi)
            else:
                fig.savefig(plotfpath)
            plt.close()
        else:
            return fig

        if verbose:
            LOGINFO('checkplot done -> %s' % plotfpath)
        return plotfpath

    # otherwise, there's no valid data for this plot
    else:

        LOGWARNING('no good data')

        for periodind in range(5):

            axes[periodind+2].text(
                0.5,0.5,
                ('no best aperture light curve available'),
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[periodind+2].transAxes
            )

        fig.set_tight_layout(True)

        if plotfpath.endswith('.png'):
            fig.savefig(plotfpath, dpi=plotdpi)
        else:
            fig.savefig(plotfpath)

        plt.close()

        if verbose:
            LOGINFO('checkplot done -> %s' % plotfpath)
        return plotfpath
