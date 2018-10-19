#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''checkplot.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017
License: MIT.

Contains functions to make checkplots: quick views for determining periodic
variability for light curves and sanity-checking results from period-finding
functions (e.g., from periodbase).

The `checkplot_pickle` function takes, for a single object, an arbitrary number
of results from independent period-finding functions (e.g. BLS, PDM, AoV, GLS,
etc.) in periodbase, and generates a pickle file that contains object and
variability information, finder chart, mag series plot, and for each
period-finding result: a periodogram and phased mag series plots for an
arbitrary number of 'best periods'. This is intended for use with an external
checkplot viewer: the Tornado webapp `checkplotserver.py`, but you can also use
the `checkplot_pickle_to_png` function to render this to a PNG that will look
something like:

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


The `checkplot_png` function takes a single period-finding result and makes the
following 3 x 3 grid and writes to a PNG:

    [LSP plot + objectinfo] [     unphased LC     ] [ period 1 phased LC ]
    [period 1 phased LC /2] [period 1 phased LC x2] [ period 2 phased LC ]
    [ period 3 phased LC  ] [period 4 phased LC   ] [ period 5 phased LC ]


The `twolsp_checkplot_png` function makes a similar plot for two independent
period-finding routines and writes to a PNG:

    [ pgram1 + objectinfo ] [        pgram2       ] [     unphased LC     ]
    [ pgram1 P1 phased LC ] [ pgram1 P2 phased LC ] [ pgram1 P3 phased LC ]
    [ pgram2 P1 phased LC ] [ pgram2 P2 phased LC ] [ pgram2 P3 phased LC ]

    where:

    pgram1 is the plot for the periodogram in the lspinfo1 dict
    pgram1 P1, P2, and P3 are the best three periods from lspinfo1
    pgram2 is the plot for the periodogram in the lspinfo2 dict
    pgram2 P1, P2, and P3 are the best three periods from lspinfo2

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
import gzip
import base64
import sys
import hashlib
import json

try:
    import cPickle as pickle
    import cStringIO
    from cStringIO import StringIO as strio
except Exception as e:
    import pickle
    from io import BytesIO as strio

import numpy as np
from numpy import nan as npnan, isfinite as npisfinite, \
    min as npmin, max as npmax, abs as npabs, ravel as npravel

# we're going to plot using Agg only
import matplotlib
MPLVERSION = tuple([int(x) for x in matplotlib.__version__.split('.')])
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import this to check if stimes, smags, serrs are Column objects
from astropy.table import Column as astcolumn

# import this to get neighbors and their x,y coords from the Skyview FITS
from astropy.wcs import WCS

# import from Pillow to generate pngs from checkplot dicts
from PIL import Image, ImageDraw, ImageFont

# import sps.cKDTree for external catalog xmatches
from scipy.spatial import cKDTree

from tornado.escape import squeeze


###################
## LOCAL IMPORTS ##
###################

from .lcmath import phase_magseries, phase_bin_magseries, \
    normalize_magseries, sigclip_magseries
from .varbase.lcfit import spline_fit_magseries, savgol_fit_magseries
from .varclass.varfeatures import all_nonperiodic_features

from .varclass import starfeatures
starfeatures.set_logger_parent(__name__)
from .varclass.starfeatures import coord_features, color_features, \
    color_classification, neighbor_gaia_features

from .plotbase import skyview_stamp, \
    PLOTYLABELS, METHODLABELS, METHODSHORTLABELS
from .coordutils import total_proper_motion, reduced_proper_motion


#######################
## UTILITY FUNCTIONS ##
#######################

def _make_periodogram(axes,
                      lspinfo,
                      objectinfo,
                      findercmap,
                      finderconvolve,
                      verbose=True,
                      findercachedir='~/.astrobase/stamp-cache'):
    '''makes the periodogram, objectinfo, and finder tile.

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
            inset.axvline(x=150,ymin=0.375,ymax=0.45,linewidth=2.0,color='b')
            inset.axhline(y=150,xmin=0.375,xmax=0.45,linewidth=2.0,color='b')

        except OSError as e:

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


        except Exception as e:
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

        # add in proper motion stuff if available in objectinfo
        if ('pmra' in objectinfo and objectinfo['pmra'] and
            'pmdecl' in objectinfo and objectinfo['pmdecl']):

            pm = total_proper_motion(objectinfo['pmra'],
                                     objectinfo['pmdecl'],
                                     objectinfo['decl'])

            axes.text(0.05,0.67,r'$\mu$ = %.2f mas yr$^{-1}$' % pm,
                      ha='left',va='center',transform=axes.transAxes,
                      fontsize=18.0)

            if 'jmag' in objectinfo and objectinfo['jmag']:

                rpm = reduced_proper_motion(objectinfo['jmag'],pm)
                axes.text(0.05,0.63,'$H_J$ = %.2f' % rpm,
                          ha='left',va='center',transform=axes.transAxes,
                          fontsize=18.0)



def _make_magseries_plot(axes,
                         stimes,
                         smags,
                         serrs,
                         magsarefluxes=False):
    '''makes the magseries plot tile.

    '''

    scaledplottime = stimes - npmin(stimes)

    axes.plot(scaledplottime,
              smags,
              marker='o',
              ms=2.0, ls='None',mew=0,
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
                                verbose=True):
    '''makes the phased magseries plot tile.

    if xliminsetmode = True, then makes a zoomed-in plot with the provided
    plotxlim as the main x limits, and the full plot as an inset.

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

        except Exception as e:

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

    elif isinstance(varepoch, list):

        try:

            if twolspmode:

                thisvarepochlist = varepoch[lspmethodind]
                plotvarepoch = thisvarepochlist[periodind]

            else:
                plotvarepoch = varepoch[periodind]

        except Exception as e:
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
              ms=2.0, ls='None',mew=0,
              color='gray',
              rasterized=True)

    # overlay the binned phased LC plot if we're making one
    if phasebin:
        axes.plot(binplotphase,
                  binplotmags,
                  marker='o',
                  ms=4.0, ls='None',mew=0,
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

    axes.set_title(plottitle)

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
                  magsarefluxes=False,
                  objectinfo=None,
                  findercmap='gray_r',
                  finderconvolve=None,
                  findercachedir='~/.astrobase/stamp-cache',
                  normto='globalmedian',
                  normmingap=4.0,
                  outfile=None,
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
                  verbose=True):
    '''This makes a checkplot for an info dict from a period-finding routine.

    A checkplot is a 3 x 3 grid of plots like so:

    [LSP plot + objectinfo] [     unphased LC     ] [ period 1 phased LC ]
    [period 1 phased LC /2] [period 1 phased LC x2] [ period 2 phased LC ]
    [ period 3 phased LC  ] [period 4 phased LC   ] [ period 5 phased LC ]

    This is used to sanity check the five best periods obtained from an LSP
    function in periodbase.

    lspinfo is either a dict or a Python pickle filename containing a dict that
    should look something like the dict below, containing the output from your
    period search routine. The key 'lspvals' is the spectral power or SNR
    obtained from Lomb-Scargle, PDM, AoV, or BLS. The keys 'nbestperiods' and
    'nbestlspvals' contain the best five periods and their respective peaks
    chosen by your period search routine (usually the highest SNR or highest
    power peaks in the spectrum).

    {'bestperiod':7.7375425564838061,
     'lspvals':array([ 0.00892461,  0.0091704 ,  0.00913682,...]),
     'periods':array([ 8.      ,  7.999936,  7.999872, ...]),
     'nbestperiods':[7.7375425564838061,
                     7.6370856881010738,
                     7.837604827964415,
                     7.5367037472486667,
                     7.9377048920074627],
     'nbestlspvals':[0.071409790831114872,
                     0.055157963469682415,
                     0.055126754408175715,
                     0.023441268126990749,
                     0.023239128705778048],
     'method':'gls'}

    The 'method' key-val pair decides what kind of period finding method was
    run. This is used to label the periodogram plot correctly. The following
    values are recognized.

    'gls' -> generalized Lomb-Scargle (e.g., from periodbase.pgen_lsp)
    'pdm' -> Stellingwerf PDM (e.g., from periodbase.stellingwerf_pdm)
    'aov' -> Schwarzenberg-Czerny AoV (e.g., from periodbase.aov_periodfind)
    'bls' -> Box Least-squared Search (e.g., from periodbase.bls_parallel_pfind)
    'sls' -> Lomb-Scargle from Scipy (e.g., from periodbase.scipylsp_parallel)

    magsarefluxes = True means the values provided in the mags input array are
    actually fluxes; this affects the sigma-clipping and plotting of light
    curves.

    If a dict is passed to objectinfo, this function will use it to figure out
    where in the sky the checkplotted object is, and put the finding chart plus
    some basic info into the checkplot. The objectinfo dict should look
    something like those produced for HAT light curves using the reader
    functions in the astrobase.hatlc module, e.g.:

    {'bmag': 17.669,
     'decl': -63.933598,
     'hatid': 'HAT-786-0021445',
     'objectid': 'HAT-786-0021445',
     'hmag': 13.414,
     'jmag': 14.086,
     'kmag': 13.255,
     'ndet': 10850,
     'network': 'HS',
     'pmdecl': -19.4,
     'pmdecl_err': 5.1,
     'pmra': 29.3,
     'pmra_err': 4.1,
     'ra': 23.172678,
     'sdssg': 17.093,
     'sdssi': 15.382,
     'sdssr': 15.956,
     'stations': 'HS02,HS04,HS06',
     'twomassid': '01324144-6356009 ',
     'ucac4id': '12566701',
     'vmag': 16.368}

    At a minimum, you must have the following fields: 'objectid', 'ra',
    'decl'. If 'jmag', 'kmag', 'bmag', 'vmag', 'sdssr', and 'sdssi' are also
    present, the following quantities will be calculated: B-V, J-K, and i-J. If
    'pmra' and 'pmdecl' are present as well, the total proper motion and reduced
    J magnitude proper motion will be calculated.

    findercmap sets the matplotlib colormap of the downloaded finder chart:

    http://matplotlib.org/examples/color/colormaps_reference.html

    finderconvolve convolves the finder FITS image with the given
    astropy.convolution kernel:

    http://docs.astropy.org/en/stable/convolution/kernels.html

    This can be useful to see effects of wide-field telescopes with large pixel
    sizes (like HAT) on the blending of sources.

    varepoch sets the time of minimum light finding strategy for the checkplot:

                                               the epoch used for all phased
    if varepoch is None                     -> light curve plots will be
                                               min(times)

    if varepoch is a single string == 'min' -> automatic epoch finding for all
                                               periods using light curve fits

    if varepoch is a single float           -> this epoch will be used for all
                                               phased light curve plots

    if varepoch is a list of floats            each epoch will be applied to
    with length == len(nbestperiods)        -> the phased light curve for each
    from period-finder results                 period specifically

    NOTE: for checkplot_png, if you use a list for varepoch, it must be of
    length len(lspinfo['nbestperiods']) + 2, because we insert half and twice
    the period into the best periods list to make those phased LC plots.

    findercachedir is the directory where the downloaded stamp FITS files
    go. Repeated calls to this function will then use the cached version of the
    stamp if the finder coordinates don't change.

    bestperiodhighlight sets whether user wants a background on the phased light
    curve from each periodogram type to distinguish them from others. this is an
    HTML hex color specification. If this is None, no highlight will be added.

    xliminsetmode = True sets up the phased mag series plot to show a zoomed-in
    portion (set by plotxlim) as the main plot and an inset version of the full
    phased light curve from phase 0.0 to 1.0. This can be useful if searching
    for small dips near phase 0.0 caused by planetary transits for example.

    verbose = False turns off many of the informational messages. Useful for
    when an external function is driving lots of checkplot calls.

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
                      findercachedir=findercachedir)

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
                                        verbose=verbose)

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
                         magsarefluxes=False,
                         objectinfo=None,
                         findercmap='gray_r',
                         finderconvolve=None,
                         findercachedir='~/.astrobase/stamp-cache',
                         normto='globalmedian',
                         normmingap=4.0,
                         outfile=None,
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
                         verbose=True):
    '''This makes a checkplot using results from two independent period-finders.

    Adapted from Luke Bouma's implementation of the same. This makes a special
    checkplot that uses two lspinfo dictionaries, from two independent
    period-finding methods. For EBs, it's probably best to use Stellingwerf PDM
    or Schwarzenberg-Czerny AoV as one of these, and the Box Least-squared
    Search method as the other one.

    The checkplot layout in this case is:

    [ pgram1 + objectinfo ] [        pgram2       ] [     unphased LC     ]
    [ pgram1 P1 phased LC ] [ pgram1 P2 phased LC ] [ pgram1 P3 phased LC ]
    [ pgram2 P1 phased LC ] [ pgram2 P2 phased LC ] [ pgram2 P3 phased LC ]

    where:

    pgram1 is the plot for the periodogram in the lspinfo1 dict
    pgram1 P1, P2, and P3 are the best three periods from lspinfo1
    pgram2 is the plot for the periodogram in the lspinfo2 dict
    pgram2 P1, P2, and P3 are the best three periods from lspinfo2

    All other args and kwargs are the same as checkplot_png. Note that we take
    the output file name from lspinfo1 if lspinfo1 is a string filename pointing
    to a (gzipped) pickle containing the results dict from a period-finding
    routine similar to those in periodbase.

    varepoch sets the time of minimum light finding strategy for the checkplot:

                                               the epoch used for all phased
    if varepoch is None                     -> light curve plots will be
                                               min(times)

    if varepoch is a single string == 'min' -> automatic epoch finding for all
                                               periods using light curve fits

    if varepoch is a single float           -> this epoch will be used for all
                                               phased light curve plots

    if varepoch is a list of lists             each epoch will be applied each
    of floats with length == 3              -> to the phased light curve for
    (i.e. for each of the best 3 periods       each period from each
     from the two period-finder results)       period-finder specifically

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
    fig.set_size_inches(30,24)

    ######################################################################
    ## PLOT 1 is the LSP from lspinfo1, including objectinfo and finder ##
    ######################################################################

    _make_periodogram(axes[0], lspinfo1, objectinfo,
                      findercmap, finderconvolve,
                      verbose=verbose,
                      findercachedir=findercachedir)

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
                             magsarefluxes=magsarefluxes)

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
                                        verbose=verbose)

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
                                        verbose=verbose)

        # end of plotting for each ax

        # save the plot to disk
        fig.set_tight_layout(True)
        if plotfpath.endswith('.png'):
            fig.savefig(plotfpath,dpi=plotdpi)
        else:
            fig.savefig(plotfpath)
        plt.close()

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


#########################################
## PICKLE CHECKPLOT UTILITY FUNCTIONS  ##
#########################################

def _xyzdist_to_distarcsec(xyzdist):
    '''
    This just inverts the xyz unit vector distance -> angular distance relation.

    '''

    return np.degrees(2.0*np.arcsin(xyzdist/2.0))*3600.0


def _base64_to_file(b64str, outfpath, writetostrio=False):
    '''
    This converts the base64 encoded string to a file.

    '''

    try:

        filebytes = base64.b64decode(b64str)

        # if we're writing back to a stringio object
        if writetostrio:

            outobj = strio(filebytes)
            return outobj

        # otherwise, we're writing to an actual file
        else:

            with open(outfpath,'wb') as outfd:
                outfd.write(filebytes)

            if os.path.exists(outfpath):
                return outfpath
            else:
                LOGERROR('could not write output file: %s' % outfpath)
                return None

    except Exception as e:

        LOGEXCEPTION('failed while trying to convert '
                     'b64 string to file %s' % outfpath)
        return None



def _pkl_finder_objectinfo(objectinfo,
                           varinfo,
                           findercmap,
                           finderconvolve,
                           sigclip,
                           normto,
                           normmingap,
                           deredden_object=True,
                           custom_bandpasses=None,
                           lclistpkl=None,
                           nbrradiusarcsec=30.0,
                           maxnumneighbors=5,
                           plotdpi=100,
                           findercachedir='~/.astrobase/stamp-cache',
                           verbose=True,
                           gaia_submit_timeout=10.0,
                           gaia_submit_tries=3,
                           gaia_max_timeout=180.0,
                           gaia_mirror='cds',
                           fast_mode=False,
                           complete_query_later=True):
    '''This returns the finder chart and object information as a dict.

    '''

    # optional mode to hit external services and fail fast if they timeout
    if fast_mode is True:
        skyview_timeout = 10.0
        skyview_retry_failed = False
        dust_timeout = 10.0
        gaia_submit_timeout = 5.0
        gaia_max_timeout = 10.0
        gaia_submit_tries = 1
        complete_query_later = False
        search_simbad = False

    elif isinstance(fast_mode, (int, float)) and fast_mode > 0.0:
        skyview_timeout = fast_mode
        skyview_retry_failed = False
        dust_timeout = fast_mode
        gaia_submit_timeout = 0.66*fast_mode
        gaia_max_timeout = fast_mode
        gaia_submit_tries = 1
        complete_query_later = False
        search_simbad = False

    else:
        skyview_timeout = 10.0
        skyview_retry_failed = True
        dust_timeout = 10.0
        search_simbad = True


    if (isinstance(objectinfo, dict) and
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

        # get the finder chart
        try:

            try:

                # generate the finder chart
                finder, finderheader = skyview_stamp(
                    objectinfo['ra'],
                    objectinfo['decl'],
                    convolvewith=finderconvolve,
                    verbose=verbose,
                    flip=False,
                    cachedir=findercachedir,
                    timeout=skyview_timeout,
                    retry_failed=skyview_retry_failed,
                )

            except OSError as e:

                if not fast_mode:

                    LOGERROR('finder image appears to be corrupt, retrying...')

                    # generate the finder chart
                    finder, finderheader = skyview_stamp(
                        objectinfo['ra'],
                        objectinfo['decl'],
                        convolvewith=finderconvolve,
                        verbose=verbose,
                        flip=False,
                        cachedir=findercachedir,
                        forcefetch=True,
                        timeout=skyview_timeout,
                        retry_failed=False  # do not start an infinite loop
                    )


            finderfig = plt.figure(figsize=(3,3),dpi=plotdpi)

            # initialize the finder WCS
            finderwcs = WCS(finderheader)

            # use the WCS transform for the plot
            ax = finderfig.add_subplot(111, frameon=False)
            ax.imshow(finder, cmap=findercmap, origin='lower')

            # skip down to after nbr stuff for the rest of the finderchart...

            # search around the target's location and get its neighbors if
            # lclistpkl is provided and it exists
            if (lclistpkl is not None and
                nbrradiusarcsec is not None and
                nbrradiusarcsec > 0.0):

                # if lclistpkl is a string, open it as a pickle
                if isinstance(lclistpkl, str) and os.path.exists(lclistpkl):

                    if lclistpkl.endswith('.gz'):
                        infd = gzip.open(lclistpkl,'rb')
                    else:
                        infd = open(lclistpkl,'rb')

                    lclist = pickle.load(infd)
                    infd.close()

                # otherwise, if it's a dict, we get it directly
                elif isinstance(lclistpkl, dict):

                    lclist = lclistpkl

                # finally, if it's nothing we recognize, ignore it
                else:

                    LOGERROR('could not understand lclistpkl kwarg, '
                             'not getting neighbor info')

                    lclist = dict()

                # check if we have a KDTree to use
                # if we don't, skip neighbor stuff
                if 'kdtree' not in lclist:

                    LOGERROR('neighbors within %.1f arcsec for %s could '
                             'not be found, no kdtree in lclistpkl: %s'
                             % (objectid, lclistpkl))
                    neighbors = None
                    kdt = None

                # otherwise, do neighbor processing
                else:

                    kdt = lclist['kdtree']

                    obj_cosdecl = np.cos(np.radians(objectinfo['decl']))
                    obj_sindecl = np.sin(np.radians(objectinfo['decl']))
                    obj_cosra = np.cos(np.radians(objectinfo['ra']))
                    obj_sinra = np.sin(np.radians(objectinfo['ra']))

                    obj_xyz = np.column_stack((obj_cosra*obj_cosdecl,
                                               obj_sinra*obj_cosdecl,
                                               obj_sindecl))
                    match_xyzdist = (
                        2.0 * np.sin(np.radians(nbrradiusarcsec/3600.0)/2.0)
                    )
                    matchdists, matchinds = kdt.query(
                        obj_xyz,
                        k=maxnumneighbors+1,  # get maxnumneighbors + tgt
                        distance_upper_bound=match_xyzdist
                    )

                    # sort by matchdist
                    mdsorted = np.argsort(matchdists[0])
                    matchdists = matchdists[0][mdsorted]
                    matchinds = matchinds[0][mdsorted]

                    # luckily, the indices to the kdtree are the same as that
                    # for the objects (I think)
                    neighbors = []

                    nbrind = 0

                    for md, mi in zip(matchdists, matchinds):

                        if np.isfinite(md) and md > 0.0:

                            # generate the xy for the finder we'll use a HTML5
                            # canvas and these pixcoords to highlight each
                            # neighbor when we mouse over its row in the
                            # neighbors tab
                            pixcoords = finderwcs.all_world2pix(
                                np.array([[lclist['objects']['ra'][mi],
                                           lclist['objects']['decl'][mi]]]),
                                1
                            )

                            # each elem is {'objectid',
                            #               'ra','decl',
                            #               'xpix','ypix',
                            #               'dist','lcfpath'}
                            thisnbr = {
                                'objectid':lclist['objects']['objectid'][mi],
                                'ra':lclist['objects']['ra'][mi],
                                'decl':lclist['objects']['decl'][mi],
                                'xpix':pixcoords[0,0],
                                'ypix':300.0 - pixcoords[0,1],
                                'dist':_xyzdist_to_distarcsec(md),
                                'lcfpath': lclist['objects']['lcfname'][mi]
                            }
                            neighbors.append(thisnbr)
                            nbrind = nbrind+1

                            # put in a nice marker for this neighbor into the
                            # overall finder chart
                            annotatex = pixcoords[0,0]
                            annotatey = pixcoords[0,1]

                            if ((300.0 - annotatex) > 50.0):
                                offx = annotatex + 30.0
                                xha = 'center'
                            else:
                                offx = annotatex - 30.0
                                xha = 'center'
                            if ((300.0 - annotatey) > 50.0):
                                offy = annotatey - 30.0
                                yha = 'center'
                            else:
                                offy = annotatey + 30.0
                                yha = 'center'

                            ax.annotate('N%s' % nbrind,
                                        (annotatex, annotatey),
                                        xytext=(offx, offy),
                                        arrowprops={'facecolor':'blue',
                                                    'edgecolor':'blue',
                                                    'width':1.0,
                                                    'headwidth':1.0,
                                                    'headlength':0.1,
                                                    'shrink':0.0},
                                        color='blue',
                                        horizontalalignment=xha,
                                        verticalalignment=yha)

            # if there are no neighbors, set the 'neighbors' key to None
            else:

                neighbors = None
                kdt = None

            #
            # finish up the finder chart after neighbors are processed
            #
            ax.set_xticks([])
            ax.set_yticks([])

            # add a reticle pointing to the object's coordinates
            object_pixcoords = finderwcs.all_world2pix([[objectinfo['ra'],
                                                         objectinfo['decl']]],1)

            ax.axvline(
                # x=150.0,
                x=object_pixcoords[0,0],
                ymin=0.375,
                ymax=0.45,
                linewidth=1,
                color='b'
            )
            ax.axhline(
                # y=150.0,
                y=object_pixcoords[0,1],
                xmin=0.375,
                xmax=0.45,
                linewidth=1,
                color='b'
            )
            ax.set_frame_on(False)

            # this is the output instance
            finderpng = strio()
            finderfig.savefig(finderpng,
                              bbox_inches='tight',
                              pad_inches=0.0, format='png')
            plt.close()

            # encode the finderpng instance to base64
            finderpng.seek(0)
            finderb64 = base64.b64encode(finderpng.read())

            # close the stringio buffer
            finderpng.close()

        except Exception as e:

            LOGEXCEPTION('could not fetch a DSS stamp for this '
                         'object %s using coords (%.3f,%.3f)' %
                         (objectid, objectinfo['ra'], objectinfo['decl']))
            finderb64 = None
            neighbors = None
            kdt = None

    # if we don't have ra, dec info, then everything is none up to this point
    else:

        finderb64 = None
        neighbors = None
        kdt = None

    #
    # end of finder chart operations
    #

    # now that we have the finder chart, get the rest of the object
    # information

    # get the rest of the features, these don't necessarily rely on ra, dec and
    # should degrade gracefully if these aren't provided
    if isinstance(objectinfo, dict):

        if 'objectid' not in objectinfo and 'hatid' in objectinfo:
            objectid = objectinfo['hatid']
            objectinfo['objectid'] = objectid
        elif 'objectid' in objectinfo:
            objectid = objectinfo['objectid']
        else:
            objectid = os.urandom(12).hex()[:7]
            objectinfo['objectid'] = objectid
            LOGWARNING('no objectid found in objectinfo dict, '
                       'making up a random one: %s')


        # first, the color features
        colorfeat = color_features(objectinfo,
                                   deredden=deredden_object,
                                   custom_bandpasses=custom_bandpasses,
                                   dust_timeout=dust_timeout)

        # next, get the neighbor features and GAIA info
        nbrfeat = neighbor_gaia_features(
            objectinfo,
            kdt,
            nbrradiusarcsec,
            verbose=False,
            gaia_submit_timeout=gaia_submit_timeout,
            gaia_submit_tries=gaia_submit_tries,
            gaia_max_timeout=gaia_max_timeout,
            gaia_mirror=gaia_mirror,
            complete_query_later=complete_query_later,
            search_simbad=search_simbad
        )

        # see if the objectinfo dict has pmra/pmdecl entries.  if it doesn't,
        # then we'll see if the nbrfeat dict has pmra/pmdecl from GAIA. we'll
        # set the appropriate provenance keys as well so we know where the PM
        # came from
        if ( ('pmra' not in objectinfo) or
             ( ('pmra' in objectinfo) and
               ( (objectinfo['pmra'] is None) or
                 (not np.isfinite(objectinfo['pmra'])) ) ) ):

            if 'ok' in nbrfeat['gaia_status']:

                objectinfo['pmra'] = nbrfeat['gaia_pmras'][0]
                objectinfo['pmra_err'] = nbrfeat['gaia_pmra_errs'][0]
                objectinfo['pmra_source'] = 'gaia'

                if verbose:
                    LOGWARNING('pmRA not found in provided objectinfo dict, '
                               'using value from GAIA')

        else:
            objectinfo['pmra_source'] = 'light curve'

        if ( ('pmdecl' not in objectinfo) or
             ( ('pmdecl' in objectinfo) and
               ( (objectinfo['pmdecl'] is None) or
                 (not np.isfinite(objectinfo['pmdecl'])) ) ) ):

            if 'ok' in nbrfeat['gaia_status']:

                objectinfo['pmdecl'] = nbrfeat['gaia_pmdecls'][0]
                objectinfo['pmdecl_err'] = nbrfeat['gaia_pmdecl_errs'][0]
                objectinfo['pmdecl_source'] = 'gaia'

                if verbose:
                    LOGWARNING('pmDEC not found in provided objectinfo dict, '
                               'using value from GAIA')

        else:
            objectinfo['pmdecl_source'] = 'light curve'


        # try to get the object's coord features next
        coordfeat = coord_features(objectinfo)

        # finally, get the object's color classification
        colorclass = color_classification(colorfeat, coordfeat)

        # update the objectinfo dict with everything
        objectinfo.update(colorfeat)
        objectinfo.update(coordfeat)
        objectinfo.update(colorclass)
        objectinfo.update(nbrfeat)

        # update GAIA info so it's available at the first level
        if 'ok' in objectinfo['gaia_status']:
            objectinfo['gaiamag'] = objectinfo['gaia_mags'][0]
            objectinfo['gaia_absmag'] = objectinfo['gaia_absolute_mags'][0]
            objectinfo['gaia_parallax'] = objectinfo['gaia_parallaxes'][0]
            objectinfo['gaia_parallax_err'] = (
                objectinfo['gaia_parallax_errs'][0]
            )
            objectinfo['gaia_pmra'] = objectinfo['gaia_pmras'][0]
            objectinfo['gaia_pmra_err'] = objectinfo['gaia_pmra_errs'][0]
            objectinfo['gaia_pmdecl'] = objectinfo['gaia_pmdecls'][0]
            objectinfo['gaia_pmdecl_err'] = objectinfo['gaia_pmdecl_errs'][0]

        else:
            objectinfo['gaiamag'] = np.nan
            objectinfo['gaia_absmag'] = np.nan
            objectinfo['gaia_parallax'] = np.nan
            objectinfo['gaia_parallax_err'] = np.nan
            objectinfo['gaia_pmra'] = np.nan
            objectinfo['gaia_pmra_err'] = np.nan
            objectinfo['gaia_pmdecl'] = np.nan
            objectinfo['gaia_pmdecl_err'] = np.nan

        # put together the initial checkplot pickle dictionary
        # this will be updated by the functions below as appropriate
        # and will written out as a gzipped pickle at the end of processing
        checkplotdict = {'objectid':objectid,
                         'neighbors':neighbors,
                         'objectinfo':objectinfo,
                         'finderchart':finderb64,
                         'sigclip':sigclip,
                         'normto':normto,
                         'normmingap':normmingap}

        # add the objecttags key to objectinfo
        checkplotdict['objectinfo']['objecttags'] = None

    # if there's no objectinfo, we can't do anything.
    else:

        # empty objectinfo dict
        checkplotdict = {'objectid':None,
                         'neighbors':None,
                         'objectinfo':{
                             'available_bands':[],
                             'available_band_labels':[],
                             'available_dereddened_bands':[],
                             'available_dereddened_band_labels':[],
                             'available_colors':[],
                             'available_color_labels':[],
                             'bmag':None,
                             'bmag-vmag':None,
                             'decl':None,
                             'hatid':None,
                             'hmag':None,
                             'imag-jmag':None,
                             'jmag-kmag':None,
                             'jmag':None,
                             'kmag':None,
                             'ndet':None,
                             'network':None,
                             'objecttags':None,
                             'pmdecl':None,
                             'pmdecl_err':None,
                             'pmra':None,
                             'pmra_err':None,
                             'propermotion':None,
                             'ra':None,
                             'rpmj':None,
                             'sdssg':None,
                             'sdssi':None,
                             'sdssr':None,
                             'stations':None,
                             'twomassid':None,
                             'ucac4id':None,
                             'vmag':None
                         },
                         'finderchart':None,
                         'sigclip':sigclip,
                         'normto':normto,
                         'normmingap':normmingap}

    # end of objectinfo processing

    # add the varinfo dict
    if isinstance(varinfo, dict):
        checkplotdict['varinfo'] = varinfo
    else:
        checkplotdict['varinfo'] = {
            'objectisvar':None,
            'vartags':None,
            'varisperiodic':None,
            'varperiod':None,
            'varepoch':None,
        }

    return checkplotdict



def _pkl_periodogram(lspinfo,
                     plotdpi=100,
                     override_pfmethod=None):
    '''This returns the periodogram plot PNG as base64, plus info as a dict.

    '''

    # get the appropriate plot ylabel
    pgramylabel = PLOTYLABELS[lspinfo['method']]

    # get the periods and lspvals from lspinfo
    periods = lspinfo['periods']
    lspvals = lspinfo['lspvals']
    bestperiod = lspinfo['bestperiod']
    nbestperiods = lspinfo['nbestperiods']
    nbestlspvals = lspinfo['nbestlspvals']

    # open the figure instance
    pgramfig = plt.figure(figsize=(7.5,4.8),dpi=plotdpi)

    # make the plot
    plt.plot(periods,lspvals)

    plt.xscale('log',basex=10)
    plt.xlabel('Period [days]')
    plt.ylabel(pgramylabel)
    plottitle = '%s - %.6f d' % (METHODLABELS[lspinfo['method']],
                                 bestperiod)
    plt.title(plottitle)

    # show the best five peaks on the plot
    for xbestperiod, xbestpeak in zip(nbestperiods,
                                      nbestlspvals):
        plt.annotate('%.6f' % xbestperiod,
                     xy=(xbestperiod, xbestpeak), xycoords='data',
                     xytext=(0.0,25.0), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->"),fontsize='14.0')

    # make a grid
    plt.grid(color='#a9a9a9',
             alpha=0.9,
             zorder=0,
             linewidth=1.0,
             linestyle=':')

    # this is the output instance
    pgrampng = strio()
    pgramfig.savefig(pgrampng,
                     # bbox_inches='tight',
                     pad_inches=0.0, format='png')
    plt.close()

    # encode the finderpng instance to base64
    pgrampng.seek(0)
    pgramb64 = base64.b64encode(pgrampng.read())

    # close the stringio buffer
    pgrampng.close()

    if not override_pfmethod:

        # this is the dict to return
        checkplotdict = {
            lspinfo['method']:{
                'periods':periods,
                'lspvals':lspvals,
                'bestperiod':bestperiod,
                'nbestperiods':nbestperiods,
                'nbestlspvals':nbestlspvals,
                'periodogram':pgramb64,
            }
        }

    else:

        # this is the dict to return
        checkplotdict = {
            override_pfmethod:{
                'periods':periods,
                'lspvals':lspvals,
                'bestperiod':bestperiod,
                'nbestperiods':nbestperiods,
                'nbestlspvals':nbestlspvals,
                'periodogram':pgramb64,
            }
        }

    return checkplotdict



def _pkl_magseries_plot(stimes, smags, serrs,
                        plotdpi=100,
                        magsarefluxes=False):
    '''This returns the magseries plot PNG as base64, plus arrays as dict.

    '''

    scaledplottime = stimes - npmin(stimes)

    # open the figure instance
    magseriesfig = plt.figure(figsize=(7.5,4.8),dpi=plotdpi)

    plt.plot(scaledplottime,
             smags,
             marker='o',
             ms=2.0, ls='None',mew=0,
             color='green',
             rasterized=True)

    # flip y axis for mags
    if not magsarefluxes:
        plot_ylim = plt.ylim()
        plt.ylim((plot_ylim[1], plot_ylim[0]))

    # set the x axis limit
    plt.xlim((npmin(scaledplottime)-2.0,
              npmax(scaledplottime)+2.0))

    # make a grid
    plt.grid(color='#a9a9a9',
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

    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)

    # fix the yaxis ticks (turns off offset and uses the full
    # value of the yaxis tick)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    # this is the output instance
    magseriespng = strio()
    magseriesfig.savefig(magseriespng,
                         # bbox_inches='tight',
                         pad_inches=0.05, format='png')
    plt.close()

    # encode the finderpng instance to base64
    magseriespng.seek(0)
    magseriesb64 = base64.b64encode(magseriespng.read())

    # close the stringio buffer
    magseriespng.close()

    checkplotdict = {
        'magseries':{
            'plot':magseriesb64,
            'times':stimes,
            'mags':smags,
            'errs':serrs
        }
    }

    return checkplotdict



def _pkl_phased_magseries_plot(checkplotdict,
                               lspmethod,
                               periodind,
                               stimes, smags, serrs,
                               varperiod, varepoch,
                               lspmethodind=0,
                               phasewrap=True,
                               phasesort=True,
                               phasebin=0.002,
                               minbinelems=7,
                               plotxlim=(-0.8,0.8),
                               plotdpi=100,
                               bestperiodhighlight=None,
                               xgridlines=None,
                               xliminsetmode=False,
                               magsarefluxes=False,
                               directreturn=False,
                               overplotfit=None,
                               verbose=True,
                               override_pfmethod=None):
    '''This returns the phased magseries plot PNG as base64 plus info as a dict.

    checkplotdict is an existing checkplotdict to update. If it's None or
    directreturn = True, then the generated dict result for this magseries plot
    will be returned directly.

    lspmethod is a string indicating the type of period-finding algorithm that
    produced the period. If this is not in METHODSHORTLABELS, it will be used
    verbatim.

    periodind is the index of the period.

      If == 0  -> best period and bestperiodhighlight is applied if not None
      If > 0   -> some other peak of the periodogram
      If == -1 -> special mode w/ no periodogram labels and enabled highlight

    overplotfit is a result dict returned from one of the XXXX_fit_magseries
    functions in astrobase.varbase.lcfit. If this is not None, then the fit will
    be overplotted on the phased light curve plot.

    overplotfit must have the following structure and at least the keys below if
    not originally from one of these functions:

    {'fittype':<str: name of fit method>,
     'fitchisq':<float: the chi-squared value of the fit>,
     'fitredchisq':<float: the reduced chi-squared value of the fit>,
     'fitinfo':{'fitmags':<ndarray: model mags or fluxes from fit function>},
     'magseries':{'times':<ndarray: times at which the fitmags are evaluated>}}

    fitmags and times should all be of the same size. overplotfit is copied over
    to the checkplot dict for each specific phased LC plot to save all of this
    information.

    '''
    # open the figure instance
    phasedseriesfig = plt.figure(figsize=(7.5,4.8),dpi=plotdpi)

    plotvarepoch = None

    # figure out the epoch, if it's None, use the min of the time
    if varepoch is None:
        plotvarepoch = npmin(stimes)

    # if the varepoch is 'min', then fit a spline to the light curve
    # phased using the min of the time, find the fit mag minimum and use
    # the time for that as the varepoch
    elif isinstance(varepoch,str) and varepoch == 'min':

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
                plotvarepoch = plotvarepoch[0]


        except Exception as e:

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

    # special case with varepoch lists per each period-finder method
    elif isinstance(varepoch, list):

        try:
            thisvarepochlist = varepoch[lspmethodind]
            plotvarepoch = thisvarepochlist[periodind]
        except Exception as e:
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
        LOGINFO('plotting %s phased LC with period %s: %.6f, epoch: %.5f' %
                (lspmethod, periodind, varperiod, plotvarepoch))

    # make the plot title based on the lspmethod
    if periodind == 0:
        plottitle = '%s best period: %.6f d - epoch: %.5f' % (
            (METHODSHORTLABELS[lspmethod] if lspmethod in METHODSHORTLABELS
             else lspmethod),
            varperiod,
            plotvarepoch
        )
    elif periodind > 0:
        plottitle = '%s peak %s: %.6f d - epoch: %.5f' % (
            (METHODSHORTLABELS[lspmethod] if lspmethod in METHODSHORTLABELS
             else lspmethod),
            periodind+1,
            varperiod,
            plotvarepoch
        )
    elif periodind == -1:
        plottitle = '%s period: %.6f d - epoch: %.5f' % (
            lspmethod,
            varperiod,
            plotvarepoch
        )


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

    else:
        binplotphase = None
        binplotmags = None


    # finally, make the phased LC plot
    plt.plot(plotphase,
             plotmags,
             marker='o',
             ms=2.0, ls='None',mew=0,
             color='gray',
             rasterized=True)

    # overlay the binned phased LC plot if we're making one
    if phasebin:
        plt.plot(binplotphase,
                 binplotmags,
                 marker='o',
                 ms=4.0, ls='None',mew=0,
                 color='#1c1e57',
                 rasterized=True)


    # if we're making a overplotfit, then plot the fit over the other stuff
    if overplotfit and isinstance(overplotfit, dict):

        fitmethod = overplotfit['fittype']
        fitredchisq = overplotfit['fitredchisq']

        plotfitmags = overplotfit['fitinfo']['fitmags']
        plotfittimes = overplotfit['magseries']['times']

        # phase the fit magseries
        fitphasedlc = phase_magseries(plotfittimes,
                                      plotfitmags,
                                      varperiod,
                                      plotvarepoch,
                                      wrap=phasewrap,
                                      sort=phasesort)
        plotfitphase = fitphasedlc['phase']
        plotfitmags = fitphasedlc['mags']

        plotfitlabel = (r'%s fit ${\chi}^2/{\mathrm{dof}} = %.3f$' %
                        (fitmethod, fitredchisq))

        # plot the fit phase and mags
        plt.plot(plotfitphase, plotfitmags,'k-',
                 linewidth=3, rasterized=True,label=plotfitlabel)

        plt.legend(loc='upper left', frameon=False)

    # flip y axis for mags
    if not magsarefluxes:
        plot_ylim = plt.ylim()
        plt.ylim((plot_ylim[1], plot_ylim[0]))

    # set the x axis limit
    if not plotxlim:
        plt.xlim((npmin(plotphase)-0.1,
                  npmax(plotphase)+0.1))
    else:
        plt.xlim((plotxlim[0],plotxlim[1]))

    # make a grid
    ax = plt.gca()
    if isinstance(xgridlines, (list, tuple)):
        ax.set_xticks(xgridlines, minor=False)

    plt.grid(color='#a9a9a9',
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

    plt.xlabel(plot_xlabel)
    plt.ylabel(plot_ylabel)

    # fix the yaxis ticks (turns off offset and uses the full
    # value of the yaxis tick)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    # set the plot title
    plt.title(plottitle)

    # make sure the best period phased LC plot stands out
    if (periodind == 0 or periodind == -1) and bestperiodhighlight:
        if MPLVERSION >= (2,0,0):
            plt.gca().set_facecolor(bestperiodhighlight)
        else:
            plt.gca().set_axis_bgcolor(bestperiodhighlight)

    # if we're making an inset plot showing the full range
    if (plotxlim and isinstance(plotxlim, (list, tuple)) and
        len(plotxlim) == 2 and xliminsetmode is True):

        # bump the ylim of the plot so that the inset can fit in this axes plot
        axesylim = plt.gca().get_ylim()

        if magsarefluxes:
            plt.gca().set_ylim(
                axesylim[0],
                axesylim[1] + 0.5*npabs(axesylim[1]-axesylim[0])
            )
        else:
            plt.gca().set_ylim(
                axesylim[0],
                axesylim[1] - 0.5*npabs(axesylim[1]-axesylim[0])
            )

        # put the inset axes in
        inset = inset_axes(plt.gca(), width="40%", height="40%", loc=1)

        # make the scatter plot for the phased LC plot
        inset.plot(plotphase,
                   plotmags,
                   marker='o',
                   ms=2.0, ls='None',mew=0,
                   color='gray',
                   rasterized=True)

        if phasebin:
            # make the scatter plot for the phased LC plot
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
        inset.text(0.5,0.9,'full phased light curve',
                   ha='center',va='center',transform=inset.transAxes)
        # don't show axes labels or ticks
        inset.set_xticks([])
        inset.set_yticks([])

    # this is the output instance
    phasedseriespng = strio()
    phasedseriesfig.savefig(phasedseriespng,
                            # bbox_inches='tight',
                            pad_inches=0.0, format='png')
    plt.close()

    # encode the finderpng instance to base64
    phasedseriespng.seek(0)
    phasedseriesb64 = base64.b64encode(phasedseriespng.read())

    # close the stringio buffer
    phasedseriespng.close()

    # this includes a fitinfo dict if one is provided in overplotfit
    retdict = {
        'plot':phasedseriesb64,
        'period':varperiod,
        'epoch':plotvarepoch,
        'phase':plotphase,
        'phasedmags':plotmags,
        'binphase':binplotphase,
        'binphasedmags':binplotmags,
        'phasewrap':phasewrap,
        'phasesort':phasesort,
        'phasebin':phasebin,
        'minbinelems':minbinelems,
        'plotxlim':plotxlim,
        'lcfit':overplotfit,
    }

    # if we're returning stuff directly, i.e. not being used embedded within
    # the checkplot_dict function
    if directreturn or checkplotdict is None:

        return retdict

    # this requires the checkplotdict to be present already, we'll just update
    # it at the appropriate lspmethod and periodind
    else:

        if override_pfmethod:
            checkplotdict[override_pfmethod][periodind] = retdict
        else:
            checkplotdict[lspmethod][periodind] = retdict

        return checkplotdict



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



########################
## READ/WRITE PICKLES ##
########################

def _write_checkplot_picklefile(checkplotdict,
                                outfile=None,
                                protocol=None,
                                outgzip=False):

    '''This writes the checkplotdict to a (gzipped) pickle file.

    If outfile is None, writes a (gzipped) pickle file of the form:

    checkplot-{objectid}.pkl(.gz)

    to the current directory.

    protocol sets the pickle protocol:

    4 -> default in Python >= 3.4 - way faster but incompatible with Python 2
    3 -> default in Python 3.0-3.3
    2 -> default in Python 2 - very slow, but compatible with Python 2 and 3

    The default protocol kwarg is None, this will make an automatic choice for
    pickle protocol that's best suited for the version of Python in use. Note
    that this will make pickles generated by Py3 incompatible with Py2.

    '''

    # figure out which protocol to use
    # for Python >= 3.4; use v4 by default
    if ((sys.version_info[0:2] >= (3,4) and not protocol) or
        (protocol > 2)):
        protocol = 4

    elif ((sys.version_info[0:2] >= (3,0) and not protocol) or
          (protocol > 2)):
        protocol = 3

    # for Python == 2.7; use v2
    elif sys.version_info[0:2] == (2,7) and not protocol:
        protocol = 2

    # otherwise, if left unspecified, use the slowest but most compatible
    # protocol. this will be readable by all (most?) Pythons
    elif not protocol:
        protocol = 0


    if outgzip:

        if not outfile:

            outfile = (
                'checkplot-{objectid}.pkl.gz'.format(
                    objectid=squeeze(checkplotdict['objectid']).replace(' ','-')
                )
            )

        with gzip.open(outfile,'wb') as outfd:
            pickle.dump(checkplotdict,outfd,protocol=protocol)

    else:

        if not outfile:

            outfile = (
                'checkplot-{objectid}.pkl'.format(
                    objectid=squeeze(checkplotdict['objectid']).replace(' ','-')
                )
            )

        # make sure to do the right thing if '.gz' is in the filename but
        # outgzip was False
        if outfile.endswith('.gz'):

            LOGWARNING('output filename ends with .gz but kwarg outgzip=False. '
                       'will use gzip to compress the output pickle')
            with gzip.open(outfile,'wb') as outfd:
                pickle.dump(checkplotdict,outfd,protocol=protocol)

        else:
            with open(outfile,'wb') as outfd:
                pickle.dump(checkplotdict,outfd,protocol=protocol)

    return os.path.abspath(outfile)



def _read_checkplot_picklefile(checkplotpickle):
    '''This reads a checkplot gzipped pickle file back into a dict.

    NOTE: the try-except is for Python 2 pickles that have numpy arrays in
    them. Apparently, these aren't compatible with Python 3. See here:

    http://stackoverflow.com/q/11305790

    The workaround is noted in this answer:

    http://stackoverflow.com/a/41366785

    '''

    if checkplotpickle.endswith('.gz'):

        try:
            with gzip.open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd)

        except UnicodeDecodeError:

            with gzip.open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd, encoding='latin1')

    else:

        try:
            with open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd)

        except UnicodeDecodeError:

            with open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd, encoding='latin1')

    return cpdict



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
                   gaia_mirror='cds',
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
    gaia_submit_tries = 1
    complete_query_later = False

    If fast_mode is a positive integer or float, timeouts will be set to
    fast_mode and the gaia_submit_timeout will be set to
    0.66*fast_mode. gaia_submit_timeout and gaia_max_timeout are re-used for
    SIMBAD as well.

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
    if isinstance(stimes, astcolumn):
        stimes = stimes.data
        LOGWARNING('times is an astropy.table.Column object, '
                   'changing to numpy array because of '
                   'potential unpickling issues')
    if isinstance(smags, astcolumn):
        smags = smags.data
        LOGWARNING('mags is an astropy.table.Column object, '
                   'changing to numpy array because of '
                   'potential unpickling issues')
    if isinstance(serrs, astcolumn):
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

                # this updates things as it runs
                checkplotdict = _pkl_phased_magseries_plot(
                    checkplotdict,
                    lspinfo['method'],
                    nbpind,
                    stimes, smags, serrs,
                    nbperiod, varepoch,
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
            if 'altsnr' in lspinfo:
                if override_pfmethod in checkplotdict:
                    checkplotdict[override_pfmethod]['altsnr'] = (
                        lspinfo['altsnr']
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
                     gaia_mirror='cds',
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
    gaia_submit_tries = 1
    complete_query_later = False

    If fast_mode is a positive integer or float, timeouts will be set to
    fast_mode and the gaia_submit_timeout will be set to
    0.66*fast_mode. gaia_submit_timeout and gaia_max_timeout are re-used for
    SIMBAD as well.

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



def checkplot_pickle_to_png(checkplotin,
                            outfile,
                            extrarows=None):
    '''This reads the pickle provided, and writes out a PNG.

    checkplotin is either a checkplot dict produced by checkplot_pickle above or
    a pickle file produced by the same function.

    The PNG has 4 x N tiles, as below:

    [    finder    ] [  objectinfo  ] [ varinfo/comments ] [ unphased LC  ]
    [ periodogram1 ] [ phased LC P1 ] [   phased LC P2   ] [ phased LC P3 ]
    [ periodogram2 ] [ phased LC P1 ] [   phased LC P2   ] [ phased LC P3 ]
                                     .
                                     .
    [ periodogramN ] [ phased LC P1 ] [ phased LC P2 ] [ phased LC P3 ]

    for N independent period-finding methods producing:

    - periodogram1,2,3...N: the periodograms from each method
    - phased LC P1,P2,P3: the phased lightcurves using the best 3 peaks in each
                          periodogram

    outfile is the output PNG file to generate.

    extrarows is a list of 4-element tuples containing paths to PNG files that
    will be added to the end of the rows generated from the checkplotin
    pickle/dict. Each tuple represents a row in the final output PNG file. If
    there are less than 4 elements per tuple, the missing elements will be
    filled in with white-space. If there are more than 4 elements per tuple,
    only the first four will be used.

    The purpose of this kwarg is to incorporate periodograms and phased LC plots
    (in the form of PNGs) generated from an external period-finding function or
    program (like vartools) to allow for comparison with astrobase results.

    Each external PNG will be resized to 750 x 480 pixels to fit into an output
    image cell.

    By convention, each 4-element tuple should contain:

    a periodiogram PNG
    phased LC PNG with 1st best peak period from periodogram
    phased LC PNG with 2nd best peak period from periodogram
    phased LC PNG with 3rd best peak period from periodogram

    example of extrarows:

    extrarows = [('/path/to/external/bls-periodogram.png',
                  '/path/to/external/bls-phasedlc-plot-bestpeak.png',
                  '/path/to/external/bls-phasedlc-plot-peak2.png',
                  '/path/to/external/bls-phasedlc-plot-peak3.png'),
                 ('/path/to/external/pdm-periodogram.png',
                  '/path/to/external/pdm-phasedlc-plot-bestpeak.png',
                  '/path/to/external/pdm-phasedlc-plot-peak2.png',
                  '/path/to/external/pdm-phasedlc-plot-peak3.png'),
                  ...]


    '''

    # figure out if the checkplotpickle is a filename
    # python 3
    if sys.version_info[:2] > (3,2):

        if (isinstance(checkplotin, str) and os.path.exists(checkplotin)):
            cpd = _read_checkplot_picklefile(checkplotin)
        elif isinstance(checkplotin, dict):
            cpd = checkplotin
        else:
            LOGERROR('checkplotin: %s of type %s is not a '
                     'valid checkplot filename (or does not exist), or a dict' %
                     (os.path.abspath(checkplotin), type(checkplotin)))
            return None

    # check for unicode in python 2.7
    else:

        # get the current checkplotdict
        if ((isinstance(checkplotin, str) or
             isinstance(checkplotin, unicode)) and
            os.path.exists(checkplotin)):
            cpd = _read_checkplot_picklefile(checkplotin)
        elif isinstance(checkplotin,dict):
            cpd = checkplotin
        else:
            LOGERROR('checkplotin: %s of type %s is not a '
                     'valid checkplot filename (or does not exist), or a dict' %
                     (os.path.abspath(checkplotin), type(checkplotin)))
            return None

    # figure out the dimensions of the output png
    # each cell is 750 x 480 pixels
    # a row is made of four cells
    # - the first row is for object info
    # - the rest are for periodograms and phased LCs, one row per method
    # if there are more than three phased LC plots per method, we'll only plot 3
    if 'pfmethods' in cpd:
        cplspmethods = cpd['pfmethods']
    else:
        cplspmethods = []
        for pfm in METHODSHORTLABELS:
            if pfm in cpd:
                cplspmethods.append(pfm)


    cprows = len(cplspmethods)

    # add in any extra rows from neighbors
    if 'neighbors' in cpd and cpd['neighbors'] and len(cpd['neighbors']) > 0:
        nbrrows = len(cpd['neighbors'])
    else:
        nbrrows = 0

    # add in any extra rows from keyword arguments
    if extrarows and len(extrarows) > 0:
        erows = len(extrarows)
    else:
        erows = 0

    # add in any extra rows from the checkplot dict
    if ('externalplots' in cpd and
        cpd['externalplots'] and
        len(cpd['externalplots']) > 0):
        cpderows = len(cpd['externalplots'])
    else:
        cpderows = 0

    totalwidth = 3000
    totalheight = 480 + (cprows + erows + nbrrows + cpderows)*480

    # this is the output PNG
    outimg = Image.new('RGBA',(totalwidth, totalheight),(255,255,255,255))

    # now fill in the rows of the output png. we'll use Pillow to build up the
    # output image from the already stored plots and stuff in the checkplot
    # dict.

    ###############################
    # row 1, cell 1: finder chart #
    ###############################

    if cpd['finderchart']:
        finder = Image.open(
            _base64_to_file(cpd['finderchart'], None, writetostrio=True)
        )
        bigfinder = finder.resize((450,450), Image.ANTIALIAS)
        outimg.paste(bigfinder,(150,20))

    #####################################
    # row 1, cell 2: object information #
    #####################################

    # find the font we need from the package data
    fontpath = os.path.join(os.path.dirname(__file__),
                            'cpserver',
                            'cps-assets',
                            'DejaVuSans.ttf')
    # load the font
    if os.path.exists(fontpath):
        cpfontnormal = ImageFont.truetype(fontpath, 20)
        cpfontlarge = ImageFont.truetype(fontpath, 28)
    else:
        LOGWARNING('could not find bundled '
                   'DejaVu Sans font in the astrobase package '
                   'data, using ugly defaults...')
        cpfontnormal = ImageFont.load_default()
        cpfontlarge = ImageFont.load_default()

    # the image draw object
    objinfodraw = ImageDraw.Draw(outimg)

    # write out the object information

    # objectid
    objinfodraw.text(
        (625, 25),
        cpd['objectid'] if cpd['objectid'] else 'no objectid',
        font=cpfontlarge,
        fill=(0,0,255,255)
    )
    # twomass id
    if 'twomassid' in cpd['objectinfo']:
        objinfodraw.text(
            (625, 60),
            ('2MASS J%s' % cpd['objectinfo']['twomassid']
             if cpd['objectinfo']['twomassid']
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    # ndet
    if 'ndet' in cpd['objectinfo']:
        objinfodraw.text(
            (625, 85),
            ('LC points: %s' % cpd['objectinfo']['ndet']
             if cpd['objectinfo']['ndet'] is not None
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (625, 85),
            ('LC points: %s' % cpd['magseries']['times'].size),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    # coords and PM
    objinfodraw.text(
        (625, 125),
        ('Coords and PM'),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )
    if 'ra' in cpd['objectinfo'] and 'decl' in cpd['objectinfo']:
        objinfodraw.text(
            (900, 125),
            (('RA, Dec: %.3f, %.3f' %
              (cpd['objectinfo']['ra'], cpd['objectinfo']['decl']))
             if (cpd['objectinfo']['ra'] is not None and
                 cpd['objectinfo']['decl'] is not None)
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (900, 125),
            'RA, Dec: nan, nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    if 'propermotion' in cpd['objectinfo']:
        objinfodraw.text(
            (900, 150),
            (('Total PM: %.5f mas/yr' % cpd['objectinfo']['propermotion'])
             if (cpd['objectinfo']['propermotion'] is not None)
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (900, 150),
            'Total PM: nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    if 'rpmj' in cpd['objectinfo']:
        objinfodraw.text(
            (900, 175),
            (('Reduced PM [Jmag]: %.3f' % cpd['objectinfo']['rpmj'])
             if (cpd['objectinfo']['rpmj'] is not None)
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (900, 175),
            'Reduced PM [Jmag]: nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # here, we have to deal with two generations of objectinfo dicts

    # first, deal with the new generation of objectinfo dicts
    if 'available_dereddened_bands' in cpd['objectinfo']:

        #
        # first, we deal with the bands and mags
        #
        # magnitudes
        objinfodraw.text(
            (625, 200),
            'Magnitudes',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        # process the various bands
        # if dereddened mags aren't available, use the observed mags
        if len(cpd['objectinfo']['available_bands']) > 0:

            # we'll get all the available mags
            for bandind, band, label in zip(
                    range(len(cpd['objectinfo']['available_bands'])),
                    cpd['objectinfo']['available_bands'],
                    cpd['objectinfo']['available_band_labels']
            ):

                thisbandmag = cpd['objectinfo'][band]

                # we'll draw stuff in three rows depending on the number of
                # bands we have to use
                if bandind in (0,1,2,3,4):

                    thispos = (900+125*bandind, 200)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (label, thisbandmag),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                elif bandind in (5,6,7,8,9):

                    rowbandind = bandind - 5

                    thispos = (900+125*rowbandind, 225)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (label, thisbandmag),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                else:

                    rowbandind = bandind - 10

                    thispos = (900+125*rowbandind, 250)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (label, thisbandmag),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )


        #
        # next, deal with the colors
        #
        # colors
        if ('dereddened' in cpd['objectinfo'] and
            cpd['objectinfo']['dereddened'] is True):
            deredlabel = "(dereddened)"
        else:
            deredlabel = ""

        objinfodraw.text(
            (625, 275),
            'Colors %s' % deredlabel,
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        if len(cpd['objectinfo']['available_colors']) > 0:

            # we'll get all the available mags (dereddened versions preferred)
            for colorind, color, colorlabel in zip(
                    range(len(cpd['objectinfo']['available_colors'])),
                    cpd['objectinfo']['available_colors'],
                    cpd['objectinfo']['available_color_labels']
            ):

                thiscolor = cpd['objectinfo'][color]

                # we'll draw stuff in three rows depending on the number of
                # bands we have to use
                if colorind in (0,1,2,3,4):

                    thispos = (900+150*colorind, 275)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (colorlabel, thiscolor),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                elif colorind in (5,6,7,8,9):

                    thisrowind = colorind - 5
                    thispos = (900+150*thisrowind, 300)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (colorlabel, thiscolor),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                elif colorind in (10,11,12,13,14):

                    thisrowind = colorind - 10
                    thispos = (900+150*thisrowind, 325)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (colorlabel, thiscolor),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                else:

                    thisrowind = colorind - 15
                    thispos = (900+150*thisrowind, 350)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (colorlabel, thiscolor),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

    # otherwise, deal with older generation of checkplots
    else:

        objinfodraw.text(
            (625, 200),
            ('Magnitudes'),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        objinfodraw.text(
            (900, 200),
            ('gri: %.3f, %.3f, %.3f' %
             ((cpd['objectinfo']['sdssg'] if
               ('sdssg' in cpd['objectinfo'] and
                cpd['objectinfo']['sdssg'] is not None)
               else npnan),
              (cpd['objectinfo']['sdssr'] if
               ('sdssr' in cpd['objectinfo'] and
                cpd['objectinfo']['sdssr'] is not None)
               else npnan),
              (cpd['objectinfo']['sdssi'] if
               ('sdssi' in cpd['objectinfo'] and
                cpd['objectinfo']['sdssi'] is not None)
               else npnan))),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
        objinfodraw.text(
            (900, 225),
            ('JHK: %.3f, %.3f, %.3f' %
             ((cpd['objectinfo']['jmag'] if
               ('jmag' in cpd['objectinfo'] and
                cpd['objectinfo']['jmag'] is not None)
               else npnan),
              (cpd['objectinfo']['hmag'] if
               ('hmag' in cpd['objectinfo'] and
                cpd['objectinfo']['hmag'] is not None)
               else npnan),
              (cpd['objectinfo']['kmag'] if
               ('kmag' in cpd['objectinfo'] and
                cpd['objectinfo']['kmag'] is not None)
               else npnan))),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
        objinfodraw.text(
            (900, 250),
            ('BV: %.3f, %.3f' %
             ((cpd['objectinfo']['bmag'] if
               ('bmag' in cpd['objectinfo'] and
                cpd['objectinfo']['bmag'] is not None)
               else npnan),
              (cpd['objectinfo']['vmag'] if
               ('vmag' in cpd['objectinfo'] and
                cpd['objectinfo']['vmag'] is not None)
               else npnan))),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        # colors
        if ('dereddened' in cpd['objectinfo'] and
            cpd['objectinfo']['dereddened'] is True):
            deredlabel = "(dereddened)"
        else:
            deredlabel = ""

        objinfodraw.text(
            (625, 275),
            'Colors %s' % deredlabel,
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        objinfodraw.text(
            (900, 275),
            ('B - V: %.3f, V - K: %.3f' %
             ( (cpd['objectinfo']['bvcolor'] if
                ('bvcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['bvcolor'] is not None)
                else npnan),
               (cpd['objectinfo']['vkcolor'] if
                ('vkcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['vkcolor'] is not None)
                else npnan) )),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
        objinfodraw.text(
            (900, 300),
            ('i - J: %.3f, g - K: %.3f' %
             ( (cpd['objectinfo']['ijcolor'] if
                ('ijcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['ijcolor'] is not None)
                else npnan),
               (cpd['objectinfo']['gkcolor'] if
                ('gkcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['gkcolor'] is not None)
                else npnan) )),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
        objinfodraw.text(
            (900, 325),
            ('J - K: %.3f' %
             ( (cpd['objectinfo']['jkcolor'] if
                ('jkcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['jkcolor'] is not None)
                else npnan),) ),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    #
    # rest of the object information
    #

    # color classification
    if ('color_classes' in cpd['objectinfo'] and
        cpd['objectinfo']['color_classes']):

        objinfodraw.text(
            (625, 375),
            ('star classification by color: %s' %
             (', '.join(cpd['objectinfo']['color_classes']))),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # GAIA neighbors
    if ( ('gaia_neighbors' in cpd['objectinfo']) and
         (cpd['objectinfo']['gaia_neighbors'] is not None) and
         (np.isfinite(cpd['objectinfo']['gaia_neighbors'])) and
         ('searchradarcsec' in cpd['objectinfo']) and
         (cpd['objectinfo']['searchradarcsec']) ):

        objinfodraw.text(
            (625, 400),
            ('%s GAIA close neighbors within %.1f arcsec' %
             (cpd['objectinfo']['gaia_neighbors'],
              cpd['objectinfo']['searchradarcsec'])),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # closest GAIA neighbor
    if ( ('gaia_closest_distarcsec' in cpd['objectinfo']) and
         (cpd['objectinfo']['gaia_closest_distarcsec'] is not None) and
         (np.isfinite(cpd['objectinfo']['gaia_closest_distarcsec'])) and
         ('gaia_closest_gmagdiff' in cpd['objectinfo']) and
         (cpd['objectinfo']['gaia_closest_gmagdiff'] is not None) and
         (np.isfinite(cpd['objectinfo']['gaia_closest_gmagdiff'])) ):

        objinfodraw.text(
            (625, 425),
            ('closest GAIA neighbor is %.1f arcsec away, '
             'GAIA mag (obj-nbr): %.3f' %
             (cpd['objectinfo']['gaia_closest_distarcsec'],
              cpd['objectinfo']['gaia_closest_gmagdiff'])),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # object tags
    if 'objecttags' in cpd['objectinfo'] and cpd['objectinfo']['objecttags']:

        objtagsplit = cpd['objectinfo']['objecttags'].split(',')

        # write three tags per line
        nobjtaglines = int(np.ceil(len(objtagsplit)/3.0))

        for objtagline in range(nobjtaglines):
            objtagslice = ','.join(objtagsplit[objtagline*3:objtagline*3+3])
            objinfodraw.text(
                (625, 450+objtagline*25),
                objtagslice,
                font=cpfontnormal,
                fill=(135, 54, 0, 255)
            )



    ################################################
    # row 1, cell 3: variability info and comments #
    ################################################

    # objectisvar
    objisvar = cpd['varinfo']['objectisvar']

    if objisvar == '0':
        objvarflag = 'Variable star flag not set'
    elif objisvar == '1':
        objvarflag = 'Object is probably a variable star'
    elif objisvar == '2':
        objvarflag = 'Object is probably not a variable star'
    elif objisvar == '3':
        objvarflag = 'Not sure if this object is a variable star'
    elif objisvar is None:
        objvarflag = 'Variable star flag not set'
    elif objisvar is True:
        objvarflag = 'Object is probably a variable star'
    elif objisvar is False:
        objvarflag = 'Object is probably not a variable star'
    else:
        objvarflag = 'Variable star flag: %s' % objisvar

    objinfodraw.text(
        (1650, 125),
        objvarflag,
        font=cpfontnormal,
        fill=(0,0,0,255)
    )

    # period
    objinfodraw.text(
        (1650, 150),
        ('Period [days]: %.6f' %
         (cpd['varinfo']['varperiod']
          if cpd['varinfo']['varperiod'] is not None
          else np.nan)),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )

    # epoch
    objinfodraw.text(
        (1650, 175),
        ('Epoch [JD]: %.6f' %
         (cpd['varinfo']['varepoch']
          if cpd['varinfo']['varepoch'] is not None
          else np.nan)),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )

    # variability tags
    if cpd['varinfo']['vartags']:

        vartagsplit = cpd['varinfo']['vartags'].split(',')

        # write three tags per line
        nvartaglines = int(np.ceil(len(vartagsplit)/3.0))

        for vartagline in range(nvartaglines):
            vartagslice = ','.join(vartagsplit[vartagline*3:vartagline*3+3])
            objinfodraw.text(
                (1650, 225+vartagline*25),
                vartagslice,
                font=cpfontnormal,
                fill=(135, 54, 0, 255)
            )

    # object comments
    if 'comments' in cpd and cpd['comments']:

        commentsplit = cpd['comments'].split(' ')

        # write 10 words per line
        ncommentlines = int(np.ceil(len(commentsplit)/10.0))

        for commentline in range(ncommentlines):
            commentslice = ' '.join(
                commentsplit[commentline*10:commentline*10+10]
            )
            objinfodraw.text(
                (1650, 325+commentline*25),
                commentslice,
                font=cpfontnormal,
                fill=(0,0,0,255)
            )

    # this handles JSON-ified checkplots returned by LCC server
    elif 'objectcomments' in cpd and cpd['objectcomments']:

        commentsplit = cpd['objectcomments'].split(' ')

        # write 10 words per line
        ncommentlines = int(np.ceil(len(commentsplit)/10.0))

        for commentline in range(ncommentlines):
            commentslice = ' '.join(
                commentsplit[commentline*10:commentline*10+10]
            )
            objinfodraw.text(
                (1650, 325+commentline*25),
                commentslice,
                font=cpfontnormal,
                fill=(0,0,0,255)
            )

    #######################################
    # row 1, cell 4: unphased light curve #
    #######################################

    if (cpd['magseries'] and
        'plot' in cpd['magseries'] and
        cpd['magseries']['plot']):
        magseries = Image.open(
            _base64_to_file(cpd['magseries']['plot'], None, writetostrio=True)
        )
        outimg.paste(magseries,(750*3,0))

    # this handles JSON-ified checkplots from LCC server
    elif ('magseries' in cpd and isinstance(cpd['magseries'],str)):

        magseries = Image.open(
            _base64_to_file(cpd['magseries'], None, writetostrio=True)
        )
        outimg.paste(magseries,(750*3,0))


    ###############################
    # the rest of the rows in cpd #
    ###############################
    for lspmethodind, lspmethod in enumerate(cplspmethods):

        ###############################
        # the periodogram comes first #
        ###############################

        if (cpd[lspmethod] and cpd[lspmethod]['periodogram']):

            pgram = Image.open(
                _base64_to_file(cpd[lspmethod]['periodogram'], None,
                                writetostrio=True)
            )
            outimg.paste(pgram,(0,480 + 480*lspmethodind))

        #############################
        # best phased LC comes next #
        #############################

        if (cpd[lspmethod] and 0 in cpd[lspmethod] and cpd[lspmethod][0]):

            plc1 = Image.open(
                _base64_to_file(cpd[lspmethod][0]['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc1,(750,480 + 480*lspmethodind))

        # this handles JSON-ified checkplots from LCC server
        elif (cpd[lspmethod] and 'phasedlc0' in cpd[lspmethod] and
              isinstance(cpd[lspmethod]['phasedlc0']['plot'], str)):

            plc1 = Image.open(
                _base64_to_file(cpd[lspmethod]['phasedlc0']['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc1,(750,480 + 480*lspmethodind))

        #################################
        # 2nd best phased LC comes next #
        #################################

        if (cpd[lspmethod] and 1 in cpd[lspmethod] and cpd[lspmethod][1]):

            plc2 = Image.open(
                _base64_to_file(cpd[lspmethod][1]['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc2,(750*2,480 + 480*lspmethodind))

        # this handles JSON-ified checkplots from LCC server
        elif (cpd[lspmethod] and 'phasedlc1' in cpd[lspmethod] and
              isinstance(cpd[lspmethod]['phasedlc1']['plot'], str)):

            plc2 = Image.open(
                _base64_to_file(cpd[lspmethod]['phasedlc1']['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc2,(750*2,480 + 480*lspmethodind))

        #################################
        # 3rd best phased LC comes next #
        #################################

        if (cpd[lspmethod] and 2 in cpd[lspmethod] and cpd[lspmethod][2]):

            plc3 = Image.open(
                _base64_to_file(cpd[lspmethod][2]['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc3,(750*3,480 + 480*lspmethodind))

        # this handles JSON-ified checkplots from LCC server
        elif (cpd[lspmethod] and 'phasedlc2' in cpd[lspmethod] and
              isinstance(cpd[lspmethod]['phasedlc2']['plot'], str)):

            plc3 = Image.open(
                _base64_to_file(cpd[lspmethod]['phasedlc2']['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc3,(750*3,480 + 480*lspmethodind))


    ################################
    ## ALL DONE WITH BUILDING PNG ##
    ################################

    #########################
    # add in any extra rows #
    #########################

    # from the keyword arguments
    if erows > 0:

        for erowind, erow in enumerate(extrarows):

            # make sure we never go above 4 plots in a row
            for ecolind, ecol in enumerate(erow[:4]):

                eplot = Image.open(ecol)
                eplotresized = eplot.resize((750,480), Image.ANTIALIAS)
                outimg.paste(eplotresized,
                             (750*ecolind,
                              (cprows+1)*480 + 480*erowind))

    # from the checkplotdict
    if cpderows > 0:

        for cpderowind, cpderow in enumerate(cpd['externalplots']):

            # make sure we never go above 4 plots in a row
            for cpdecolind, cpdecol in enumerate(cpderow[:4]):

                cpdeplot = Image.open(cpdecol)
                cpdeplotresized = cpdeplot.resize((750,480), Image.ANTIALIAS)
                outimg.paste(cpdeplotresized,
                             (750*cpdecolind,
                              (cprows+1)*480 + (erows*480) + 480*cpderowind))


    # from neighbors:
    if nbrrows > 0:

        # we have four tiles
        # tile 1: neighbor objectid, ra, decl, distance, unphased LC
        # tile 2: phased LC for gls
        # tile 3: phased LC for pdm
        # tile 4: phased LC for any other period finding method
        #         the priority is like so: ['bls','mav','aov','win']

        for nbrind, nbr in enumerate(cpd['neighbors']):

            # figure out which period finding methods are available for this
            # neighbor. make sure to match the ones from the actual object in
            # order of priority: 'gls','pdm','bls','aov','mav','acf','win'
            nbrlspmethods = []

            for lspmethod in cpd['pfmethods']:
                if lspmethod in nbr:
                    nbrlspmethods.append(lspmethod)

            # restrict to top three in priority
            nbrlspmethods = nbrlspmethods[:3]

            try:

                # first panel: neighbor objectid, ra, decl, distance, unphased
                # LC
                nbrlc = Image.open(
                    _base64_to_file(
                        nbr['magseries']['plot'], None, writetostrio=True
                    )
                )
                outimg.paste(nbrlc,
                             (750*0,
                              (cprows+1)*480 + (erows*480) + (cpderows*480) +
                              480*nbrind))

                # overlay the objectinfo
                objinfodraw.text(
                    (98,
                     (cprows+1)*480 + (erows*480) + (cpderows*480) +
                     480*nbrind + 15),
                    ('N%s: %s' % (nbrind + 1, nbr['objectid'])),
                    font=cpfontlarge,
                    fill=(0,0,255,255)
                )
                # overlay the objectinfo
                objinfodraw.text(
                    (98,
                     (cprows+1)*480 + (erows*480) + (cpderows*480) +
                     480*nbrind + 50),
                    ('(RA, DEC) = (%.3f, %.3f), distance: %.1f arcsec' %
                     (nbr['ra'], nbr['decl'], nbr['dist'])),
                    font=cpfontnormal,
                    fill=(0,0,255,255)
                )

                # second panel: phased LC for gls
                lsp1lc = Image.open(
                    _base64_to_file(
                        nbr[nbrlspmethods[0]][0]['plot'], None,
                        writetostrio=True
                    )
                )
                outimg.paste(lsp1lc,
                             (750*1,
                              (cprows+1)*480 + (erows*480) + (cpderows*480) +
                              480*nbrind))

                # second panel: phased LC for gls
                lsp2lc = Image.open(
                    _base64_to_file(
                        nbr[nbrlspmethods[1]][0]['plot'], None,
                        writetostrio=True
                    )
                )
                outimg.paste(lsp2lc,
                             (750*2,
                              (cprows+1)*480 + (erows*480) + (cpderows*480) +
                              480*nbrind))

                # second panel: phased LC for gls
                lsp3lc = Image.open(
                    _base64_to_file(
                        nbr[nbrlspmethods[2]][0]['plot'], None,
                        writetostrio=True
                    )
                )
                outimg.paste(lsp3lc,
                             (750*3,
                              (cprows+1)*480 + (erows*480) + (cpderows*480) +
                              480*nbrind))

            except Exception as e:

                LOGERROR('neighbor %s does not have a magseries plot, '
                         'measurements are probably all nan' % nbr['objectid'])

                # overlay the objectinfo
                objinfodraw.text(
                    (98,
                     (cprows+1)*480 + (erows*480) + (cpderows*480) +
                     480*nbrind + 15),
                    ('N%s: %s' %
                     (nbrind + 1, nbr['objectid'])),
                    font=cpfontlarge,
                    fill=(0,0,255,255)
                )

                if 'ra' in nbr and 'decl' in nbr and 'dist' in nbr:

                    # overlay the objectinfo
                    objinfodraw.text(
                        (98,
                         (cprows+1)*480 + (erows*480) + (cpderows*480) +
                         480*nbrind + 50),
                        ('(RA, DEC) = (%.3f, %.3f), distance: %.1f arcsec' %
                         (nbr['ra'], nbr['decl'], nbr['dist'])),
                        font=cpfontnormal,
                        fill=(0,0,255,255)
                    )

                elif 'objectinfo' in nbr:

                    # overlay the objectinfo
                    objinfodraw.text(
                        (98,
                         (cprows+1)*480 + (erows*480) + (cpderows*480) +
                         480*nbrind + 50),
                        ('(RA, DEC) = (%.3f, %.3f), distance: %.1f arcsec' %
                         (nbr['objectinfo']['ra'],
                          nbr['objectinfo']['decl'],
                          nbr['objectinfo']['distarcsec'])),
                        font=cpfontnormal,
                        fill=(0,0,255,255)
                    )


    #####################
    ## WRITE FINAL PNG ##
    #####################

    # check if the output filename is actually an instance of StringIO
    if sys.version_info[:2] < (3,0):

        is_strio = isinstance(outfile, cStringIO.InputType)

    else:

        is_strio = isinstance(outfile, strio)


    if not is_strio:

        # check if we've stupidly copied over the same filename as the input
        # pickle to expected output file
        if outfile.endswith('pkl'):
            LOGWARNING('expected output PNG filename ends with .pkl, '
                       'changed to .png')
            outfile = outfile.replace('.pkl','.png')

    outimg.save(outfile, format='PNG', optimize=True)


    if not is_strio:
        if os.path.exists(outfile):
            LOGINFO('checkplot pickle -> checkplot PNG: %s OK' % outfile)
            return outfile
        else:
            LOGERROR('failed to write checkplot PNG')
            return None

    else:
        LOGINFO('checkplot pickle -> StringIO instance OK')
        return outfile



def cp2png(checkplotin, extrarows=None):
    '''
    This is just a shortened form of the function above for convenience.

    This only handles pickle files.

    '''

    if checkplotin.endswith('.gz'):
        outfile = checkplotin.replace('.pkl.gz','.png')
    else:
        outfile = checkplotin.replace('.pkl','.png')

    return checkplot_pickle_to_png(checkplotin, outfile, extrarows=extrarows)



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
                                gaia_mirror='cds',
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

    varinfo = cpd['varinfo'].copy()

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

    cpd.update(newcpd)
    cpd['objectinfo']['objecttags'] = objecttags
    cpd['comments'] = comments

    newcpf = _write_checkplot_picklefile(cpd, outfile=cpf)

    return newcpf



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
