#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''checkplot.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017
License: MIT.

Contains functions to make checkplots: quick views for determining periodic
variability for light curves and sanity-checking results from period-finding
functions (e.g., from periodbase).

The checkplot_png function makes the following 3 x 3 grid and writes to a PNG:

    [LSP plot + objectinfo] [     unphased LC     ] [ period 1 phased LC ]
    [period 1 phased LC /2] [period 1 phased LC x2] [ period 2 phased LC ]
    [ period 3 phased LC  ] [period 4 phased LC   ] [ period 5 phased LC ]

The twolsp_checkplot_png function makes a similar plot for two independent
period-finding routines and writes to a PNG:

    [ pgram1 + objectinfo ] [        pgram2       ] [     unphased LC     ]
    [ pgram1 P1 phased LC ] [ pgram1 P2 phased LC ] [ pgram1 P3 phased LC ]
    [ pgram2 P1 phased LC ] [ pgram2 P2 phased LC ] [ pgram2 P3 phased LC ]

    where:

    pgram1 is the plot for the periodogram in the lspinfo1 dict
    pgram1 P1, P2, and P3 are the best three periods from lspinfo1
    pgram2 is the plot for the periodogram in the lspinfo2 dict
    pgram2 P1, P2, and P3 are the best three periods from lspinfo2

The checkplot_pickle function takes, for a single object, an arbitrary number of
results from independent period-finding functions (e.g. BLS, PDM, AoV, GLS) in
periodbase, and generates a gzipped pickle file that contains object and
variability information, finder chart, mag series plot, and for each
period-finding result: a periodogram and phased mag series plots for up to
arbitrary number of 'best periods'. This is intended for use with an external
checkplot viewer: the Tornado webapp checkplotserver.py, but you can also use
the checkplot_pickle_to_png function to render this to a PNG similar to those
above. In this case, the PNG will look something like:

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

import os
import os.path
import gzip
import base64
import sys
import hashlib
import sys

try:
    import cPickle as pickle
    from cStringIO import StringIO as strio
except:
    import pickle
    from io import BytesIO as strio

import numpy as np
from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs, \
    ravel as npravel

# we're going to plot using Agg only
import matplotlib
MPLVERSION = tuple([int(x) for x in matplotlib.__version__.split('.')])
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import logging
from datetime import datetime as dtime
from traceback import format_exc

# import this to check if stimes, smags, serrs are Column objects
from astropy.table import Column as astcolumn

# import this to get neighbors and their x,y coords from the Skyview FITS
from astropy.wcs import WCS

# import from Pillow to generate pngs from checkplot dicts
from PIL import Image, ImageDraw, ImageFont



#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.checkplot' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (dtime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (dtime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (dtime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (dtime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                dtime.utcnow().isoformat(),
                message, format_exc()
                )
            )


###################
## LOCAL IMPORTS ##
###################

from .lcmath import phase_magseries, phase_bin_magseries, \
    normalize_magseries, sigclip_magseries
from .varbase.lcfit import spline_fit_magseries
from .varbase.features import all_nonperiodic_features
from .coordutils import total_proper_motion, reduced_proper_motion
from .plotbase import skyview_stamp, \
    PLOTYLABELS, METHODLABELS, METHODSHORTLABELS


############
## CONFIG ##
############



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
        ('objectid' in objectinfo or 'hatid' in objectinfo)
        and 'ra' in objectinfo and 'decl' in objectinfo and
        objectinfo['ra'] and objectinfo['decl']):

        if 'objectid' not in objectinfo:
            objectid = objectinfo['hatid']
        else:
            objectid = objectinfo['objectid']

        if verbose:
            LOGINFO('adding in object information and '
                    'finder chart for %s at RA: %.3f, DEC: %.3f' %
                    (objectid, objectinfo['ra'], objectinfo['decl']))

        # FIXME: get mag info from astroquery or HATDS if needed


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
            dss = skyview_stamp(objectinfo['ra'],
                                objectinfo['decl'],
                                convolvewith=finderconvolve,
                                cachedir=findercachedir,
                                verbose=verbose)
            stamp = dss

            # inset plot it on the current axes
            inset = inset_axes(axes, width="40%", height="40%", loc=1)
            inset.imshow(stamp,cmap=findercmap)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_frame_on(False)

            # grid lines pointing to the center of the frame
            inset.axvline(x=150,ymin=0.2,ymax=0.4,linewidth=2.0,color='k')
            inset.axhline(y=150,xmin=0.2,xmax=0.4,linewidth=2.0,color='k')

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

            axes.text(0.05,0.67,'$\mu$ = %.2f mas yr$^{-1}$' % pm,
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
    plot_xlim = axes.get_xlim()
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
                                stimes, smags,
                                varperiod, varepoch,
                                phasewrap, phasesort,
                                phasebin, minbinelems,
                                plotxlim,
                                lspmethod,
                                xliminsetmode=False,
                                twolspmode=False,
                                magsarefluxes=False):
    '''makes the phased magseries plot tile.

    if xliminsetmode = True, then makes a zoomed-in plot with the provided
    plotxlim as the main x limits, and the full plot as an inset.

    '''

    # phase the magseries
    phasedlc = phase_magseries(stimes,
                               smags,
                               varperiod,
                               varepoch,
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
        plot_xlim = axes.get_xlim()
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
            varepoch
        )
    elif periodind == 1 and not twolspmode:
        plottitle = '%s best period x 0.5: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            varperiod,
            varepoch
        )
    elif periodind == 2 and not twolspmode:
        plottitle = '%s best period x 2: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            varperiod,
            varepoch
        )
    elif periodind > 2 and not twolspmode:
        plottitle = '%s peak %s: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            periodind-1,
            varperiod,
            varepoch
        )
    elif periodind > 0:
        plottitle = '%s peak %s: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
            periodind+1,
            varperiod,
            varepoch
        )

    axes.set_title(plottitle)

    # if we're making an inset plot showing the full range
    if (plotxlim and isinstance(plotxlim, list) and
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
                  plotxlim=[-0.8,0.8],
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

        periods = lspinfo['periods']
        lspvals = lspinfo['lspvals']
        bestperiod = lspinfo['bestperiod']
        nbestperiods = lspinfo['nbestperiods']
        nbestlspvals = lspinfo['nbestlspvals']
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

            # figure out the epoch, if it's None, use the min of the time
            if varepoch is None:
                varepoch = npmin(stimes)

            # if the varepoch is 'min', then fit a spline to the light curve
            # phased using the min of the time, find the fit mag minimum and use
            # the time for that as the varepoch
            elif isinstance(varepoch,str) and varepoch == 'min':

                try:
                    spfit = spline_fit_magseries(stimes,
                                                 smags,
                                                 serrs,
                                                 varperiod,
                                                 sigclip=None,
                                                 magsarefluxes=magsarefluxes,
                                                 verbose=verbose)
                    varepoch = spfit['fitinfo']['fitepoch']
                    if len(varepoch) != 1:
                        varepoch = varepoch[0]
                except Exception as e:
                    LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                    varepoch = npmin(stimes)

            if verbose:
                LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                        (varperiod, varepoch))

            # make sure the best period phased LC plot stands out
            if periodind == 0 and bestperiodhighlight:
                if MPLVERSION >= (2,0,0):
                    axes[periodind+2].set_facecolor(bestperiodhighlight)
                else:
                    axes[periodind+2].set_axis_bgcolor(bestperiodhighlight)

            _make_phased_magseries_plot(axes[periodind+2],
                                        periodind,
                                        stimes, smags,
                                        varperiod, varepoch,
                                        phasewrap, phasesort,
                                        phasebin, minbinelems,
                                        plotxlim, lspmethod,
                                        xliminsetmode=xliminsetmode,
                                        magsarefluxes=magsarefluxes)

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
                         plotxlim=[-0.8,0.8],
                         xliminsetmode=False,
                         plotdpi=100,
                         bestperiodhighlight=None,
                         verbose=True):
    '''This makes a checkplot using results from two independent period-finders.

    Adapted from Luke Bouma's implementation of the same. This makes a special
    checkplot that uses two lspinfo dictionaries, from two independent
    period-finding methods. For EBs, it's probably best to use Stellingwerf PDM
    or Schwarzenberg-Czerny AoV as one of these, and the Box Least-squared Search
    method as the other one.

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

    '''

    # generate the plot filename
    if not outfile and isinstance(lspinfo1,str):
        plotfpath = os.path.join(
            os.path.dirname(lspinfo),
            'twolsp-checkplot-%s.png' % (
                os.path.basename(lspinfo),
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

        periods1 = lspinfo1['periods']
        lspvals1 = lspinfo1['lspvals']
        bestperiod1 = lspinfo1['bestperiod']
        nbestperiods1 = lspinfo1['nbestperiods']
        nbestlspvals1 = lspinfo1['nbestlspvals']
        lspmethod1 = lspinfo1['method']

        periods2 = lspinfo2['periods']
        lspvals2 = lspinfo2['lspvals']
        bestperiod2 = lspinfo2['bestperiod']
        nbestperiods2 = lspinfo2['nbestperiods']
        nbestlspvals2 = lspinfo2['nbestlspvals']
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

            # figure out the epoch, if it's None, use the min of the time
            if varepoch is None:
                varepoch = npmin(stimes)

            # if the varepoch is 'min', then fit a spline to the light curve
            # phased using the min of the time, find the fit mag minimum and use
            # the time for that as the varepoch
            elif isinstance(varepoch,str) and varepoch == 'min':

                try:
                    spfit = spline_fit_magseries(stimes,
                                                 smags,
                                                 serrs,
                                                 varperiod,
                                                 sigclip=None,
                                                 magsarefluxes=magsarefluxes,
                                                 verbose=verbose)
                    varepoch = spfit['fitinfo']['fitepoch']
                    if len(varepoch) != 1:
                        varepoch = varepoch[0]
                except Exception as e:
                    LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                    varepoch = npmin(stimes)

            if verbose:
                LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                        (varperiod, varepoch))

            # make sure the best period phased LC plot stands out
            if periodind == 0 and bestperiodhighlight:
                if MPLVERSION >= (2,0,0):
                    plotaxes.set_facecolor(bestperiodhighlight)
                else:
                    plotaxes.set_axis_bgcolor(bestperiodhighlight)

            _make_phased_magseries_plot(plotaxes,
                                        periodind,
                                        stimes, smags,
                                        varperiod, varepoch,
                                        phasewrap, phasesort,
                                        phasebin, minbinelems,
                                        plotxlim, lspmethod1,
                                        twolspmode=True,
                                        magsarefluxes=magsarefluxes,
                                        xliminsetmode=xliminsetmode)

        ##########################################################
        ### NOW PLOT PHASED LCS FOR 3 BEST PERIODS IN LSPINFO2 ###
        ##########################################################
        for periodind, varperiod, plotaxes in zip([0,1,2],
                                                  lspbestperiods2[:3],
                                                  [axes[6], axes[7], axes[8]]):

            # figure out the epoch, if it's None, use the min of the time
            if varepoch is None:
                varepoch = npmin(stimes)

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
                    varepoch = spfit['fitinfo']['fitepoch']
                    if len(varepoch) != 1:
                        varepoch = varepoch[0]
                except Exception as e:
                    LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                    varepoch = npmin(stimes)

            if verbose:
                LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                        (varperiod, varepoch))

            # make sure the best period phased LC plot stands out
            if periodind == 0 and bestperiodhighlight:
                if MPLVERSION >= (2,0,0):
                    plotaxes.set_facecolor(bestperiodhighlight)
                else:
                    plotaxes.set_axis_bgcolor(bestperiodhighlight)

            _make_phased_magseries_plot(plotaxes,
                                        periodind,
                                        stimes, smags,
                                        varperiod, varepoch,
                                        phasewrap, phasesort,
                                        phasebin, minbinelems,
                                        plotxlim, lspmethod2,
                                        twolspmode=True,
                                        magsarefluxes=magsarefluxes,
                                        xliminsetmode=xliminsetmode)

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
                           lclistpkl=None,
                           nbrradiusarcsec=30.0,
                           plotdpi=100,
                           findercachedir='~/.astrobase/stamp-cache',
                           verbose=True):
    '''This returns the finder chart and object information as a dict.

    '''

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

            # generate the finder chart
            finder, finderheader = skyview_stamp(objectinfo['ra'],
                                                 objectinfo['decl'],
                                                 convolvewith=finderconvolve,
                                                 verbose=verbose,
                                                 cachedir=findercachedir)
            finderfig = plt.figure(figsize=(3,3),dpi=plotdpi,frameon=False)

            plt.imshow(finder, cmap=findercmap)

            # skip down to after nbr stuff for the rest of the finderchart...

            # search around the target's location and get its neighbors if
            # lclistpkl is provided and it exists
            if (lclistpkl is not None and
                os.path.exists(lclistpkl) and
                nbrradiusarcsec is not None and
                nbrradiusarcsec > 0.0):

                if lclistpkl.endswith('.gz'):
                    infd = gzip.open(lclistpkl,'rb')
                else:
                    infd = open(lclistpkl,'rb')

                lclist = pickle.load(infd)
                infd.close()

                if not 'kdtree' in lclist:

                    LOGERROR('neighbors within %.1f arcsec for %s could '
                             'not be found, no kdtree in lclistpkl: %s'
                             % (objectid, lclistpkl))
                    neighbors = None

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
                        k=6, # get closest 5 neighbors + tgt
                        distance_upper_bound=match_xyzdist
                    )

                    # sort by matchdist
                    mdsorted = np.argsort(matchdists[0])
                    matchdists = matchdists[0][mdsorted]
                    matchinds = matchinds[0][mdsorted]

                    # luckily, the indices to the kdtree are the same as that
                    # for the objects (I think)
                    neighbors = []

                    # initialize the finder WCS
                    finderwcs = WCS(finderheader)

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
                            annotatey = 300.0 - pixcoords[0,1]

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

                            plt.annotate('N%s' % nbrind,
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

            #
            # finish up the finder chart after neighbors are processed
            #
            plt.xticks([])
            plt.yticks([])
            # grid lines pointing to the center of the frame
            plt.axvline(x=150,ymin=0.2,ymax=0.4,linewidth=2.0,color='b')
            plt.axhline(y=149,xmin=0.2,xmax=0.4,linewidth=2.0,color='b')
            plt.gca().set_frame_on(False)

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

        # now that we have the finder chart, get the rest of the object
        # information

        # FIXME: get this stuff from astroquery or HATDS if it's missing and we
        # have the ra/decl

        if ('bmag' in objectinfo and objectinfo['bmag'] is not None and
            'vmag' in objectinfo and objectinfo['vmag'] is not None):
            objectinfo['bvcolor'] = objectinfo['bmag'] - objectinfo['vmag']
        else:
            objectinfo['bvcolor'] = None

        if ('sdssi' in objectinfo and objectinfo['sdssi'] is not None and
            'jmag' in objectinfo and objectinfo['jmag'] is not None):
            objectinfo['ijcolor'] = objectinfo['sdssi'] - objectinfo['jmag']
        else:
            objectinfo['ijcolor'] = None

        if ('jmag' in objectinfo and objectinfo['jmag'] is not None and
            'kmag' in objectinfo and objectinfo['kmag'] is not None):
            objectinfo['jkcolor'] = objectinfo['jmag'] - objectinfo['kmag']
        else:
            objectinfo['jkcolor'] = None

        # add in proper motion stuff if available in objectinfo
        if ('pmra' in objectinfo and objectinfo['pmra'] and
            'pmdecl' in objectinfo and objectinfo['pmdecl']):

            objectinfo['propermotion'] = total_proper_motion(
                objectinfo['pmra'],
                objectinfo['pmdecl'],
                objectinfo['decl']
            )
        else:
            objectinfo['propermotion'] = None

        if ('jmag' in objectinfo and objectinfo['jmag'] and
            objectinfo['propermotion']):

            objectinfo['reducedpropermotion'] = reduced_proper_motion(
                objectinfo['jmag'],
                objectinfo['propermotion']
            )
        else:
            objectinfo['reducedpropermotion'] = None

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

        # put together the initial checkplot pickle dictionary
        # this will be updated by the functions below as appropriate
        # and will written out as a gzipped pickle at the end of processing
        checkplotdict = {'objectid':None,
                         'neighbors':None,
                         'objectinfo':{'bmag':None,
                                       'bvcolor':None,
                                       'decl':None,
                                       'hatid':None,
                                       'hmag':None,
                                       'ijcolor':None,
                                       'jkcolor':None,
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
                                       'reducedpropermotion':None,
                                       'sdssg':None,
                                       'sdssi':None,
                                       'sdssr':None,
                                       'stations':None,
                                       'twomassid':None,
                                       'ucac4id':None,
                                       'vmag':None},
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



def _pkl_periodogram(lspinfo, plotdpi=100):
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
    plot_xlim = plt.xlim()
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



def _pkl_phased_magseries_plot(checkplotdict, lspmethod, periodind,
                               stimes, smags, serrs,
                               varperiod, varepoch,
                               phasewrap=True,
                               phasesort=True,
                               phasebin=0.002,
                               minbinelems=7,
                               plotxlim=[-0.8,0.8],
                               plotdpi=100,
                               bestperiodhighlight=None,
                               xgridlines=None,
                               xliminsetmode=False,
                               magsarefluxes=False,
                               directreturn=False,
                               overplotfit=None,
                               verbose=True):
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

    # figure out the epoch, if it's None, use the min of the time
    if varepoch is None:
        varepoch = npmin(stimes)

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
            varepoch = spfit['fitinfo']['fitepoch']
            if len(varepoch) != 1:
                varepoch = varepoch[0]
        except Exception as e:
            LOGEXCEPTION('spline fit failed, using min(times) as epoch')
            varepoch = npmin(stimes)

    if verbose:
        LOGINFO('plotting %s phased LC with period %s: %.6f, epoch: %.5f' %
                (lspmethod, periodind, varperiod, varepoch))

    # make the plot title based on the lspmethod
    if periodind == 0:
        plottitle = '%s best period: %.6f d - epoch: %.5f' % (
            (METHODSHORTLABELS[lspmethod] if lspmethod in METHODSHORTLABELS
             else lspmethod),
            varperiod,
            varepoch
        )
    elif periodind > 0:
        plottitle = '%s peak %s: %.6f d - epoch: %.5f' % (
            (METHODSHORTLABELS[lspmethod] if lspmethod in METHODSHORTLABELS
             else lspmethod),
            periodind+1,
            varperiod,
            varepoch
        )
    elif periodind == -1:
        plottitle = '%s period: %.6f d - epoch: %.5f' % (
            lspmethod,
            varperiod,
            varepoch
        )


    # phase the magseries
    phasedlc = phase_magseries(stimes,
                               smags,
                               varperiod,
                               varepoch,
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
        fitchisq = overplotfit['fitchisq']
        fitredchisq = overplotfit['fitredchisq']

        plotfitmags = overplotfit['fitinfo']['fitmags']
        plotfittimes = overplotfit['magseries']['times']

        # phase the fit magseries
        fitphasedlc = phase_magseries(plotfittimes,
                                      plotfitmags,
                                      varperiod,
                                      varepoch,
                                      wrap=phasewrap,
                                      sort=phasesort)
        plotfitphase = fitphasedlc['phase']
        plotfitmags = fitphasedlc['mags']

        plotfitlabel = ('%s fit ${\chi}^2/{\mathrm{dof}} = %.3f$' %
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
        plot_xlim = plt.xlim()
        plt.xlim((npmin(plotphase)-0.1,
                       npmax(plotphase)+0.1))
    else:
        plt.xlim((plotxlim[0],plotxlim[1]))

    # make a grid
    ax = plt.gca()
    if isinstance(xgridlines,list):
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
    if (plotxlim and isinstance(plotxlim, list) and
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
        'epoch':varepoch,
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

        checkplotdict[lspmethod][periodind] = retdict
        return checkplotdict



def _write_checkplot_picklefile(checkplotdict,
                                outfile=None,
                                protocol=2,
                                outgzip=False):
    '''This writes the checkplotdict to a (gzipped) pickle file.

    If outfile is None, writes a (gzipped) pickle file of the form:

    checkplot-{objectid}.pkl(.gz)

    to the current directory.

    protocol sets the pickle protocol:

    3 -> default in Python 3 - way faster but incompatible with Python 2
    2 -> default in Python 2 - very slow, but compatible with Python 2 and 3

    the default protocol is 2 so that pickle files generated by newer Pythons
    can still be read by older ones. if this isn't a concern, set protocol to 3.

    '''

    if outgzip:

        if not outfile:

            outfile = (
                'checkplot-{objectid}.pkl.gz'.format(
                    objectid=checkplotdict['objectid']
                )
            )

        with gzip.open(outfile,'wb') as outfd:
            pickle.dump(checkplotdict,outfd,protocol=protocol)

    else:

        if not outfile:

            outfile = (
                'checkplot-{objectid}.pkl'.format(
                    objectid=checkplotdict['objectid']
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

    But not sure how robust this is. We should probably move to another format
    for these checkplots.

    '''

    if checkplotpickle.endswith('.gz'):

        try:
            with gzip.open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd)

        except UnicodeDecodeError:

            with gzip.open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd, encoding='latin1')

            LOGWARNING('pickle %s was probably from Python 2 '
                       'and failed to load without using "latin1" encoding. '
                       'This is probably a numpy issue: '
                       'http://stackoverflow.com/q/11305790' % checkplotpickle)

    else:

        try:
            with open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd)

        except UnicodeDecodeError:

            with open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd, encoding='latin1')

            LOGWARNING('pickle %s was probably from Python 2 '
                       'and failed to load without using "latin1" encoding. '
                       'This is probably a numpy issue: '
                       'http://stackoverflow.com/q/11305790' % checkplotpickle)

    return cpdict



#############################
## CHECKPLOT DICT FUNCTION ##
#############################

def checkplot_dict(lspinfolist,
                   times,
                   mags,
                   errs,
                   magsarefluxes=False,
                   nperiodstouse=3,
                   objectinfo=None,
                   varinfo=None,
                   getvarfeatures=True,
                   lclistpkl=None,
                   nbrradiusarcsec=30.0,
                   lcfitfunc=None,
                   lcfitparams={},
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
                   plotxlim=[-0.8,0.8],
                   xliminsetmode=False,
                   plotdpi=100,
                   bestperiodhighlight=None,
                   xgridlines=None,
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

    sigclip is either a single float or a list of two floats. in the first case,
    the sigclip is applied symmetrically. in the second case, the first sigclip
    in the list is applied to +ve magnitude deviations (fainter) and the second
    sigclip in the list is appleid to -ve magnitude deviations (brighter).
    An example list would be `[10.,-3.]` (for 10 sigma dimmings, 3 sigma
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

    # 0. get the objectinfo and finder chart and initialize the checkplotdict
    checkplotdict = _pkl_finder_objectinfo(objectinfo,
                                           varinfo,
                                           findercmap,
                                           finderconvolve,
                                           sigclip,
                                           normto,
                                           normmingap,
                                           lclistpkl=lclistpkl,
                                           nbrradiusarcsec=nbrradiusarcsec,
                                           plotdpi=plotdpi,
                                           verbose=verbose,
                                           findercachedir=findercachedir)

    # if an objectinfo dict is absent, we'll generate a fake objectid based on
    # the second five time and mag array values. this should be OK to ID the
    # object across repeated runs of this function with the same times, mags,
    # errs, but should provide enough uniqueness otherwise (across different
    # times/mags array inputs). this is all done so we can still save checkplots
    # correctly to pickles after reviewing them using checkplotserver

    # try again to get the right objectid
    if (objectinfo and isinstance(objectinfo, dict) and
        'objectid' in objectinfo and objectinfo['objectid']):
        checkplotdict['objectid'] = objectinfo['objectid']

    # if this doesn't work, generate a random one
    if checkplotdict['objectid'] is None:
        try:
            objuuid = hashlib.sha512(times[5:10].tostring() +
                                     mags[5:10].tostring()).hexdigest()[:5]
        except Exception as e:
            LOGWARNING('times, mags, and errs may have too few items')
            objuuid = hashlib.sha512(times.tostring() +
                                     mags.tostring()).hexdigest()[:5]

        LOGWARNING('no objectid provided in objectinfo keyword arg, '
                   'generated from times[5:10] + mags[5:10]: %s' % objuuid)
        checkplotdict['objectid'] = objuuid


    # filter the input times, mags, errs; do sigclipping and normalization
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

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
    if len(stimes) > 49:

        # 1. get the mag series plot using these filtered stimes, smags, serrs
        magseriesdict = _pkl_magseries_plot(stimes, smags, serrs,
                                            plotdpi=plotdpi,
                                            magsarefluxes=magsarefluxes)

        # update the checkplotdict
        checkplotdict.update(magseriesdict)

        # 2. for each lspinfo in lspinfolist, read it in (from pkl or pkl.gz
        # if necessary), make the periodogram, make the phased mag series plots
        # for each of the nbestperiods in each lspinfo dict
        for lspinfo in lspinfolist:

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
            periodogramdict = _pkl_periodogram(lspinfo,plotdpi=plotdpi)

            # update the checkplotdict.

            # NOTE: periodograms and phased light curves are indexed by
            # lspmethod. this means if you have multiple lspinfo objects of the
            # same lspmethod, the latest one will always overwrite the earlier
            # ones.
            checkplotdict.update(periodogramdict)

            # now, make the phased light curve plots for each of the
            # nbestperiods from this periodogram
            for nbpind, nbperiod in enumerate(
                    lspinfo['nbestperiods'][:nperiodstouse]
                    ):

                # if there's a function to use for fitting, do the fit
                if lcfitfunc:
                    try:
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
                    verbose=verbose
                )

            # if there's an snr key for this lspmethod, add the info in it to
            # the checkplotdict as well
            if 'snr' in lspinfo:
                checkplotdict[lspinfo['method']]['snr'] = lspinfo['snr']
            if 'altsnr' in lspinfo:
                checkplotdict[lspinfo['method']]['altsnr'] = lspinfo['altsnr']
            if 'transitdepth' in lspinfo:
                checkplotdict[lspinfo['method']]['transitdepth'] = (
                    lspinfo['transitdepth']
                )
            if 'transitduration' in lspinfo:
                checkplotdict[lspinfo['method']]['transitduration'] = (
                    lspinfo['transitduration']
                )

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
                magsarefluxes=magsarefluxes
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

        # the checkplotdict now contains everything we need
        contents = sorted(list(checkplotdict.keys()))
        checkplotdict['status'] = 'ok: contents are %s' % contents

        if verbose:
            LOGINFO('checkplot dict complete for %s' % checkplotdict['objectid'])
            LOGINFO('checkplot dict contents: %s' % contents)

    # otherwise, we don't have enough LC points, return nothing
    else:

        LOGERROR('not enough light curve points for %s' %
                 checkplotdict['objectid'])
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
                     magsarefluxes=False,
                     nperiodstouse=3,
                     objectinfo=None,
                     lcfitfunc=None,
                     lcfitparams={},
                     varinfo=None,
                     getvarfeatures=True,
                     lclistpkl=None,
                     nbrradiusarcsec=30.0,
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
                     plotxlim=[-0.8,0.8],
                     xliminsetmode=False,
                     plotdpi=100,
                     returndict=False,
                     pickleprotocol=None,
                     bestperiodhighlight=None,
                     xgridlines=None,
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
    sigclip in the list is appleid to -ve magnitude deviations (brighter).
    An example list would be `[10.,-3.]` (for 10 sigma dimmings, 3 sigma
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

    if outgzip:

        # generate the outfile filename
        if not outfile and isinstance(lspinfolist[0],str):
            plotfpath = os.path.join(
                os.path.dirname(lspinfolist[0]),
                'checkplot-%s.pkl.gz' % (
                    os.path.basename(
                        lspinfolist[0].replace('.pkl','').replace('.gz','')
                    )
                )
            )
        elif outfile:
            plotfpath = outfile
        else:
            plotfpath = 'checkplot.pkl.gz'

    else:

        # generate the outfile filename
        if not outfile and isinstance(lspinfolist[0],str):
            plotfpath = os.path.join(
                os.path.dirname(lspinfolist[0]),
                'checkplot-%s.pkl' % (
                    os.path.basename(
                        lspinfolist[0].replace('.pkl','').replace('.gz','')
                    )
                )
            )
        elif outfile:
            plotfpath = outfile
        else:
            plotfpath = 'checkplot.pkl'


    # call checkplot_dict for most of the work
    checkplotdict = checkplot_dict(
        lspinfolist,
        times,
        mags,
        errs,
        magsarefluxes=magsarefluxes,
        nperiodstouse=nperiodstouse,
        objectinfo=objectinfo,
        varinfo=varinfo,
        getvarfeatures=getvarfeatures,
        lclistpkl=lclistpkl,
        nbrradiusarcsec=nbrradiusarcsec,
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
        verbose=verbose
    )


    # figure out which protocol to use
    # for Python >= 3.4; use v3
    if ((sys.version_info[0:2] >= (3,4) and not pickleprotocol) or
        (pickleprotocol == 3)):
        pickleprotocol = 3
        if verbose:
            LOGWARNING('the output pickle uses protocol v3 '
                       'which IS NOT backwards compatible with Python 2.7')

    # for Python == 2.7; use v2
    elif sys.version_info[0:2] == (2,7) and not pickleprotocol:
        pickleprotocol = 2

    # otherwise, if left unspecified, use the slowest but most compatible
    # protocol. this will be readable by all (most?) Pythons
    elif not pickleprotocol:
        pickleprotocol = 0

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


    # break out python 2.7 and > 3 nonsense
    if sys.version_info[:2] > (3,2):

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

        # get the current checkplotdict
        if ((isinstance(currentcp, str) or isinstance(currentcp, unicode))
            and os.path.exists(currentcp)):
            cp_current = _read_checkplot_picklefile(currentcp)
        elif isinstance(currentcp,dict):
            cp_current = currentcp
        else:
            LOGERROR('currentcp: %s of type %s is not a '
                     'valid checkplot filename (or does not exist), or a dict' %
                     (os.path.abspath(currentcp), type(currentcp)))
            return None

        # get the updated checkplotdict
        if ((isinstance(updatedcp, str) or isinstance(updatedcp, unicode))
            and os.path.exists(updatedcp)):
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

    # figure out which protocol to use
    # for Python >= 3.4; use v4 by default
    if ((sys.version_info[0:2] >= (3,4) and not pickleprotocol) or
        (pickleprotocol > 2)):
        pickleprotocol = 3
        if verbose:
            LOGWARNING('the output pickle uses protocol v3 '
                       'which IS NOT backwards compatible with Python 2.7')

    # for Python == 2.7; use v2
    elif sys.version_info[0:2] == (2,7) and not pickleprotocol:
        pickleprotocol = 2

    # otherwise, if left unspecified, use the slowest but most compatible
    # protocol. this will be readable by all (most?) Pythons
    elif not pickleprotocol:
        pickleprotocol = 0

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
        if ((isinstance(checkplotin, str) or isinstance(checkplotin, unicode))
            and os.path.exists(checkplotin)):
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
    cplspmethods = []
    cprows = 0

    # get checkplot pickle rows
    for lspmethod in ('gls','pdm','bls','aov','fch','mav'):
        if lspmethod in cpd:
            cplspmethods.append(lspmethod)
            cprows = cprows + 1

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
                            'data',
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
        (875, 25),
        cpd['objectid'] if cpd['objectid'] else 'no objectid',
        font=cpfontlarge,
        fill=(0,0,255,255)
    )
    # twomass id
    if 'twomassid' in cpd['objectinfo']:
        objinfodraw.text(
            (875, 60),
            ('2MASS J%s' % cpd['objectinfo']['twomassid']
             if cpd['objectinfo']['twomassid']
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    # ndet
    if 'ndet' in cpd['objectinfo']:
        objinfodraw.text(
            (875, 85),
            ('LC points: %s' % cpd['objectinfo']['ndet']
             if cpd['objectinfo']['ndet'] is not None
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (875, 85),
            ('LC points: %s' % cpd['magseries']['times'].size),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    # coords and PM
    objinfodraw.text(
        (875, 125),
        ('Coords and PM'),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )
    if 'ra' in cpd['objectinfo'] and 'decl' in cpd['objectinfo']:
        objinfodraw.text(
            (1125, 125),
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
            (1125, 125),
            'RA, Dec: nan, nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    if 'propermotion' in cpd['objectinfo']:
        objinfodraw.text(
            (1125, 150),
            (('Total PM: %.5f mas/yr' % cpd['objectinfo']['propermotion'])
             if (cpd['objectinfo']['propermotion'] is not None)
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (1125, 150),
            'Total PM: nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    if 'reducedpropermotion' in cpd['objectinfo']:
        objinfodraw.text(
            (1125, 175),
            (('Reduced PM: %.3f' % cpd['objectinfo']['reducedpropermotion'])
             if (cpd['objectinfo']['reducedpropermotion'] is not None)
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (1125, 175),
            'Reduced PM: nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # magnitudes
    objinfodraw.text(
        (875, 200),
        ('Magnitudes'),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )
    objinfodraw.text(
        (1125, 200),
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
        (1125, 225),
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
        (1125, 250),
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
    objinfodraw.text(
        (875, 275),
        ('Colors'),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )
    objinfodraw.text(
        (1125, 275),
        ('B - V: %.3f' %
         (cpd['objectinfo']['bvcolor'] if
          ('bvcolor' in cpd['objectinfo'] and
           cpd['objectinfo']['bvcolor'] is not None)
          else npnan)),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )
    objinfodraw.text(
        (1125, 300),
        ('i - J: %.3f' %
         (cpd['objectinfo']['ijcolor'] if
          ('ijcolor' in cpd['objectinfo'] and
           cpd['objectinfo']['ijcolor'] is not None)
          else npnan)),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )
    objinfodraw.text(
        (1125, 325),
        ('J - K: %.3f' %
         (cpd['objectinfo']['jkcolor'] if
          ('jkcolor' in cpd['objectinfo'] and
           cpd['objectinfo']['jkcolor'] is not None)
          else npnan)),
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
                (875, 375+objtagline*25),
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
        (1600, 125),
        objvarflag,
        font=cpfontnormal,
        fill=(0,0,0,255)
    )

    # period
    objinfodraw.text(
        (1600, 150),
        ('Period [days]: %.6f' %
         (cpd['varinfo']['varperiod']
          if cpd['varinfo']['varperiod'] is not None
          else np.nan)),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )

    # epoch
    objinfodraw.text(
        (1600, 175),
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
                (1600, 225+vartagline*25),
                vartagslice,
                font=cpfontnormal,
                fill=(135, 54, 0, 255)
            )

    # object comments
    if cpd['comments']:

        commentsplit = cpd['comments'].split(' ')

        # write 10 words per line
        ncommentlines = int(np.ceil(len(commentsplit)/10.0))

        for commentline in range(ncommentlines):
            commentslice = ' '.join(
                commentsplit[commentline*10:commentline*10+10]
            )
            objinfodraw.text(
                (1600, 325+commentline*25),
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

        if (cpd[lspmethod] and cpd[lspmethod][0]):

            plc1 = Image.open(
            _base64_to_file(cpd[lspmethod][0]['plot'], None, writetostrio=True)
            )
            outimg.paste(plc1,(750,480 + 480*lspmethodind))

        #################################
        # 2nd best phased LC comes next #
        #################################

        if (cpd[lspmethod] and cpd[lspmethod][1]):

            plc2 = Image.open(
            _base64_to_file(cpd[lspmethod][1]['plot'], None, writetostrio=True)
            )
            outimg.paste(plc2,(750*2,480 + 480*lspmethodind))

        #################################
        # 3rd best phased LC comes next #
        #################################

        if (cpd[lspmethod] and cpd[lspmethod][2]):

            plc3 = Image.open(
            _base64_to_file(cpd[lspmethod][2]['plot'], None, writetostrio=True)
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
        # tile 4: phased LC for bls

        for nbrind, nbr in enumerate(cpd['neighbors']):

            # first panel: neighbor objectid, ra, decl, distance, unphased LC
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
            glslc = Image.open(
                _base64_to_file(
                    nbr['gls'][0]['plot'], None, writetostrio=True
                )
            )
            outimg.paste(glslc,
                         (750*1,
                          (cprows+1)*480 + (erows*480) + (cpderows*480) +
                          480*nbrind))

            # second panel: phased LC for gls
            pdmlc = Image.open(
                _base64_to_file(
                    nbr['pdm'][0]['plot'], None, writetostrio=True
                )
            )
            outimg.paste(pdmlc,
                         (750*2,
                          (cprows+1)*480 + (erows*480) + (cpderows*480) +
                          480*nbrind))

            # second panel: phased LC for gls
            blslc = Image.open(
                _base64_to_file(
                    nbr['bls'][0]['plot'], None, writetostrio=True
                )
            )
            outimg.paste(blslc,
                         (750*3,
                          (cprows+1)*480 + (erows*480) + (cpderows*480) +
                          480*nbrind))


    #####################
    ## WRITE FINAL PNG ##
    #####################

    # check if we've stupidly copied over the same filename as the input pickle
    # to expected output file
    if outfile.endswith('pkl'):
        LOGWARNING('expected output PNG filename ends with .pkl, '
                   'changed to .png')
        outfile = outfile.replace('.pkl','.png')

    outimg.save(outfile)

    if os.path.exists(outfile):
        LOGINFO('checkplot pickle -> checkplot PNG: %s OK' % outfile)
        return outfile
    else:
        LOGERROR('failed to write checkplot PNG')
        return None



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
