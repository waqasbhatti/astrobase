#!/usr/bin/env python
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
import uuid

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
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import logging
from datetime import datetime
from traceback import format_exc



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
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


###################
## LOCAL IMPORTS ##
###################

from .lcmath import phase_magseries, phase_bin_magseries, \
    normalize_magseries, sigclip_magseries
from .varbase.lcfit import spline_fit_magseries
from .coordutils import total_proper_motion, reduced_proper_motion
from .plotbase import astroquery_skyview_stamp, \
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
                      finderconvolve):
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
            dss = astroquery_skyview_stamp(objectinfo['ra'],objectinfo['decl'])
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

    axes.scatter(scaledplottime,
                 smags,
                 marker='o',
                 s=2,
                 color='green')

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
                                phasewrap, phasesort, phasebin,
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
                                          binsize=phasebin)
        binplotphase = binphasedlc['binnedphases']
        binplotmags = binphasedlc['binnedmags']


    # finally, make the phased LC plot
    axes.scatter(plotphase,
                 plotmags,
                 marker='o',
                 s=2,
                 color='gray')

    # overlay the binned phased LC plot if we're making one
    if phasebin:
        axes.scatter(binplotphase,
                     binplotmags,
                     marker='o',
                     s=20,
                     color='blue')

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
                          axesylim[1] + 0.75*npabs(axesylim[1]-axesylim[0]))
        else:
            axes.set_ylim(axesylim[0],
                          axesylim[1] - 0.75*npabs(axesylim[1]-axesylim[0]))

        # put the inset axes in
        inset = inset_axes(axes, width="40%", height="40%", loc=1)

        # make the scatter plot for the phased LC plot
        inset.scatter(plotphase,
                      plotmags,
                      marker='o',
                      s=2,
                      color='gray')

        # overlay the binned phased LC plot if we're making one
        if phasebin:
            inset.scatter(binplotphase,
                          binplotmags,
                          marker='o',
                          s=20,
                          color='blue')

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
                  normto='globalmedian',
                  normmingap=4.0,
                  outfile=None,
                  sigclip=4.0,
                  varepoch='min',
                  phasewrap=True,
                  phasesort=True,
                  phasebin=0.002,
                  plotxlim=[-0.8,0.8],
                  xliminsetmode=False,
                  plotdpi=100,
                  bestperiodhighlight='#adff2f'):
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
    'aov' -> Schwarzenberg-Cerny AoV (e.g., from periodbase.aov_periodfind)
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

    xliminsetmode = True sets up the phased mag series plot to show a zoomed-in
    portion (set by plotxlim) as the main plot and an inset version of the full
    phased light curve from phase 0.0 to 1.0. This can be useful if searching
    for small dips near phase 0.0 caused by planetary transits for example.

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
                      findercmap, finderconvolve)

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
                                                 magsarefluxes=magsarefluxes)
                    varepoch = spfit['fitinfo']['fitepoch']
                    if len(varepoch) != 1:
                        varepoch = varepoch[0]
                except Exception as e:
                    LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                    varepoch = npmin(stimes)

            LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                    (varperiod, varepoch))

            # make sure the best period phased LC plot stands out
            if periodind == 0 and bestperiodhighlight:
                axes[periodind+2].set_axis_bgcolor(bestperiodhighlight)

            _make_phased_magseries_plot(axes[periodind+2],
                                        periodind,
                                        stimes, smags,
                                        varperiod, varepoch,
                                        phasewrap, phasesort, phasebin,
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
        plt.close()

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
                         normto='globalmedian',
                         normmingap=4.0,
                         outfile=None,
                         sigclip=4.0,
                         varepoch='min',
                         phasewrap=True,
                         phasesort=True,
                         phasebin=0.002,
                         plotxlim=[-0.8,0.8],
                         xliminsetmode=False,
                         plotdpi=100,
                         bestperiodhighlight='#adff2f'):
    '''This makes a checkplot using results from two independent period-finders.

    Adapted from Luke Bouma's implementation of the same. This makes a special
    checkplot that uses two lspinfo dictionaries, from two independent
    period-finding methods. For EBs, it's probably best to use Stellingwerf PDM
    or Schwarzenberg-Cerny AoV as one of these, and the Box Least-squared Search
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
        LOGINFO('loading LSP info from pickle %s' % lspinfo1)

        if '.gz' in lspinfo1:
            with gzip.open(lspinfo1,'rb') as infd:
                lspinfo1 = pickle.load(infd)
        else:
            with open(lspinfo1,'rb') as infd:
                lspinfo1 = pickle.load(infd)


    # get the second LSP from a pickle file transparently
    if isinstance(lspinfo2,str) and os.path.exists(lspinfo2):
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
                      findercmap, finderconvolve)

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
                                                 magsarefluxes=magsarefluxes)
                    varepoch = spfit['fitinfo']['fitepoch']
                    if len(varepoch) != 1:
                        varepoch = varepoch[0]
                except Exception as e:
                    LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                    varepoch = npmin(stimes)

            LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                    (varperiod, varepoch))

            # make sure the best period phased LC plot stands out
            if periodind == 0 and bestperiodhighlight:
                plotaxes.set_axis_bgcolor(bestperiodhighlight)

            _make_phased_magseries_plot(plotaxes,
                                        periodind,
                                        stimes, smags,
                                        varperiod, varepoch,
                                        phasewrap, phasesort, phasebin,
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
                    spfit = spline_fit_magseries(stimes, smags, serrs,
                                                 varperiod)
                    varepoch = spfit['fitinfo']['fitepoch']
                    if len(varepoch) != 1:
                        varepoch = varepoch[0]
                except Exception as e:
                    LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                    varepoch = npmin(stimes)

            LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                    (varperiod, varepoch))

            # make sure the best period phased LC plot stands out
            if periodind == 0:
                plotaxes.set_axis_bgcolor('#adff2f')

            _make_phased_magseries_plot(plotaxes,
                                        periodind,
                                        stimes, smags,
                                        varperiod, varepoch,
                                        phasewrap, phasesort, phasebin,
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

        LOGINFO('checkplot done -> %s' % plotfpath)
        return plotfpath


#########################################
## PICKLE CHECKPLOT UTILITY FUNCTIONS  ##
#########################################

def _base64_to_file(b64str, outfpath):
    '''
    This converts the base64 encoded string to a file.

    '''

    try:

        filebytes = base64.b64decode(b64str)
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
                           plotdpi=100):
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

        LOGINFO('adding in object information and '
                'finder chart for %s at RA: %.3f, DEC: %.3f' %
                (objectid, objectinfo['ra'], objectinfo['decl']))

        # get the finder chart
        try:
            finder = astroquery_skyview_stamp(objectinfo['ra'],
                                              objectinfo['decl'],
                                              convolvewith=finderconvolve)
            finderfig = plt.figure(figsize=(3,3),dpi=plotdpi,frameon=False)
            plt.imshow(finder, cmap=findercmap)
            plt.xticks([])
            plt.yticks([])
            # grid lines pointing to the center of the frame
            plt.axvline(x=150,ymin=0.2,ymax=0.4,linewidth=2.0,color='k')
            plt.axhline(y=149,xmin=0.2,xmax=0.4,linewidth=2.0,color='k')
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

        # now that we have the finder chart, get the rest of the object
        # information

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
                         'objectinfo':objectinfo,
                         'finderchart':finderb64,
                         'sigclip':sigclip,
                         'normto':normto,
                         'normmingap':normmingap}

        # add the objecttags key to objectinfo
        checkplotdict['objectinfo']['objecttags'] = None

    # if there's no objectinfo, we can't do anything.  we'll generate a random
    # objectid so we can still save checkplots to pickles using checkplotserver
    else:

        # put together the initial checkplot pickle dictionary
        # this will be updated by the functions below as appropriate
        # and will written out as a gzipped pickle at the end of processing
        objuuid = uuid.uuid4().hex[-8:]

        LOGWARNING('no object ID provided, '
                   'using a randomly generated one: %s' % objuuid)

        checkplotdict = {'objectid':objuuid,
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

    plt.scatter(scaledplottime,
                 smags,
                 marker='o',
                 s=2,
                 color='green')

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
                               phasewrap, phasesort, phasebin,
                               plotxlim,
                               plotdpi=100,
                               bestperiodhighlight='#adff2f',
                               xgridlines=None,
                               xliminsetmode=False,
                               magsarefluxes=False):
    '''This returns the phased magseries plot PNG as base64 plus info as a dict.

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
                                         sigclip=None)
            varepoch = spfit['fitinfo']['fitepoch']
            if len(varepoch) != 1:
                varepoch = varepoch[0]
        except Exception as e:
            LOGEXCEPTION('spline fit failed, using min(times) as epoch')
            varepoch = npmin(stimes)

    LOGINFO('plotting %s phased LC with period %s: %.6f, epoch: %.5f' %
            (lspmethod, periodind, varperiod, varepoch))

    # make the plot title based on the lspmethod
    if periodind == 0:
        plottitle = '%s best period: %.6f d - epoch: %.5f' % (
            METHODSHORTLABELS[lspmethod],
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
                                          binsize=phasebin)
        binplotphase = binphasedlc['binnedphases']
        binplotmags = binphasedlc['binnedmags']

    else:
        binplotphase = None
        binplotmags = None


    # finally, make the phased LC plot
    plt.scatter(plotphase,
                plotmags,
                marker='o',
                s=2,
                color='gray')

    # overlay the binned phased LC plot if we're making one
    if phasebin:
        plt.scatter(binplotphase,
                    binplotmags,
                    marker='o',
                    s=10,
                    color='blue')

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
    if periodind == 0 and bestperiodhighlight:
        plt.gca().set_axis_bgcolor(bestperiodhighlight)

    # if we're making an inset plot showing the full range
    if (plotxlim and isinstance(plotxlim, list) and
        len(plotxlim) == 2 and xliminsetmode is True):

        # bump the ylim of the plot so that the inset can fit in this axes plot
        axesylim = plt.gca().get_ylim()

        if magsarefluxes:
            plt.gca().set_ylim(
                axesylim[0],
                axesylim[1] + 0.75*npabs(axesylim[1]-axesylim[0])
            )
        else:
            plt.gca().set_ylim(
                axesylim[0],
                axesylim[1] - 0.75*npabs(axesylim[1]-axesylim[0])
            )

        # put the inset axes in
        inset = inset_axes(plt.gca(), width="40%", height="40%", loc=1)

        # make the scatter plot for the phased LC plot
        inset.scatter(plotphase,
                      plotmags,
                      marker='o',
                      s=2,
                      color='gray')

        if phasebin:
            # make the scatter plot for the phased LC plot
            inset.scatter(binplotphase,
                          binplotmags,
                          marker='o',
                          s=2,
                          color='gray')

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

    # this requires the checkplotdict to be present already, we'll just update
    # it at the appropriate lspmethod and periodind
    checkplotdict[lspmethod][periodind] = {
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
        'plotxlim':plotxlim
    }

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
            with gzip.open(checkplotpickle) as infd:
                cpdict = pickle.load(infd)

        except UnicodeDecodeError:

            with gzip.open(checkplotpickle) as infd:
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
                   findercmap='gray_r',
                   finderconvolve=None,
                   normto='globalmedian',
                   normmingap=4.0,
                   sigclip=4.0,
                   varepoch='min',
                   phasewrap=True,
                   phasesort=True,
                   phasebin=0.002,
                   plotxlim=[-0.8,0.8],
                   xliminsetmode=False,
                   plotdpi=100,
                   bestperiodhighlight='#adff2f',
                   xgridlines=None):

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

    sigclip is either a single float or a list of two floats. in the first case,
    the sigclip is applied symmetrically. in the second case, the first sigclip
    in the list is applied to +ve magnitude deviations (fainter) and the second
    sigclip in the list is appleid to -ve magnitude deviations (brighter).
    An example list would be `[10.,-3.]` (for 10 sigma dimmings, 3 sigma
    brightenings).

    bestperiodhighlight (boolean) sets whether user wants a green background on
    bestperiod from each periodogram.

    xgridlines (default None) can be a list, e.g., [-0.5,0.,0.5] that sets the
    x-axis grid lines on plotted phased LCs for easy visual identification of
    important features.

    xliminsetmode = True sets up the phased mag series plot to show a zoomed-in
    portion (set by plotxlim) as the main plot and an inset version of the full
    phased light curve from phase 0.0 to 1.0. This can be useful if searching
    for small dips near phase 0.0 caused by planetary transits for example.

    '''

    # first, get the objectinfo and finder chart
    # and initialize the checkplotdict
    checkplotdict = _pkl_finder_objectinfo(objectinfo,
                                           varinfo,
                                           findercmap,
                                           finderconvolve,
                                           sigclip,
                                           normto,
                                           normmingap,
                                           plotdpi=plotdpi)



    # filter the input times, mags, errs; do sigclipping and normalization
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # report on how sigclip went
    LOGINFO('sigclip = %s: before = %s observations, '
            'after = %s observations' %
            (sigclip, len(times), len(stimes)))


    # take care of the normalization
    if normto is not False:
        stimes, smags = normalize_magseries(stimes, smags,
                                            normto=normto,
                                            mingap=normmingap)

    # make sure we have some lightcurve points to plot after sigclip
    if len(stimes) > 49:

        # next, get the mag series plot using these filtered stimes, smags,
        # serrs
        magseriesdict = _pkl_magseries_plot(stimes, smags, serrs,
                                            plotdpi=plotdpi,
                                            magsarefluxes=magsarefluxes)

        # update the checkplotdict
        checkplotdict.update(magseriesdict)

        # next, for each lspinfo in lspinfolist, read it in (from pkl or pkl.gz
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

                # this updates things as it runs
                checkplotdict = _pkl_phased_magseries_plot(
                    checkplotdict,
                    lspinfo['method'],
                    nbpind,
                    stimes, smags, serrs,
                    nbperiod, varepoch,
                    phasewrap, phasesort, phasebin,
                    plotxlim,
                    plotdpi=plotdpi,
                    bestperiodhighlight=bestperiodhighlight,
                    magsarefluxes=magsarefluxes,
                    xliminsetmode=xliminsetmode,
                    xgridlines=xgridlines
                )

        # the checkplotdict now contains everything we need
        LOGINFO('checkplot dict complete for %s' % checkplotdict['objectid'])
        contents = sorted(list(checkplotdict.keys()))
        LOGINFO('checkplot dict contents: %s' % contents)
        checkplotdict['status'] = 'ok: contents are %s' % contents

        # add a comments key:val
        checkplotdict['comments'] = None

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
                     varinfo=None,
                     findercmap='gray_r',
                     finderconvolve=None,
                     normto='globalmedian',
                     normmingap=4.0,
                     outfile=None,
                     outgzip=False,
                     sigclip=4.0,
                     varepoch='min',
                     phasewrap=True,
                     phasesort=True,
                     phasebin=0.002,
                     plotxlim=[-0.8,0.8],
                     xliminsetmode=False,
                     plotdpi=100,
                     returndict=False,
                     pickleprotocol=None,
                     bestperiodhighlight='#adff2f',
                     xgridlines=None):

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

    gzip controls whether to gzip the output pickle. it turns out that this is
    the slowest bit in the output process, so if you're after speed, best not to
    use this. this is False by default since it turns out that gzip actually
    doesn't save that much space (29 MB vs. 35 MB for the average checkplot
    pickle).

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

    if returndict is True, will return the checkplotdict created and the path to
    the output checkplot pickle file as a tuple. if returndict is False, will
    only return the path to the output checkplot pickle.

    pickleprotocol sets the protocol version of the output pickle. Anything with
    version > 2 can't be read by Python 2.7 or earlier, but is much faster to
    dump, load, and is smaller on disk. This function will detect your Python
    version and attempt to use version 3 if Python > 3 or version 2 if Python <
    3. It will emit a warning if it uses protocol version 3 that these pickles
    won't work on older Pythons.

    sigclip is either a single float or a list of two floats. in the first case,
    the sigclip is applied symmetrically. in the second case, the first sigclip
    in the list is applied to +ve magnitude deviations (fainter) and the second
    sigclip in the list is appleid to -ve magnitude deviations (brighter).
    An example list would be `[10.,-3.]` (for 10 sigma dimmings, 3 sigma
    brightenings).

    bestperiodhighlight (boolean) sets whether user wants a green background on
    bestperiod from each periodogram.

    xgridlines (default None) can be a list, e.g., [-0.5,0.,0.5] that sets the
    x-axis grid lines on plotted phased LCs for easy visual identification of
    important features.

    xliminsetmode = True sets up the phased mag series plot to show a zoomed-in
    portion (set by plotxlim) as the main plot and an inset version of the full
    phased light curve from phase 0.0 to 1.0. This can be useful if searching
    for small dips near phase 0.0 caused by planetary transits for example.

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
        findercmap=findercmap,
        finderconvolve=finderconvolve,
        normto=normto,
        normmingap=normmingap,
        sigclip=sigclip,
        varepoch=varepoch,
        phasewrap=phasewrap,
        phasesort=phasesort,
        phasebin=phasebin,
        plotxlim=plotxlim,
        xliminsetmode=xliminsetmode,
        plotdpi=plotdpi,
        bestperiodhighlight=bestperiodhighlight,
        xgridlines=xgridlines
    )


    # figure out which protocol to use
    # for Python >= 3.4; use v3
    if ((sys.version_info[0:2] >= (3,4) and not pickleprotocol) or
        (pickleprotocol == 3)):
        pickleprotocol = 3
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
        LOGINFO('checkplot done -> %s' % picklefname)
        return checkplotdict, picklefname

    # otherwise, just return the filename
    else:
        # just to make sure: free up space
        del checkplotdict
        LOGINFO('checkplot done -> %s' % picklefname)
        return picklefname



def checkplot_pickle_update(currentcp, updatedcp,
                            outfile=None,
                            outgzip=False,
                            pickleprotocol=None):
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

    if ((isinstance(updatedcp, str) or isinstance(updatedcp, unicode))
        and os.path.exists(updatedcp)):
        cp_updated = _read_checkplot_picklefile(updatedcp)
    elif isinstance(updatedcp, dict):
        cp_updated = updatedcp
    else:
        LOGERROR('currentcp: %s of type %s is not a '
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



def checkplot_pickle_to_png(checkplotpickle, outfpath):
    '''This reads the pickle provided, and writes out a PNG.

    checkplotpickle is either a checkplot dict produced by checkplot_pickle
    above or a gzipped pickle file produced by the same function.

    The PNG has 4 x N tiles, as below:

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

    FIXME: to be implemented

    '''
