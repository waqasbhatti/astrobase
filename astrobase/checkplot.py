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


'''
import os
import os.path
import gzip
import base64

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

# check the DISPLAY variable to see if we can plot stuff interactively
try:
    dispok = os.environ['DISPLAY']
except KeyError:
    import matplotlib
    matplotlib.use('Agg')
    dispok = False

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

from .lcmath import phase_magseries, phase_bin_magseries, normalize_magseries
from .varbase import spline_fit_magseries
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
                         serrs):
    '''makes the magseries plot tile.

    '''

    scaledplottime = stimes - npmin(stimes)

    axes.scatter(scaledplottime,
                 smags,
                 marker='o',
                 s=2,
                 color='green')

    # flip y axis for mags
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
                                twolspmode=False):
    '''makes the phased magseries plot tile.

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



############################################
## CHECKPLOT FUNCTIONS THAT WRITE TO PNGS ##
############################################

def checkplot_png(lspinfo,
                  times,
                  mags,
                  errs,
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
                  plotdpi=100):
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

    # remove nans
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[find], mags[find], errs[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next
    if sigclip:

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs


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

        _make_magseries_plot(axes[1], stimes, smags, serrs)


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
                    spfit = spline_fit_magseries(stimes, smags, serrs,
                                                 varperiod)
                    varepoch = spfit['fitepoch']
                    if len(varepoch) != 1:
                        varepoch = varepoch[0]
                except Exception as e:
                    LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                    varepoch = npmin(stimes)

            LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                    (varperiod, varepoch))

            # make sure the best period phased LC plot stands out
            if periodind == 0:
                axes[periodind+2].set_axis_bgcolor('#adff2f')

            _make_phased_magseries_plot(axes[periodind+2],
                                        periodind,
                                        stimes, smags,
                                        varperiod, varepoch,
                                        phasewrap, phasesort, phasebin,
                                        plotxlim, lspmethod)

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
                         plotdpi=100):
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

    # remove nans
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[find], mags[find], errs[find]

    # get the median and stdev = 1.483 x MAD
    median_mag = npmedian(fmags)
    stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

    # sigclip next
    if sigclip:

        sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

        stimes = ftimes[sigind]
        smags = fmags[sigind]
        serrs = ferrs[sigind]

        LOGINFO('sigclip = %s: before = %s observations, '
                'after = %s observations' %
                (sigclip, len(times), len(stimes)))

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs


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

        _make_magseries_plot(axes[2], stimes, smags, serrs)

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
                    spfit = spline_fit_magseries(stimes, smags, serrs,
                                                 varperiod)
                    varepoch = spfit['fitepoch']
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
                                        plotxlim, lspmethod1,
                                        twolspmode=True)

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
                    varepoch = spfit['fitepoch']
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
                                        twolspmode=True)

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


################################################
## CHECKPLOT FUNCTIONS THAT WORK WITH PICKLES ##
################################################

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



def _pkl_finder_objectinfo(objectinfo, findercmap, finderconvolve,
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
            finderfig.savefig(finderpng, bbox_inches='tight',
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
                         'finderchart':finderb64}


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
    for bestperiod, bestpeak in zip(nbestperiods,
                                    nbestlspvals):
        plt.annotate('%.6f' % bestperiod,
                      xy=(bestperiod, bestpeak), xycoords='data',
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
    pgramfig.savefig(pgrampng, bbox_inches='tight',
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



def _pkl_magseries_plot(stimes, smags, serrs, plotdpi=100):
    '''This returns the magseries plot PNG as base64, plus arrays as dict.

    '''



def _pkl_phased_magseries_plot(periodind, stimes, smags,
                               varperiod, varepoch,
                               phasewrap, phasesort, phasebin,
                               plotxlim, lspmethod, plotdpi=100):
    '''This returns the phased magseries plot PNG as base64 plus info as a dict.

    '''


def multilsp_checkplot_pickle(lspinfolist,
                              times,
                              mags,
                              errs,
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
                              plotdpi=100):

    '''This writes a multiple lspinfo checkplot to a gzipped pickle file.

    The gzipped pickle file contains all the plots (magseries and phased
    magseries), periodograms, object information, variability information, light
    curves, and phased light curves. This is intended to be used with an
    external viewer app (e.g. checkplotserver.py), or by using the
    checkplot_pickle_to_png function below.

    All other options are the same as for checkplot_png. This function can take
    input from multiple lspinfo dicts (e.g. a list of output dicts or gzipped
    pickles of dicts from the BLS, PDM, AoV, or GLS period-finders in
    periodbase).

    '''



def checkplot_pickle_to_dict(checkplotpickle):
    '''
    This reads the checkplot gzipped pickle into a dict.

    '''



def checkplot_pickle_to_png(checkplotpickle):
    '''This reads the pickle provided, and writes out a PNG.

    The PNG has 4 x N tiles, as below:

    [ finderchart  ] [ objectinfo   ] [ variableinfo ] [ unphased LC  ]
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



def checkplot_pickle_update(current, updated,
                            outfile=None):
    '''This updates the current checkplot dict with updated values provided.

    Writes out the new checkplot gzipped pickle file to outfile. Mostly only
    useful for checkplotserver.py.

    '''
