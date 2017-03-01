#!/usr/bin/env python

'''
plotbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2016
License: MIT.

Contains various useful functions for plotting light curves and associated data.


'''
import os
import os.path
import gzip

try:
    import cPickle as pickle
except:
    import pickle

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

import logging
from datetime import datetime
from traceback import format_exc

try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve

# for downloading DSS stamps from NASA GSFC SkyView
from astroquery.skyview import SkyView

# for convolving DSS stamps to simulate seeing effects
import astropy.convolution as aconv

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.plotbase' % parent_name)

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

from .lcmath import phase_magseries, phase_magseries_with_errs, \
    phase_bin_magseries, phase_bin_magseries_with_errs, \
    time_bin_magseries, time_bin_magseries_with_errs, sigclip_magseries, \
    normalize_magseries, find_lc_timegroups

from .varbase.lcfit import spline_fit_magseries

from .coordutils import total_proper_motion, reduced_proper_motion

#########################
## SIMPLE LIGHT CURVES ##
#########################

def plot_mag_series(times,
                    mags,
                    magsarefluxes=False,
                    errs=None,
                    outfile=None,
                    sigclip=30.0,
                    normto='globalmedian',
                    normmingap=4.0,
                    timebin=None,
                    yrange=None,
                    segmentmingap=100.0,
                    plotdpi=100):
    '''This plots a magnitude time series.

    If magsarefluxes = False, then this function reverses the y-axis as is
    customary for magnitudes. If magsarefluxes = True, then this isn't done.

    If outfile is none, then plots to matplotlib interactive window. If outfile
    is a string denoting a filename, uses that to write a png/eps/pdf figure.

    timebin is either a float indicating binsize in seconds, or None indicating
    no time-binning is required.

    sigclip is either a single float or a list of two floats. in the first case,
    the sigclip is applied symmetrically. in the second case, the first sigclip
    in the list is applied to +ve magnitude deviations (fainter) and the second
    sigclip in the list is appleid to -ve magnitude deviations (brighter).

    normto is either 'globalmedian', 'zero' or a float to normalize the mags
    to. If it's False, no normalization will be done on the magnitude time
    series. normmingap controls the minimum gap required to find possible
    groupings in the light curve that may belong to a different instrument (so
    may be displaced vertically)

    segmentmingap controls the minimum length of time (in days) required to
    consider a timegroup in the light curve as a separate segment. This is
    useful when the light curve consists of measurements taken over several
    seasons, so there's lots of dead space in the plot that can be cut out to
    zoom in on the interesting stuff. If segmentmingap is not None, the
    magseries plot will be cut in this way.

    plotdpi sets the DPI for PNG plots (default = 100).

    '''

    # sigclip the magnitude timeseries
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # now we proceed to binning
    if timebin and errs is not None:

        binned = time_bin_magseries_with_errs(stimes, smags, serrs,
                                              binsize=timebin)
        btimes, bmags, berrs = (binned['binnedtimes'],
                                binned['binnedmags'],
                                binned['binnederrs'])

    elif timebin and errs is None:

        binned = time_bin_magseries(stimes, smags,
                                    binsize=timebin)
        btimes, bmags, berrs = binned['binnedtimes'], binned['binnedmags'], None

    else:

        btimes, bmags, berrs = stimes, smags, serrs


    # check if we need to normalize
    if normto is not False:
        btimes, bmags = normalize_magseries(btimes, bmags,
                                            normto=normto,
                                            mingap=normmingap)

    btimeorigin = btimes.min()
    btimes = btimes - btimeorigin

    ##################################
    ## FINALLY PLOT THE LIGHT CURVE ##
    ##################################

    # if we're going to plot with segment gaps highlighted, then find the gaps
    if segmentmingap is not None:
        ntimegroups, timegroups = find_lc_timegroups(btimes,
                                                     mingap=segmentmingap)

    # get the yrange for all the plots if it's given
    if yrange and isinstance(yrange,list) and len(yrange) == 2:
        ymin, ymax = yrange

    # if it's not given, figure it out
    else:

        # the plot y limits are just 0.05 mags on each side if mags are used
        if not magsarefluxes:
            ymin, ymax = (bmags.min() - 0.05,
                          bmags.max() + 0.05)
        # if we're dealing with fluxes, limits are 2% of the flux range per side
        else:
            ycov = bmags.max() - bmags.min()
            ymin = bmags.min() - 0.02*ycov
            ymax = bmags.max() + 0.02*ycov

    # if we're supposed to make the plot segment-aware (i.e. gaps longer than
    # segmentmingap will be cut out)
    if segmentmingap and ntimegroups > 1:

        LOGINFO('%s time groups found' % ntimegroups)

        # our figure is now a multiple axis plot
        # the aspect ratio is a bit wider
        fig, axes = plt.subplots(1,ntimegroups,sharey=True)
        fig.set_size_inches(9.6,4.8)
        axes = np.ravel(axes)

        # now go through each axis and make the plots for each timegroup
        for timegroup, ax, axind in zip(timegroups, axes, range(len(axes))):

            tgtimes = btimes[timegroup]
            tgmags = bmags[timegroup]

            if berrs:
                tgerrs = berrs[timegroup]
            else:
                tgerrs = None

            LOGINFO('axes: %s, timegroup %s: JD %.3f to %.3f' % (
                axind,
                axind+1,
                btimeorigin + tgtimes.min(),
                btimeorigin + tgtimes.max())
            )

            ax.errorbar(tgtimes, tgmags, fmt='go', yerr=tgerrs,
                         markersize=2.0, markeredgewidth=0.0, ecolor='grey',
                         capsize=0)

            # don't use offsets on any xaxis
            ax.get_xaxis().get_major_formatter().set_useOffset(False)

            # fix the ticks to use no yoffsets and remove right spines for first
            # axes instance
            if axind == 0:
                ax.get_yaxis().get_major_formatter().set_useOffset(False)
                ax.spines['right'].set_visible(False)
                ax.yaxis.tick_left()
            # remove the right and left spines for the other axes instances
            elif 0 < axind < (len(axes)-1):
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(right='off', labelright='off',
                               left='off',labelleft='off')
            # make the left spines invisible for the last axes instance
            elif axind == (len(axes)-1):
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(True)
                ax.yaxis.tick_right()

            # set the yaxis limits
            if not magsarefluxes:
                ax.set_ylim(ymax, ymin)
            else:
                ax.set_ylim(ymin, ymax)

            # now figure out the xaxis ticklabels and ranges
            tgrange = tgtimes.max() - tgtimes.min()

            if tgrange < 10.0:
                ticklocations = [tgrange/2.0]
                ax.set_xlim(npmin(tgtimes) - 0.5, npmax(tgtimes) + 0.5)
            elif 10.0 < tgrange < 30.0:
                ticklocations = np.linspace(tgtimes.min()+5.0,
                                            tgtimes.max()-5.0,
                                            num=2)
                ax.set_xlim(npmin(tgtimes) - 2.0, npmax(tgtimes) + 2.0)

            elif 30.0 < tgrange < 100.0:
                ticklocations = np.linspace(tgtimes.min()+10.0,
                                            tgtimes.max()-10.0,
                                            num=3)
                ax.set_xlim(npmin(tgtimes) - 2.5, npmax(tgtimes) + 2.5)
            else:
                ticklocations = np.linspace(tgtimes.min()+20.0,
                                            tgtimes.max()-20.0,
                                            num=3)
                ax.set_xlim(npmin(tgtimes) - 3.0, npmax(tgtimes) + 3.0)

            ax.xaxis.set_ticks([int(x) for x in ticklocations])

        # done with plotting all the sub axes

        # make the distance between sub plots smaller
        plt.subplots_adjust(wspace=0.07)

        # make the overall x and y labels
        fig.text(0.5, 0.00, 'JD - %.3f (not showing gaps)' % btimeorigin,
                 ha='center')
        if not magsarefluxes:
            fig.text(0.02, 0.5, 'magnitude', va='center', rotation='vertical')
        else:
            fig.text(0.02, 0.5, 'flux', va='center', rotation='vertical')


    # make normal figure otherwise
    else:

        fig = plt.figure()
        fig.set_size_inches(7.5,4.8)

        plt.errorbar(btimes, bmags, fmt='go', yerr=berrs,
                     markersize=2.0, markeredgewidth=0.0, ecolor='grey',
                     capsize=0)

        # make a grid
        plt.grid(color='#a9a9a9',
                 alpha=0.9,
                 zorder=0,
                 linewidth=1.0,
                 linestyle=':')

        # fix the ticks to use no offsets
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

        plt.xlabel('JD - %.3f' % btimeorigin)

        # set the yaxis limits and labels
        if not magsarefluxes:
            plt.ylim(ymax, ymin)
            plt.ylabel('magnitude')
        else:
            plt.ylim(ymin, ymax)
            plt.ylabel('flux')

    # write the plot out to a file if requested
    if outfile and isinstance(outfile, str):

        if outfile.endswith('.png'):
            plt.savefig(outfile,bbox_inches='tight',dpi=plotdpi)
        else:
            plt.savefig(outfile,bbox_inches='tight')
        plt.close()
        return os.path.abspath(outfile)

    elif dispok:

        plt.show()
        plt.close()
        return

    else:

        LOGWARNING('no output file specified and no $DISPLAY set, '
                   'saving to magseries-plot.png in current directory')
        outfile = 'magseries-plot.png'
        plt.savefig(outfile,bbox_inches='tight',dpi=plotdpi)
        plt.close()
        return os.path.abspath(outfile)



#########################
## PHASED LIGHT CURVES ##
#########################

def plot_phased_mag_series(times,
                           mags,
                           period,
                           magsarefluxes=False,
                           errs=None,
                           normto='globalmedian',
                           normmingap=4.0,
                           epoch='min',
                           outfile=None,
                           sigclip=30.0,
                           phasewrap=True,
                           phasesort=True,
                           phasebin=None,
                           plotphaselim=[-0.8,0.8],
                           fitknotfrac=0.01,
                           yrange=None,
                           plotdpi=100):
    '''This plots a phased magnitude time series using the period provided.

    If epoch is None, uses the min(times) as the epoch.

    If epoch is a string 'min', then fits a cubic spline to the phased light
    curve using min(times), finds the magnitude minimum from the fitted light
    curve, then uses the corresponding time value as the epoch.

    If epoch is a float, then uses that directly to phase the light curve and as
    the epoch of the phased mag series plot.

    If outfile is none, then plots to matplotlib interactive window. If outfile
    is a string denoting a filename, uses that to write a png/eps/pdf figure.

    plotdpi sets the DPI for PNG plots.

    '''

    # sigclip the magnitude timeseries
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)


    # check if we need to normalize
    if normto is not False:
        stimes, smags = normalize_magseries(stimes, smags,
                                            normto=normto,
                                            mingap=normmingap)

    # figure out the epoch, if it's None, use the min of the time
    if epoch is None:
        epoch = stimes.min()

    # if the epoch is 'min', then fit a spline to the light curve phased
    # using the min of the time, find the fit mag minimum and use the time for
    # that as the epoch
    elif isinstance(epoch,str) and epoch == 'min':

        try:
            spfit = spline_fit_magseries(stimes, smags, serrs, period,
                                         knotfraction=fitknotfrac)
            epoch = spfit['fitinfo']['fitepoch']
            if len(epoch) != 1:
                epoch = epoch[0]
        except Exception as e:
            LOGEXCEPTION('spline fit failed, using min(times) as epoch')
            epoch = npmin(stimes)


    # now phase (and optionally, phase bin the light curve)
    if errs is not None:

        # phase the magseries
        phasedlc = phase_magseries_with_errs(stimes,
                                             smags,
                                             serrs,
                                             period,
                                             epoch,
                                             wrap=phasewrap,
                                             sort=phasesort)
        plotphase = phasedlc['phase']
        plotmags = phasedlc['mags']
        ploterrs = phasedlc['errs']

        # if we're supposed to bin the phases, do so
        if phasebin:

            binphasedlc = phase_bin_magseries_with_errs(plotphase,
                                                        plotmags,
                                                        ploterrs,
                                                        binsize=phasebin)
            plotphase = binphasedlc['binnedphases']
            plotmags = binphasedlc['binnedmags']
            ploterrs = binphasedlc['binnederrs']

    else:

        # phase the magseries
        phasedlc = phase_magseries(stimes,
                                   smags,
                                   period,
                                   epoch,
                                   wrap=phasewrap,
                                   sort=phasesort)
        plotphase = phasedlc['phase']
        plotmags = phasedlc['mags']
        ploterrs = None

        # if we're supposed to bin the phases, do so
        if phasebin:

            binphasedlc = phase_bin_magseries(plotphase,
                                              plotmags,
                                              binsize=phasebin)
            plotphase = binphasedlc['binnedphases']
            plotmags = binphasedlc['binnedmags']
            ploterrs = None


    # finally, make the plots

    # initialize the plot
    fig = plt.figure()
    fig.set_size_inches(7.5,4.8)

    plt.errorbar(plotphase, plotmags, fmt='bo', yerr=ploterrs,
                 markersize=2.0, markeredgewidth=0.0, ecolor='#B2BEB5',
                 capsize=0)

    # make a grid
    plt.grid(color='#a9a9a9',
             alpha=0.9,
             zorder=0,
             linewidth=1.0,
             linestyle=':')

    # make lines for phase 0.0, 0.5, and -0.5
    plt.axvline(0.0,alpha=0.9,linestyle='dashed',color='g')
    plt.axvline(-0.5,alpha=0.9,linestyle='dashed',color='g')
    plt.axvline(0.5,alpha=0.9,linestyle='dashed',color='g')

    # fix the ticks to use no offsets
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    # get the yrange
    if yrange and isinstance(yrange,list) and len(yrange) == 2:
        ymin, ymax = yrange
    else:
        ymin, ymax = plt.ylim()

    # set the y axis labels and range
    if not magsarefluxes:
        plt.ylim(ymax, ymin)
        yaxlabel = 'magnitude'
    else:
        plt.ylim(ymin, ymax)
        yaxlabel = 'flux'

    # set the x axis limit
    if not plotphaselim:
        plot_xlim = plt.xlim()
        plt.xlim((npmin(plotphase)-0.1,
                  npmax(plotphase)+0.1))
    else:
        plt.xlim((plotphaselim[0],plotphaselim[1]))

    # set up the axis labels and plot title
    plt.xlabel('phase')
    plt.ylabel(yaxlabel)
    plt.title('period: %.6f d - epoch: %.6f' % (period, epoch))

    LOGINFO('using period: %.6f d and epoch: %.6f' % (period, epoch))

    # make the figure
    if outfile and isinstance(outfile, str):

        if outfile.endswith('.png'):
            plt.savefig(outfile, bbox_inches='tight', dpi=plotdpi)
        else:
            plt.savefig(outfile, bbox_inches='tight')
        plt.close()
        return period, epoch, os.path.abspath(outfile)

    elif dispok:

        plt.show()
        plt.close()
        return period, epoch

    else:

        LOGWARNING('no output file specified and no $DISPLAY set, '
                   'saving to magseries-phased-plot.png in current directory')
        outfile = 'magseries-phased-plot.png'
        plt.savefig(outfile, bbox_inches='tight', dpi=plotdpi)
        plt.close()
        return period, epoch, os.path.abspath(outfile)



###################
## OBJECT STAMPS ##
###################

def astroquery_skyview_stamp(
        ra, decl, survey='DSS2 Red',
        flip=True,
        convolvewith=None
):
    '''This uses astroquery's SkyView connector to get stamps.

    flip = True will flip the image top to bottom.

    if convolvewith is an astropy.convolution kernel:

    http://docs.astropy.org/en/stable/convolution/kernels.html

    this will return the stamp convolved with that kernel. This can be useful to
    see effects of wide-field telescopes (like the HATNet and HATSouth lenses)
    degrading the nominal 1 arcsec/px of DSS, causing blending of targets and
    any variability.


    '''

    position = '{ra:.3f}d{decl:+.3f}d'.format(ra=ra,decl=decl)

    imglist = SkyView.get_images(position=position,
                                 survey=[survey],
                                 coordinates='J2000')

    # this frame is usually upside down (at least for DSS), flip it if asked for
    frame = imglist[0][0].data

    if flip:
        frame = np.flipud(frame)

    for x in imglist:
        x.close()

    if convolvewith:
        convolved = aconv.convolve(frame, convolvewith)
        return frame

    else:
        return frame



def get_dss_stamp(ra, decl, outfile, stampsize=5.0):
    '''This gets a DSS stamp from the HAT data server.

    These are a bit nicer than the DSS stamps direct from STScI because they
    have crosshairs and annotations.

    '''

    stampsurl = (
        "https://hatsurveys.org/lightcurves/stamps/direct?coords={ra},{decl}"
        "&stampsize={stampsize}"
        ).format(ra=ra,
                 decl=decl,
                 stampsize=stampsize)

    downloaded, msg = urlretrieve(stampsurl, outfile)

    return downloaded


##################
## PERIODOGRAMS ##
##################

PLOTYLABELS = {'gls':'Generalized Lomb-Scargle normalized power',
               'pdm':'Stellingwerf PDM $\Theta$',
               'aov':'Schwarzenberg-Cerny AoV $\Theta$',
               'bls':'Box Least-squared Search SR',
               'sls':'Lomb-Scargle normalized power'}

METHODLABELS = {'gls':'Generalized Lomb-Scargle periodogram',
                'pdm':'Stellingwerf phase-dispersion minimization',
                'aov':'Schwarzenberg-Cerny analysis of variance',
                'bls':'Box Least-squared Search',
                'sls':'Lomb-Scargle periodogram (Scipy)'}

METHODSHORTLABELS = {'gls':'Generalized L-S',
                     'pdm':'Stellingwerf PDM',
                     'aov':'Schwarzenberg-Cerny AoV',
                     'bls':'BLS',
                     'sls':'L-S (Scipy)'}


def plot_periodbase_lsp(lspinfo, outfile=None, plotdpi=100):

    '''Makes a plot of periodograms obtained from periodbase functions.

    If lspinfo is a dictionary, uses the information directly. If it's a
    filename string ending with .pkl, then this assumes it's a periodbase LSP
    pickle and loads the corresponding info from it.

    '''

    # get the lspinfo from a pickle file transparently
    if isinstance(lspinfo,str) and os.path.exists(lspinfo):
        LOGINFO('loading LSP info from pickle %s' % lspinfo)
        with open(lspinfo,'rb') as infd:
            lspinfo = pickle.load(infd)

    try:

        # get the things to plot out of the data
        periods = lspinfo['periods']
        lspvals = lspinfo['lspvals']
        bestperiod = lspinfo['bestperiod']
        lspmethod = lspinfo['method']

        # make the LSP plot on the first subplot
        plt.plot(periods, lspvals)
        plt.xscale('log',basex=10)
        plt.xlabel('Period [days]')
        plt.ylabel(PLOTYLABELS[lspmethod])
        plottitle = '%s best period: %.6f d' % (METHODSHORTLABELS[lspmethod],
                                                bestperiod)
        plt.title(plottitle)

        # show the best five peaks on the plot
        for bestperiod, bestpeak in zip(lspinfo['nbestperiods'],
                                        lspinfo['nbestlspvals']):

            plt.annotate('%.6f' % bestperiod,
                         xy=(bestperiod, bestpeak), xycoords='data',
                         xytext=(0.0,25.0), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"),fontsize='x-small')

        # make a grid
        plt.grid(color='#a9a9a9',
                 alpha=0.9,
                 zorder=0,
                 linewidth=1.0,
                 linestyle=':')

        # make the figure
        if outfile and isinstance(outfile, str):

            if outfile.endswith('.png'):
                plt.savefig(outfile,bbox_inches='tight',dpi=plotdpi)
            else:
                plt.savefig(outfile,bbox_inches='tight')

            plt.close()
            return os.path.abspath(outfile)

        elif dispok:

            plt.show()
            plt.close()
            return

        else:

            LOGWARNING('no output file specified and no $DISPLAY set, '
                       'saving to lsp-plot.png in current directory')
            outfile = 'lsp-plot.png'
            plt.savefig(outfile,bbox_inches='tight',dpi=plotdpi)
            plt.close()
            return os.path.abspath(outfile)

    except Exception as e:

        LOGEXCEPTION('could not plot this LSP, appears to be empty')
        return
