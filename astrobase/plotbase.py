#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plotbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2016
# License: MIT.

'''
Contains various useful functions for plotting light curves and associated data.

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
import pickle
from io import BytesIO as Strio

import numpy as np
from numpy import min as npmin, max as npmax

# FIXME: enforce no display for now
import matplotlib
matplotlib.use('Agg')
dispok = False

import matplotlib.axes
import matplotlib.pyplot as plt

# for convolving DSS stamps to simulate seeing effects
import astropy.convolution as aconv

from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.visualization import (
    ZScaleInterval,
    ImageNormalize,
    LinearStretch
)


###################
## LOCAL IMPORTS ##
###################

from .lcmath import phase_magseries, phase_magseries_with_errs, \
    phase_bin_magseries, phase_bin_magseries_with_errs, \
    time_bin_magseries, time_bin_magseries_with_errs, sigclip_magseries, \
    normalize_magseries, find_lc_timegroups

from .lcfit.nonphysical import spline_fit_magseries

from .services.skyview import get_stamp


#########################
## SIMPLE LIGHT CURVES ##
#########################

def plot_magseries(times,
                   mags,
                   magsarefluxes=False,
                   errs=None,
                   out=None,
                   sigclip=30.0,
                   normto='globalmedian',
                   normmingap=4.0,
                   timebin=None,
                   yrange=None,
                   segmentmingap=100.0,
                   plotdpi=100):
    '''This plots a magnitude/flux time-series.

    Parameters
    ----------

    times,mags : np.array
        The mag/flux time-series to plot as a function of time.

    magsarefluxes : bool
        Indicates if the input `mags` array is actually an array of flux
        measurements instead of magnitude measurements. If this is set to True,
        then the plot y-axis will be set as appropriate for mag or fluxes. In
        addition:

        - if `normto` is 'zero', then the median flux is divided from each
          observation's flux value to yield normalized fluxes with 1.0 as the
          global median.
        - if `normto` is 'globalmedian', then the global median flux value
          across the entire time series is multiplied with each measurement.
        - if `norm` is set to a `float`, then this number is multiplied with the
          flux value for each measurement.

    errs : np.array or None
        If this is provided, contains the measurement errors associated with
        each measurement of flux/mag in time-series. Providing this kwarg will
        add errbars to the output plot.

    out : str or StringIO/BytesIO object or None
        Sets the output type and target:

        - If `out` is a string, will save the plot to the specified file name.
        - If `out` is a StringIO/BytesIO object, will save the plot to that file
          handle. This can be useful to carry out additional operations on the
          output binary stream, or convert it to base64 text for embedding in
          HTML pages.
        - If `out` is None, will save the plot to a file called
          'magseries-plot.png' in the current working directory.

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

    normto : {'globalmedian', 'zero'} or a float
        Sets the normalization target::

          'globalmedian' -> norms each mag to the global median of the LC column
          'zero'         -> norms each mag to zero
          a float        -> norms each mag to this specified float value.

    normmingap : float
        This defines how much the difference between consecutive measurements is
        allowed to be to consider them as parts of different timegroups. By
        default it is set to 4.0 days.

    timebin : float or None
        The bin size to use to group together measurements closer than this
        amount in time. This is in seconds. If this is None, no time-binning
        will be performed.

    yrange : list of two floats or None
        This is used to provide a custom y-axis range to the plot. If None, will
        automatically determine y-axis range.

    segmentmingap : float or None
        This controls the minimum length of time (in days) required to consider
        a timegroup in the light curve as a separate segment. This is useful
        when the light curve consists of measurements taken over several
        seasons, so there's lots of dead space in the plot that can be cut out
        to zoom in on the interesting stuff. If `segmentmingap` is not None, the
        magseries plot will be cut in this way and the x-axis will show these
        breaks.

    plotdpi : int
        Sets the resolution in DPI for PNG plots (default = 100).

    Returns
    -------

    str or BytesIO/StringIO object
        Returns based on the input:

        - If `out` is a str or None, the path to the generated plot file is
          returned.
        - If `out` is a StringIO/BytesIO object, will return the
          StringIO/BytesIO object to which the plot was written.

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
                                            magsarefluxes=magsarefluxes,
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
    if yrange and isinstance(yrange,(list,tuple)) and len(yrange) == 2:
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
        fig.set_size_inches(10,4.8)
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
        fig.text(0.5, 0.00, 'JD - %.3f (not showing gaps > %.2f d)' %
                 (btimeorigin, segmentmingap), ha='center')
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

    is_Strio = isinstance(out, Strio)

    # write the plot out to a file if requested
    if out and not is_Strio:

        if out.endswith('.png'):
            plt.savefig(out,bbox_inches='tight',dpi=plotdpi)
        else:
            plt.savefig(out,bbox_inches='tight')
        plt.close()
        return os.path.abspath(out)

    elif out and is_Strio:

        plt.savefig(out, bbox_inches='tight', dpi=plotdpi, format='png')
        return out

    elif not out and dispok:

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

def plot_phased_magseries(times,
                          mags,
                          period,
                          epoch='min',
                          fitknotfrac=0.01,
                          errs=None,
                          magsarefluxes=False,
                          normto='globalmedian',
                          normmingap=4.0,
                          sigclip=30.0,
                          phasewrap=True,
                          phasesort=True,
                          phasebin=None,
                          plotphaselim=(-0.8,0.8),
                          yrange=None,
                          xtimenotphase=False,
                          xaxlabel='phase',
                          yaxlabel=None,
                          modelmags=None,
                          modeltimes=None,
                          modelerrs=None,
                          outfile=None,
                          plotdpi=100):
    '''Plots a phased magnitude/flux time-series using the period provided.

    Parameters
    ----------

    times,mags : np.array
        The mag/flux time-series to plot as a function of phase given `period`.

    period : float
        The period to use to phase-fold the time-series. Should be the same unit
        as `times` (usually in days)

    epoch : 'min' or float or None
        This indicates how to get the epoch to use for phasing the light curve:

        - If None, uses the `min(times)` as the epoch for phasing.

        - If epoch is the string 'min', then fits a cubic spline to the phased
          light curve using `min(times)` as the initial epoch, finds the
          magnitude/flux minimum of this phased light curve fit, and finally
          uses the that time value as the epoch. This is useful for plotting
          planetary transits and eclipsing binary phased light curves so that
          phase 0.0 corresponds to the mid-center time of primary eclipse (or
          transit).

        - If epoch is a float, then uses that directly to phase the light
          curve and as the epoch of the phased mag series plot.

    fitknotfrac : float
        If `epoch='min'`, this function will attempt to fit a cubic spline to
        the phased light curve to find a time of light minimum as phase
        0.0. This kwarg sets the number of knots to generate the spline as a
        fraction of the total number of measurements in the input
        time-series. By default, this is set so that 100 knots are used to
        generate a spline for fitting the phased light curve consisting of 10000
        measurements.

    errs : np.array or None
        If this is provided, contains the measurement errors associated with
        each measurement of flux/mag in time-series. Providing this kwarg will
        add errbars to the output plot.

    magsarefluxes : bool
        Indicates if the input `mags` array is actually an array of flux
        measurements instead of magnitude measurements. If this is set to True,
        then the plot y-axis will be set as appropriate for mag or fluxes.

    normto : {'globalmedian', 'zero'} or a float
        Sets the normalization target::

          'globalmedian' -> norms each mag to the global median of the LC column
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

    plotphaselim : sequence of two floats or None
        The x-axis limits to use when making the phased light curve plot. By
        default, this is (-0.8, 0.8), which places phase 0.0 at the center of
        the plot and covers approximately two cycles in phase to make any trends
        clear.

    yrange : list of two floats or None
        This is used to provide a custom y-axis range to the plot. If None, will
        automatically determine y-axis range.

    xtimenotphase : bool
        If True, the x-axis gets units of time (multiplies phase by period).

    xaxlabel : str
        Sets the label for the x-axis.

    yaxlabel : str or None
        Sets the label for the y-axis. If this is None, the appropriate label
        will be used based on the value of the `magsarefluxes` kwarg.

    modeltimes,modelmags,modelerrs : np.array or None
        If all of these are provided, then this function will overplot the
        values of modeltimes and modelmags on top of the actual phased light
        curve. This is useful for plotting variability models on top of the
        light curve (e.g. plotting a Mandel-Agol transit model over the actual
        phased light curve. These arrays will be phased using the already
        provided period and epoch.

    outfile : str or StringIO/BytesIO or matplotlib.axes.Axes or None
        - a string filename for the file where the plot will be written.
        - a StringIO/BytesIO object to where the plot will be written.
        - a matplotlib.axes.Axes object to where the plot will be written.
        - if None, plots to 'magseries-phased-plot.png' in current dir.

    plotdpi : int
        Sets the resolution in DPI for PNG plots (default = 100).

    Returns
    -------

    str or StringIO/BytesIO or matplotlib.axes.Axes
        This returns based on the input:

        - If `outfile` is a str or None, the path to the generated plot file is
          returned.
        - If `outfile` is a StringIO/BytesIO object, will return the
          StringIO/BytesIO object to which the plot was written.
        - If `outfile` is a matplotlib.axes.Axes object, will return the Axes
          object with the plot elements added to it. One can then directly
          include this Axes object in some other Figure.

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
                                            magsarefluxes=magsarefluxes,
                                            mingap=normmingap)

        if ( isinstance(modelmags, np.ndarray) and
             isinstance(modeltimes, np.ndarray) ):

            stimes, smags = normalize_magseries(modeltimes, modelmags,
                                                normto=normto,
                                                magsarefluxes=magsarefluxes,
                                                mingap=normmingap)

    # figure out the epoch, if it's None, use the min of the time
    if epoch is None:
        epoch = stimes.min()

    # if the epoch is 'min', then fit a spline to the light curve phased
    # using the min of the time, find the fit mag minimum and use the time for
    # that as the epoch
    elif isinstance(epoch, str) and epoch == 'min':

        try:
            spfit = spline_fit_magseries(stimes, smags, serrs, period,
                                         knotfraction=fitknotfrac)
            epoch = spfit['fitinfo']['fitepoch']
            if len(epoch) != 1:
                epoch = epoch[0]
        except Exception:
            LOGEXCEPTION('spline fit failed, using min(times) as epoch')
            epoch = npmin(stimes)

    # now phase the data light curve (and optionally, phase bin the light curve)
    if errs is not None:

        phasedlc = phase_magseries_with_errs(stimes, smags, serrs, period,
                                             epoch, wrap=phasewrap,
                                             sort=phasesort)
        plotphase = phasedlc['phase']
        plotmags = phasedlc['mags']
        ploterrs = phasedlc['errs']

        # if we're supposed to bin the phases, do so
        if phasebin:

            binphasedlc = phase_bin_magseries_with_errs(plotphase, plotmags,
                                                        ploterrs,
                                                        binsize=phasebin)
            binplotphase = binphasedlc['binnedphases']
            binplotmags = binphasedlc['binnedmags']
            binploterrs = binphasedlc['binnederrs']

    else:

        phasedlc = phase_magseries(stimes, smags, period, epoch,
                                   wrap=phasewrap, sort=phasesort)
        plotphase = phasedlc['phase']
        plotmags = phasedlc['mags']
        ploterrs = None

        # if we're supposed to bin the phases, do so
        if phasebin:

            binphasedlc = phase_bin_magseries(plotphase,
                                              plotmags,
                                              binsize=phasebin)
            binplotphase = binphasedlc['binnedphases']
            binplotmags = binphasedlc['binnedmags']
            binploterrs = None

    # phase the model light curve
    modelplotphase, modelplotmags = None, None

    if ( isinstance(modelerrs,np.ndarray) and
         isinstance(modeltimes,np.ndarray) and
         isinstance(modelmags,np.ndarray) ):

        modelphasedlc = phase_magseries_with_errs(modeltimes, modelmags,
                                                  modelerrs, period, epoch,
                                                  wrap=phasewrap,
                                                  sort=phasesort)
        modelplotphase = modelphasedlc['phase']
        modelplotmags = modelphasedlc['mags']

    # note that we never will phase-bin the model (no point).
    elif ( not isinstance(modelerrs,np.ndarray) and
           isinstance(modeltimes,np.ndarray) and
           isinstance(modelmags,np.ndarray) ):

        modelphasedlc = phase_magseries(modeltimes, modelmags, period, epoch,
                                        wrap=phasewrap, sort=phasesort)
        modelplotphase = modelphasedlc['phase']
        modelplotmags = modelphasedlc['mags']

    # finally, make the plots

    # check if the outfile is actually an Axes object
    if isinstance(outfile, matplotlib.axes.Axes):
        ax = outfile

    # otherwise, it's just a normal file or StringIO/BytesIO
    else:
        fig = plt.figure()
        fig.set_size_inches(7.5,4.8)
        ax = plt.gca()

    if xtimenotphase:
        plotphase *= period

    if phasebin:
        ax.errorbar(plotphase, plotmags, fmt='o',
                    color='#B2BEB5',
                    yerr=ploterrs,
                    markersize=3.0,
                    markeredgewidth=0.0,
                    ecolor='#B2BEB5',
                    capsize=0)
        if xtimenotphase:
            binplotphase *= period
        ax.errorbar(binplotphase, binplotmags, fmt='bo', yerr=binploterrs,
                    markersize=5.0, markeredgewidth=0.0, ecolor='#B2BEB5',
                    capsize=0)

    else:
        ax.errorbar(plotphase, plotmags, fmt='ko', yerr=ploterrs,
                    markersize=3.0, markeredgewidth=0.0, ecolor='#B2BEB5',
                    capsize=0)

    if (isinstance(modelplotphase, np.ndarray) and
        isinstance(modelplotmags, np.ndarray)):

        if xtimenotphase:
            modelplotphase *= period
        ax.plot(modelplotphase, modelplotmags, zorder=5, linewidth=0.5,
                alpha=0.9, color='#181c19')

    # make a grid
    ax.grid(color='#a9a9a9',
            alpha=0.9,
            zorder=0,
            linewidth=1.0,
            linestyle=':')

    # make lines for phase 0.0, 0.5, and -0.5
    ax.axvline(0.0,alpha=0.9,linestyle='dashed',color='g')
    if not xtimenotphase:
        ax.axvline(-0.5,alpha=0.9,linestyle='dashed',color='g')
        ax.axvline(0.5,alpha=0.9,linestyle='dashed',color='g')
    else:
        ax.axvline(-period*0.5,alpha=0.9,linestyle='dashed',color='g')
        ax.axvline(period*0.5,alpha=0.9,linestyle='dashed',color='g')

    # fix the ticks to use no offsets
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    # get the yrange
    if yrange and isinstance(yrange,(list,tuple)) and len(yrange) == 2:
        ymin, ymax = yrange
    else:
        ymin, ymax = ax.get_ylim()

    # set the y axis labels and range
    if not yaxlabel:
        if not magsarefluxes:
            ax.set_ylim(ymax, ymin)
            yaxlabel = 'magnitude'
        else:
            ax.set_ylim(ymin, ymax)
            yaxlabel = 'flux'

    # set the x axis limit
    if not plotphaselim:
        ax.set_xlim((npmin(plotphase)-0.1,
                     npmax(plotphase)+0.1))
    else:
        if xtimenotphase:
            ax.set_xlim((period*plotphaselim[0],period*plotphaselim[1]))
        else:
            ax.set_xlim((plotphaselim[0],plotphaselim[1]))

    # set up the axis labels and plot title
    ax.set_xlabel(xaxlabel)
    ax.set_ylabel(yaxlabel)
    ax.set_title('period: %.6f d - epoch: %.6f' % (period, epoch))

    LOGINFO('using period: %.6f d and epoch: %.6f' % (period, epoch))

    # check if the output filename is actually an instance of StringIO
    is_Strio = isinstance(outfile, Strio)

    # make the figure
    if (outfile and
        not is_Strio and
        not isinstance(outfile, matplotlib.axes.Axes)):

        if outfile.endswith('.png'):
            fig.savefig(outfile, bbox_inches='tight', dpi=plotdpi)
        else:
            fig.savefig(outfile, bbox_inches='tight')
        plt.close()
        return period, epoch, os.path.abspath(outfile)

    elif outfile and is_Strio:

        fig.savefig(outfile, bbox_inches='tight', dpi=plotdpi, format='png')
        return outfile

    elif outfile and isinstance(outfile, matplotlib.axes.Axes):

        return outfile

    elif not outfile and dispok:

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


##########################
## PLOTTING FITS IMAGES ##
##########################

def skyview_stamp(ra, decl,
                  survey='DSS2 Red',
                  scaling='Linear',
                  sizepix=300,
                  flip=True,
                  convolvewith=None,
                  forcefetch=False,
                  cachedir='~/.astrobase/stamp-cache',
                  timeout=10.0,
                  retry_failed=False,
                  savewcsheader=True,
                  verbose=False):
    '''This downloads a DSS FITS stamp centered on the coordinates specified.

    This wraps the function :py:func:`astrobase.services.skyview.get_stamp`,
    which downloads Digitized Sky Survey stamps in FITS format from the NASA
    SkyView service:

    https://skyview.gsfc.nasa.gov/current/cgi/query.pl

    Also adds some useful operations on top of the FITS file returned.

    Parameters
    ----------

    ra,decl : float
        The center coordinates for the stamp in decimal degrees.

    survey : str
        The survey name to get the stamp from. This is one of the
        values in the 'SkyView Surveys' option boxes on the SkyView
        webpage. Currently, we've only tested using 'DSS2 Red' as the value for
        this kwarg, but the other ones should work in principle.

    scaling : str
        This is the pixel value scaling function to use. Can be any of the
        strings ("Log", "Linear", "Sqrt", "HistEq").

    sizepix : int
        Size of the requested stamp, in pixels. (DSS scale is ~1arcsec/px).

    flip : bool
        Will flip the downloaded image top to bottom. This should usually be
        True because matplotlib and FITS have different image coord origin
        conventions. Alternatively, set this to False and use the
        `origin='lower'` in any call to `matplotlib.pyplot.imshow` when plotting
        this image.

    convolvewith : astropy.convolution Kernel object or None
        If `convolvewith` is an astropy.convolution Kernel object from:

        http://docs.astropy.org/en/stable/convolution/kernels.html

        then, this function will return the stamp convolved with that
        kernel. This can be useful to see effects of wide-field telescopes (like
        the HATNet and HATSouth lenses) degrading the nominal 1 arcsec/px of
        DSS, causing blending of targets and any variability.

    forcefetch : bool
        If True, will disregard any existing cached copies of the stamp already
        downloaded corresponding to the requested center coordinates and
        redownload the FITS from the SkyView service.

    cachedir : str
        This is the path to the astrobase cache directory. All downloaded FITS
        stamps are stored here as .fits.gz files so we can immediately respond
        with the cached copy when a request is made for a coordinate center
        that's already been downloaded.

    timeout : float
        Sets the timeout in seconds to wait for a response from the NASA SkyView
        service.

    retry_failed : bool
        If the initial request to SkyView fails, and this is True, will retry
        until it succeeds.

    savewcsheader : bool
        If this is True, also returns the WCS header of the downloaded FITS
        stamp in addition to the FITS image itself. Useful for projecting object
        coordinates onto image xy coordinates for visualization.

    verbose : bool
        If True, indicates progress.

    Returns
    -------

    tuple or array or None
        This returns based on the value of `savewcsheader`:

        - If `savewcsheader=True`, returns a tuple:
          (FITS stamp image as a numpy array, FITS header)
        - If `savewcsheader=False`, returns only the FITS stamp image as numpy
          array.
        - If the stamp retrieval fails, returns None.

    '''

    stampdict = get_stamp(ra, decl,
                          survey=survey,
                          scaling=scaling,
                          sizepix=sizepix,
                          forcefetch=forcefetch,
                          cachedir=cachedir,
                          timeout=timeout,
                          retry_failed=retry_failed,
                          verbose=verbose)
    #
    # DONE WITH FETCHING STUFF
    #
    if stampdict:

        # open the frame
        stampfits = pyfits.open(stampdict['fitsfile'])
        header = stampfits[0].header
        frame = stampfits[0].data
        stampfits.close()

        # finally, we can process the frame
        if flip:
            frame = np.flipud(frame)

        if verbose:
            LOGINFO('fetched stamp successfully for (%.3f, %.3f)'
                    % (ra, decl))

        if convolvewith:

            convolved = aconv.convolve(frame, convolvewith)
            if savewcsheader:
                return convolved, header
            else:
                return convolved

        else:

            if savewcsheader:
                return frame, header
            else:
                return frame

    else:
        LOGERROR('could not fetch the requested stamp for '
                 'coords: (%.3f, %.3f) from survey: %s and scaling: %s'
                 % (ra, decl, survey, scaling))
        return None


def fits_finder_chart(
        fitsfile,
        outfile,
        fitsext=0,
        wcsfrom=None,
        scale=ZScaleInterval(),
        stretch=LinearStretch(),
        colormap=plt.cm.gray_r,
        findersize=None,
        finder_coordlimits=None,
        overlay_ra=None,
        overlay_decl=None,
        overlay_pltopts={'marker':'o',
                         'markersize':10.0,
                         'markerfacecolor':'none',
                         'markeredgewidth':2.0,
                         'markeredgecolor':'red'},
        overlay_zoomcontain=False,
        grid=False,
        gridcolor='k'
):
    '''This makes a finder chart for a given FITS with an optional object
    position overlay.

    Parameters
    ----------

    fitsfile : str
        `fitsfile` is the FITS file to use to make the finder chart.

    outfile : str
        `outfile` is the name of the output file. This can be a png or pdf or
        whatever else matplotlib can write given a filename and extension.

    fitsext : int
        Sets the FITS extension in `fitsfile` to use to extract the image array
        from.

    wcsfrom : str or None
        If `wcsfrom` is None, the WCS to transform the RA/Dec to pixel x/y will
        be taken from the FITS header of `fitsfile`. If this is not None, it
        must be a FITS or similar file that contains a WCS header in its first
        extension.

    scale : astropy.visualization.Interval object
        `scale` sets the normalization for the FITS pixel values. This is an
        astropy.visualization Interval object.
        See http://docs.astropy.org/en/stable/visualization/normalization.html
        for details on `scale` and `stretch` objects.

    stretch : astropy.visualization.Stretch object
        `stretch` sets the stretch function for mapping FITS pixel values to
        output pixel values. This is an astropy.visualization Stretch object.
        See http://docs.astropy.org/en/stable/visualization/normalization.html
        for details on `scale` and `stretch` objects.

    colormap : matplotlib Colormap object
        `colormap` is a matplotlib color map object to use for the output image.

    findersize : None or tuple of two ints
        If `findersize` is None, the output image size will be set by the NAXIS1
        and NAXIS2 keywords in the input `fitsfile` FITS header. Otherwise,
        `findersize` must be a tuple with the intended x and y size of the image
        in inches (all output images will use a DPI = 100).

    finder_coordlimits : list of four floats or None
        If not None, `finder_coordlimits` sets x and y limits for the plot,
        effectively zooming it in if these are smaller than the dimensions of
        the FITS image. This should be a list of the form: [minra, maxra,
        mindecl, maxdecl] all in decimal degrees.

    overlay_ra, overlay_decl : np.array or None
        `overlay_ra` and `overlay_decl` are ndarrays containing the RA and Dec
        values to overplot on the image as an overlay. If these are both None,
        then no overlay will be plotted.

    overlay_pltopts : dict
        `overlay_pltopts` controls how the overlay points will be plotted. This
        a dict with standard matplotlib marker, etc. kwargs as key-val pairs,
        e.g. 'markersize', 'markerfacecolor', etc. The default options make red
        outline circles at the location of each object in the overlay.

    overlay_zoomcontain : bool
        `overlay_zoomcontain` controls if the finder chart will be zoomed to
        just contain the overlayed points. Everything outside the footprint of
        these points will be discarded.

    grid : bool
        `grid` sets if a grid will be made on the output image.

    gridcolor : str
        `gridcolor` sets the color of the grid lines. This is a usual matplotib
        color spec string.

    Returns
    -------

    str or None
        The filename of the generated output image if successful. None
        otherwise.

    '''

    # read in the FITS file
    if wcsfrom is None:

        hdulist = pyfits.open(fitsfile)
        img, hdr = hdulist[fitsext].data, hdulist[fitsext].header
        hdulist.close()

        frameshape = (hdr['NAXIS1'], hdr['NAXIS2'])
        w = WCS(hdr)

    elif os.path.exists(wcsfrom):

        hdulist = pyfits.open(fitsfile)
        img, hdr = hdulist[fitsext].data, hdulist[fitsext].header
        hdulist.close()

        frameshape = (hdr['NAXIS1'], hdr['NAXIS2'])
        w = WCS(wcsfrom)

    else:

        LOGERROR('could not determine WCS info for input FITS: %s' %
                 fitsfile)
        return None

    # use the frame shape to set the output PNG's dimensions
    if findersize is None:
        fig = plt.figure(figsize=(frameshape[0]/100.0,
                                  frameshape[1]/100.0))
    else:
        fig = plt.figure(figsize=findersize)

    # set the coord limits if zoomcontain is True
    # we'll leave 30 arcseconds of padding on each side
    if (overlay_zoomcontain and
        overlay_ra is not None and
        overlay_decl is not None):

        finder_coordlimits = [overlay_ra.min()-30.0/3600.0,
                              overlay_ra.max()+30.0/3600.0,
                              overlay_decl.min()-30.0/3600.0,
                              overlay_decl.max()+30.0/3600.0]

    # set the coordinate limits if provided
    if finder_coordlimits and isinstance(finder_coordlimits, (list,tuple)):

        minra, maxra, mindecl, maxdecl = finder_coordlimits
        cntra, cntdecl = (minra + maxra)/2.0, (mindecl + maxdecl)/2.0

        pixelcoords = w.all_world2pix([[minra, mindecl],
                                       [maxra, maxdecl],
                                       [cntra, cntdecl]],1)
        x1, y1, x2, y2 = (int(pixelcoords[0,0]),
                          int(pixelcoords[0,1]),
                          int(pixelcoords[1,0]),
                          int(pixelcoords[1,1]))

        xmin = x1 if x1 < x2 else x2
        xmax = x2 if x2 > x1 else x1

        ymin = y1 if y1 < y2 else y2
        ymax = y2 if y2 > y1 else y1

        # create a new WCS with the same transform but new center coordinates
        whdr = w.to_header()
        whdr['CRPIX1'] = (xmax - xmin)/2
        whdr['CRPIX2'] = (ymax - ymin)/2
        whdr['CRVAL1'] = cntra
        whdr['CRVAL2'] = cntdecl
        whdr['NAXIS1'] = xmax - xmin
        whdr['NAXIS2'] = ymax - ymin
        w = WCS(whdr)

    else:
        xmin, xmax, ymin, ymax = 0, hdr['NAXIS2'], 0, hdr['NAXIS1']

    # add the axes with the WCS projection
    # this should automatically handle subimages because we fix the WCS
    # appropriately above for these
    fig.add_subplot(111,projection=w)

    if scale is not None and stretch is not None:

        norm = ImageNormalize(img,
                              interval=scale,
                              stretch=stretch)

        plt.imshow(img[ymin:ymax,xmin:xmax],
                   origin='lower',
                   cmap=colormap,
                   norm=norm)

    else:

        plt.imshow(img[ymin:ymax,xmin:xmax],
                   origin='lower',
                   cmap=colormap)

    # handle additional options
    if grid:
        plt.grid(color=gridcolor,ls='solid',lw=1.0)

    # handle the object overlay
    if overlay_ra is not None and overlay_decl is not None:

        our_pltopts = dict(
            transform=plt.gca().get_transform('fk5'),
            marker='o',
            markersize=10.0,
            markerfacecolor='none',
            markeredgewidth=2.0,
            markeredgecolor='red',
            rasterized=True,
            linestyle='none'
        )
        if overlay_pltopts is not None and isinstance(overlay_pltopts,
                                                      dict):
            our_pltopts.update(overlay_pltopts)

        plt.gca().set_autoscale_on(False)
        plt.gca().plot(overlay_ra, overlay_decl,
                       **our_pltopts)

    plt.xlabel('Right Ascension [deg]')
    plt.ylabel('Declination [deg]')

    # get the x and y axes objects to fix the ticks
    xax = plt.gca().coords[0]
    yax = plt.gca().coords[1]

    yax.set_major_formatter('d.ddd')
    xax.set_major_formatter('d.ddd')

    # save the figure
    plt.savefig(outfile, dpi=100.0)
    plt.close('all')

    return outfile


##################
## PERIODOGRAMS ##
##################

PLOTYLABELS = {'gls':'Generalized Lomb-Scargle normalized power',
               'pdm':r'Stellingwerf PDM $\Theta$',
               'aov':r'Schwarzenberg-Czerny AoV $\Theta$',
               'mav':r'Schwarzenberg-Czerny AoVMH $\Theta$',
               'bls':'Box Least-squared Search SR',
               'acf':'Autocorrelation Function',
               'win':'Lomb-Scargle normalized power',
               'ext':'External period-finder power',
               'tls':'Transit Least-Squares SDE'}

METHODLABELS = {'gls':'Generalized Lomb-Scargle periodogram',
                'pdm':'Stellingwerf phase-dispersion minimization',
                'aov':'Schwarzenberg-Czerny AoV',
                'mav':'Schwarzenberg-Czerny AoV multi-harmonic',
                'bls':'Box Least-squared Search',
                'acf':'McQuillan+ ACF Period Search',
                'win':'Timeseries Sampling Lomb-Scargle periodogram',
                'ext':'External period-finder periodogram',
                'tls':'Transit Least-Squares periodogram'}

METHODSHORTLABELS = {'gls':'Generalized L-S',
                     'pdm':'Stellingwerf PDM',
                     'aov':'Schwarzenberg-Czerny AoV',
                     'mav':'Schwarzenberg-Czerny AoVMH',
                     'acf':'McQuillan+ ACF',
                     'bls':'BLS',
                     'win':'Sampling L-S',
                     'ext':'External period-finder',
                     'tls':'TLS'}


def plot_periodbase_lsp(lspinfo, outfile=None, plotdpi=100):

    '''Makes a plot of periodograms obtained from `periodbase` functions.

    This takes the output dict produced by any `astrobase.periodbase`
    period-finder function or a pickle filename containing such a dict and makes
    a periodogram plot.

    Parameters
    ----------

    lspinfo : dict or str
        If lspinfo is a dict, it must be a dict produced by an
        `astrobase.periodbase` period-finder function or a dict from your own
        period-finder function or routine that is of the form below with at
        least these keys::

            {'periods': np.array of all periods searched by the period-finder,
             'lspvals': np.array of periodogram power value for each period,
             'bestperiod': a float value that is the period with the highest
                           peak in the periodogram, i.e. the most-likely actual
                           period,
             'method': a three-letter code naming the period-finder used; must
                       be one of the keys in the `METHODLABELS` dict above,
             'nbestperiods': a list of the periods corresponding to periodogram
                             peaks (`nbestlspvals` below) to annotate on the
                             periodogram plot so they can be called out
                             visually,
             'nbestlspvals': a list of the power values associated with
                             periodogram peaks to annotate on the periodogram
                             plot so they can be called out visually; should be
                             the same length as `nbestperiods` above}

        If lspinfo is a str, then it must be a path to a pickle file that
        contains a dict of the form described above.

    outfile : str or None
        If this is a str, will write the periodogram plot to the file specified
        by this string. If this is None, will write to a file called
        'lsp-plot.png' in the current working directory.

    plotdpi : int
        Sets the resolution in DPI of the output periodogram plot PNG file.

    Returns
    -------

    str
        Absolute path to the periodogram plot file created.

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
                         arrowprops=dict(arrowstyle="->"),
                         fontsize='x-small')

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

    except Exception:

        LOGEXCEPTION('could not plot this LSP, appears to be empty')
        return
