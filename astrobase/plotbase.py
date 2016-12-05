#!/usr/bin/env python

'''
plotbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2016
License: MIT.

Contains various useful functions for plotting light curves and associated data.


'''
import os
import os.path

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

from urllib import urlretrieve

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
    normalize_magseries

from .varbase import spline_fit_magseries

from .coordutils import total_proper_motion, reduced_proper_motion

#########################
## SIMPLE LIGHT CURVES ##
#########################

def plot_mag_series(times,
                    mags,
                    errs=None,
                    outfile=None,
                    sigclip=30.0,
                    normto='globalmedian',
                    normmingap=4.0,
                    timebin=None,
                    yrange=None):
    '''This plots a magnitude time series.

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
    series.

    '''

    if errs is not None:

        # remove nans
        find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
        ftimes, fmags, ferrs = times[find], mags[find], errs[find]

        # get the median and stdev = 1.483 x MAD
        median_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

        # sigclip next for a single sigclip value
        if sigclip and isinstance(sigclip,float):

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

            LOGINFO('sigclip = %s: before = %s observations, '
                    'after = %s observations' %
                    (sigclip, len(times), len(stimes)))

        # this handles sigclipping for asymmetric +ve and -ve clip values
        elif sigclip and isinstance(sigclip,list) and len(sigclip) == 2:

            sigposind = (fmags - median_mag) < (sigclip[0]*stddev_mag)
            signegind = (fmags - median_mag) > (sigclip[1]*stddev_mag)
            sigind = sigposind & signegind

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

    else:

        # remove nans
        find = npisfinite(times) & npisfinite(mags)
        ftimes, fmags, ferrs = times[find], mags[find], None

        # get the median and stdev = 1.483 x MAD
        median_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = None

            LOGINFO('sigclip = %s: before = %s observations, '
                    'after = %s observations' %
                    (sigclip, len(times), len(stimes)))

        else:

            stimes = ftimes
            smags = fmags
            serrs = None

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

    # finally, proceed with plotting
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

    # get the yrange
    if yrange and isinstance(yrange,list) and len(yrange) == 2:
        ymin, ymax = yrange
    else:
        ymin, ymax = plt.ylim()
    plt.ylim(ymax,ymin)

    plt.xlim(npmin(btimes) - 0.001*npmin(btimes),
             npmax(btimes) + 0.001*npmin(btimes))

    plt.xlabel('time [JD]')
    plt.ylabel('magnitude')

    if outfile and isinstance(outfile, str):

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
        plt.savefig(outfile,bbox_inches='tight')
        plt.close()
        return os.path.abspath(outfile)



#########################
## PHASED LIGHT CURVES ##
#########################

def plot_phased_mag_series(times,
                           mags,
                           period,
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
                           yrange=None):
    '''This plots a phased magnitude time series using the period provided.

    If epoch is None, uses the min(times) as the epoch.

    If epoch is a string 'min', then fits a cubic spline to the phased light
    curve using min(times), finds the magnitude minimum from the fitted light
    curve, then uses the corresponding time value as the epoch.

    If epoch is a float, then uses that directly to phase the light curve and as
    the epoch of the phased mag series plot.

    If outfile is none, then plots to matplotlib interactive window. If outfile
    is a string denoting a filename, uses that to write a png/eps/pdf figure.

    '''

    if errs is not None:

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

    else:

        # remove nans
        find = npisfinite(times) & npisfinite(mags)
        ftimes, fmags, ferrs = times[find], mags[find], None

        # get the median and stdev = 1.483 x MAD
        median_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = None

            LOGINFO('sigclip = %s: before = %s observations, '
                    'after = %s observations' %
                    (sigclip, len(times), len(stimes)))

        else:

            stimes = ftimes
            smags = fmags
            serrs = None


    # check if we need to normalize
    if normto is not False:
        stimes, smags = normalize_magseries(stimes, smags,
                                            normto=normto,
                                            mingap=normmingap)

    # figure out the epoch, if it's None, use the min of the time
    if epoch is None:

        epoch = npmin(stimes)

    # if the epoch is 'min', then fit a spline to the light curve phased
    # using the min of the time, find the fit mag minimum and use the time for
    # that as the epoch
    elif isinstance(epoch,str) and epoch == 'min':

        try:
            spfit = spline_fit_magseries(stimes, smags, serrs, period,
                                         knotfraction=fitknotfrac)
            epoch = spfit['fitepoch']
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
    plt.ylim(ymax,ymin)

    # set the x axis limit
    if not plotphaselim:
        plot_xlim = plt.xlim()
        plt.xlim((npmin(plotphase)-0.1,
                  npmax(plotphase)+0.1))
    else:
        plt.xlim((plotphaselim[0],plotphaselim[1]))

    # set up the labels
    plt.xlabel('phase')
    plt.ylabel('magnitude')
    plt.title('using period: %.6f d and epoch: %.6f' % (period, epoch))

    LOGINFO('using period: %.6f d and epoch: %.6f' % (period, epoch))

    # make the figure
    if outfile and isinstance(outfile, str):

        plt.savefig(outfile,bbox_inches='tight')
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
        plt.savefig(outfile,bbox_inches='tight')
        plt.close()
        return period, epoch, os.path.abspath(outfile)


###################
## OBJECT STAMPS ##
###################

def astroquery_skyview_stamp(ra, decl, survey='DSS2 Red',
                             flip=True,
                             convolvewith=None
):
    '''This uses astroquery's SkyView connector to get stamps.

    flip = True will flip the image top to bottom.

    if convolvwith is an astropy.convolution kernel:

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

    These are a bit nicer than the DSS stamps from STScI because they have
    crosshairs and annotations.

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

def plot_periodbase_lsp(lspinfo, outfile=None):

    '''Makes a plot of the L-S periodogram obtained from periodbase functions.

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

        # make the LSP plot on the first subplot
        plt.plot(periods,lspvals)

        plt.xscale('log',basex=10)
        plt.xlabel('Period [days]')
        plt.ylabel('LSP power')
        plottitle = 'best period = %.6f d' % bestperiod
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
            plt.savefig(outfile,bbox_inches='tight')
            plt.close()
            return os.path.abspath(outfile)

    except Exception as e:

        LOGEXCEPTION('could not plot this LSP, appears to be empty')
        return



def make_checkplot(lspinfo,
                   times,
                   mags,
                   errs,
                   objectinfo=None,
                   findercmap='gray_r',
                   normto='globalmedian',
                   normmingap=4.0,
                   outfile=None,
                   sigclip=4.0,
                   varepoch='min',
                   phasewrap=True,
                   phasesort=True,
                   phasebin=0.002,
                   plotxlim=[-0.8,0.8]):
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
    obtained from Lomb-Scargle or BLS. The keys 'nbestperiods' and
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
                     0.023239128705778048]}

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

    '''

    if not outfile and isinstance(lspinfo,str):
        # generate the plot filename
        plotfpath = 'phasedlc-checkplot-%s.png' % (
            lspinfo,
        )
    elif outfile:
        plotfpath = outfile
    else:
        plotfpath = 'checkplot.png'

    # get the lspinfo from a pickle file transparently
    if isinstance(lspinfo,str) and os.path.exists(lspinfo):
        LOGINFO('loading LSP info from pickle %s' % lspinfo)
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

    elif ('periods' in lspinfo and
          'strlens' in lspinfo and
          'bestperiod' in lspinfo):

        periods = lspinfo['periods']
        lspvals = lspinfo['strlens']
        bestperiod = lspinfo['bestperiod']
        nbestperiods = lspinfo['nbestperiods'].tolist()
        nbestlspvals = lspinfo['nbeststrlens'].tolist()

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

    # make the LSP plot on the first subplot
    axes[0].plot(periods,lspvals)

    axes[0].set_xscale('log',basex=10)
    axes[0].set_xlabel('Period [days]')
    axes[0].set_ylabel('LSP power')
    plottitle = '%.6f d' % bestperiod
    axes[0].set_title(plottitle)

    # show the best five peaks on the plot
    for bestperiod, bestpeak in zip(nbestperiods,
                                    nbestlspvals):
        axes[0].annotate('%.6f' % bestperiod,
                         xy=(bestperiod, bestpeak), xycoords='data',
                         xytext=(0.0,25.0), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->"),fontsize='14.0')

    # make a grid
    axes[0].grid(color='#a9a9a9',
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

        # figure out dss stamp output path
        if not outfile:
            dsspath = 'dss-stamp-%s.jpg' % objectid
        else:
            dsspath = 'dss-stamp-%s.jpg' % outfile.rstrip('.png')


        LOGINFO('adding in object information for %s' % objectid)

        # calculate colors
        if ('bmag' in objectinfo and 'vmag' in objectinfo and
            'jmag' in objectinfo and 'kmag' in objectinfo and
            'sdssi' in objectinfo and
            objectinfo['bmag'] and objectinfo['vmag'] and
            objectinfo['jmag'] and objectinfo['kmag'] and
            objectinfo['sdssg']):
            bvcolor = objectinfo['bmag'] - objectinfo['vmag']
            jkcolor = objectinfo['jmag'] - objectinfo['kmag']
            ijcolor = objectinfo['sdssi'] - objectinfo['jmag']
        else:
            bvcolor = None
            jkcolor = None
            ijcolor = None

        # bump the ylim of the LSP plot so that the overplotted finder and
        # objectinfo can fit in this axes plot
        lspylim = axes[0].get_ylim()
        axes[0].set_ylim(lspylim[0], lspylim[1]+0.75*lspylim[1])

        # get the stamp
        try:
            dss = astroquery_skyview_stamp(objectinfo['ra'],objectinfo['decl'])
            stamp = dss

            # inset plot it on the current axes
            from mpl_toolkits.axes_grid.inset_locator import inset_axes
            inset = inset_axes(axes[0], width="40%", height="40%", loc=1)
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
        axes[0].text(
            0.05,0.95,
            '%s' % objectid,
            ha='left',va='center',transform=axes[0].transAxes,
            fontsize=18.0
        )

        axes[0].text(
            0.05,0.91,
            'RA = %.3f, DEC = %.3f' % (objectinfo['ra'], objectinfo['decl']),
            ha='left',va='center',transform=axes[0].transAxes,
            fontsize=18.0
        )

        if bvcolor:
            axes[0].text(0.05,0.87,
                         '$B - V$ = %.3f, $V$ = %.3f' % (bvcolor,
                                                         objectinfo['vmag']),
                         ha='left',va='center',transform=axes[0].transAxes,
                         fontsize=18.0)
        elif 'vmag' in objectinfo and objectinfo['vmag']:
            axes[0].text(0.05,0.87,
                         '$V$ = %.3f' % (objectinfo['vmag'],),
                         ha='left',va='center',transform=axes[0].transAxes,
                         fontsize=18.0)

        if ijcolor:
            axes[0].text(0.05,0.83,
                         '$i - J$ = %.3f, $J$ = %.3f' % (ijcolor,
                                                         objectinfo['jmag']),
                         ha='left',va='center',transform=axes[0].transAxes,
                         fontsize=18.0)
        elif 'jmag' in objectinfo and objectinfo['jmag']:
            axes[0].text(0.05,0.83,
                         '$J$ = %.3f' % (objectinfo['jmag'],),
                         ha='left',va='center',transform=axes[0].transAxes,
                         fontsize=18.0)

        if jkcolor:
            axes[0].text(0.05,0.79,
                         '$J - K$ = %.3f, $K$ = %.3f' % (jkcolor,
                                                         objectinfo['kmag']),
                         ha='left',va='center',transform=axes[0].transAxes,
                         fontsize=18.0)
        elif 'kmag' in objectinfo and objectinfo['kmag']:
            axes[0].text(0.05,0.79,
                         '$K$ = %.3f' % (objectinfo['kmag'],),
                         ha='left',va='center',transform=axes[0].transAxes,
                         fontsize=18.0)

        if 'sdssr' in objectinfo and objectinfo['sdssr']:
            axes[0].text(0.05,0.75,'SDSS $r$ = %.3f' % objectinfo['sdssr'],
                         ha='left',va='center',transform=axes[0].transAxes,
                         fontsize=18.0)

        # add in proper motion stuff if available in objectinfo
        if ('pmra' in objectinfo and objectinfo['pmra'] and
            'pmdecl' in objectinfo and objectinfo['pmdecl']):

            pm = total_proper_motion(objectinfo['pmra'],
                                     objectinfo['pmdecl'],
                                     objectinfo['decl'])

            axes[0].text(0.05,0.67,'$\mu$ = %.2f mas yr$^{-1}$' % pm,
                         ha='left',va='center',transform=axes[0].transAxes,
                         fontsize=18.0)

            if 'jmag' in objectinfo and objectinfo['jmag']:

                rpm = reduced_proper_motion(objectinfo['jmag'],pm)
                axes[0].text(0.05,0.63,'$H_J$ = %.2f' % rpm,
                             ha='left',va='center',transform=axes[0].transAxes,
                             fontsize=18.0)


        # once done with adding objectinfo, delete the downloaded stamp
        if os.path.exists(dsspath):
            os.remove(dsspath)

    # end of adding in objectinfo

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

        scaledplottime = stimes - npmin(stimes)

        axes[1].scatter(scaledplottime,
                        smags,
                        marker='o',
                        s=2,
                        color='green')

        # flip y axis for mags
        plot_ylim = axes[1].get_ylim()
        axes[1].set_ylim((plot_ylim[1], plot_ylim[0]))

        # set the x axis limit
        plot_xlim = axes[1].get_xlim()
        axes[1].set_xlim((npmin(scaledplottime)-1.0,
                          npmax(scaledplottime)+1.0))

        # make a grid
        axes[1].grid(color='#a9a9a9',
                     alpha=0.9,
                     zorder=0,
                     linewidth=1.0,
                     linestyle=':')

       # make the x and y axis labels
        plot_xlabel = 'JD - %.3f' % npmin(stimes)
        plot_ylabel = 'magnitude'

        axes[1].set_xlabel(plot_xlabel)
        axes[1].set_ylabel(plot_ylabel)

        # fix the yaxis ticks (turns off offset and uses the full
        # value of the yaxis tick)
        axes[1].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[1].get_xaxis().get_major_formatter().set_useOffset(False)


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
                    spfit = spline_fit_magseries(stimes, smags, serrs, varperiod)
                    varepoch = spfit['fitepoch']
                    if len(varepoch) != 1:
                        varepoch = varepoch[0]
                except Exception as e:
                    LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                    varepoch = npmin(stimes)

            LOGINFO('plotting phased LC with period %.6f, epoch %.5f' %
                    (varperiod, varepoch))

            #########################################
            ## PLOT 3 is the best-period phased LC ##
            #########################################

            # make sure the best period phased LC plot stands out
            if periodind == 0:
                axes[periodind+2].set_axis_bgcolor('#adff2f')

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
            axes[periodind+2].scatter(plotphase,
                                      plotmags,
                                      marker='o',
                                      s=2,
                                      color='gray')

            # overlay the binned phased LC plot if we're making one
            if phasebin:
                axes[periodind+2].scatter(binplotphase,
                                          binplotmags,
                                          marker='o',
                                          s=20,
                                          color='blue')

            # flip y axis for mags
            plot_ylim = axes[periodind+2].get_ylim()
            axes[periodind+2].set_ylim((plot_ylim[1], plot_ylim[0]))

            # set the x axis limit
            if not plotxlim:
                plot_xlim = axes[periodind+2].get_xlim()
                axes[periodind+2].set_xlim((npmin(plotphase)-0.1,
                                            npmax(plotphase)+0.1))
            else:
                axes[periodind+2].set_xlim((plotxlim[0],plotxlim[1]))

            # make a grid
            axes[periodind+2].grid(color='#a9a9a9',
                                   alpha=0.9,
                                   zorder=0,
                                   linewidth=1.0,
                                   linestyle=':')

           # make the x and y axis labels
            plot_xlabel = 'phase'
            plot_ylabel = 'magnitude'

            axes[periodind+2].set_xlabel(plot_xlabel)
            axes[periodind+2].set_ylabel(plot_ylabel)

            # fix the yaxis ticks (turns off offset and uses the full
            # value of the yaxis tick)
            axes[
                periodind+2
            ].get_yaxis().get_major_formatter().set_useOffset(False)
            axes[
                periodind+2
            ].get_xaxis().get_major_formatter().set_useOffset(False)

            # make the plot title
            if periodind == 0:
                plottitle = 'best period -> %.6f d - epoch %.5f' % (
                    varperiod,
                    varepoch
                )
            elif periodind == 1:
                plottitle = 'best period x 0.5 -> %.6f d - epoch %.5f' % (
                    varperiod,
                    varepoch
                )
            elif periodind == 2:
                plottitle = 'best period x 2 ->  %.6f d - epoch %.5f' % (
                    varperiod,
                    varepoch
                )
            else:
                plottitle = 'LSP peak %s -> %.6f d - epoch %.5f' % (
                    periodind-1,
                    varperiod,
                    varepoch
                )

            axes[periodind+2].set_title(plottitle)

        # end of plotting for each ax
        # save the plot to disk
        fig.set_tight_layout(True)
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
                ('no best aperture %s light curve available' %
                 ylabelcol[0].upper()),
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[periodind+2].transAxes
            )

        fig.set_tight_layout(True)
        fig.savefig(plotfpath)
        plt.close()

        LOGINFO('checkplot done -> %s' % plotfpath)
        return plotfpath
