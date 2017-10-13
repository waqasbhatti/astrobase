#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''saov.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Contains the Schwarzenberg-Czerny Analysis of Variance period-search algorithm
implementation for periodbase.

'''


from multiprocessing import Pool, cpu_count
import logging
from datetime import datetime
from traceback import format_exc

import numpy as np

# import these to avoid lookup overhead
from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, \
    arctan as nparctan, nanargmax as npnanargmax, nanargmin as npnanargmin, \
    empty as npempty, ceil as npceil, mean as npmean, \
    digitize as npdigitize, unique as npunique, \
    argmax as npargmax, argmin as npargmin


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.saov' % parent_name)

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

from ..lcmath import phase_magseries, sigclip_magseries, time_bin_magseries, \
    phase_bin_magseries

from . import get_frequency_grid


############
## CONFIG ##
############

NCPUS = cpu_count()


#####################################################
## ANALYSIS of VARIANCE (Schwarzenberg-Czerny 1989) ##
#####################################################

def aov_theta(times, mags, errs, frequency,
              binsize=0.05, minbin=9):
    '''Calculates the Schwarzenberg-Czerny AoV statistic at a test frequency.

    '''

    period = 1.0/frequency
    fold_time = times[0]

    phased = phase_magseries(times,
                             mags,
                             period,
                             fold_time,
                             wrap=False,
                             sort=True)

    phases = phased['phase']
    pmags = phased['mags']
    bins = np.arange(0.0, 1.0, binsize)
    nbins = bins.size
    ndets = phases.size

    binnedphaseinds = npdigitize(phases, bins)

    bin_s1_tops = []
    bin_s2_tops = []
    binndets = []
    goodbins = 0

    all_xbar = npmedian(pmags)

    for x in npunique(binnedphaseinds):

        thisbin_inds = binnedphaseinds == x
        thisbin_phases = phases[thisbin_inds]
        thisbin_mags = pmags[thisbin_inds]

        if thisbin_mags.size > minbin:

            thisbin_ndet = thisbin_mags.size
            thisbin_xbar = npmedian(thisbin_mags)

            # get s1
            thisbin_s1_top = (
                thisbin_ndet *
                (thisbin_xbar - all_xbar) *
                (thisbin_xbar - all_xbar)
            )

            # get s2
            thisbin_s2_top = npsum((thisbin_mags - all_xbar) *
                                   (thisbin_mags - all_xbar))

            bin_s1_tops.append(thisbin_s1_top)
            bin_s2_tops.append(thisbin_s2_top)
            binndets.append(thisbin_ndet)
            goodbins = goodbins + 1


    # turn the quantities into arrays
    bin_s1_tops = nparray(bin_s1_tops)
    bin_s2_tops = nparray(bin_s2_tops)
    binndets = nparray(binndets)

    # calculate s1 first
    s1 = npsum(bin_s1_tops)/(goodbins - 1.0)

    # then calculate s2
    s2 = npsum(bin_s2_tops)/(ndets - goodbins)

    theta_aov = s1/s2

    return theta_aov



def aov_worker(task):
    '''
    This is a parallel worker for the function below.

    task[0] = times
    task[1] = mags
    task[2] = errs
    task[3] = frequency
    task[4] = binsize
    task[5] = minbin

    '''

    times, mags, errs, frequency, binsize, minbin = task

    try:

        theta = aov_theta(times, mags, errs, frequency,
                          binsize=binsize, minbin=minbin)

        return theta

    except Exception as e:

        return npnan



def aov_periodfind(times,
                   mags,
                   errs,
                   magsarefluxes=False,
                   autofreq=True,
                   startp=None,
                   endp=None,
                   normalize=True,
                   stepsize=1.0e-4,
                   phasebinsize=0.05,
                   mindetperbin=9,
                   nbestpeaks=5,
                   periodepsilon=0.1, # 0.1
                   sigclip=10.0,
                   nworkers=None,
                   verbose=True):
    '''This runs a parallel AoV period search.

    NOTE: normalize = True here as recommended by Schwarzenberg-Czerny 1996,
    i.e. mags will be normalized to zero and rescaled so their variance = 1.0

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # get the frequencies to use
        if startp:
            endf = 1.0/startp
        else:
            # default start period is 0.1 day
            endf = 1.0/0.1

        if endp:
            startf = 1.0/endp
        else:
            # default end period is length of time series
            startf = 1.0/(stimes.max() - stimes.min())

        # if we're not using autofreq, then use the provided frequencies
        if not autofreq:
            frequencies = np.arange(startf, endf, stepsize)
            if verbose:
                LOGINFO(
                    'using %s frequency points, start P = %.3f, end P = %.3f' %
                    (frequencies.size, 1.0/endf, 1.0/startf)
                )
        else:
            # this gets an automatic grid of frequencies to use
            frequencies = get_frequency_grid(stimes,
                                             minfreq=startf,
                                             maxfreq=endf)
            if verbose:
                LOGINFO(
                    'using autofreq with %s frequency points, '
                    'start P = %.3f, end P = %.3f' %
                    (frequencies.size,
                     1.0/frequencies.max(),
                     1.0/frequencies.min())
                )

        # map to parallel workers
        if (not nworkers) or (nworkers > NCPUS):
            nworkers = NCPUS
            if verbose:
                LOGINFO('using %s workers...' % nworkers)

        pool = Pool(nworkers)

        # renormalize the working mags to zero and scale them so that the
        # variance = 1 for use with our LSP functions
        if normalize:
            nmags = (smags - npmedian(smags))/npstd(smags)
        else:
            nmags = smags

        tasks = [(stimes, nmags, serrs, x, phasebinsize, mindetperbin)
                 for x in frequencies]

        lsp = pool.map(aov_worker, tasks)

        pool.close()
        pool.join()
        del pool

        lsp = nparray(lsp)
        periods = 1.0/frequencies

        # find the nbestpeaks for the periodogram: 1. sort the lsp array by
        # highest value first 2. go down the values until we find five
        # values that are separated by at least periodepsilon in period

        # make sure to filter out non-finite values
        finitepeakind = npisfinite(lsp)
        finlsp = lsp[finitepeakind]
        finperiods = periods[finitepeakind]

        # make sure that finlsp has finite values before we work on it
        try:

            bestperiodind = npargmax(finlsp)

        except ValueError:

            LOGERROR('no finite periodogram values '
                     'for this mag series, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'method':'aov',
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'normalize':normalize,
                              'phasebinsize':phasebinsize,
                              'mindetperbin':mindetperbin,
                              'autofreq':autofreq,
                              'periodepsilon':periodepsilon,
                              'nbestpeaks':nbestpeaks,
                              'sigclip':sigclip}}

        sortedlspind = np.argsort(finlsp)[::-1]
        sortedlspperiods = finperiods[sortedlspind]
        sortedlspvals = finlsp[sortedlspind]

        prevbestlspval = sortedlspvals[0]
        # now get the nbestpeaks
        nbestperiods, nbestlspvals, peakcount = (
            [finperiods[bestperiodind]],
            [finlsp[bestperiodind]],
            1
        )
        prevperiod = sortedlspperiods[0]

        # find the best nbestpeaks in the lsp and their periods
        for period, lspval in zip(sortedlspperiods, sortedlspvals):

            if peakcount == nbestpeaks:
                break
            perioddiff = abs(period - prevperiod)
            bestperiodsdiff = [abs(period - x) for x in nbestperiods]

            # print('prevperiod = %s, thisperiod = %s, '
            #       'perioddiff = %s, peakcount = %s' %
            #       (prevperiod, period, perioddiff, peakcount))

            # this ensures that this period is different from the last
            # period and from all the other existing best periods by
            # periodepsilon to make sure we jump to an entire different peak
            # in the periodogram
            if (perioddiff > periodepsilon and
                all(x > periodepsilon for x in bestperiodsdiff)):
                nbestperiods.append(period)
                nbestlspvals.append(lspval)
                peakcount = peakcount + 1

            prevperiod = period


        return {'bestperiod':finperiods[bestperiodind],
                'bestlspval':finlsp[bestperiodind],
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':nbestlspvals,
                'nbestperiods':nbestperiods,
                'lspvals':lsp,
                'periods':periods,
                'method':'aov',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'phasebinsize':phasebinsize,
                          'mindetperbin':mindetperbin,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}

    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'aov',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'phasebinsize':phasebinsize,
                          'mindetperbin':mindetperbin,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}
