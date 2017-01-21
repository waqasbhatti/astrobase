#!/usr/bin/env python

'''
periodbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2015

Contains various useful tools for period finding.


'''


from multiprocessing import Pool, cpu_count
import ctypes
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

from scipy.signal import lombscargle, find_peaks_cwt

# experimental numba speedup stuff
try:
    from numba import jit
    HAVENUMBA = True
except:
    HAVENUMBA = False


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.periodbase' % parent_name)

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

from .lcmath import phase_magseries, sigclip_magseries, time_bin_magseries, \
    phase_bin_magseries

from .glsp import generalized_lsp_value as glspval, \
    generalized_lsp_value_notau as glspvalnt

from bls import eebls


############
## CONFIG ##
############

NCPUS = cpu_count()


#######################
## UTILITY FUNCTIONS ##
#######################

def get_frequency_grid(times,
                       samplesperpeak=5,
                       nyquistfactor=5,
                       minfreq=None,
                       maxfreq=None,
                       returnf0dfnf=False):
    '''This calculates a frequency grid for the period finding functions in this
    module.

    Based on the autofrequency function in astropy.stats.lombscargle.

    http://docs.astropy.org/en/stable/_modules/astropy/stats/lombscargle/core.html#LombScargle.autofrequency

    '''

    baseline = times.max() - times.min()
    nsamples = times.size

    df = 1. / baseline / samplesperpeak

    if minfreq is not None:
        f0 = minfreq
    else:
        f0 = 0.5 * df

    if maxfreq is not None:
        Nf = int(npceil((maxfreq - f0) / df))
    else:
        Nf = int(0.5 * samplesperpeak * nyquistfactor * nsamples)


    if returnf0dfnf:
        return f0, df, Nf, f0 + df * nparange(Nf)
    else:
        return f0 + df * nparange(Nf)


####################################################################
## PHASE DISPERSION MINIMIZATION (Stellingwerf+ 1978, 2011, 2013) ##
####################################################################

def stellingwerf_pdm_theta(times, mags, errs, frequency,
                           binsize=0.05, minbin=9):
    '''
    This calculates the Stellingwerf PDM theta value at a test frequency.

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

    binnedphaseinds = npdigitize(phases, bins)

    binvariances = []
    binndets = []
    goodbins = 0

    for x in npunique(binnedphaseinds):

        thisbin_inds = binnedphaseinds == x
        thisbin_phases = phases[thisbin_inds]
        thisbin_mags = pmags[thisbin_inds]

        if thisbin_mags.size > minbin:
            thisbin_variance = npvar(thisbin_mags,ddof=1)
            binvariances.append(thisbin_variance)
            binndets.append(thisbin_mags.size)
            goodbins = goodbins + 1

    # now calculate theta
    binvariances = nparray(binvariances)
    binndets = nparray(binndets)

    theta_top = npsum(binvariances*(binndets - 1)) / (npsum(binndets) -
                                                      goodbins)
    theta_bot = npvar(pmags,ddof=1)
    theta = theta_top/theta_bot

    return theta



def stellingwerf_pdm_worker(task):
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

        theta = stellingwerf_pdm_theta(times, mags, errs, frequency,
                                       binsize=binsize, minbin=minbin)

        return theta

    except Exception as e:

        return npnan



def stellingwerf_pdm(times,
                     mags,
                     errs,
                     autofreq=True,
                     startp=None,
                     endp=None,
                     normalize=False,
                     stepsize=1.0e-4,
                     phasebinsize=0.05,
                     mindetperbin=9,
                     nbestpeaks=5,
                     periodepsilon=0.1, # 0.1
                     sigclip=10.0,
                     nworkers=None):
    '''This runs a parallel Stellingwerf PDM period search.

    '''

    # get rid of nans first
    find = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    if len(ftimes) > 9 and len(fmags) > 9 and len(ferrs) > 9:

        # get the median and stdev = 1.483 x MAD
        median_mag = np.median(fmags)
        stddev_mag = (np.median(np.abs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (np.abs(fmags - median_mag)) < (sigclip * stddev_mag)

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
                # default end period is length of time series divided by 2
                startf = 1.0/(stimes.max() - stimes.min())

            # if we're not using autofreq, then use the provided frequencies
            if not autofreq:
                frequencies = np.arange(startf, endf, stepsize)
                LOGINFO(
                    'using %s frequency points, start P = %.3f, end P = %.3f' %
                    (frequencies.size, 1.0/endf, 1.0/startf)
                )
            else:
                # this gets an automatic grid of frequencies to use
                frequencies = get_frequency_grid(stimes,
                                                 minfreq=startf,
                                                 maxfreq=endf)
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

            lsp = pool.map(stellingwerf_pdm_worker, tasks)

            pool.close()
            pool.join()
            del pool

            lsp = nparray(lsp)
            periods = 1.0/frequencies

            # find the nbestpeaks for the periodogram: 1. sort the lsp array by
            # lowest value first 2. go down the values until we find five values
            # that are separated by at least periodepsilon in period

            # make sure to filter out the non-finite values of lsp
            finitepeakind = npisfinite(lsp)
            finlsp = lsp[finitepeakind]
            finperiods = periods[finitepeakind]


            bestperiodind = npargmin(finlsp)

            sortedlspind = np.argsort(finlsp)
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
                    'method':'pdm'}

        else:

            LOGERROR('no good detections for these times and mags, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'method':'pdm'}
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'pdm'}



###########################################################
## ANALYSIS of VARIANCE (Schwarzenberg-Cerny 1989, 1996) ##
###########################################################

def aov_theta(times, mags, errs, frequency,
              binsize=0.05, minbin=9):
    '''Calculates the Schwarzenberg-Cerny AoV statistic at a test frequency.

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
                   nworkers=None):
    '''This runs a parallel AoV period search.

    NOTE: normalize = True here as recommended by Schwarzenberg-Cerny 1996,
    i.e. mags will be normalized to zero and rescaled so their variance = 1.0

    '''

    # get rid of nans first
    find = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    if len(ftimes) > 9 and len(fmags) > 9 and len(ferrs) > 9:

        # get the median and stdev = 1.483 x MAD
        median_mag = np.median(fmags)
        stddev_mag = (np.median(np.abs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (np.abs(fmags - median_mag)) < (sigclip * stddev_mag)

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
                # default end period is length of time series divided by 2
                startf = 1.0/(stimes.max() - stimes.min())

            # if we're not using autofreq, then use the provided frequencies
            if not autofreq:
                frequencies = np.arange(startf, endf, stepsize)
                LOGINFO(
                    'using %s frequency points, start P = %.3f, end P = %.3f' %
                    (frequencies.size, 1.0/endf, 1.0/startf)
                )
            else:
                # this gets an automatic grid of frequencies to use
                frequencies = get_frequency_grid(stimes,
                                                 minfreq=startf,
                                                 maxfreq=endf)
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

            bestperiodind = npargmax(lsp)

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
                    'method':'aov'}

        else:

            LOGERROR('no good detections for these times and mags, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'method':'aov'}
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'aov'}



##############################
## GENERALIZED LOMB-SCARGLE ##
##############################

def glsp_worker(task):
    '''This is a worker to wrap the generalized Lomb-Scargle single-frequency
    function.

    '''

    try:
        return glspval(*task)
    except Exception as e:
        return npnan



def glsp_worker_notau(task):
    '''This is a worker to wrap the generalized Lomb-Scargle single-freq func.

    This version doesn't use tau.

    '''

    try:
        return glspvalnt(*task)
    except Exception as e:
        return npnan



def pgen_lsp(
        times,
        mags,
        errs,
        startp=None,
        endp=None,
        autofreq=True,
        nbestpeaks=5,
        periodepsilon=0.1, # 0.1
        stepsize=1.0e-4,
        nworkers=None,
        sigclip=10.0,
        glspfunc=glsp_worker,
):
    '''This calculates the generalized LSP given times, mags, errors.

    Uses the algorithm from Zechmeister and Kurster (2009). By default, this
    calculates a frequency grid to use automatically, based on the autofrequency
    function from astropy.stats.lombscargle. If startp and endp are provided,
    will generate a frequency grid based on these instead.

    '''

    # get rid of nans first
    find = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    if len(ftimes) > 9 and len(fmags) > 9 and len(ferrs) > 9:

        # get the median and stdev = 1.483 x MAD
        median_mag = np.median(fmags)
        stddev_mag = (np.median(np.abs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (np.abs(fmags - median_mag)) < (sigclip * stddev_mag)

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
                # default end period is length of time series divided by 2
                startf = 1.0/(stimes.max() - stimes.min())

            # if we're not using autofreq, then use the provided frequencies
            if not autofreq:
                omegas = 2*np.pi*np.arange(startf, endf, stepsize)
                LOGINFO(
                    'using %s frequency points, start P = %.3f, end P = %.3f' %
                    (omegas.size, 1.0/endf, 1.0/startf)
                )
            else:
                # this gets an automatic grid of frequencies to use
                freqs = get_frequency_grid(stimes,
                                           minfreq=startf,
                                           maxfreq=endf)
                omegas = 2*np.pi*freqs
                LOGINFO(
                    'using autofreq with %s frequency points, '
                    'start P = %.3f, end P = %.3f' %
                    (omegas.size, 1.0/freqs.max(), 1.0/freqs.min())
                )

            # map to parallel workers
            if (not nworkers) or (nworkers > NCPUS):
                nworkers = NCPUS
                LOGINFO('using %s workers...' % nworkers)

            pool = Pool(nworkers)

            tasks = [(stimes, smags, serrs, x) for x in omegas]
            lsp = pool.map(glspfunc, tasks)

            pool.close()
            pool.join()
            del pool

            lsp = np.array(lsp)
            periods = 2.0*np.pi/omegas

            # find the nbestpeaks for the periodogram: 1. sort the lsp array by
            # highest value first 2. go down the values until we find five
            # values that are separated by at least periodepsilon in period

            # make sure to filter out non-finite values of lsp

            finitepeakind = npisfinite(lsp)
            finlsp = lsp[finitepeakind]
            finperiods = periods[finitepeakind]

            bestperiodind = npargmax(finlsp)

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
                    'omegas':omegas,
                    'periods':periods,
                    'method':'gls'}

        else:

            LOGERROR('no good detections for these times and mags, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'omegas':None,
                    'periods':None,
                    'method':'gls'}
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'omegas':None,
                'periods':None,
                'method':'gls'}



######################################
## BLS (Kovacs, Zucker, Mazeh 2002) ##
######################################

def _bls_runner(times,
                mags,
                nfreq,
                freqmin,
                stepsize,
                nbins,
                minduration,
                maxduration):
    '''
    This runs the bls.eebls function using the given inputs.

    '''

    workarr_u = np.ones(times.size)
    workarr_v = np.ones(times.size)

    blsresult = eebls(times, mags,
                      workarr_u, workarr_v,
                      nfreq, freqmin, stepsize,
                      nbins, minduration, maxduration)

    return {'power':blsresult[0],
            'bestperiod':blsresult[1],
            'bestpower':blsresult[2],
            'transdepth':blsresult[3],
            'transduration':blsresult[4],
            'transingressbin':blsresult[5],
            'transegressbin':blsresult[6]}



def parallel_bls_worker(task):
    '''
    This wraps _bls_runner for the parallel function below.

    task[0] = times
    task[1] = mags
    task[2] = nfreq
    task[3] = freqmin
    task[4] = stepsize
    task[5] = nbins
    task[6] = minduration
    task[7] = maxduration

    '''

    try:
        return _bls_runner(*task)
    except Exception as e:
        LOGEXCEPTION('BLS failed for task %s' % repr(task[2:]))
    return {'power':np.array([npnan for x in range(nfreq)]),
            'bestperiod':npnan,
            'bestpower':npnan,
            'transdepth':npnan,
            'transduration':npnan,
            'transingressbin':npnan,
            'transegressbin':npnan}




def bls_serial_pfind(times, mags, errs,
                     startp=0.1, # search from 0.1 d to...
                     endp=100.0, # ... 100.0 d -- don't search full timebase
                     stepsize=5.0e-4,
                     mintransitduration=0.01, # minimum transit length in phase
                     maxtransitduration=0.8,  # maximum transit length in phase
                     nphasebins=200,
                     autofreq=True, # figure out f0, nf, and df automatically
                     periodepsilon=0.1,
                     nbestpeaks=5,
                     sigclip=10.0):
    '''Runs the Box Least Squares Fitting Search for transit-shaped signals.

    Based on eebls.f from Kovacs et al. 2002 and python-bls from Foreman-Mackey
    et al. 2015. This is the serial version (which is good enough in most cases
    because BLS in Fortran is fairly fast). If nfreq > 5e5, this will take a
    while.

    '''

    # get rid of nans first
    find = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    if len(ftimes) > 9 and len(fmags) > 9 and len(ferrs) > 9:

        # get the median and stdev = 1.483 x MAD
        median_mag = np.median(fmags)
        stddev_mag = (np.median(np.abs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (np.abs(fmags - median_mag)) < (sigclip * stddev_mag)

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

        # make sure there are enough points to calculate a spectrum
        if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

            # if we're setting up everything automatically
            if autofreq:

                # use heuristic to figure out best timestep
                # see http://www.astro.princeton.edu/~jhartman/vartools.html
                stepsize = 0.25*mintransitduration/(times[-1]-times[0])

                # now figure out the frequencies to use
                minfreq = 1.0/endp
                maxfreq = 1.0/startp
                nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))

                # figure out the best number of phasebins to use
                nphasebins = int(np.ceil(2.0/mintransitduration))

                # say what we're using
                LOGINFO('autofreq: using stepsize: %s, min P: %s, '
                        'max P: %s, nfreq: %s, nphasebins: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, startp, endp, nfreq, nphasebins,
                         mintransitduration, maxtransitduration))
                LOGINFO('autofreq: minfreq: %s, maxfreq: %s' % (minfreq,
                                                                maxfreq))

            else:

                minfreq = 1.0/endp
                maxfreq = 1.0/startp
                nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))

                # say what we're using
                LOGINFO('manualfreq: using stepsize: %s, min P: %s, '
                        'max P: %s, nfreq: %s, nphasebins: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, startp, endp, nfreq, nphasebins,
                         mintransitduration, maxtransitduration))
                LOGINFO('manualfreq: minfreq: %s, maxfreq: %s' %
                        (minfreq,maxfreq))


            if nfreq > 5.0e5:

                LOGWARNING('more than 5.0e5 frequencies to go through; '
                           'this will take a while. '
                           'you might want to use the '
                           'periodbase.bls_parallel_pfind function instead')

            if minfreq < (1.0/(ftimes.max() - ftimes.min())):

                LOGWARNING('the requested max P = %.3f is larger than '
                           'the time base of the observations = %.3f, '
                           ' will make minfreq = 2 x 1/timebase'
                           % (endp, ftimes.max() - ftimes.min()))
                minfreq = 2.0/(ftimes.max() - ftimes.min())
                LOGINFO('new minfreq: %s, maxfreq: %s' %
                        (minfreq, maxfreq))


            # run BLS
            try:

                blsresult = _bls_runner(ftimes,
                                        fmags,
                                        nfreq,
                                        minfreq,
                                        stepsize,
                                        nphasebins,
                                        mintransitduration,
                                        maxtransitduration)

                # find the peaks in the BLS. this uses wavelet transforms to
                # smooth the spectrum and find peaks. a similar thing would be
                # to do a convolution with a gaussian kernel or a tophat
                # function, calculate d/dx(result), then get indices where this
                # is zero
                # blspeakinds = find_peaks_cwt(blsresults['power'],
                #                              nparray([2.0,3.0,4.0,5.0]))



                frequencies = minfreq + nparange(nfreq)*stepsize
                periods = 1.0/frequencies
                lsp = blsresult['power']

                # find the nbestpeaks for the periodogram: 1. sort the lsp array
                # by highest value first 2. go down the values until we find
                # five values that are separated by at least periodepsilon in
                # period
                # make sure to get only the finite peaks in the periodogram
                # this is needed because BLS may produce infs for some peaks
                finitepeakind = npisfinite(lsp)
                finlsp = lsp[finitepeakind]
                finperiods = periods[finitepeakind]

                bestperiodind = np.argmax(finlsp)
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
                    # periodepsilon to make sure we jump to an entire different
                    # peak in the periodogram
                    if (perioddiff > periodepsilon and
                        all(x > periodepsilon for x in bestperiodsdiff)):
                        nbestperiods.append(period)
                        nbestlspvals.append(lspval)
                        peakcount = peakcount + 1

                    prevperiod = period


                # generate the return dict
                resultdict = {
                    'bestperiod':finperiods[bestperiodind],
                    'bestlspval':finlsp[bestperiodind],
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':nbestlspvals,
                    'nbestperiods':nbestperiods,
                    'lspvals':lsp,
                    'frequencies':frequencies,
                    'periods':periods,
                    'blsresult':blsresult,
                    'method':'bls',
                }

                return resultdict

            except Exception as e:

                LOGEXCEPTION('BLS failed!')
                return {'bestperiod':npnan,
                        'bestlspval':npnan,
                        'nbestpeaks':nbestpeaks,
                        'nbestlspvals':None,
                        'nbestperiods':None,
                        'lspvals':None,
                        'periods':None,
                        'method':'bls'}


        else:

            LOGERROR('no good detections for these times and mags, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'method':'bls'}
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'bls'}



def bls_parallel_pfind(
        times, mags, errs,
        startp=0.1, # by default, search from 0.1 d to...
        endp=100.0, # ... 100.0 d -- don't search full timebase
        stepsize=1.0e-4,
        mintransitduration=0.01, # minimum transit length in phase
        maxtransitduration=0.8,  # maximum transit length in phase
        nphasebins=200,
        autofreq=True, # figure out f0, nf, and df automatically
        nbestpeaks=5,
        periodepsilon=0.1, # 0.1
        nworkers=None,
        sigclip=10.0
):
    '''Runs the Box Least Squares Fitting Search for transit-shaped signals.

    Based on eebls.f from Kovacs et al. 2002 and python-bls from Foreman-Mackey
    et al. 2015. Breaks up the full frequency space into chunks and passes them
    to parallel BLS workers.

    NOTE: the combined BLS spectrum produced by this function is not identical
    to that produced by running BLS in one shot for the entire frequency
    space. There are differences on the order of 1.0e-3 or so in the respective
    peak values, but peaks appear at the same frequencies for both methods. This
    is likely due to different aliasing caused by smaller chunks of the
    frequency space used by the parallel workers in this function. When in
    doubt, confirm results for this parallel implementation by comparing to
    those from the serial implementation above.

    '''

    # get rid of nans first
    find = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    if len(ftimes) > 9 and len(fmags) > 9 and len(ferrs) > 9:

        # get the median and stdev = 1.483 x MAD
        median_mag = np.median(fmags)
        stddev_mag = (np.median(np.abs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (np.abs(fmags - median_mag)) < (sigclip * stddev_mag)

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

        # make sure there are enough points to calculate a spectrum
        if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

            # if we're setting up everything automatically
            if autofreq:

                # use heuristic to figure out best timestep
                # see http://www.astro.princeton.edu/~jhartman/vartools.html
                stepsize = 0.25*mintransitduration/(times[-1]-times[0])

                # now figure out the frequencies to use
                minfreq = 1.0/endp
                maxfreq = 1.0/startp
                nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))

                # figure out the best number of phasebins to use
                # see http://www.astro.princeton.edu/~jhartman/vartools.html
                nphasebins = int(np.ceil(2.0/mintransitduration))

                # say what we're using
                LOGINFO('autofreq: using stepsize: %s, min P: %s, '
                        'max P: %s, nfreq: %s, nphasebins: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, startp, endp, nfreq, nphasebins,
                         mintransitduration, maxtransitduration))
                LOGINFO('autofreq: minfreq: %s, maxfreq: %s' % (minfreq,
                                                                maxfreq))

            else:

                minfreq = 1.0/endp
                maxfreq = 1.0/startp
                nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))

                # say what we're using
                LOGINFO('manualfreq: using stepsize: %s, min P: %s, '
                        'max P: %s, nfreq: %s, nphasebins: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, startp, endp, nfreq, nphasebins,
                         mintransitduration, maxtransitduration))
                LOGINFO('manualfreq: minfreq: %s, maxfreq: %s' %
                        (minfreq,maxfreq))

            if minfreq < (1.0/(ftimes.max() - ftimes.min())):

                LOGWARNING('the requested max P = %.3f is larger than '
                           'the time base of the observations = %.3f, '
                           ' will make minfreq = 2 x 1/timebase'
                           % (endp, ftimes.max() - ftimes.min()))
                minfreq = 2.0/(ftimes.max() - ftimes.min())
                LOGINFO('new minfreq: %s, maxfreq: %s' %
                        (minfreq, maxfreq))


            #############################
            ## NOW RUN BLS IN PARALLEL ##
            #############################

            # fix number of CPUs if needed
            if not nworkers or nworkers > NCPUS:
                nworkers = NCPUS
                LOGINFO('using %s workers...' % nworkers)

            # break up the tasks into chunks
            frequencies = minfreq + nparange(nfreq)*stepsize
            chunksize = int(float(len(frequencies))/nworkers) + 1
            chunk_minfreqs = [frequencies[x*chunksize] for x in range(nworkers)]
            chunk_nfreqs = [frequencies[x*chunksize:x*chunksize+chunksize].size
                            for x in range(nworkers)]


            # populate the tasks list
            tasks = [(stimes, smags,
                      chunk_minf, chunk_nf,
                      stepsize, nphasebins,
                      mintransitduration, maxtransitduration)
                     for (chunk_nf, chunk_minf)
                     in zip(chunk_minfreqs, chunk_nfreqs)]

            for ind, task in enumerate(tasks):
                LOGINFO('worker %s: minfreq = %.3f, nfreqs = %s' %
                        (ind+1, task[3], task[2]))
            LOGINFO('running...')

            # return tasks

            # start the pool
            pool = Pool(nworkers)
            results = pool.map(parallel_bls_worker, tasks)

            pool.close()
            pool.join()
            del pool

            # now concatenate the output lsp arrays
            lsp = np.concatenate([x['power'] for x in results])
            periods = 1.0/frequencies

            # find the nbestpeaks for the periodogram: 1. sort the lsp array
            # by highest value first 2. go down the values until we find
            # five values that are separated by at least periodepsilon in
            # period

            # make sure to get only the finite peaks in the periodogram
            # this is needed because BLS may produce infs for some peaks
            finitepeakind = npisfinite(lsp)
            finlsp = lsp[finitepeakind]
            finperiods = periods[finitepeakind]

            bestperiodind = np.argmax(finlsp)
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
                # periodepsilon to make sure we jump to an entire different
                # peak in the periodogram
                if (perioddiff > periodepsilon and
                    all(x > periodepsilon for x in bestperiodsdiff)):
                    nbestperiods.append(period)
                    nbestlspvals.append(lspval)
                    peakcount = peakcount + 1

                prevperiod = period


            # generate the return dict
            resultdict = {
                'bestperiod':finperiods[bestperiodind],
                'bestlspval':finlsp[bestperiodind],
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':nbestlspvals,
                'nbestperiods':nbestperiods,
                'lspvals':lsp,
                'frequencies':frequencies,
                'periods':periods,
                'blsresult':results,
                'method':'bls',
            }

            return resultdict

        else:

            LOGERROR('no good detections for these times and mags, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'method':'bls'}
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'bls'}
