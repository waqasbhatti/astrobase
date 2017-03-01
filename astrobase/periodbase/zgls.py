#!/usr/bin/env python

'''zgls.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Contains the Zechmeister & Kurster (2002) Generalized Lomb-Scargle period-search
algorithm implementation for periodbase.

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
    globals()['LOGGER'] = logging.getLogger('%s.zgls' % parent_name)

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

from .glsp import generalized_lsp_value as glspval, \
    generalized_lsp_value_notau as glspvalnt

from . import get_frequency_grid


############
## CONFIG ##
############

NCPUS = cpu_count()


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
        magsarefluxes=False,
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
