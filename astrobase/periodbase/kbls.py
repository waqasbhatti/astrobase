#!/usr/bin/env python

'''kbls.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Contains the Kovacs, et al. (2002) Box-Least-squart-Search  period-search
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
    globals()['LOGGER'] = logging.getLogger('%s.kbls' % parent_name)

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

from pyeebls import eebls


############
## CONFIG ##
############

NCPUS = cpu_count()



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
                     magsarefluxes=False,
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



    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # if we're setting up everything automatically
        if autofreq:

            # use heuristic to figure out best timestep
            # see http://www.astro.princeton.edu/~jhartman/vartools.html
            stepsize = 0.25*mintransitduration/(stimes.max()-stimes.min())

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

        if minfreq < (1.0/(stimes.max() - stimes.min())):

            LOGWARNING('the requested max P = %.3f is larger than '
                       'the time base of the observations = %.3f, '
                       ' will make minfreq = 2 x 1/timebase'
                       % (endp, stimes.max() - stimes.min()))
            minfreq = 2.0/(stimes.max() - stimes.min())
            LOGINFO('new minfreq: %s, maxfreq: %s' %
                    (minfreq, maxfreq))


        # run BLS
        try:

            blsresult = _bls_runner(stimes,
                                    smags,
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
                'stepsize':stepsize,
                'nfreq':nfreq,
                'nphasebins':nphasebins,
                'mintransitduration':mintransitduration,
                'maxtransitduration':maxtransitduration,
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
                    'stepsize':stepsize,
                    'nfreq':nfreq,
                    'nphasebins':nphasebins,
                    'mintransitduration':mintransitduration,
                    'maxtransitduration':maxtransitduration,
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
                'stepsize':stepsize,
                'nfreq':None,
                'nphasebins':None,
                'mintransitduration':mintransitduration,
                'maxtransitduration':maxtransitduration,
                'method':'bls'}



def bls_parallel_pfind(
        times, mags, errs,
        magsarefluxes=False,
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

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # if we're setting up everything automatically
        if autofreq:

            # use heuristic to figure out best timestep
            # see http://www.astro.princeton.edu/~jhartman/vartools.html
            stepsize = 0.25*mintransitduration/(stimes.max()-stimes.min())

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

        if minfreq < (1.0/(stimes.max() - stimes.min())):

            LOGWARNING('the requested max P = %.3f is larger than '
                       'the time base of the observations = %.3f, '
                       ' will make minfreq = 2 x 1/timebase'
                       % (endp, stimes.max() - stimes.min()))
            minfreq = 2.0/(stimes.max() - stimes.min())
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
            'stepsize':stepsize,
            'nfreq':nfreq,
            'nphasebins':nphasebins,
            'mintransitduration':mintransitduration,
            'maxtransitduration':maxtransitduration,
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
                'blsresult':None,
                'stepsize':stepsize,
                'nfreq':None,
                'nphasebins':None,
                'mintransitduration':mintransitduration,
                'maxtransitduration':maxtransitduration,
                'method':'bls'}
