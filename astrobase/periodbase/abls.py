#!/usr/bin/env python

'''abls.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Contains the Kovacs, et al. (2002) Box-Least-squared-Search period-search
algorithm implementation for periodbase. This uses the implementation in Astropy
3.1, so requires that version.

This will be used preferentially if we have Python >= 3.6 and Astropy >= 3.1.

'''

#############
## LOGGING ##
#############

import logging
from datetime import datetime
from traceback import format_exc

# setup a logger
LOGGER = None
LOGMOD = __name__
DEBUG = False

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.%s' % (parent_name, LOGMOD))

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('[%s - DBUG] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('[%s - INFO] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('[%s - ERR!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('[%s - WRN!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '[%s - EXC!] %s\nexception was: %s' % (
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                message, format_exc()
            )
        )


#############
## IMPORTS ##
#############

from multiprocessing import Pool, cpu_count

from math import fmod

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


###################
## LOCAL IMPORTS ##
###################

from ..lcmath import phase_magseries, sigclip_magseries, \
    time_bin_magseries, phase_bin_magseries, \
    phase_magseries_with_errs, phase_bin_magseries_with_errs



from pyeebls import eebls

from ..varbase.lcfit import savgol_fit_magseries, \
    traptransit_fit_magseries


############
## CONFIG ##
############

NCPUS = cpu_count()


#######################
## UTILITY FUNCTIONS ##
#######################

def auto_transit_duration(min_radius_hint,
                          max_radius_hint):
    '''
    This figures out the minimum and max transit duration (q) automatically.

    q ~ 0.076 x R**(2/3) x P**(-2/3)

    P = period in days
    R = stellar radius in solar radii

    '''




def bls_serial_pfind(times, mags, errs,
                     magsarefluxes=False,
                     startp=0.1,  # search from 0.1 d to...
                     endp=100.0,  # ... 100.0 d -- don't search full timebase
                     stepsize=5.0e-4,
                     mintransitduration=0.01,  # minimum transit length in phase
                     maxtransitduration=0.4,   # maximum transit length in phase
                     nphasebins=200,
                     autofreq=True,  # figure out f0, nf, and df automatically
                     periodepsilon=0.1,
                     nbestpeaks=5,
                     sigclip=10.0,
                     verbose=True):
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

            # figure out the best number of phasebins to use
            nphasebins = int(np.ceil(2.0/mintransitduration))

            # use heuristic to figure out best timestep
            stepsize = 0.25*mintransitduration/(stimes.max()-stimes.min())

            # now figure out the frequencies to use
            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))

            # say what we're using
            if verbose:
                LOGINFO('min P: %s, max P: %s, nfreq: %s, '
                        'minfreq: %s, maxfreq: %s' % (startp, endp, nfreq,
                                                      minfreq, maxfreq))
                LOGINFO('autofreq = True: using AUTOMATIC values for '
                        'freq stepsize: %s, nphasebins: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, nphasebins,
                         mintransitduration, maxtransitduration))

        else:

            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))

            # say what we're using
            if verbose:
                LOGINFO('min P: %s, max P: %s, nfreq: %s, '
                        'minfreq: %s, maxfreq: %s' % (startp, endp, nfreq,
                                                      minfreq, maxfreq))
                LOGINFO('autofreq = False: using PROVIDED values for '
                        'freq stepsize: %s, nphasebins: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, nphasebins,
                         mintransitduration, maxtransitduration))


        if nfreq > 5.0e5:

            if verbose:
                LOGWARNING('more than 5.0e5 frequencies to go through; '
                           'this will take a while. '
                           'you might want to use the '
                           'periodbase.bls_parallel_pfind function instead')

        if minfreq < (1.0/(stimes.max() - stimes.min())):

            if verbose:
                LOGWARNING('the requested max P = %.3f is larger than '
                           'the time base of the observations = %.3f, '
                           ' will make minfreq = 2 x 1/timebase'
                           % (endp, stimes.max() - stimes.min()))
            minfreq = 2.0/(stimes.max() - stimes.min())
            if verbose:
                LOGINFO('new minfreq: %s, maxfreq: %s' %
                        (minfreq, maxfreq))


        # run BLS
        try:

            blsresult = 1

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
                        'method':'bls',
                        'kwargs':{'startp':startp,
                                  'endp':endp,
                                  'stepsize':stepsize,
                                  'mintransitduration':mintransitduration,
                                  'maxtransitduration':maxtransitduration,
                                  'nphasebins':nphasebins,
                                  'autofreq':autofreq,
                                  'periodepsilon':periodepsilon,
                                  'nbestpeaks':nbestpeaks,
                                  'sigclip':sigclip,
                                  'magsarefluxes':magsarefluxes}}

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
                if (perioddiff > (periodepsilon*prevperiod) and
                    all(x > (periodepsilon*prevperiod)
                        for x in bestperiodsdiff)):
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
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'mintransitduration':mintransitduration,
                          'maxtransitduration':maxtransitduration,
                          'nphasebins':nphasebins,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip,
                          'magsarefluxes':magsarefluxes}}

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
                    'method':'bls',
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'mintransitduration':mintransitduration,
                              'maxtransitduration':maxtransitduration,
                              'nphasebins':nphasebins,
                              'autofreq':autofreq,
                              'periodepsilon':periodepsilon,
                              'nbestpeaks':nbestpeaks,
                              'sigclip':sigclip,
                              'magsarefluxes':magsarefluxes}}


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
                'method':'bls',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'mintransitduration':mintransitduration,
                          'maxtransitduration':maxtransitduration,
                          'nphasebins':nphasebins,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip,
                          'magsarefluxes':magsarefluxes}}



def bls_parallel_pfind(
        times, mags, errs,
        magsarefluxes=False,
        startp=0.1,  # by default, search from 0.1 d to...
        endp=100.0,  # ... 100.0 d -- don't search full timebase
        stepsize=1.0e-4,
        mintransitduration=0.01,  # minimum transit length in phase
        maxtransitduration=0.4,   # maximum transit length in phase
        nphasebins=200,
        autofreq=True,  # figure out f0, nf, and df automatically
        nbestpeaks=5,
        periodepsilon=0.1,  # 0.1
        nworkers=None,
        sigclip=10.0,
        verbose=True
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

            # figure out the best number of phasebins to use
            nphasebins = int(np.ceil(2.0/mintransitduration))

            # use heuristic to figure out best timestep
            stepsize = 0.25*mintransitduration/(stimes.max()-stimes.min())

            # now figure out the frequencies to use
            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))

            # say what we're using
            if verbose:
                LOGINFO('min P: %s, max P: %s, nfreq: %s, '
                        'minfreq: %s, maxfreq: %s' % (startp, endp, nfreq,
                                                      minfreq, maxfreq))
                LOGINFO('autofreq = True: using AUTOMATIC values for '
                        'freq stepsize: %s, nphasebins: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, nphasebins,
                         mintransitduration, maxtransitduration))

        else:

            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(np.ceil((maxfreq - minfreq)/stepsize))

            # say what we're using
            if verbose:
                LOGINFO('min P: %s, max P: %s, nfreq: %s, '
                        'minfreq: %s, maxfreq: %s' % (startp, endp, nfreq,
                                                      minfreq, maxfreq))
                LOGINFO('autofreq = False: using PROVIDED values for '
                        'freq stepsize: %s, nphasebins: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, nphasebins,
                         mintransitduration, maxtransitduration))

        # check the minimum frequency
        if minfreq < (1.0/(stimes.max() - stimes.min())):

            minfreq = 2.0/(stimes.max() - stimes.min())
            if verbose:
                LOGWARNING('the requested max P = %.3f is larger than '
                           'the time base of the observations = %.3f, '
                           ' will make minfreq = 2 x 1/timebase'
                           % (endp, stimes.max() - stimes.min()))
                LOGINFO('new minfreq: %s, maxfreq: %s' %
                        (minfreq, maxfreq))


        #############################
        ## NOW RUN BLS IN PARALLEL ##
        #############################

        # fix number of CPUs if needed
        if not nworkers or nworkers > NCPUS:
            nworkers = NCPUS
            if verbose:
                LOGINFO('using %s workers...' % nworkers)

        # break up the tasks into chunks
        frequencies = minfreq + nparange(nfreq)*stepsize

        csrem = int(fmod(nfreq, nworkers))
        csint = int(float(nfreq/nworkers))

        chunk_minfreqs, chunk_nfreqs = [], []

        for x in range(nworkers):

            this_minfreqs = frequencies[x*csint]

            # handle usual nfreqs
            if x < (nworkers - 1):
                this_nfreqs = frequencies[x*csint:x*csint+csint].size
            else:
                this_nfreqs = frequencies[x*csint:x*csint+csint+csrem].size

            chunk_minfreqs.append(this_minfreqs)
            chunk_nfreqs.append(this_nfreqs)

        # chunk_minfreqs = [frequencies[x*chunksize] for x in range(nworkers)]
        # chunk_nfreqs = [frequencies[x*chunksize:x*chunksize+chunksize].size
        #                 for x in range(nworkers)]


        # populate the tasks list
        tasks = [(stimes, smags,
                  chunk_minf, chunk_nf,
                  stepsize, nphasebins,
                  mintransitduration, maxtransitduration)
                 for (chunk_nf, chunk_minf)
                 in zip(chunk_minfreqs, chunk_nfreqs)]

        if verbose:
            for ind, task in enumerate(tasks):
                LOGINFO('worker %s: minfreq = %.6f, nfreqs = %s' %
                        (ind+1, task[3], task[2]))
            LOGINFO('running...')

        # return tasks


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
                'method':'bls',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'mintransitduration':mintransitduration,
                          'maxtransitduration':maxtransitduration,
                          'nphasebins':nphasebins,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip,
                          'magsarefluxes':magsarefluxes}}



def bls_snr(blsdict,
            times,
            mags,
            errs,
            magsarefluxes=False,
            sigclip=10.0,
            perioddeltapercent=10,
            npeaks=None,
            assumeserialbls=False,
            verbose=True):
    '''Calculates the signal to noise ratio for each best peak in the BLS
    periodogram.

    SNR = transit model depth / RMS of light curve with transit model subtracted
          * sqrt(number of points in transit)

    blsdict is the output of either bls_parallel_pfind or bls_serial_pfind.

    times, mags, errs are ndarrays containing the magnitude series.

    perioddeltapercent controls the period interval used by a bls_serial_pfind
    run around each peak period to figure out the transit depth, duration, and
    ingress/egress bins for eventual calculation of the SNR of the peak.

    npeaks controls how many of the periods in blsdict['nbestperiods'] to find
    the SNR for. If it's None, then this will calculate the SNR for all of
    them. If it's an integer between 1 and len(blsdict['nbestperiods']), will
    calculate for only the specified number of peak periods, starting from the
    best period.

    If assumeserialbls is True, will not rerun bls_serial_pfind to figure out
    the transit depth, duration, and ingress/egress bins for eventual
    calculation of the SNR of the peak. This is normally False because we assume
    that the user will be using bls_parallel_pfind, which works on chunks of
    frequency space so returns multiple values of transit depth, duration,
    ingress/egress bin specific to those chunks. These may not be valid for the
    global best peaks in the periodogram, so we need to rerun bls_serial_pfind
    around each peak in blsdict['nbestperiods'] to get correct values for these.

    FIXME: for now, we're only doing simple RMS. Need to calculate red and
    white-noise RMS as outlined below:

      - calculate the white noise rms and the red noise rms of the residual.

        - the white noise rms is just the rms of the residual
        - the red noise rms = sqrt(binnedrms^2 - expectedbinnedrms^2)

      - calculate the SNR using:

        sqrt(delta^2 / ((sigma_w ^2 / nt) + (sigma_r ^2 / Nt))))

        where:

        delta = transit depth
        sigma_w = white noise rms
        sigma_r = red noise rms
        nt = number of in-transit points
        Nt = number of distinct transits sampled

    '''

def bls_stats_singleperiod(times, mags, errs, period,
                           magsarefluxes=False,
                           sigclip=10.0,
                           perioddeltapercent=10,
                           nphasebins=200,
                           mintransitduration=0.01,
                           maxtransitduration=0.4,
                           verbose=True):
    '''This calculates the SNR, refit period, and time of center-transit for a
    single period.

    times, mags, errs are numpy arrays containing these values.

    period is the period for which the SNR, refit period, and refit epoch should
    be calculated.

    sigclip is the amount of sigmaclip to apply to the magnitude time-series.

    perioddeltapercent is used to set the search window around the specified
    period, which will be used to rerun BLS to get the transit ingress and
    egress bins.

    nphasebins is the number of phase bins to use for the BLS process. This
    should be equal to the value of nphasebins you used for your initial BLS run
    to find the specified period.

    verbose indicates whether this function should report its progress.

    This returns a dict similar to bls_snr above.

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)
