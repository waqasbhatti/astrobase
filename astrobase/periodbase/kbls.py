#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kbls.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''
Contains the Kovacs, et al. (2002) Box-Least-squared-Search period-search
algorithm implementation for periodbase.

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

from math import fmod
from multiprocessing import Pool, cpu_count

from numpy import (
    nan as npnan, arange as nparange, ones as npones, array as nparray,
    isfinite as npisfinite, argmax as npargmax, floor as npfloor,
    linspace as nplinspace, digitize as npdigitize, where as npwhere,
    abs as npabs, min as npmin, full_like as npfull_like, median as npmedian,
    std as npstd, sqrt as npsqrt, ceil as npceil, argsort as npargsort,
    concatenate as npconcatenate, ndarray as npndarray, inf as npinf
)

###################
## LOCAL IMPORTS ##
###################

from pyeebls import eebls

from ..lcmath import sigclip_magseries, phase_magseries_with_errs
from ..lcfit.nonphysical import savgol_fit_magseries
from ..lcfit.transits import traptransit_fit_magseries

from .utils import resort_by_time

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
    '''This runs the pyeebls.eebls function using the given inputs.

    Parameters
    ----------

    times,mags : np.array
        The input magnitude time-series to search for transits.

    nfreq : int
        The number of frequencies to use when searching for transits.

    freqmin : float
        The minimum frequency of the period-search -> max period that will be
        used for the search.

    stepsize : float
        The step-size in frequency to use to generate a frequency-grid.

    nbins : int
        The number of phase bins to use.

    minduration : float
        The minimum fractional transit duration that will be considered.

    maxduration : float
        The maximum fractional transit duration that will be considered.

    Returns
    -------

    dict
        Returns a dict of the form::

            {
                'power':           the periodogram power array,
                'bestperiod':      the best period found,
                'bestpower':       the highest peak of the periodogram power,
                'transdepth':      transit depth found by eebls.f,
                'transduration':   transit duration found by eebls.f,
                'transingressbin': transit ingress bin found by eebls.f,
                'transegressbin':  transit egress bin found by eebls.f,
            }

    '''

    workarr_u = npones(times.size)
    workarr_v = npones(times.size)

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


def _parallel_bls_worker(task):
    '''
    This wraps the BLS function for the parallel driver below.

    Parameters
    ----------

    tasks : tuple
        This is of the form::

            task[0] = times
            task[1] = mags
            task[2] = nfreq
            task[3] = freqmin
            task[4] = stepsize
            task[5] = nbins
            task[6] = minduration
            task[7] = maxduration

    Returns
    -------

    dict
        Returns a dict of the form::

            {
                'power':           the periodogram power array,
                'bestperiod':      the best period found,
                'bestpower':       the highest peak of the periodogram power,
                'transdepth':      transit depth found by eebls.f,
                'transduration':   transit duration found by eebls.f,
                'transingressbin': transit ingress bin found by eebls.f,
                'transegressbin':  transit egress bin found by eebls.f,
            }

    '''

    try:

        return _bls_runner(*task)

    except Exception:

        LOGEXCEPTION('BLS failed for task %s' % repr(task[2:]))

        return {
            'power':nparray([npnan for x in range(task[2])]),
            'bestperiod':npnan,
            'bestpower':npnan,
            'transdepth':npnan,
            'transduration':npnan,
            'transingressbin':npnan,
            'transegressbin':npnan
        }


def bls_serial_pfind(
        times, mags, errs,
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
        endp_timebase_check=True,
        verbose=True,
        get_stats=True,
):
    '''Runs the Box Least Squares Fitting Search for transit-shaped signals.

    Based on eebls.f from Kovacs et al. 2002 and python-bls from Foreman-Mackey
    et al. 2015. This is the serial version (which is good enough in most cases
    because BLS in Fortran is fairly fast). If nfreq > 5e5, this will take a
    while.

    Parameters
    ----------

    times,mags,errs : np.array
        The magnitude/flux time-series to search for transits.

    magsarefluxes : bool
        If the input measurement values in `mags` and `errs` are in fluxes, set
        this to True.

    startp,endp : float
        The minimum and maximum periods to consider for the transit search.

    stepsize : float
        The step-size in frequency to use when constructing a frequency grid for
        the period search.

    mintransitduration,maxtransitduration : float
        The minimum and maximum transitdurations (in units of phase) to consider
        for the transit search.

    nphasebins : int
        The number of phase bins to use in the period search.

    autofreq : bool
        If this is True, the values of `stepsize` and `nphasebins` will be
        ignored, and these, along with a frequency-grid, will be determined
        based on the following relations::

            nphasebins = int(ceil(2.0/mintransitduration))
            if nphasebins > 3000:
                nphasebins = 3000

            stepsize = 0.25*mintransitduration/(times.max()-times.min())

            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(ceil((maxfreq - minfreq)/stepsize))

    periodepsilon : float
        The fractional difference between successive values of 'best' periods
        when sorting by periodogram power to consider them as separate periods
        (as opposed to part of the same periodogram peak). This is used to avoid
        broad peaks in the periodogram and make sure the 'best' periods returned
        are all actually independent.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

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

    endp_timebase_check : bool
        If True, will check if the ``endp`` value is larger than the time-base
        of the observations. If it is, will change the ``endp`` value such that
        it is half of the time-base. If False, will allow an ``endp`` larger
        than the time-base of the observations.

    verbose : bool
        If this is True, will indicate progress and details about the frequency
        grid used for the period search.

    get_stats : bool
        If True, runs :py:func:`.bls_stats_singleperiod` for each of the best
        periods in the output and injects the output into the output dict so you
        only have to run this function to get the periods and their stats.

        The output dict from this function will then contain a 'stats' key
        containing a list of dicts with statistics for each period in
        ``resultdict['nbestperiods']``. These dicts will contain fit values of
        transit parameters after a trapezoid transit model is fit to the phased
        light curve at each period in ``resultdict['nbestperiods']``, i.e. fit
        values for period, epoch, transit depth, duration, ingress duration, and
        the SNR of the transit.

        NOTE: make sure to check the 'fit_status' key for each
        ``resultdict['stats']`` item to confirm that the trapezoid transit model
        fit succeeded and that the stats calculated are valid.

    Returns
    -------

    dict
        This function returns a dict, referred to as an `lspinfo` dict in other
        astrobase functions that operate on periodogram results. This is a
        standardized format across all astrobase period-finders, and is of the
        form below::

            {'bestperiod': the best period value in the periodogram,
             'bestlspval': the periodogram peak associated with the best period,
             'nbestpeaks': the input value of nbestpeaks,
             'nbestlspvals': nbestpeaks-size list of best period peak values,
             'nbestperiods': nbestpeaks-size list of best periods,
             'stats': BLS stats for each best period,
             'lspvals': the full array of periodogram powers,
             'frequencies': the full array of frequencies considered,
             'periods': the full array of periods considered,
             'blsresult': the result dict from the eebls.f wrapper function,
             'stepsize': the actual stepsize used,
             'nfreq': the actual nfreq used,
             'nphasebins': the actual nphasebins used,
             'mintransitduration': the input mintransitduration,
             'maxtransitduration': the input maxtransitdurations,
             'method':'bls' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # resort by time
    stimes, smags, errs = resort_by_time(stimes, smags, serrs)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # if we're setting up everything automatically
        if autofreq:

            # figure out the best number of phasebins to use
            nphasebins = int(npceil(2.0/mintransitduration))
            if nphasebins > 3000:
                nphasebins = 3000

            # use heuristic to figure out best timestep
            stepsize = 0.25*mintransitduration/(stimes.max()-stimes.min())

            # now figure out the frequencies to use
            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(npceil((maxfreq - minfreq)/stepsize))

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
            nfreq = int(npceil((maxfreq - minfreq)/stepsize))

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

        if ((minfreq < (1.0/(stimes.max() - stimes.min()))) and
            endp_timebase_check):

            LOGWARNING('the requested max P = %.3f is larger than '
                       'the time base of the observations = %.3f, '
                       ' will make minfreq = 2 x 1/timebase'
                       % (endp, stimes.max() - stimes.min()))
            minfreq = 2.0/(stimes.max() - stimes.min())
            LOGWARNING('new minfreq: %s, maxfreq: %s' %
                       (minfreq, maxfreq))

        #
        # run BLS
        #
        try:

            blsresult = _bls_runner(stimes,
                                    smags,
                                    nfreq,
                                    minfreq,
                                    stepsize,
                                    nphasebins,
                                    mintransitduration,
                                    maxtransitduration)

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

            sortedlspind = npargsort(finlsp)[::-1]
            sortedlspperiods = finperiods[sortedlspind]
            sortedlspvals = finlsp[sortedlspind]

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

                # this ensures that this period is different from the last
                # period and from all the other existing best periods by
                # periodepsilon to make sure we jump to an entire different
                # peak in the periodogram
                if (perioddiff > (periodepsilon*prevperiod) and
                    all(x > (periodepsilon*period)
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
                          'magsarefluxes':magsarefluxes}
            }

            # get stats if requested
            if get_stats:
                resultdict['stats'] = []
                for bp in nbestperiods:

                    if verbose:
                        LOGINFO("Getting stats for best period: %.6f" % bp)

                    this_pstats = bls_stats_singleperiod(
                        stimes, smags, serrs, bp,
                        magsarefluxes=resultdict['kwargs']['magsarefluxes'],
                        sigclip=resultdict['kwargs']['sigclip'],
                        nphasebins=resultdict['nphasebins'],
                        mintransitduration=resultdict['mintransitduration'],
                        maxtransitduration=resultdict['maxtransitduration'],
                        verbose=verbose,
                    )
                    resultdict['stats'].append(this_pstats)

            return resultdict

        except Exception:

            LOGEXCEPTION('BLS failed!')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'blsresult':None,
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
        sigclip=10.0,
        endp_timebase_check=True,
        verbose=True,
        nworkers=None,
        get_stats=True,
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

    Parameters
    ----------

    times,mags,errs : np.array
        The magnitude/flux time-series to search for transits.

    magsarefluxes : bool
        If the input measurement values in `mags` and `errs` are in fluxes, set
        this to True.

    startp,endp : float
        The minimum and maximum periods to consider for the transit search.

    stepsize : float
        The step-size in frequency to use when constructing a frequency grid for
        the period search.

    mintransitduration,maxtransitduration : float
        The minimum and maximum transitdurations (in units of phase) to consider
        for the transit search.

    nphasebins : int
        The number of phase bins to use in the period search.

    autofreq : bool
        If this is True, the values of `stepsize` and `nphasebins` will be
        ignored, and these, along with a frequency-grid, will be determined
        based on the following relations::

            nphasebins = int(ceil(2.0/mintransitduration))
            if nphasebins > 3000:
                nphasebins = 3000

            stepsize = 0.25*mintransitduration/(times.max()-times.min())

            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(ceil((maxfreq - minfreq)/stepsize))

    periodepsilon : float
        The fractional difference between successive values of 'best' periods
        when sorting by periodogram power to consider them as separate periods
        (as opposed to part of the same periodogram peak). This is used to avoid
        broad peaks in the periodogram and make sure the 'best' periods returned
        are all actually independent.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

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

    endp_timebase_check : bool
        If True, will check if the ``endp`` value is larger than the time-base
        of the observations. If it is, will change the ``endp`` value such that
        it is half of the time-base. If False, will allow an ``endp`` larger
        than the time-base of the observations.

    verbose : bool
        If this is True, will indicate progress and details about the frequency
        grid used for the period search.

    nworkers : int or None
        The number of parallel workers to launch for period-search. If None,
        nworkers = NCPUS.

    get_stats : bool
        If True, runs :py:func:`.bls_stats_singleperiod` for each of the best
        periods in the output and injects the output into the output dict so you
        only have to run this function to get the periods and their stats.

        The output dict from this function will then contain a 'stats' key
        containing a list of dicts with statistics for each period in
        ``resultdict['nbestperiods']``. These dicts will contain fit values of
        transit parameters after a trapezoid transit model is fit to the phased
        light curve at each period in ``resultdict['nbestperiods']``, i.e. fit
        values for period, epoch, transit depth, duration, ingress duration, and
        the SNR of the transit.

        NOTE: make sure to check the 'fit_status' key for each
        ``resultdict['stats']`` item to confirm that the trapezoid transit model
        fit succeeded and that the stats calculated are valid.

    Returns
    -------

    dict
        This function returns a dict, referred to as an `lspinfo` dict in other
        astrobase functions that operate on periodogram results. This is a
        standardized format across all astrobase period-finders, and is of the
        form below::

            {'bestperiod': the best period value in the periodogram,
             'bestlspval': the periodogram peak associated with the best period,
             'nbestpeaks': the input value of nbestpeaks,
             'nbestlspvals': nbestpeaks-size list of best period peak values,
             'nbestperiods': nbestpeaks-size list of best periods,
             'stats': list of stats dicts returned for each best period,
             'lspvals': the full array of periodogram powers,
             'frequencies': the full array of frequencies considered,
             'periods': the full array of periods considered,
             'blsresult': list of result dicts from eebls.f wrapper functions,
             'stepsize': the actual stepsize used,
             'nfreq': the actual nfreq used,
             'nphasebins': the actual nphasebins used,
             'mintransitduration': the input mintransitduration,
             'maxtransitduration': the input maxtransitdurations,
             'method':'bls' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # resort by time
    stimes, smags, errs = resort_by_time(stimes, smags, serrs)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # if we're setting up everything automatically
        if autofreq:

            # figure out the best number of phasebins to use
            nphasebins = int(npceil(2.0/mintransitduration))
            if nphasebins > 3000:
                nphasebins = 3000

            # use heuristic to figure out best timestep
            stepsize = 0.25*mintransitduration/(stimes.max()-stimes.min())

            # now figure out the frequencies to use
            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(npceil((maxfreq - minfreq)/stepsize))

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
            nfreq = int(npceil((maxfreq - minfreq)/stepsize))

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
        if ((minfreq < (1.0/(stimes.max() - stimes.min()))) and
            endp_timebase_check):

            LOGWARNING('the requested max P = %.3f is larger than '
                       'the time base of the observations = %.3f, '
                       ' will make minfreq = 2 x 1/timebase'
                       % (endp, stimes.max() - stimes.min()))
            minfreq = 2.0/(stimes.max() - stimes.min())
            LOGWARNING('new minfreq: %s, maxfreq: %s' %
                       (minfreq, maxfreq))

        #############################
        ## NOW RUN BLS IN PARALLEL ##
        #############################

        # fix number of CPUs if needed
        if not nworkers or nworkers > NCPUS:
            nworkers = NCPUS
            if verbose:
                LOGINFO('using %s workers...' % nworkers)

        # the frequencies array to be searched
        frequencies = minfreq + nparange(nfreq)*stepsize

        # break up the tasks into chunks
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

        # start the pool
        pool = Pool(nworkers)
        results = pool.map(_parallel_bls_worker, tasks)

        pool.close()
        pool.join()
        del pool

        # now concatenate the output lsp arrays
        lsp = npconcatenate([x['power'] for x in results])
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
                    'blsresult':None,
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

        sortedlspind = npargsort(finlsp)[::-1]
        sortedlspperiods = finperiods[sortedlspind]
        sortedlspvals = finlsp[sortedlspind]

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

            # this ensures that this period is different from the last
            # period and from all the other existing best periods by
            # periodepsilon to make sure we jump to an entire different
            # peak in the periodogram
            if (perioddiff > (periodepsilon*prevperiod) and
                all(x > (periodepsilon*period) for x in bestperiodsdiff)):
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
                      'magsarefluxes':magsarefluxes}
        }

        # get stats if requested
        if get_stats:

            resultdict['stats'] = []

            for bp in nbestperiods.copy():

                if verbose:
                    LOGINFO("Getting stats for best period: %.6f" % bp)

                this_pstats = bls_stats_singleperiod(
                    stimes, smags, serrs, bp,
                    magsarefluxes=resultdict['kwargs']['magsarefluxes'],
                    sigclip=resultdict['kwargs']['sigclip'],
                    nphasebins=resultdict['nphasebins'],
                    mintransitduration=resultdict['mintransitduration'],
                    maxtransitduration=resultdict['maxtransitduration'],
                    verbose=verbose,
                )
                resultdict['stats'].append(this_pstats)

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


def _get_bls_stats(stimes,
                   smags,
                   serrs,
                   thistransdepth,
                   thistransduration,
                   ingressdurationfraction,
                   nphasebins,
                   thistransingressbin,
                   thistransegressbin,
                   thisbestperiod,
                   thisnphasebins,
                   magsarefluxes=False,
                   verbose=False):
    '''
    Actually calculates the stats.

    '''

    try:

        # try getting the minimum light epoch using the phase bin method
        me_epochbin = int((thistransegressbin +
                           thistransingressbin)/2.0)

        me_phases = (
            (stimes - stimes.min())/thisbestperiod -
            npfloor((stimes - stimes.min())/thisbestperiod)
        )
        me_phases_sortind = npargsort(me_phases)
        me_sorted_phases = me_phases[me_phases_sortind]
        me_sorted_times = stimes[me_phases_sortind]

        me_bins = nplinspace(0.0, 1.0, thisnphasebins)
        me_bininds = npdigitize(me_sorted_phases, me_bins)

        me_centertransit_ind = me_bininds == me_epochbin
        me_centertransit_phase = (
            npmedian(me_sorted_phases[me_centertransit_ind])
        )
        me_centertransit_timeloc = npwhere(
            npabs(me_sorted_phases - me_centertransit_phase) ==
            npmin(npabs(me_sorted_phases - me_centertransit_phase))
        )
        me_centertransit_time = me_sorted_times[
            me_centertransit_timeloc
        ]

        if me_centertransit_time.size > 1:
            LOGWARNING('multiple possible times-of-center transits '
                       'found for period %.7f, picking the first '
                       'one from: %s' %
                       (thisbestperiod, repr(me_centertransit_time)))

        thisminepoch = me_centertransit_time[0]

    except Exception:

        LOGEXCEPTION(
            'could not determine the center time of transit for '
            'the phased LC, trying SavGol fit instead...'
        )
        # fit a Savitsky-Golay instead and get its minimum
        savfit = savgol_fit_magseries(stimes, smags, serrs,
                                      thisbestperiod,
                                      magsarefluxes=magsarefluxes,
                                      verbose=verbose,
                                      sigclip=None)
        thisminepoch = savfit['fitinfo']['fitepoch']

    if isinstance(thisminepoch, npndarray):
        if verbose:
            LOGWARNING('minimum epoch is actually an array:\n'
                       '%s\n'
                       'instead of a float, '
                       'are there duplicate time values '
                       'in the original input? '
                       'will use the first value in this array.'
                       % repr(thisminepoch))
        thisminepoch = thisminepoch[0]

    # make sure the INITIAL value of the ingress duration isn't more than half
    # of the total duration of the transit
    model_ingressduration = ingressdurationfraction*thistransduration
    if model_ingressduration > (0.5*thistransduration):
        model_ingressduration = 0.5*thistransduration

    #
    # set the depth bounds appropriately for the type of mag input
    #

    # require positive depth for fluxes
    if magsarefluxes:
        transit_depth_bounds = (0.0, npinf)

    # require negative depth for mags
    else:
        transit_depth_bounds = (-npinf, 0.0)

    # set up trapezoid transit model to fit for this LC
    transitparams = [
        thisbestperiod,
        thisminepoch,
        thistransdepth,
        thistransduration,
        model_ingressduration
    ]

    #
    # run the model fit
    #
    modelfit = traptransit_fit_magseries(
        stimes,
        smags,
        serrs,
        transitparams,
        sigclip=None,
        magsarefluxes=magsarefluxes,
        param_bounds={
            'period':(0.0, npinf),
            'epoch':(0.0, npinf),
            'depth':transit_depth_bounds,
            'duration':(0.0,1.0),
            # allow ingress duration during fitting to float up to 0.5
            # FIXME: this should technically always be a fn of duration but
            # curve_fit doesn't support this kind of constraint
            # use scipy.optimize.minimize instead?
            'ingressduration':(0.0,0.5),
        },
        verbose=verbose
    )

    # if the model fit succeeds, calculate SNR using the trapezoid model fit
    if modelfit and modelfit['fitinfo']['finalparams'] is not None:

        fitparams = modelfit['fitinfo']['finalparams']
        fiterrs = modelfit['fitinfo']['finalparamerrs']
        modelmags, actualmags, modelphase = (
            modelfit['fitinfo']['fitmags'],
            modelfit['magseries']['mags'],
            modelfit['magseries']['phase']
        )
        subtractedmags = actualmags - modelmags
        subtractedrms = npstd(subtractedmags)
        fit_period, fit_epoch, fit_depth, fit_duration, fit_ingress_dur = (
            fitparams
        )

        npts_in_transit = modelfit['fitinfo']['ntransitpoints']
        transit_snr = (
            npsqrt(npts_in_transit) * npabs(fit_depth/subtractedrms)
        )

        if verbose:

            LOGINFO('refit best period: %.6f, '
                    'refit center of transit: %.5f' %
                    (fit_period, fit_epoch))

            LOGINFO('npoints in transit: %s' % npts_in_transit)

            LOGINFO('transit depth (delta): %.5f, '
                    'frac transit length (q): %.3f, '
                    ' SNR: %.3f' %
                    (fit_depth,
                     fit_duration,
                     transit_snr))

        return {'period':fit_period,
                'epoch':fit_epoch,
                'snr':transit_snr,
                'transitdepth':fit_depth,
                'transitduration':fit_duration,
                'ingressduration':fit_ingress_dur,
                'npoints_in_transit':npts_in_transit,
                'fitparams':fitparams,
                'fiterrs':fiterrs,
                'fit_status':'ok',
                'nphasebins':nphasebins,
                'transingressbin':thistransingressbin,
                'transegressbin':thistransegressbin,
                'blsmodel':modelmags,
                'subtractedmags':subtractedmags,
                'phasedmags':actualmags,
                'phases':modelphase,
                'fitinfo':modelfit}

    # if the model fit doesn't work, then do the SNR calculation the old way
    else:

        # phase using this epoch
        phased_magseries = phase_magseries_with_errs(stimes,
                                                     smags,
                                                     serrs,
                                                     thisbestperiod,
                                                     thisminepoch,
                                                     wrap=False,
                                                     sort=True)

        tphase = phased_magseries['phase']
        tmags = phased_magseries['mags']

        # use the transit depth and duration to subtract the BLS transit
        # model from the phased mag series. we're centered about 0.0 as the
        # phase of the transit minimum so we need to look at stuff from
        # [0.0, transitphase] and [1.0-transitphase, 1.0]
        transitphase = thistransduration/2.0

        transitindices = ((tphase < transitphase) |
                          (tphase > (1.0 - transitphase)))

        # this is the BLS model
        # constant = median(tmags) outside transit
        # constant = thistransitdepth inside transit
        blsmodel = npfull_like(tmags, npmedian(tmags))

        if magsarefluxes:

            # eebls.f returns +ve transit depth for fluxes
            # so we need to subtract here to get fainter fluxes in transit
            blsmodel[transitindices] = (
                blsmodel[transitindices] - thistransdepth
            )
        else:

            # eebls.f returns -ve transit depth for magnitudes
            # so we need to subtract here to get fainter mags in transits
            blsmodel[transitindices] = (
                blsmodel[transitindices] - thistransdepth
            )

        # see __init__/get_snr_of_dip docstring for description of transit
        # SNR equation, which is what we use for `thissnr`.
        subtractedmags = tmags - blsmodel
        subtractedrms = npstd(subtractedmags)
        npts_in_transit = len(tmags[transitindices])
        thissnr = (
            npsqrt(npts_in_transit) * npabs(thistransdepth/subtractedrms)
        )

        # tell user about stuff if verbose = True
        if verbose:

            LOGINFO('refit best period: %.6f, '
                    'refit center of transit: %.5f' %
                    (thisbestperiod, thisminepoch))

            LOGINFO('transit ingress phase = %.3f to %.3f' % (1.0 -
                                                              transitphase,
                                                              1.0))
            LOGINFO('transit egress phase = %.3f to %.3f' % (0.0,
                                                             transitphase))
            LOGINFO('npoints in transit: %s' % tmags[transitindices].size)

            LOGINFO('transit depth (delta): %.5f, '
                    'frac transit length (q): %.3f, '
                    ' SNR: %.3f' %
                    (thistransdepth,
                     thistransduration,
                     thissnr))

        return {'period':thisbestperiod,
                'epoch':thisminepoch,
                'snr':thissnr,
                'transitdepth':thistransdepth,
                'transitduration':thistransduration,
                'npoints_in_transit':npts_in_transit,
                'fit_status':'trapezoid model fit failed, using box model',
                'nphasebins':nphasebins,
                'transingressbin':thistransingressbin,
                'transegressbin':thistransegressbin,
                'blsmodel':blsmodel,
                'subtractedmags':subtractedmags,
                'phasedmags':tmags,
                'phases':tphase}


def bls_stats_singleperiod(times, mags, errs, period,
                           magsarefluxes=False,
                           sigclip=10.0,
                           perioddeltapercent=10,
                           nphasebins=200,
                           mintransitduration=0.01,
                           maxtransitduration=0.4,
                           ingressdurationfraction=0.1,
                           verbose=True):
    '''This calculates the SNR, depth, duration, a refit period, and time of
    center-transit for a single period.

    The equation used for SNR is::

        SNR = (transit model depth / RMS of LC with transit model subtracted)
              * sqrt(number of points in transit)

    NOTE: you should set the kwargs `sigclip`, `nphasebins`,
    `mintransitduration`, `maxtransitduration` to what you used for an initial
    BLS run to detect transits in the input light curve to match those input
    conditions.

    Parameters
    ----------

    times,mags,errs : np.array
        These contain the magnitude/flux time-series and any associated errors.

    period : float
        The period to search around and refit the transits. This will be used to
        calculate the start and end periods of a rerun of BLS to calculate the
        stats.

    magsarefluxes : bool
        Set to True if the input measurements in `mags` are actually fluxes and
        not magnitudes.

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

    perioddeltapercent : float
        The fraction of the period provided to use to search around this
        value. This is a percentage. The period range searched will then be::

            [period - (perioddeltapercent/100.0)*period,
             period + (perioddeltapercent/100.0)*period]

    nphasebins : int
        The number of phase bins to use in the BLS run.

    mintransitduration : float
        The minimum transit duration in phase to consider.

    maxtransitduration : float
        The maximum transit duration to consider.

    ingressdurationfraction : float
        The fraction of the transit duration to use to generate an initial value
        of the transit ingress duration for the BLS model refit. This will be
        fit by this function.

    verbose : bool
        If True, will indicate progress and any problems encountered.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'period': the refit best period,
             'epoch': the refit epoch (i.e. mid-transit time),
             'snr':the SNR of the transit,
             'transitdepth':the depth of the transit,
             'transitduration':the duration of the transit,
             'ingressduration':if trapezoid fit OK, is the ingress duration,
             'npoints_in_transit':the number of LC points in transit,
             'fit_status': 'ok' or 'trapezoid model fit failed,...',
             'nphasebins':the input value of nphasebins,
             'transingressbin':the phase bin containing transit ingress,
             'transegressbin':the phase bin containing transit egress,
             'blsmodel':the full BLS model used along with its parameters,
             'subtractedmags':BLS model - phased light curve,
             'phasedmags':the phase light curve,
             'phases': the phase values}

        You should check the 'fit_status' key in this returned dict for a value
        of 'ok'. If it is 'trapezoid model fit failed, using box model', you may
        not want to trust the transit period and epoch found.

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # get the period interval
        startp = period - perioddeltapercent*period/100.0

        if startp < 0:
            startp = period

        endp = period + perioddeltapercent*period/100.0

        # rerun BLS in serial mode around the specified period to get the
        # transit depth, duration, ingress and egress bins
        blsres = bls_serial_pfind(stimes, smags, serrs,
                                  verbose=verbose,
                                  startp=startp,
                                  endp=endp,
                                  nphasebins=nphasebins,
                                  mintransitduration=mintransitduration,
                                  maxtransitduration=maxtransitduration,
                                  magsarefluxes=magsarefluxes,
                                  get_stats=False,
                                  sigclip=None)

        if (not blsres or
            'blsresult' not in blsres or
            blsres['blsresult'] is None):
            LOGERROR("BLS failed during a period-search "
                     "performed around the input best period: %.6f. "
                     "Can't continue. " % period)
            return None

        thistransdepth = blsres['blsresult']['transdepth']
        thistransduration = blsres['blsresult']['transduration']
        thisbestperiod = blsres['bestperiod']
        thistransingressbin = blsres['blsresult']['transingressbin']
        thistransegressbin = blsres['blsresult']['transegressbin']
        thisnphasebins = nphasebins

        stats = _get_bls_stats(stimes,
                               smags,
                               serrs,
                               thistransdepth,
                               thistransduration,
                               ingressdurationfraction,
                               nphasebins,
                               thistransingressbin,
                               thistransegressbin,
                               thisbestperiod,
                               thisnphasebins,
                               magsarefluxes=magsarefluxes,
                               verbose=verbose)

        return stats

    # if there aren't enough points in the mag series, bail out
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return None


def bls_snr(blsdict,
            times,
            mags,
            errs,
            assumeserialbls=False,
            magsarefluxes=False,
            sigclip=10.0,
            npeaks=None,
            perioddeltapercent=10,
            ingressdurationfraction=0.1,
            verbose=True):
    '''Calculates the signal to noise ratio for each best peak in the BLS
    periodogram, along with transit depth, duration, and refit period and epoch.

    The following equation is used for SNR::

        SNR = (transit model depth / RMS of LC with transit model subtracted)
              * sqrt(number of points in transit)

    Parameters
    ----------

    blsdict : dict
        This is an lspinfo dict produced by either `bls_parallel_pfind` or
        `bls_serial_pfind` in this module, or by your own BLS function. If you
        provide results in a dict from an external BLS function, make sure this
        matches the form below::

            {'bestperiod': the best period value in the periodogram,
             'bestlspval': the periodogram peak associated with the best period,
             'nbestpeaks': the input value of nbestpeaks,
             'nbestlspvals': nbestpeaks-size list of best period peak values,
             'nbestperiods': nbestpeaks-size list of best periods,
             'lspvals': the full array of periodogram powers,
             'frequencies': the full array of frequencies considered,
             'periods': the full array of periods considered,
             'blsresult': list of result dicts from eebls.f wrapper functions,
             'stepsize': the actual stepsize used,
             'nfreq': the actual nfreq used,
             'nphasebins': the actual nphasebins used,
             'mintransitduration': the input mintransitduration,
             'maxtransitduration': the input maxtransitdurations,
             'method':'bls' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    times,mags,errs : np.array
        These contain the magnitude/flux time-series and any associated errors.

    assumeserialbls : bool
        If this is True, this function will not rerun BLS around each best peak
        in the input lspinfo dict to refit the periods and epochs. This is
        usally required for `bls_parallel_pfind` so set this to False if you use
        results from that function. The parallel method breaks up the frequency
        space into chunks for speed, and the results may not exactly match those
        from a regular BLS run.

    magsarefluxes : bool
        Set to True if the input measurements in `mags` are actually fluxes and
        not magnitudes.

    npeaks : int or None
        This controls how many of the periods in `blsdict['nbestperiods']` to
        find the SNR for. If it's None, then this will calculate the SNR for all
        of them. If it's an integer between 1 and
        `len(blsdict['nbestperiods'])`, will calculate for only the specified
        number of peak periods, starting from the best period.

    perioddeltapercent : float
        The fraction of the period provided to use to search around this
        value. This is a percentage. The period range searched will then be::

            [period - (perioddeltapercent/100.0)*period,
             period + (perioddeltapercent/100.0)*period]

    ingressdurationfraction : float
        The fraction of the transit duration to use to generate an initial value
        of the transit ingress duration for the BLS model refit. This will be
        fit by this function.

    verbose : bool
        If True, will indicate progress and any problems encountered.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'npeaks: the number of periodogram peaks requested to get SNR for,
             'period': list of refit best periods for each requested peak,
             'epoch': list of refit epochs (i.e. mid-transit times),
             'snr':list of SNRs of the transit for each requested peak,
             'transitdepth':list of depths of the transits,
             'transitduration':list of durations of the transits,
             'nphasebins':the input value of nphasebins,
             'transingressbin':the phase bin containing transit ingress,
             'transegressbin':the phase bin containing transit egress,
             'allblsmodels':the full BLS models used along with its parameters,
             'allsubtractedmags':BLS models - phased light curves,
             'allphasedmags':the phase light curves,
             'allphases': the phase values}

    '''

    # figure out how many periods to work on
    if (npeaks and (0 < npeaks < len(blsdict['nbestperiods']))):
        nperiods = npeaks
    else:
        if verbose:
            LOGWARNING('npeaks not specified or invalid, '
                       'getting SNR for all %s BLS peaks' %
                       len(blsdict['nbestperiods']))
        nperiods = len(blsdict['nbestperiods'])

    nbestperiods = blsdict['nbestperiods'][:nperiods]

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        nbestsnrs = []
        transitdepth, transitduration = [], []
        ingressduration, points_in_transit, fit_status = [], [], []
        nphasebins, transingressbin, transegressbin = [], [], []

        # keep these around for diagnostics
        allsubtractedmags = []
        allphasedmags = []
        allphases = []
        allblsmodels = []

        # these are refit periods and epochs
        refitperiods = []
        refitepochs = []

        for period in nbestperiods:

            # get the period interval
            startp = period - perioddeltapercent*period/100.0

            if startp < 0:
                startp = period

            endp = period + perioddeltapercent*period/100.0

            # see if we need to rerun bls_serial_pfind
            if not assumeserialbls:

                # run bls_serial_pfind with the kwargs copied over from the
                # initial run. replace only the startp, endp, verbose, sigclip
                # kwarg values
                prevkwargs = blsdict['kwargs'].copy()
                prevkwargs['verbose'] = verbose
                prevkwargs['startp'] = startp
                prevkwargs['endp'] = endp
                prevkwargs['sigclip'] = None

                blsres = bls_serial_pfind(stimes,
                                          smags,
                                          serrs,
                                          **prevkwargs)

            else:
                blsres = blsdict

            thistransdepth = blsres['blsresult']['transdepth']
            thistransduration = blsres['blsresult']['transduration']
            thisbestperiod = blsres['bestperiod']
            thistransingressbin = blsres['blsresult']['transingressbin']
            thistransegressbin = blsres['blsresult']['transegressbin']
            thisnphasebins = blsdict['kwargs']['nphasebins']

            stats = _get_bls_stats(stimes,
                                   smags,
                                   serrs,
                                   thistransdepth,
                                   thistransduration,
                                   ingressdurationfraction,
                                   nphasebins,
                                   thistransingressbin,
                                   thistransegressbin,
                                   thisbestperiod,
                                   thisnphasebins,
                                   magsarefluxes=magsarefluxes,
                                   verbose=verbose)

            # update the lists with results from this peak
            nbestsnrs.append(stats['snr'])
            transitdepth.append(stats['transitdepth'])
            transitduration.append(stats['transitduration'])
            ingressduration.append(stats['ingressduration'])
            points_in_transit.append(stats['npoints_in_transit'])
            fit_status.append(stats['fit_status'])

            transingressbin.append(stats['transingressbin'])
            transegressbin.append(stats['transegressbin'])
            nphasebins.append(stats['nphasebins'])

            # update the refit periods and epochs
            refitperiods.append(stats['period'])
            refitepochs.append(stats['epoch'])

            # update the diagnostics
            allsubtractedmags.append(stats['subtractedmags'])
            allphasedmags.append(stats['phasedmags'])
            allphases.append(stats['phases'])
            allblsmodels.append(stats['blsmodel'])

        #
        # done with working on each peak
        #

    # if there aren't enough points in the mag series, bail out
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        nbestsnrs = None
        transitdepth, transitduration = None, None
        ingressduration, points_in_transit, fit_status = None, None, None
        nphasebins, transingressbin, transegressbin = None, None, None
        allsubtractedmags, allphases, allphasedmags = None, None, None

    return {'npeaks':npeaks,
            'period':refitperiods,
            'epoch':refitepochs,
            'snr':nbestsnrs,
            'transitdepth':transitdepth,
            'transitduration':transitduration,
            'ingressduration':ingressduration,
            'npoints_in_transit':points_in_transit,
            'fit_status':fit_status,
            'nphasebins':nphasebins,
            'transingressbin':transingressbin,
            'transegressbin':transegressbin,
            'allblsmodels':allblsmodels,
            'allsubtractedmags':allsubtractedmags,
            'allphasedmags':allphasedmags,
            'allphases':allphases}
