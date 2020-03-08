#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# abls.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''
Contains the Kovacs, et al. (2002) Box-Least-squared-Search period-search
algorithm implementation for periodbase. This uses the implementation in Astropy
3.1, so requires that version.

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

from multiprocessing import Pool, cpu_count

from math import fmod

from numpy import (
    nan as npnan, arange as nparange, array as nparray,
    isfinite as npisfinite, argmax as npargmax, linspace as nplinspace,
    ceil as npceil, argsort as npargsort, concatenate as npconcatenate
)

try:
    from astropy.stats import BoxLeastSquares
except ImportError:
    from astropy.timeseries import BoxLeastSquares

from astropy import units as u


###################
## LOCAL IMPORTS ##
###################

from ..lcmath import sigclip_magseries
from .utils import resort_by_time

############
## CONFIG ##
############

NCPUS = cpu_count()


#######################
## UTILITY FUNCTIONS ##
#######################

def bls_serial_pfind(times, mags, errs,
                     magsarefluxes=False,
                     startp=0.1,  # search from 0.1 d to...
                     endp=100.0,  # ... 100.0 d -- don't search full timebase
                     stepsize=5.0e-4,
                     mintransitduration=0.01,  # minimum transit length in phase
                     maxtransitduration=0.4,   # maximum transit length in phase
                     ndurations=100,
                     autofreq=True,  # figure out f0, nf, and df automatically
                     blsobjective='likelihood',
                     blsmethod='fast',
                     blsoversample=10,
                     blsmintransits=3,
                     blsfreqfactor=10.0,
                     periodepsilon=0.1,
                     nbestpeaks=5,
                     sigclip=10.0,
                     endp_timebase_check=True,
                     verbose=True,
                     raiseonfail=False):
    '''Runs the Box Least Squares Fitting Search for transit-shaped signals.

    Based on the version of BLS in Astropy 3.1:
    `astropy.stats.BoxLeastSquares`. If you don't have Astropy 3.1, this module
    will fail to import. Note that by default, this implementation of
    `bls_serial_pfind` doesn't use the `.autoperiod()` function from
    `BoxLeastSquares` but uses the same auto frequency-grid generation as the
    functions in `periodbase.kbls`. If you want to use Astropy's implementation,
    set the value of `autofreq` kwarg to 'astropy'.

    The dict returned from this function contains a `blsmodel` key, which is the
    generated model from Astropy's BLS. Use the `.compute_stats()` method to
    calculate the required stats like SNR, depth, duration, etc.

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

    ndurations : int
        The number of transit durations to use in the period-search.

    autofreq : bool or str
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

        If this is False, you must set `startp`, `endp`, and `stepsize` as
        appropriate.

        If this is str == 'astropy', will use the
        `astropy.stats.BoxLeastSquares.autoperiod()` function to calculate the
        frequency grid instead of the kbls method.

    blsobjective : {'likelihood','snr'}
        Sets the type of objective to optimize in the `BoxLeastSquares.power()`
        function.

    blsmethod : {'fast','slow'}
        Sets the type of method to use in the `BoxLeastSquares.power()`
        function.

    blsoversample : {'likelihood','snr'}
        Sets the `oversample` kwarg for the `BoxLeastSquares.power()` function.

    blsmintransits : int
        Sets the `min_n_transits` kwarg for the `BoxLeastSquares.autoperiod()`
        function.

    blsfreqfactor : float
        Sets the `frequency_factor` kwarg for the `BoxLeastSquares.autperiod()`
        function.

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

    raiseonfail : bool
        If True, raises an exception if something goes wrong. Otherwise, returns
        None.

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
             'lspvals': the full array of periodogram powers,
             'frequencies': the full array of frequencies considered,
             'periods': the full array of periods considered,
             'durations': the array of durations used to run BLS,
             'blsresult': Astropy BLS result object (BoxLeastSquaresResult),
             'blsmodel': Astropy BLS BoxLeastSquares object used for work,
             'stepsize': the actual stepsize used,
             'nfreq': the actual nfreq used,
             'durations': the durations array used,
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
        if isinstance(autofreq, bool) and autofreq:

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
                        'freq stepsize: %s, ndurations: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, ndurations,
                         mintransitduration, maxtransitduration))

            use_autoperiod = False

        elif isinstance(autofreq, bool) and not autofreq:

            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(npceil((maxfreq - minfreq)/stepsize))

            # say what we're using
            if verbose:
                LOGINFO('min P: %s, max P: %s, nfreq: %s, '
                        'minfreq: %s, maxfreq: %s' % (startp, endp, nfreq,
                                                      minfreq, maxfreq))
                LOGINFO('autofreq = False: using PROVIDED values for '
                        'freq stepsize: %s, ndurations: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, ndurations,
                         mintransitduration, maxtransitduration))

            use_autoperiod = False

        elif isinstance(autofreq, str) and autofreq == 'astropy':

            use_autoperiod = True
            minfreq = 1.0/endp
            maxfreq = 1.0/startp

        else:

            LOGERROR("unknown autofreq kwarg encountered. can't continue...")
            return None

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

        # run BLS
        try:

            # astropy's BLS requires durations in units of time
            durations = nplinspace(mintransitduration*startp,
                                   maxtransitduration*startp,
                                   ndurations)

            # set up the correct units for the BLS model
            if magsarefluxes:

                blsmodel = BoxLeastSquares(
                    stimes*u.day,
                    smags*u.dimensionless_unscaled,
                    dy=serrs*u.dimensionless_unscaled
                )

            else:

                blsmodel = BoxLeastSquares(
                    stimes*u.day,
                    smags*u.mag,
                    dy=serrs*u.mag
                )

            # use autoperiod if requested
            if use_autoperiod:
                periods = nparray(
                    blsmodel.autoperiod(
                        durations,
                        minimum_period=startp,
                        maximum_period=endp,
                        minimum_n_transit=blsmintransits,
                        frequency_factor=blsfreqfactor
                    )
                )
                nfreq = periods.size

                if verbose:
                    LOGINFO(
                        "autofreq = 'astropy', used .autoperiod() with "
                        "minimum_n_transit = %s, freq_factor = %s "
                        "to generate the frequency grid" %
                        (blsmintransits, blsfreqfactor)
                    )
                    LOGINFO('stepsize = %.5f, nfreq = %s, minfreq = %.5f, '
                            'maxfreq = %.5f, ndurations = %s' %
                            (abs(1.0/periods[1] - 1.0/periods[0]),
                             nfreq,
                             1.0/periods.max(),
                             1.0/periods.min(),
                             durations.size))

            # otherwise, use kbls method
            else:
                frequencies = minfreq + nparange(nfreq)*stepsize
                periods = 1.0/frequencies

            if nfreq > 5.0e5:
                if verbose:
                    LOGWARNING('more than 5.0e5 frequencies to go through; '
                               'this will take a while. '
                               'you might want to use the '
                               'abls.bls_parallel_pfind function instead')

            # run the periodogram
            blsresult = blsmodel.power(
                periods*u.day,
                durations*u.day,
                objective=blsobjective,
                method=blsmethod,
                oversample=blsoversample
            )

            # get the peak values
            lsp = nparray(blsresult.power)

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
                        'nbestinds':None,
                        'nbestlspvals':None,
                        'nbestperiods':None,
                        'lspvals':None,
                        'periods':None,
                        'durations':None,
                        'method':'bls',
                        'blsresult':None,
                        'blsmodel':None,
                        'kwargs':{'startp':startp,
                                  'endp':endp,
                                  'stepsize':stepsize,
                                  'mintransitduration':mintransitduration,
                                  'maxtransitduration':maxtransitduration,
                                  'ndurations':ndurations,
                                  'blsobjective':blsobjective,
                                  'blsmethod':blsmethod,
                                  'blsoversample':blsoversample,
                                  'blsntransits':blsmintransits,
                                  'blsfreqfactor':blsfreqfactor,
                                  'autofreq':autofreq,
                                  'periodepsilon':periodepsilon,
                                  'nbestpeaks':nbestpeaks,
                                  'sigclip':sigclip,
                                  'magsarefluxes':magsarefluxes}}

            sortedlspind = npargsort(finlsp)[::-1]
            sortedlspperiods = finperiods[sortedlspind]
            sortedlspvals = finlsp[sortedlspind]

            # now get the nbestpeaks
            nbestperiods, nbestlspvals, nbestinds, peakcount = (
                [finperiods[bestperiodind]],
                [finlsp[bestperiodind]],
                [bestperiodind],
                1
            )
            prevperiod = sortedlspperiods[0]

            # find the best nbestpeaks in the lsp and their periods
            for period, lspval, ind in zip(sortedlspperiods,
                                           sortedlspvals,
                                           sortedlspind):

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
                    all(x > (periodepsilon*period)
                        for x in bestperiodsdiff)):
                    nbestperiods.append(period)
                    nbestlspvals.append(lspval)
                    nbestinds.append(ind)
                    peakcount = peakcount + 1

                prevperiod = period

            # generate the return dict
            resultdict = {
                'bestperiod':finperiods[bestperiodind],
                'bestlspval':finlsp[bestperiodind],
                'nbestpeaks':nbestpeaks,
                'nbestinds':nbestinds,
                'nbestlspvals':nbestlspvals,
                'nbestperiods':nbestperiods,
                'lspvals':lsp,
                'frequencies':frequencies,
                'periods':periods,
                'durations':durations,
                'blsresult':blsresult,
                'blsmodel':blsmodel,
                'stepsize':stepsize,
                'nfreq':nfreq,
                'mintransitduration':mintransitduration,
                'maxtransitduration':maxtransitduration,
                'method':'bls',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'mintransitduration':mintransitduration,
                          'maxtransitduration':maxtransitduration,
                          'ndurations':ndurations,
                          'blsobjective':blsobjective,
                          'blsmethod':blsmethod,
                          'blsoversample':blsoversample,
                          'blsntransits':blsmintransits,
                          'blsfreqfactor':blsfreqfactor,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip,
                          'magsarefluxes':magsarefluxes}
            }

            return resultdict

        except Exception:

            LOGEXCEPTION('BLS failed!')

            if raiseonfail:
                raise

            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestinds':None,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'durations':None,
                    'blsresult':None,
                    'blsmodel':None,
                    'stepsize':stepsize,
                    'nfreq':nfreq,
                    'mintransitduration':mintransitduration,
                    'maxtransitduration':maxtransitduration,
                    'method':'bls',
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'mintransitduration':mintransitduration,
                              'maxtransitduration':maxtransitduration,
                              'ndurations':ndurations,
                              'blsobjective':blsobjective,
                              'blsmethod':blsmethod,
                              'blsoversample':blsoversample,
                              'blsntransits':blsmintransits,
                              'blsfreqfactor':blsfreqfactor,
                              'autofreq':autofreq,
                              'periodepsilon':periodepsilon,
                              'nbestpeaks':nbestpeaks,
                              'sigclip':sigclip,
                              'magsarefluxes':magsarefluxes}}

    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestinds':None,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'durations':None,
                'blsresult':None,
                'blsmodel':None,
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
                          'ndurations':ndurations,
                          'blsobjective':blsobjective,
                          'blsmethod':blsmethod,
                          'blsoversample':blsoversample,
                          'blsntransits':blsmintransits,
                          'blsfreqfactor':blsfreqfactor,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip,
                          'magsarefluxes':magsarefluxes}}


def _parallel_bls_worker(task):
    '''
    This wraps Astropy's BoxLeastSquares for use with bls_parallel_pfind below.

    `task` is a tuple::

        task[0] = times
        task[1] = mags
        task[2] = errs
        task[3] = magsarefluxes

        task[4] = minfreq
        task[5] = nfreq
        task[6] = stepsize

        task[7] = ndurations
        task[8] = mintransitduration
        task[9] = maxtransitduration

        task[10] = blsobjective
        task[11] = blsmethod
        task[12] = blsoversample

    '''

    try:

        times, mags, errs = task[:3]
        magsarefluxes = task[3]

        minfreq, nfreq, stepsize = task[4:7]

        ndurations, mintransitduration, maxtransitduration = task[7:10]

        blsobjective, blsmethod, blsoversample = task[10:]

        frequencies = minfreq + nparange(nfreq)*stepsize
        periods = 1.0/frequencies

        # astropy's BLS requires durations in units of time
        durations = nplinspace(mintransitduration*periods.min(),
                               maxtransitduration*periods.min(),
                               ndurations)

        # set up the correct units for the BLS model
        if magsarefluxes:

            blsmodel = BoxLeastSquares(
                times*u.day,
                mags*u.dimensionless_unscaled,
                dy=errs*u.dimensionless_unscaled
            )

        else:

            blsmodel = BoxLeastSquares(
                times*u.day,
                mags*u.mag,
                dy=errs*u.mag
            )

        blsresult = blsmodel.power(
            periods*u.day,
            durations*u.day,
            objective=blsobjective,
            method=blsmethod,
            oversample=blsoversample
        )

        return {
            'blsresult': blsresult,
            'blsmodel': blsmodel,
            'durations': durations,
            'power': nparray(blsresult.power)
        }

    except Exception:

        LOGEXCEPTION('BLS for frequency chunk: (%.6f, %.6f) failed.' %
                     (frequencies[0], frequencies[-1]))

        return {
            'blsresult': None,
            'blsmodel': None,
            'durations': durations,
            'power': nparray([npnan for x in range(nfreq)]),
        }


def bls_parallel_pfind(
        times, mags, errs,
        magsarefluxes=False,
        startp=0.1,  # by default, search from 0.1 d to...
        endp=100.0,  # ... 100.0 d -- don't search full timebase
        stepsize=1.0e-4,
        mintransitduration=0.01,  # minimum transit length in phase
        maxtransitduration=0.4,   # maximum transit length in phase
        ndurations=100,
        autofreq=True,  # figure out f0, nf, and df automatically
        blsobjective='likelihood',
        blsmethod='fast',
        blsoversample=5,
        blsmintransits=3,
        blsfreqfactor=10.0,
        nbestpeaks=5,
        periodepsilon=0.1,  # 0.1
        sigclip=10.0,
        endp_timebase_check=True,
        verbose=True,
        nworkers=None,
):
    '''Runs the Box Least Squares Fitting Search for transit-shaped signals.

    Breaks up the full frequency space into chunks and passes them to parallel
    BLS workers.

    Based on the version of BLS in Astropy 3.1:
    `astropy.stats.BoxLeastSquares`. If you don't have Astropy 3.1, this module
    will fail to import. Note that by default, this implementation of
    `bls_parallel_pfind` doesn't use the `.autoperiod()` function from
    `BoxLeastSquares` but uses the same auto frequency-grid generation as the
    functions in `periodbase.kbls`. If you want to use Astropy's implementation,
    set the value of `autofreq` kwarg to 'astropy'. The generated period array
    will then be broken up into chunks and sent to the individual workers.

    NOTE: the combined BLS spectrum produced by this function is not identical
    to that produced by running BLS in one shot for the entire frequency
    space. There are differences on the order of 1.0e-3 or so in the respective
    peak values, but peaks appear at the same frequencies for both methods. This
    is likely due to different aliasing caused by smaller chunks of the
    frequency space used by the parallel workers in this function. When in
    doubt, confirm results for this parallel implementation by comparing to
    those from the serial implementation above.

    In particular, when you want to get reliable estimates of the SNR, transit
    depth, duration, etc. that Astropy's BLS gives you, rerun `bls_serial_pfind`
    with `startp`, and `endp` close to the best period you want to characterize
    the transit at. The dict returned from that function contains a `blsmodel`
    key, which is the generated model from Astropy's BLS. Use the
    `.compute_stats()` method to calculate the required stats.

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

    ndurations : int
        The number of transit durations to use in the period-search.

    autofreq : bool or str
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

        If this is False, you must set `startp`, `endp`, and `stepsize` as
        appropriate.

        If this is str == 'astropy', will use the
        `astropy.stats.BoxLeastSquares.autoperiod()` function to calculate the
        frequency grid instead of the kbls method.

    blsobjective : {'likelihood','snr'}
        Sets the type of objective to optimize in the `BoxLeastSquares.power()`
        function.

    blsmethod : {'fast','slow'}
        Sets the type of method to use in the `BoxLeastSquares.power()`
        function.

    blsoversample : {'likelihood','snr'}
        Sets the `oversample` kwarg for the `BoxLeastSquares.power()` function.

    blsmintransits : int
        Sets the `min_n_transits` kwarg for the `BoxLeastSquares.autoperiod()`
        function.

    blsfreqfactor : float
        Sets the `frequency_factor` kwarg for the `BoxLeastSquares.autoperiod()`
        function.

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
             'lspvals': the full array of periodogram powers,
             'frequencies': the full array of frequencies considered,
             'periods': the full array of periods considered,
             'durations': the array of durations used to run BLS,
             'blsresult': Astropy BLS result object (BoxLeastSquaresResult),
             'blsmodel': Astropy BLS BoxLeastSquares object used for work,
             'stepsize': the actual stepsize used,
             'nfreq': the actual nfreq used,
             'durations': the durations array used,
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
        if isinstance(autofreq, bool) and autofreq:

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
                        'freq stepsize: %s, ndurations: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, ndurations,
                         mintransitduration, maxtransitduration))

            use_autoperiod = False

        elif isinstance(autofreq, bool) and not autofreq:

            minfreq = 1.0/endp
            maxfreq = 1.0/startp
            nfreq = int(npceil((maxfreq - minfreq)/stepsize))

            # say what we're using
            if verbose:
                LOGINFO('min P: %s, max P: %s, nfreq: %s, '
                        'minfreq: %s, maxfreq: %s' % (startp, endp, nfreq,
                                                      minfreq, maxfreq))
                LOGINFO('autofreq = False: using PROVIDED values for '
                        'freq stepsize: %s, ndurations: %s, '
                        'min transit duration: %s, max transit duration: %s' %
                        (stepsize, ndurations,
                         mintransitduration, maxtransitduration))

            use_autoperiod = False

        elif isinstance(autofreq, str) and autofreq == 'astropy':

            use_autoperiod = True
            minfreq = 1.0/endp
            maxfreq = 1.0/startp

        else:

            LOGERROR("unknown autofreq kwarg encountered. can't continue...")
            return None

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

        # check if autoperiod is True and get the correct period-grid
        if use_autoperiod:

            # astropy's BLS requires durations in units of time
            durations = nplinspace(mintransitduration*startp,
                                   maxtransitduration*startp,
                                   ndurations)

            # set up the correct units for the BLS model
            if magsarefluxes:

                blsmodel = BoxLeastSquares(
                    stimes*u.day,
                    smags*u.dimensionless_unscaled,
                    dy=serrs*u.dimensionless_unscaled
                )

            else:

                blsmodel = BoxLeastSquares(
                    stimes*u.day,
                    smags*u.mag,
                    dy=serrs*u.mag
                )

            periods = nparray(
                blsmodel.autoperiod(
                    durations*u.day,
                    minimum_period=startp,
                    maximum_period=endp,
                    minimum_n_transit=blsmintransits,
                    frequency_factor=blsfreqfactor
                )
            )

            frequencies = 1.0/periods
            nfreq = frequencies.size

            if verbose:
                LOGINFO(
                    "autofreq = 'astropy', used .autoperiod() with "
                    "minimum_n_transit = %s, freq_factor = %s "
                    "to generate the frequency grid" %
                    (blsmintransits, blsfreqfactor)
                )
                LOGINFO('stepsize = %s, nfreq = %s, minfreq = %.5f, '
                        'maxfreq = %.5f, ndurations = %s' %
                        (abs(frequencies[1] - frequencies[0]),
                         nfreq,
                         1.0/periods.max(),
                         1.0/periods.min(),
                         durations.size))

            del blsmodel
            del durations

        # otherwise, use kbls method
        else:

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
        #
        # task[0] = times
        # task[1] = mags
        # task[2] = errs
        # task[3] = magsarefluxes

        # task[4] = minfreq
        # task[5] = nfreq
        # task[6] = stepsize

        # task[7] = nphasebins
        # task[8] = mintransitduration
        # task[9] = maxtransitduration

        # task[10] = blsobjective
        # task[11] = blsmethod
        # task[12] = blsoversample

        # populate the tasks list
        tasks = [(stimes, smags, serrs, magsarefluxes,
                  chunk_minf, chunk_nf, stepsize,
                  ndurations, mintransitduration, maxtransitduration,
                  blsobjective, blsmethod, blsoversample)
                 for (chunk_minf, chunk_nf)
                 in zip(chunk_minfreqs, chunk_nfreqs)]

        if verbose:
            for ind, task in enumerate(tasks):
                LOGINFO('worker %s: minfreq = %.6f, nfreqs = %s' %
                        (ind+1, task[4], task[5]))
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
                    'nbestinds':None,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'durations':None,
                    'method':'bls',
                    'blsresult':None,
                    'blsmodel':None,
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'mintransitduration':mintransitduration,
                              'maxtransitduration':maxtransitduration,
                              'ndurations':ndurations,
                              'blsobjective':blsobjective,
                              'blsmethod':blsmethod,
                              'blsoversample':blsoversample,
                              'autofreq':autofreq,
                              'periodepsilon':periodepsilon,
                              'nbestpeaks':nbestpeaks,
                              'sigclip':sigclip,
                              'magsarefluxes':magsarefluxes}}

        sortedlspind = npargsort(finlsp)[::-1]
        sortedlspperiods = finperiods[sortedlspind]
        sortedlspvals = finlsp[sortedlspind]

        # now get the nbestpeaks
        nbestperiods, nbestlspvals, nbestinds, peakcount = (
            [finperiods[bestperiodind]],
            [finlsp[bestperiodind]],
            [bestperiodind],
            1
        )
        prevperiod = sortedlspperiods[0]

        # find the best nbestpeaks in the lsp and their periods
        for period, lspval, ind in zip(sortedlspperiods,
                                       sortedlspvals,
                                       sortedlspind):

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
                nbestinds.append(ind)
                peakcount = peakcount + 1

            prevperiod = period

        # generate the return dict
        resultdict = {
            'bestperiod':finperiods[bestperiodind],
            'bestlspval':finlsp[bestperiodind],
            'nbestpeaks':nbestpeaks,
            'nbestinds':nbestinds,
            'nbestlspvals':nbestlspvals,
            'nbestperiods':nbestperiods,
            'lspvals':lsp,
            'frequencies':frequencies,
            'periods':periods,
            'durations':[x['durations'] for x in results],
            'blsresult':[x['blsresult'] for x in results],
            'blsmodel':[x['blsmodel'] for x in results],
            'stepsize':stepsize,
            'nfreq':nfreq,
            'mintransitduration':mintransitduration,
            'maxtransitduration':maxtransitduration,
            'method':'bls',
            'kwargs':{'startp':startp,
                      'endp':endp,
                      'stepsize':stepsize,
                      'mintransitduration':mintransitduration,
                      'maxtransitduration':maxtransitduration,
                      'ndurations':ndurations,
                      'blsobjective':blsobjective,
                      'blsmethod':blsmethod,
                      'blsoversample':blsoversample,
                      'autofreq':autofreq,
                      'periodepsilon':periodepsilon,
                      'nbestpeaks':nbestpeaks,
                      'sigclip':sigclip,
                      'magsarefluxes':magsarefluxes}
        }

        return resultdict

    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestinds':None,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'durations':None,
                'blsresult':None,
                'blsmodel':None,
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
                          'ndurations':ndurations,
                          'blsobjective':blsobjective,
                          'blsmethod':blsmethod,
                          'blsoversample':blsoversample,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip,
                          'magsarefluxes':magsarefluxes}}
