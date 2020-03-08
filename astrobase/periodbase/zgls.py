#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# zgls.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''
Contains the Zechmeister & Kurster (2002) Generalized Lomb-Scargle period-search
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

from multiprocessing import Pool, cpu_count

from numpy import (
    nan as npnan, arange as nparange, array as nparray, isfinite as npisfinite,
    argmax as npargmax, argsort as npargsort, sum as npsum, cos as npcos,
    sin as npsin, pi as pi_value, nonzero as npnonzero, nanmax as npnanmax,
    arctan as nparctan,
)


###################
## LOCAL IMPORTS ##
###################

from ..lcmath import sigclip_magseries
from .utils import get_frequency_grid, independent_freq_count, resort_by_time


############
## CONFIG ##
############

NCPUS = cpu_count()


######################################################
## PERIODOGRAM VALUE EXPRESSIONS FOR A SINGLE OMEGA ##
######################################################

def generalized_lsp_value(times, mags, errs, omega):
    '''Generalized LSP value for a single omega.

    The relations used are::

        P(w) = (1/YY) * (YC*YC/CC + YS*YS/SS)

        where: YC, YS, CC, and SS are all calculated at T

        and where: tan 2omegaT = 2*CS/(CC - SS)

        and where:

        Y = sum( w_i*y_i )
        C = sum( w_i*cos(wT_i) )
        S = sum( w_i*sin(wT_i) )

        YY = sum( w_i*y_i*y_i ) - Y*Y
        YC = sum( w_i*y_i*cos(wT_i) ) - Y*C
        YS = sum( w_i*y_i*sin(wT_i) ) - Y*S

        CpC = sum( w_i*cos(w_T_i)*cos(w_T_i) )
        CC = CpC - C*C
        SS = (1 - CpC) - S*S
        CS = sum( w_i*cos(w_T_i)*sin(w_T_i) ) - C*S

    Parameters
    ----------

    times,mags,errs : np.array
        The time-series to calculate the periodogram value for.

    omega : float
        The frequency to calculate the periodogram value at.

    Returns
    -------

    periodogramvalue : float
        The normalized periodogram at the specified test frequency `omega`.

    '''

    one_over_errs2 = 1.0/(errs*errs)

    W = npsum(one_over_errs2)
    wi = one_over_errs2/W

    sin_omegat = npsin(omega*times)
    cos_omegat = npcos(omega*times)

    cos2_omegat = cos_omegat*cos_omegat

    # calculate some more sums and terms
    Y = npsum( wi*mags )
    C = npsum( wi*cos_omegat )
    S = npsum( wi*sin_omegat )

    CpC = npsum( wi*cos2_omegat )
    CC = CpC - C*C
    SS = 1 - CpC - S*S  # use SpS = 1 - CpC

    YpY = npsum( wi*mags*mags)

    YpC = npsum( wi*mags*cos_omegat )
    YpS = npsum( wi*mags*sin_omegat )

    # SpS = npsum( wi*sin2_omegat )

    # the final terms
    YY = YpY - Y*Y
    YC = YpC - Y*C
    YS = YpS - Y*S

    periodogramvalue = (YC*YC/CC + YS*YS/SS)/YY

    return periodogramvalue


def generalized_lsp_value_withtau(times, mags, errs, omega):
    '''Generalized LSP value for a single omega.

    This uses tau to provide an arbitrary time-reference point.

    The relations used are::

        P(w) = (1/YY) * (YC*YC/CC + YS*YS/SS)

        where: YC, YS, CC, and SS are all calculated at T

        and where: tan 2omegaT = 2*CS/(CC - SS)

        and where:

        Y = sum( w_i*y_i )
        C = sum( w_i*cos(wT_i) )
        S = sum( w_i*sin(wT_i) )

        YY = sum( w_i*y_i*y_i ) - Y*Y
        YC = sum( w_i*y_i*cos(wT_i) ) - Y*C
        YS = sum( w_i*y_i*sin(wT_i) ) - Y*S

        CpC = sum( w_i*cos(w_T_i)*cos(w_T_i) )
        CC = CpC - C*C
        SS = (1 - CpC) - S*S
        CS = sum( w_i*cos(w_T_i)*sin(w_T_i) ) - C*S

    Parameters
    ----------

    times,mags,errs : np.array
        The time-series to calculate the periodogram value for.

    omega : float
        The frequency to calculate the periodogram value at.

    Returns
    -------

    periodogramvalue : float
        The normalized periodogram at the specified test frequency `omega`.

    '''

    one_over_errs2 = 1.0/(errs*errs)

    W = npsum(one_over_errs2)
    wi = one_over_errs2/W

    sin_omegat = npsin(omega*times)
    cos_omegat = npcos(omega*times)

    cos2_omegat = cos_omegat*cos_omegat
    sincos_omegat = sin_omegat*cos_omegat

    # calculate some more sums and terms
    Y = npsum( wi*mags )
    C = npsum( wi*cos_omegat )
    S = npsum( wi*sin_omegat )

    CpS = npsum( wi*sincos_omegat )
    CpC = npsum( wi*cos2_omegat )
    CS = CpS - C*S
    CC = CpC - C*C
    SS = 1 - CpC - S*S  # use SpS = 1 - CpC

    # calculate tau
    tan_omega_tau_top = 2.0*CS
    tan_omega_tau_bottom = CC - SS
    tan_omega_tau = tan_omega_tau_top/tan_omega_tau_bottom
    tau = nparctan(tan_omega_tau)/(2.0*omega)

    # now we need to calculate all the bits at tau
    sin_omega_tau = npsin(omega*(times - tau))
    cos_omega_tau = npcos(omega*(times - tau))
    cos2_omega_tau = cos_omega_tau*cos_omega_tau

    C_tau = npsum(wi*cos_omega_tau)
    S_tau = npsum(wi*sin_omega_tau)

    CpC_tau = npsum( wi*cos2_omega_tau )
    CC_tau = CpC_tau - C_tau*C_tau
    SS_tau = 1 - CpC_tau - S_tau*S_tau  # use SpS = 1 - CpC

    YpY = npsum( wi*mags*mags)

    YpC_tau = npsum( wi*mags*cos_omega_tau )
    YpS_tau = npsum( wi*mags*sin_omega_tau )

    # SpS = npsum( wi*sin2_omegat )

    # the final terms
    YY = YpY - Y*Y
    YC_tau = YpC_tau - Y*C_tau
    YS_tau = YpS_tau - Y*S_tau

    periodogramvalue = (YC_tau*YC_tau/CC_tau + YS_tau*YS_tau/SS_tau)/YY

    return periodogramvalue


def generalized_lsp_value_notau(times, mags, errs, omega):
    '''
    This is the simplified version not using tau.

    The relations used are::

        W = sum (1.0/(errs*errs) )
        w_i = (1/W)*(1/(errs*errs))

        Y = sum( w_i*y_i )
        C = sum( w_i*cos(wt_i) )
        S = sum( w_i*sin(wt_i) )

        YY = sum( w_i*y_i*y_i ) - Y*Y
        YC = sum( w_i*y_i*cos(wt_i) ) - Y*C
        YS = sum( w_i*y_i*sin(wt_i) ) - Y*S

        CpC = sum( w_i*cos(w_t_i)*cos(w_t_i) )
        CC = CpC - C*C
        SS = (1 - CpC) - S*S
        CS = sum( w_i*cos(w_t_i)*sin(w_t_i) ) - C*S

        D(omega) = CC*SS - CS*CS
        P(omega) = (SS*YC*YC + CC*YS*YS - 2.0*CS*YC*YS)/(YY*D)

    Parameters
    ----------

    times,mags,errs : np.array
        The time-series to calculate the periodogram value for.

    omega : float
        The frequency to calculate the periodogram value at.

    Returns
    -------

    periodogramvalue : float
        The normalized periodogram at the specified test frequency `omega`.

    '''

    one_over_errs2 = 1.0/(errs*errs)

    W = npsum(one_over_errs2)
    wi = one_over_errs2/W

    sin_omegat = npsin(omega*times)
    cos_omegat = npcos(omega*times)

    cos2_omegat = cos_omegat*cos_omegat
    sincos_omegat = sin_omegat*cos_omegat

    # calculate some more sums and terms
    Y = npsum( wi*mags )
    C = npsum( wi*cos_omegat )
    S = npsum( wi*sin_omegat )

    YpY = npsum( wi*mags*mags)

    YpC = npsum( wi*mags*cos_omegat )
    YpS = npsum( wi*mags*sin_omegat )

    CpC = npsum( wi*cos2_omegat )
    # SpS = npsum( wi*sin2_omegat )

    CpS = npsum( wi*sincos_omegat )

    # the final terms
    YY = YpY - Y*Y
    YC = YpC - Y*C
    YS = YpS - Y*S
    CC = CpC - C*C
    SS = 1 - CpC - S*S  # use SpS = 1 - CpC
    CS = CpS - C*S

    # P(omega) = (SS*YC*YC + CC*YS*YS - 2.0*CS*YC*YS)/(YY*D)
    # D(omega) = CC*SS - CS*CS
    Domega = CC*SS - CS*CS
    lspval = (SS*YC*YC + CC*YS*YS - 2.0*CS*YC*YS)/(YY*Domega)

    return lspval


def specwindow_lsp_value(times, mags, errs, omega):
    '''This calculates the peak associated with the spectral window function
    for times and at the specified omega.

    NOTE: this is classical Lomb-Scargle, not the Generalized
    Lomb-Scargle. `mags` and `errs` are silently ignored since we're calculating
    the periodogram of the observing window function. These are kept to present
    a consistent external API so the `pgen_lsp` function below can call this
    transparently.

    Parameters
    ----------

    times,mags,errs : np.array
        The time-series to calculate the periodogram value for.

    omega : float
        The frequency to calculate the periodogram value at.

    Returns
    -------

    periodogramvalue : float
        The normalized periodogram at the specified test frequency `omega`.

    '''

    norm_times = times - times.min()

    tau = (
        (1.0/(2.0*omega)) *
        nparctan( npsum(npsin(2.0*omega*norm_times)) /
                  npsum(npcos(2.0*omega*norm_times)) )
    )

    lspval_top_cos = (npsum(1.0 * npcos(omega*(norm_times-tau))) *
                      npsum(1.0 * npcos(omega*(norm_times-tau))))
    lspval_bot_cos = npsum( (npcos(omega*(norm_times-tau))) *
                            (npcos(omega*(norm_times-tau))) )

    lspval_top_sin = (npsum(1.0 * npsin(omega*(norm_times-tau))) *
                      npsum(1.0 * npsin(omega*(norm_times-tau))))
    lspval_bot_sin = npsum( (npsin(omega*(norm_times-tau))) *
                            (npsin(omega*(norm_times-tau))) )

    lspval = 0.5 * ( (lspval_top_cos/lspval_bot_cos) +
                     (lspval_top_sin/lspval_bot_sin) )

    return lspval


##############################
## GENERALIZED LOMB-SCARGLE ##
##############################

def _glsp_worker(task):
    '''This is a worker to wrap the generalized Lomb-Scargle single-frequency
    function.

    '''

    try:
        return generalized_lsp_value(*task)
    except Exception:
        return npnan


def _glsp_worker_withtau(task):
    '''This is a worker to wrap the generalized Lomb-Scargle single-frequency
    function.

    '''

    try:
        return generalized_lsp_value_withtau(*task)
    except Exception:
        return npnan


def _glsp_worker_specwindow(task):
    '''This is a worker to wrap the generalized Lomb-Scargle single-frequency
    function.

    '''

    try:
        return specwindow_lsp_value(*task)
    except Exception:
        return npnan


def _glsp_worker_notau(task):
    '''This is a worker to wrap the generalized Lomb-Scargle single-freq func.

    This version doesn't use tau.

    '''

    try:
        return generalized_lsp_value_notau(*task)
    except Exception:
        return npnan


def pgen_lsp(
        times,
        mags,
        errs,
        magsarefluxes=False,
        startp=None,
        endp=None,
        stepsize=1.0e-4,
        autofreq=True,
        nbestpeaks=5,
        periodepsilon=0.1,
        sigclip=10.0,
        nworkers=None,
        workchunksize=None,
        glspfunc=_glsp_worker_withtau,
        verbose=True
):
    '''This calculates the generalized Lomb-Scargle periodogram.

    Uses the algorithm from Zechmeister and Kurster (2009).

    Parameters
    ----------

    times,mags,errs : np.array
        The mag/flux time-series with associated measurement errors to run the
        period-finding on.

    magsarefluxes : bool
        If the input measurement values in `mags` and `errs` are in fluxes, set
        this to True.

    startp,endp : float or None
        The minimum and maximum periods to consider for the transit search.

    stepsize : float
        The step-size in frequency to use when constructing a frequency grid for
        the period search.

    autofreq : bool
        If this is True, the value of `stepsize` will be ignored and the
        :py:func:`astrobase.periodbase.get_frequency_grid` function will be used
        to generate a frequency grid based on `startp`, and `endp`. If these are
        None as well, `startp` will be set to 0.1 and `endp` will be set to
        `times.max() - times.min()`.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

    periodepsilon : float
        The fractional difference between successive values of 'best' periods
        when sorting by periodogram power to consider them as separate periods
        (as opposed to part of the same periodogram peak). This is used to avoid
        broad peaks in the periodogram and make sure the 'best' periods returned
        are all actually independent.

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

    nworkers : int
        The number of parallel workers to use when calculating the periodogram.

    workchunksize : None or int
        If this is an int, will use chunks of the given size to break up the
        work for the parallel workers. If None, the chunk size is set to 1.

    glspfunc : Python function
        The worker function to use to calculate the periodogram. This can be
        used to make this function calculate the time-series sampling window
        function instead of the time-series measurements' GLS periodogram by
        passing in `_glsp_worker_specwindow` instead of the default
        `_glsp_worker_withtau` function.

    verbose : bool
        If this is True, will indicate progress and details about the frequency
        grid used for the period search.

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
             'periods': the full array of periods considered,
             'method':'gls' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)
    stimes, smags, serrs = resort_by_time(stimes, smags, serrs)

    # get rid of zero errs
    nzind = npnonzero(serrs)
    stimes, smags, serrs = stimes[nzind], smags[nzind], serrs[nzind]

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
            omegas = 2*pi_value*nparange(startf, endf, stepsize)
            if verbose:
                LOGINFO(
                    'using %s frequency points, start P = %.3f, end P = %.3f' %
                    (omegas.size, 1.0/endf, 1.0/startf)
                )
        else:
            # this gets an automatic grid of frequencies to use
            freqs = get_frequency_grid(stimes,
                                       minfreq=startf,
                                       maxfreq=endf)
            omegas = 2*pi_value*freqs
            if verbose:
                LOGINFO(
                    'using autofreq with %s frequency points, '
                    'start P = %.3f, end P = %.3f' %
                    (omegas.size, 1.0/freqs.max(), 1.0/freqs.min())
                )

        # map to parallel workers
        if (not nworkers) or (nworkers > NCPUS):
            nworkers = NCPUS
            if verbose:
                LOGINFO('using %s workers...' % nworkers)

        pool = Pool(nworkers)

        tasks = [(stimes, smags, serrs, x) for x in omegas]
        if workchunksize:
            lsp = pool.map(glspfunc, tasks, chunksize=workchunksize)
        else:
            lsp = pool.map(glspfunc, tasks)

        pool.close()
        pool.join()
        del pool

        lsp = nparray(lsp)
        periods = 2.0*pi_value/omegas

        # find the nbestpeaks for the periodogram: 1. sort the lsp array by
        # highest value first 2. go down the values until we find five
        # values that are separated by at least periodepsilon in period

        # make sure to filter out non-finite values of lsp

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
                    'omegas':omegas,
                    'periods':None,
                    'method':'gls',
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'autofreq':autofreq,
                              'periodepsilon':periodepsilon,
                              'nbestpeaks':nbestpeaks,
                              'sigclip':sigclip}}

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

            # print('prevperiod = %s, thisperiod = %s, '
            #       'perioddiff = %s, peakcount = %s' %
            #       (prevperiod, period, perioddiff, peakcount))

            # this ensures that this period is different from the last
            # period and from all the other existing best periods by
            # periodepsilon to make sure we jump to an entire different peak
            # in the periodogram
            if (perioddiff > (periodepsilon*prevperiod) and
                all(x > (periodepsilon*period) for x in bestperiodsdiff)):
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
                'method':'gls',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
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
                'omegas':None,
                'periods':None,
                'method':'gls',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}


def specwindow_lsp(
        times,
        mags,
        errs,
        magsarefluxes=False,
        startp=None,
        endp=None,
        stepsize=1.0e-4,
        autofreq=True,
        nbestpeaks=5,
        periodepsilon=0.1,
        sigclip=10.0,
        nworkers=None,
        glspfunc=_glsp_worker_specwindow,
        verbose=True
):
    '''This calculates the spectral window function.

    Wraps the `pgen_lsp` function above to use the specific worker for
    calculating the window-function.

    Parameters
    ----------

    times,mags,errs : np.array
        The mag/flux time-series with associated measurement errors to run the
        period-finding on.

    magsarefluxes : bool
        If the input measurement values in `mags` and `errs` are in fluxes, set
        this to True.

    startp,endp : float or None
        The minimum and maximum periods to consider for the transit search.

    stepsize : float
        The step-size in frequency to use when constructing a frequency grid for
        the period search.

    autofreq : bool
        If this is True, the value of `stepsize` will be ignored and the
        :py:func:`astrobase.periodbase.get_frequency_grid` function will be used
        to generate a frequency grid based on `startp`, and `endp`. If these are
        None as well, `startp` will be set to 0.1 and `endp` will be set to
        `times.max() - times.min()`.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

    periodepsilon : float
        The fractional difference between successive values of 'best' periods
        when sorting by periodogram power to consider them as separate periods
        (as opposed to part of the same periodogram peak). This is used to avoid
        broad peaks in the periodogram and make sure the 'best' periods returned
        are all actually independent.

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

    nworkers : int
        The number of parallel workers to use when calculating the periodogram.

    glspfunc : Python function
        The worker function to use to calculate the periodogram. This is used to
        used to make the `pgen_lsp` function calculate the time-series sampling
        window function instead of the time-series measurements' GLS periodogram
        by passing in `_glsp_worker_specwindow` instead of the default
        `_glsp_worker` function.

    verbose : bool
        If this is True, will indicate progress and details about the frequency
        grid used for the period search.

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
             'periods': the full array of periods considered,
             'method':'win' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    '''

    # run the LSP using glsp_worker_specwindow as the worker
    lspres = pgen_lsp(
        times,
        mags,
        errs,
        magsarefluxes=magsarefluxes,
        startp=startp,
        endp=endp,
        autofreq=autofreq,
        nbestpeaks=nbestpeaks,
        periodepsilon=periodepsilon,
        stepsize=stepsize,
        nworkers=nworkers,
        sigclip=sigclip,
        glspfunc=glspfunc,
        verbose=verbose
    )

    # update the resultdict to indicate we're a spectral window function
    lspres['method'] = 'win'

    if lspres['lspvals'] is not None:

        # renormalize the periodogram to between 0 and 1 like the usual GLS.
        lspmax = npnanmax(lspres['lspvals'])

        if npisfinite(lspmax):

            lspres['lspvals'] = lspres['lspvals']/lspmax
            lspres['nbestlspvals'] = [
                x/lspmax for x in lspres['nbestlspvals']
            ]
            lspres['bestlspval'] = lspres['bestlspval']/lspmax

    return lspres


##########################################
## FALSE ALARM PROBABILITY CALCULATIONS ##
##########################################

def probability_peak_exceeds_value(times, peakval):
    '''This calculates the probability that periodogram values exceed the given
    peak value.

    This is from page 3 of Zechmeister and Kurster (2009)::

        Prob(p > p_best) = (1 − p_best)**((N−3)/2)

    where::

        p_best is the peak value in consideration
        N is the number of times

    Note that this is for the default normalization of the periodogram,
    e.g. P_normalized = P(omega), such that P represents the sample variance
    (see Table 1).

    Parameters
    ----------

    lspvals : np.array
        The periodogram power value array.

    peakval : float
        A single peak value to calculate the probability for.

    Returns
    -------

    prob: float
        The probability value.

    '''

    return (1.0 - peakval)**((times.size - 3.0)/2.0)


def analytic_false_alarm_probability(lspinfo,
                                     times,
                                     conservative_nfreq_eff=True,
                                     peakvals=None,
                                     inplace=True):

    '''This returns the analytic false alarm probabilities for periodogram
    peak values.

    The calculation follows that on page 3 of Zechmeister & Kurster (2009)::

        FAP = 1 − [1 − Prob(z > z0)]**M

    where::

        M is the number of independent frequencies
        Prob(z > z0) is the probability of peak with value > z0
        z0 is the peak value we're evaluating

    Parameters
    ----------

    lspinfo : dict
        The dict returned by the :py:func:`~astrobase.periodbase.zgls.pgen_lsp`
        function.

    times : np.array
        The times for which the periodogram result in ``lspinfo`` was
        calculated.

    conservative_nfreq_eff : bool
        If True, will follow the prescription given in Schwarzenberg-Czerny
        (2003):

        http://adsabs.harvard.edu/abs/2003ASPC..292..383S

        and estimate the effective number of independent frequences M_eff as::

            min(N_obs, N_freq, DELTA_f/delta_f)

    peakvals : sequence or None
        The peak values for which to evaluate the false-alarm probability. If
        None, will calculate this for each of the peak values in the
        ``nbestpeaks`` key of the ``lspinfo`` dict.

    inplace : bool
        If True, puts the results of the FAP calculation into the ``lspinfo``
        dict as a list available as ``lspinfo['falsealarmprob']``.

    Returns
    -------

    list
        The calculated false alarm probabilities for each of the peak values in
        ``peakvals``.

    '''

    frequencies = 1.0/lspinfo['periods']

    M = independent_freq_count(frequencies,
                               times,
                               conservative=conservative_nfreq_eff)

    if peakvals is None:
        peakvals = lspinfo['nbestlspvals']

    prob_exceed_vals = [
        probability_peak_exceeds_value(times, p) for p in peakvals
    ]

    false_alarm_probs = [
        1.0 - (1.0 - prob_exc)**M for prob_exc in prob_exceed_vals
    ]

    if inplace:
        lspinfo['falsealarmprob'] = false_alarm_probs

    return false_alarm_probs
