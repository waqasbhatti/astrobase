#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# smav.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''
Contains the Schwarzenberg-Czerny Analysis of Variance period-search algorithm
implementation for periodbase. This uses the multi-harmonic version presented in
Schwarzenberg-Czerny (1996).

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
    argmax as npargmax, median as npmedian, std as npstd, argsort as npargsort,
    sum as npsum, cos as npcos, sin as npsin, vdot as npvdot, pi as pi_value,
    max as npmax, abs as npabs
)


###################
## LOCAL IMPORTS ##
###################

from ..lcmath import phase_magseries_with_errs, sigclip_magseries
from .utils import get_frequency_grid, independent_freq_count, resort_by_time


############
## CONFIG ##
############

NCPUS = cpu_count()


####################################################################
## MULTIHARMONIC ANALYSIS of VARIANCE (Schwarzenberg-Czerny 1996) ##
####################################################################


def aovhm_theta(times, mags, errs, frequency,
                nharmonics, magvariance):
    '''This calculates the harmonic AoV theta statistic for a frequency.

    This is a mostly faithful translation of the inner loop in `aovper.f90`. See
    the following for details:

    - http://users.camk.edu.pl/alex/
    - Schwarzenberg-Czerny (`1996
      <http://iopscience.iop.org/article/10.1086/309985/meta>`_)

    Schwarzenberg-Czerny (1996) equation 11::

        theta_prefactor = (K - 2N - 1)/(2N)
        theta_top = sum(c_n*c_n) (from n=0 to n=2N)
        theta_bot = variance(timeseries) - sum(c_n*c_n) (from n=0 to n=2N)

        theta = theta_prefactor * (theta_top/theta_bot)

        N = number of harmonics (nharmonics)
        K = length of time series (times.size)

    Parameters
    ----------

    times,mags,errs : np.array
        The input time-series to calculate the test statistic for. These should
        all be of nans/infs and be normalized to zero.

    frequency : float
        The test frequency to calculate the statistic for.

    nharmonics : int
        The number of harmonics to calculate up to.The recommended range is 4 to
        8.

    magvariance : float
        This is the (weighted by errors) variance of the magnitude time
        series. We provide it as a pre-calculated value here so we don't have to
        re-calculate it for every worker.

    Returns
    -------

    aov_harmonic_theta : float
        THe value of the harmonic AoV theta for the specified test `frequency`.

    '''

    period = 1.0/frequency

    ndet = times.size
    two_nharmonics = nharmonics + nharmonics

    # phase with test period
    phasedseries = phase_magseries_with_errs(
        times, mags, errs, period, times[0],
        sort=True, wrap=False
    )

    # get the phased quantities
    phase = phasedseries['phase']
    pmags = phasedseries['mags']
    perrs = phasedseries['errs']

    # this is sqrt(1.0/errs^2) -> the weights
    pweights = 1.0/perrs

    # multiply by 2.0*PI (for omega*time)
    phase = phase * 2.0 * pi_value

    # this is the z complex vector
    z = npcos(phase) + 1.0j*npsin(phase)

    # multiply phase with N
    phase = nharmonics * phase

    # this is the psi complex vector
    psi = pmags * pweights * (npcos(phase) + 1j*npsin(phase))

    # this is the initial value of z^n
    zn = 1.0 + 0.0j

    # this is the initial value of phi
    phi = pweights + 0.0j

    # initialize theta to zero
    theta_aov = 0.0

    # go through all the harmonics now up to 2N
    for _ in range(two_nharmonics):

        # this is <phi, phi>
        phi_dot_phi = npsum(phi * phi.conjugate())

        # this is the alpha_n numerator
        alpha = npsum(pweights * z * phi)

        # this is <phi, psi>. make sure to use npvdot and NOT npdot to get
        # complex conjugate of first vector as expected for complex vectors
        phi_dot_psi = npvdot(phi, psi)

        # make sure phi_dot_phi is not zero
        phi_dot_phi = npmax([phi_dot_phi, 10.0e-9])

        # this is the expression for alpha_n
        alpha = alpha / phi_dot_phi

        # update theta_aov for this harmonic
        theta_aov = (theta_aov +
                     npabs(phi_dot_psi) * npabs(phi_dot_psi) / phi_dot_phi)

        # use the recurrence relation to find the next phi
        phi = phi * z - alpha * zn * phi.conjugate()

        # update z^n
        zn = zn * z

    # done with all harmonics, calculate the theta_aov for this freq
    # the max below makes sure that magvariance - theta_aov > zero
    theta_aov = ( (ndet - two_nharmonics - 1.0) * theta_aov /
                  (two_nharmonics * npmax([magvariance - theta_aov,
                                           1.0e-9])) )

    return theta_aov


def _aovhm_theta_worker(task):
    '''
    This is a parallel worker for the function below.

    Parameters
    ----------

    tasks : tuple
        This is of the form below::

            task[0] = times
            task[1] = mags
            task[2] = errs
            task[3] = frequency
            task[4] = nharmonics
            task[5] = magvariance

    Returns
    -------

    harmonic_aov_theta : float
        The value of the harmonic AoV statistic for the test frequency used.
        If something goes wrong with the calculation, nan is returned.

    '''

    times, mags, errs, frequency, nharmonics, magvariance = task

    try:

        theta = aovhm_theta(times, mags, errs, frequency,
                            nharmonics, magvariance)

        return theta

    except Exception:

        return npnan


def aovhm_periodfind(times,
                     mags,
                     errs,
                     magsarefluxes=False,
                     startp=None,
                     endp=None,
                     stepsize=1.0e-4,
                     autofreq=True,
                     normalize=True,
                     nharmonics=6,
                     nbestpeaks=5,
                     periodepsilon=0.1,
                     sigclip=10.0,
                     nworkers=None,
                     verbose=True):
    '''This runs a parallelized harmonic Analysis-of-Variance (AoV) period
    search.

    NOTE: normalize = True here as recommended by Schwarzenberg-Czerny 1996,
    i.e. mags will be normalized to zero and rescaled so their variance = 1.0.

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

    normalize : bool
        This sets if the input time-series is normalized to 0.0 and rescaled
        such that its variance = 1.0. This is the recommended procedure by
        Schwarzenberg-Czerny 1996.

    nharmonics : int
        The number of harmonics to use when calculating the AoV theta value at a
        test frequency. This should be between 4 and 8 in most cases.

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
             'method':'mav' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)
    stimes, smags, serrs = resort_by_time(stimes, smags, serrs)

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
            frequencies = nparange(startf, endf, stepsize)
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

        # figure out the weighted variance
        # www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weighvar.pdf
        magvariance_top = npsum(nmags/(serrs*serrs))
        magvariance_bot = (nmags.size - 1)*npsum(1.0/(serrs*serrs)) / nmags.size
        magvariance = magvariance_top/magvariance_bot

        tasks = [(stimes, nmags, serrs, x, nharmonics, magvariance)
                 for x in frequencies]

        lsp = pool.map(_aovhm_theta_worker, tasks)

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
                    'method':'mav',
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'normalize':normalize,
                              'nharmonics':nharmonics,
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
                'periods':periods,
                'method':'mav',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'nharmonics':nharmonics,
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
                'method':'mav',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'nharmonics':nharmonics,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}


def analytic_false_alarm_probability(lspinfo,
                                     times,
                                     conservative_nfreq_eff=True,
                                     peakvals=None,
                                     inplace=True):
    '''This returns the analytic false alarm probabilities for periodogram
    peak values.

    FIXME: this doesn't actually work. Fix later.

    The calculation follows that on page 3 of Zechmeister & Kurster (2009)::

        FAP = 1 − [1 − Prob(z > z0)]**M

    where::

        M is the number of independent frequencies
        Prob(z > z0) is the probability of peak with value > z0
        z0 is the peak value we're evaluating

    For AoV and AoV-harmonic, the Prob(z > z0) is described by the F
    distribution, according to:

    - Schwarzenberg-Czerny (1997;
      https://ui.adsabs.harvard.edu/#abs/1997ApJ...489..941S)

    - Schwarzenberg-Czerny (1996;
      http://adsabs.harvard.edu/abs/1996ApJ...460L.107S)

    This is given by::

        F( 2N, K - 2N - 1; theta_aov )

    Where::

        N = number of harmonics used for AOV_harmonic
        K = number of observations

    This translates to a scipy.stats call to the F distribution CDF::

        x = theta_aov_best
        prob_exceeds_val = scipy.stats.f.cdf(x, 2N, K - 2N - 1)

    Which we can then plug into the false alarm prob eqn above with the
    calculation of M.

    Parameters
    ----------

    lspinfo : dict
        The dict returned by the
        :py:func:`~astrobase.periodbase.spdm.aovhm_periodfind` function.

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

    from scipy.stats import f

    frequencies = 1.0/lspinfo['periods']

    M = independent_freq_count(frequencies,
                               times,
                               conservative=conservative_nfreq_eff)

    if peakvals is None:
        peakvals = lspinfo['nbestlspvals']

    nharmonics = lspinfo['kwargs']['nharmonics']
    ndet = times.size

    false_alarm_probs = []

    for peakval in peakvals:

        prob_xval = peakval
        prob_exceeds_val = f.cdf(prob_xval,
                                 2*nharmonics,
                                 ndet - 2*nharmonics - 1)
        false_alarm_probs.append(1.0 - (1.0 - prob_exceeds_val)**M)

    if inplace:
        lspinfo['falsealarmprob'] = false_alarm_probs

    return false_alarm_probs
