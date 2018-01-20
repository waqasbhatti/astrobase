#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''periodicfeatures - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This contains functions that calculate various light curve features using
information about periods and fits to phased light curves.

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

from time import time as unixtime
from itertools import combinations

import numpy as np
from scipy.signal import argrelmin, argrelmax


###################
## LOCAL IMPORTS ##
###################

from .. import lcmath
from ..varbase import lcfit
from ..lcmodels import sinusoidal, eclipses, transits
from ..periodbase.zgls import specwindow_lsp
from .varfeatures import lightcurve_ptp_measures


###################################
## FEATURE CALCULATION FUNCTIONS ##
###################################

def lcfit_features(times, mags, errs, period,
                   fourierorder=5,
                   # these are depth, duration, ingress duration
                   transitparams=[-0.01,0.1,0.1],
                   # these are depth, duration, depth ratio, secphase
                   ebparams=[-0.2,0.3,0.7,0.5],
                   sigclip=10.0,
                   magsarefluxes=False,
                   verbose=True):
    '''
    This calculates various features related to fitting models to light curves.

    - calculates R_ij and phi_ij ratios for Fourier fit amplitudes and phases
    - calculates the redchisq for fourier, EB, and planet transit fits
    - calculates the redchisq for fourier, EB, planet transit fits w/2 x period

    '''

    # get the finite values
    finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes, fmags, ferrs = times[finind], mags[finind], errs[finind]

    # get nonzero errors
    nzind = np.nonzero(ferrs)
    ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

    # get the MAD of the unphased light curve
    lightcurve_median = np.median(fmags)
    lightcurve_mad = np.median(np.abs(fmags - lightcurve_median))

    #
    # fourier fit
    #

    # we fit a Fourier series to the light curve using the best period and
    # extract the amplitudes and phases up to the 8th order to fit the LC. the
    # various ratios of the amplitudes A_ij and the differences in the phases
    # phi_ij are also used as periodic variability features

    # do the fit
    ffit = lcfit.fourier_fit_magseries(ftimes, fmags, ferrs, period,
                                       fourierorder=fourierorder,
                                       sigclip=sigclip,
                                       magsarefluxes=magsarefluxes,
                                       verbose=verbose)

    # get the coeffs and redchisq
    fourier_fitcoeffs = ffit['fitinfo']['finalparams']
    fourier_chisq = ffit['fitchisq']
    fourier_redchisq = ffit['fitredchisq']

    if fourier_fitcoeffs is not None:

        fourier_modelmags, _, _, fpmags, _ = sinusoidal.fourier_sinusoidal_func(
            [period,
             ffit['fitinfo']['fitepoch'],
             ffit['fitinfo']['finalparams'][:fourierorder],
             ffit['fitinfo']['finalparams'][fourierorder:]],
            ftimes,
            fmags,
            ferrs
        )

        fourier_residuals = fourier_modelmags - fpmags
        fourier_residual_median = np.median(fourier_residuals)
        fourier_residual_mad = np.median(np.abs(fourier_residuals -
                                                fourier_residual_median))


        # break them out into amps and phases
        famplitudes = fourier_fitcoeffs[:fourierorder]
        fphases = fourier_fitcoeffs[fourierorder:]

        famp_combos = combinations(famplitudes,2)
        famp_cinds = combinations(range(len(famplitudes)),2)

        fpha_combos = combinations(fphases,2)
        fpha_cinds = combinations(range(len(fphases)),2)

    else:

        LOGERROR('LC fit to sinusoidal series model failed, '
                 'using initial params')

        initfourieramps = [0.6] + [0.2]*(fourierorder - 1)
        initfourierphas = [0.1] + [0.1]*(fourierorder - 1)

        fourier_modelmags, _, _, fpmags, _ = sinusoidal.fourier_sinusoidal_func(
            [period,
             ffit['fitinfo']['fitepoch'],
             initfourieramps,
             initfourierphas],
            ftimes,
            fmags,
            ferrs
        )

        fourier_residuals = fourier_modelmags - fpmags
        fourier_residual_median = np.median(fourier_residuals)
        fourier_residual_mad = np.median(np.abs(fourier_residuals -
                                                fourier_residual_median))

        # break them out into amps and phases
        famplitudes = initfourieramps
        fphases = initfourierphas

        famp_combos = combinations(famplitudes,2)
        famp_cinds = combinations(range(len(famplitudes)),2)

        fpha_combos = combinations(fphases,2)
        fpha_cinds = combinations(range(len(fphases)),2)


    fampratios = {}
    fphadiffs = {}

    # get the ratios for all fourier coeff combinations
    for ampi, ampc, phai, phac in zip(famp_cinds,
                                      famp_combos,
                                      fpha_cinds,
                                      fpha_combos):

        ampratind = 'R_%s%s' % (ampi[1]+1, ampi[0]+1)
        # this is R_ij
        amprat = ampc[1]/ampc[0]
        phadiffind = 'phi_%s%s' % (phai[1]+1, phai[0]+1)
        # this is phi_ij
        phadiff = phac[1] - phai[0]*phac[0]

        fampratios[ampratind] = amprat
        fphadiffs[phadiffind] = phadiff

    # update the outdict for the Fourier fit results
    outdict = {
        'fourier_ampratios':fampratios,
        'fourier_phadiffs':fphadiffs,
        'fourier_fitparams':fourier_fitcoeffs,
        'fourier_redchisq':fourier_redchisq,
        'fourier_chisq':fourier_chisq,
        'fourier_residual_median':fourier_residual_median,
        'fourier_residual_mad':fourier_residual_mad,
        'fourier_residual_mad_over_lcmad':fourier_residual_mad/lightcurve_mad
    }

    # EB and planet fits will find the epoch automatically
    planetfitparams = [period,
                       None,
                       transitparams[0],
                       transitparams[1],
                       transitparams[2]]

    ebfitparams = [period,
                   None,
                   ebparams[0],
                   ebparams[1],
                   ebparams[2],
                   ebparams[3]]

    # do the planet and EB fit with this period
    planet_fit = lcfit.traptransit_fit_magseries(ftimes, fmags, ferrs,
                                                 planetfitparams,
                                                 sigclip=sigclip,
                                                 magsarefluxes=magsarefluxes,
                                                 verbose=verbose)

    planetfit_finalparams = planet_fit['fitinfo']['finalparams']
    planetfit_chisq = planet_fit['fitchisq']
    planetfit_redchisq = planet_fit['fitredchisq']

    if planetfit_finalparams is not None:

        planet_modelmags, _, _, ppmags, _ = transits.trapezoid_transit_func(
            planetfit_finalparams,
            ftimes,
            fmags,
            ferrs
        )

    else:

        LOGERROR('LC fit to transit planet model failed, using initial params')
        planet_modelmags, _, _, ppmags, _ = transits.trapezoid_transit_func(
            planetfitparams,
            ftimes,
            fmags,
            ferrs
        )

    planet_residuals = planet_modelmags - ppmags
    planet_residual_median = np.median(planet_residuals)
    planet_residual_mad = np.median(np.abs(planet_residuals -
                                           planet_residual_median))


    eb_fit = lcfit.gaussianeb_fit_magseries(ftimes, fmags, ferrs,
                                            ebfitparams,
                                            sigclip=sigclip,
                                            magsarefluxes=magsarefluxes,
                                            verbose=verbose)

    ebfit_finalparams = eb_fit['fitinfo']['finalparams']
    ebfit_chisq = eb_fit['fitchisq']
    ebfit_redchisq = eb_fit['fitredchisq']

    if ebfit_finalparams is not None:

        eb_modelmags, _, _, ebpmags, _ = eclipses.invgauss_eclipses_func(
            ebfit_finalparams,
            ftimes,
            fmags,
            ferrs
        )

    else:

        LOGERROR('LC fit to EB model failed, using initial params')

        eb_modelmags, _, _, ebpmags, _ = eclipses.invgauss_eclipses_func(
            ebfitparams,
            ftimes,
            fmags,
            ferrs
        )

    eb_residuals = eb_modelmags - ebpmags
    eb_residual_median = np.median(eb_residuals)
    eb_residual_mad = np.median(np.abs(eb_residuals - eb_residual_median))

    # do the EB fit with 2 x period
    ebfitparams[0] = ebfitparams[0]*2.0
    eb_fitx2 = lcfit.gaussianeb_fit_magseries(ftimes, fmags, ferrs,
                                              ebfitparams,
                                              sigclip=sigclip,
                                              magsarefluxes=magsarefluxes,
                                              verbose=verbose)

    ebfitx2_finalparams = eb_fitx2['fitinfo']['finalparams']
    ebfitx2_chisq = eb_fitx2['fitchisq']
    ebfitx2_redchisq = eb_fitx2['fitredchisq']

    if ebfitx2_finalparams is not None:

        ebx2_modelmags, _, _, ebx2pmags, _ = eclipses.invgauss_eclipses_func(
            ebfitx2_finalparams,
            ftimes,
            fmags,
            ferrs
        )

    else:

        LOGERROR('LC fit to EB model with 2xP failed, using initial params')

        ebx2_modelmags, _, _, ebx2pmags, _ = eclipses.invgauss_eclipses_func(
            ebfitparams,
            ftimes,
            fmags,
            ferrs
        )

    ebx2_residuals = ebx2_modelmags - ebx2pmags
    ebx2_residual_median = np.median(ebx2_residuals)
    ebx2_residual_mad = np.median(np.abs(ebx2_residuals -
                                         ebx2_residual_median))

    # update the outdict
    outdict.update({
        'planet_fitparams':planetfit_finalparams,
        'planet_chisq':planetfit_chisq,
        'planet_redchisq':planetfit_redchisq,
        'planet_residual_median':planet_residual_median,
        'planet_residual_mad':planet_residual_mad,
        'planet_residual_mad_over_lcmad':(
            planet_residual_mad/lightcurve_mad,
        ),
        'eb_fitparams':ebfit_finalparams,
        'eb_chisq':ebfit_chisq,
        'eb_redchisq':ebfit_redchisq,
        'eb_residual_median':eb_residual_median,
        'eb_residual_mad':eb_residual_mad,
        'eb_residual_mad_over_lcmad':(
            eb_residual_mad/lightcurve_mad,
        ),
        'ebx2_fitparams':ebfitx2_finalparams,
        'ebx2_chisq':ebfitx2_chisq,
        'ebx2_redchisq':ebfitx2_redchisq,
        'ebx2_residual_median':ebx2_residual_median,
        'ebx2_residual_mad':ebx2_residual_mad,
        'ebx2_residual_mad_over_lcmad':(
            ebx2_residual_mad/lightcurve_mad,
        ),
    })

    return outdict



def periodogram_features(pgramlist, times, mags, errs,
                         sigclip=10.0,
                         pdiff_threshold=1.0e-4,
                         sidereal_threshold=1.0e-4,
                         sampling_peak_multiplier=5.0,
                         sampling_startp=None,
                         sampling_endp=None,
                         verbose=True):
    '''This calculates various periodogram features (for each periodogram).

    pgramlist is a list of dicts returned by any of the periodfinding methods in
    astrobase.periodbase. This can also be obtained from the resulting pickle
    from the lcproc.run_pf function. Might be a good idea to make pgramlist a
    list of periodogram lists from all magnitude columns to test periodic
    variability across all magnitude columns (e.g. period diffs between EPD and
    TFA mags)

    times, mags, errs are from the object's light curve. These are used to
    recalculat the sampling L-S periodogram if one is not present in
    pgramlist. If it's present, these can all be set to None.

    sigclip is the sigclip to apply to the light curve.

    pdiff_threshold is the max diff between periods to consider them the same.

    sidereal_threshold is the max diff between any of the periods and the
    sidereal day periods to consider them the same.

    sampling_peak_multipler is the minimum multiplicative factor of a period's
    normalized periodogram peak over the sampling periodogram peak at the same
    period required to accept the period as possibly real.

    sampling_startp and sampling_endp are provided if the pgramlist doesn't have
    a spectral window LSP and this must be obtained from the times, mags, errs
    directly by running periodbase.specwindow_lsp.

    '''
    # run the sampling peak periodogram if necessary
    pfmethodlist = [pgram['method'] for pgram in pgramlist]

    if 'win' not in pfmethodlist:

        # get the finite values
        finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
        ftimes, fmags, ferrs = times[finind], mags[finind], errs[finind]

        # get nonzero errors
        nzind = np.nonzero(ferrs)
        ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

        sampling_lsp = specwindow_lsp(times, mags, errs,
                                      startp=sampling_startp,
                                      endp=sampling_endp,
                                      sigclip=sigclip,
                                      verbose=verbose)

    else:
        sampling_lsp = pgramlist[pfmethodlist.index('win')]


    # get the normalized sampling periodogram peaks
    normalized_sampling_lspvals = (
        sampling_lsp['lspvals']/(np.nanmax(sampling_lsp['lspvals']) -
                                 np.nanmin(sampling_lsp['lspvals']))
    )
    normalized_sampling_periods = sampling_lsp['periods']

    # go through the periodograms and calculate normalized peak height of best
    # periods over the normalized peak height of the sampling periodogram at the
    # same periods

    for pfm, pgram in zip(pfmethodlist, pgramlist):

        if pfm == 'pdm':

            best_peak_sampling_ratios = []
            close_to_sidereal_flag = []

            periods = pgram['periods']
            peaks = pgram['lspvals']

            normalized_peaks = (1.0 - peaks)/(np.nanmax(1.0 - peaks) -
                                              np.nanmin(1.0 - peaks))

            # get the best period normalized peaks
            if pgram['nbestperiods'] is None:
                LOGERROR('no period results for method: %s' % pfm)
                continue

            for bp in pgram['nbestperiods']:

                if np.isfinite(bp):

                    #
                    # first, get the normalized peak ratio
                    #
                    thisp_norm_pgrampeak = normalized_peaks[periods == bp]

                    thisp_sampling_pgramind = (
                        np.abs(normalized_sampling_periods -
                               bp) < pdiff_threshold
                    )
                    thisp_sampling_peaks = normalized_sampling_lspvals[
                        thisp_sampling_pgramind
                    ]
                    if thisp_sampling_peaks.size > 1:
                        thisp_sampling_ratio = (
                            thisp_norm_pgrampeak/np.mean(thisp_sampling_peaks)
                        )
                    elif thisp_sampling_peaks.size == 1:
                        thisp_sampling_ratio = (
                            thisp_norm_pgrampeak/thisp_sampling_peaks
                        )
                    else:
                        LOGERROR('sampling periodogram is not defined '
                                 'at period %.5f, '
                                 'skipping calculation of ratio' % bp)
                        thisp_sampling_ratio = np.nan

                    best_peak_sampling_ratios.append(thisp_sampling_ratio)

                    #
                    # next, see if the best periods are close to a sidereal day
                    # or any multiples of thus
                    #
                    sidereal_a_ratio = (bp - 1.0027379)/bp
                    sidereal_b_ratio = (bp - 0.9972696)/bp

                    if ((sidereal_a_ratio < sidereal_threshold) or
                        (sidereal_b_ratio < sidereal_threshold)):

                        close_to_sidereal_flag.append(True)

                    else:

                        close_to_sidereal_flag.append(False)



                else:
                    LOGERROR('period is nan')
                    best_peak_sampling_ratios.append(np.nan)
                    close_to_sidereal_flag.append(False)

            # update the pgram with these
            pgram['nbestpeakratios'] = best_peak_sampling_ratios
            pgram['siderealflags'] = close_to_sidereal_flag


        elif pfm != 'win':

            best_peak_sampling_ratios = []
            close_to_sidereal_flag = []

            periods = pgram['periods']
            peaks = pgram['lspvals']

            normalized_peaks = peaks/(np.nanmax(peaks) - np.nanmin(peaks))

            # get the best period normalized peaks
            if pgram['nbestperiods'] is None:
                LOGERROR('no period results for method: %s' % pfm)
                continue

            #
            # first, get the best period normalized peaks
            #
            for bp in pgram['nbestperiods']:

                if np.isfinite(bp):

                    thisp_norm_pgrampeak = normalized_peaks[periods == bp]

                    thisp_sampling_pgramind = (
                        np.abs(normalized_sampling_periods -
                               bp) < pdiff_threshold
                    )
                    thisp_sampling_peaks = normalized_sampling_lspvals[
                        thisp_sampling_pgramind
                    ]
                    if thisp_sampling_peaks.size > 1:
                        thisp_sampling_ratio = (
                            thisp_norm_pgrampeak/np.mean(thisp_sampling_peaks)
                        )
                    elif thisp_sampling_peaks.size == 1:
                        thisp_sampling_ratio = (
                            thisp_norm_pgrampeak/thisp_sampling_peaks
                        )
                    else:
                        LOGERROR('sampling periodogram is not defined '
                                 'at period %.5f, '
                                 'skipping calculation of ratio' % bp)
                        thisp_sampling_ratio = np.nan

                    best_peak_sampling_ratios.append(thisp_sampling_ratio)

                    #
                    # next, see if the best periods are close to a sidereal day
                    # or any multiples of thus
                    #
                    sidereal_a_ratio = (bp - 1.0027379)/bp
                    sidereal_b_ratio = (bp - 0.9972696)/bp

                    if ((sidereal_a_ratio < sidereal_threshold) or
                        (sidereal_b_ratio < sidereal_threshold)):

                        close_to_sidereal_flag.append(True)

                    else:

                        close_to_sidereal_flag.append(False)


                else:
                    LOGERROR('period is nan')
                    best_peak_sampling_ratios.append(np.nan)
                    close_to_sidereal_flag.append(False)

            # update the pgram with these
            pgram['nbestpeakratios'] = best_peak_sampling_ratios
            pgram['siderealflags'] = close_to_sidereal_flag

    #
    # done with calculations, get the features we need
    #
    # get the best periods across all the period finding methods
    all_bestperiods = np.concatenate(
        [x['nbestperiods']
         for x in pgramlist if
         (x['method'] != 'win' and x['nbestperiods'] is not None)]
    )
    all_bestperiod_diffs = np.array(
        [abs(a-b) for a,b in combinations(all_bestperiods,2)]
    )

    all_sampling_ratios = np.concatenate(
        [x['nbestpeakratios']
         for x in pgramlist if
         (x['method'] != 'win' and x['nbestperiods'] is not None)]
    )

    all_sidereal_flags = np.concatenate(
        [x['siderealflags']
         for x in pgramlist if
         (x['method'] != 'win' and x['nbestperiods'] is not None)]
    )

    # bestperiods_n_abovesampling - number of top period estimates with peaks
    #                               that are at least sampling_peak_multiplier x
    #                               sampling peak height at the same period
    bestperiods_n_abovesampling = (
        all_sampling_ratios[all_sampling_ratios >
                            sampling_peak_multiplier]
    ).size
    # bestperiods_n_sidereal - number of top period estimates that are
    #                          consistent with a 1 day period (1.0027379 and
    #                          0.9972696 actually, for sidereal day period)
    bestperiods_n_sidereal = all_sidereal_flags.sum()

    # bestperiods_diffn_threshold - the number of cross-wise period diffs from
    #                               all period finders that fall below the
    #                               pdiff_threshold
    bestperiods_diffn_threshold = (
        all_bestperiod_diffs < pdiff_threshold
    ).size

    resdict = {
        'bestperiods_n_abovesampling':bestperiods_n_abovesampling,
        'bestperiods_n_sidereal':bestperiods_n_sidereal,
        'bestperiods_diffn_threshold':bestperiods_diffn_threshold
    }

    return resdict



def phasedlc_features(times,
                      mags,
                      errs,
                      period,
                      nbrtimes=None,
                      nbrmags=None,
                      nbrerrs=None):
    '''This calculates various phased LC features for the object.

    If nbrtimes, nbrmags, and nbrerrs are all not None, they should
    be ndarrays with times, mags, errs of this object's closest neighbor (close
    within some small number x FWHM of telescope to check for blending) will
    also calculate extra features based on neighbor phased LC.

    '''
    # get the finite values
    finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes, fmags, ferrs = times[finind], mags[finind], errs[finind]

    # get nonzero errors
    nzind = np.nonzero(ferrs)
    ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

    # only operate on LC if enough points
    if ftimes.size > 49:

        # get the MAD of the unphased light curve
        lightcurve_median = np.median(fmags)
        lightcurve_mad = np.median(np.abs(fmags - lightcurve_median))

        # get p2p for raw lightcurve
        p2p_unphasedlc = lightcurve_ptp_measures(ftimes, fmags, ferrs)
        inveta_unphasedlc = 1.0/p2p_unphasedlc['eta_normal']

        # phase the light curve with the given period, assume epoch is
        # times.min()
        phasedlc = lcmath.phase_magseries_with_errs(ftimes, fmags, ferrs,
                                                    period, ftimes.min(),
                                                    wrap=False)

        phase = phasedlc['phase']
        pmags = phasedlc['mags']
        perrs = phasedlc['errs']

        # get ptp measures for best period
        ptp_bestperiod = lightcurve_ptp_measures(phase,pmags,perrs)

        # phase the light curve with the given periodx2, assume epoch is
        # times.min()
        phasedlc = lcmath.phase_magseries_with_errs(ftimes, fmags, ferrs,
                                                    period*2.0, ftimes.min(),
                                                    wrap=False)

        phasex2 = phasedlc['phase']
        pmagsx2 = phasedlc['mags']
        perrsx2 = phasedlc['errs']


        # get ptp measures for best periodx2
        ptp_bestperiodx2 = lightcurve_ptp_measures(phasex2,pmagsx2,perrsx2)

        # eta_phasedlc_bestperiod - calculate eta for the phased LC with best
        # period
        inveta_bestperiod = 1.0/ptp_bestperiod['eta_normal']

        # eta_phasedlc_bestperiodx2 - calculate eta for the phased LC with best
        #                             period x 2
        inveta_bestperiodx2 = 1.0/ptp_bestperiodx2['eta_normal']


        # eta_phased_ratio_eta_raw - eta for best period phased LC / eta for raw
        # LC
        inveta_ratio_phased_unphased = inveta_bestperiod/inveta_unphasedlc

        # eta_phasedx2_ratio_eta_raw - eta for best periodx2 phased LC/eta for
        # raw LC
        inveta_ratio_phasedx2_unphased = inveta_bestperiodx2/inveta_unphasedlc


        # freq_model_max_delta_mags - absval of magdiff btw model phased LC
        #                             maxima using period x 2. look at points
        #                             more than 10 points away for maxima
        phasedx2_maxval_ind = argrelmax(pmagsx2, order=10)
        if phasedx2_maxval_ind[0].size > 1:
            phasedx2_magdiff_maxval = (
                np.max(np.abs(np.diff(pmagsx2[phasedx2_maxval_ind[0]])))
            )
        else:
            phasedx2_magdiff_maxval = np.nan

        # freq_model_min_delta_mags - absval of magdiff btw model phased LC
        #                             minima using period x 2. look at points
        #                             more than 10 points away for minima
        phasedx2_minval_ind = argrelmin(pmagsx2, order=10)
        if phasedx2_minval_ind[0].size > 1:
            phasedx2_magdiff_minval = (
                np.max(np.abs(np.diff(pmagsx2[phasedx2_minval_ind[0]])))
            )
        else:
            phasedx2_magdiff_minval = np.nan

        # p2p_scatter_pfold_over_mad - MAD of successive absolute mag diffs of
        #                              the phased LC using best period divided
        #                              by the MAD of the unphased LC
        phased_magdiff = np.diff(pmags)
        phased_magdiff_median = np.median(phased_magdiff)
        phased_magdiff_mad = np.median(np.abs(phased_magdiff -
                                              phased_magdiff_median))

        phasedx2_magdiff = np.diff(pmagsx2)
        phasedx2_magdiff_median = np.median(phasedx2_magdiff)
        phasedx2_magdiff_mad = np.median(np.abs(phasedx2_magdiff -
                                                phasedx2_magdiff_median))

        phased_magdiffmad_unphased_mad_ratio = phased_magdiff_mad/lightcurve_mad
        phasedx2_magdiffmad_unphased_mad_ratio = (
            phasedx2_magdiff_mad/lightcurve_mad
        )

        # get the percentiles of the slopes of the adjacent mags for phasedx2
        phasedx2_slopes = np.diff(pmagsx2)/np.diff(phasex2)
        phasedx2_slope_percentiles = np.ravel(np.nanpercentile(phasedx2_slopes,
                                                               [10.0,90.0]))
        phasedx2_slope_10percentile = phasedx2_slope_percentiles[0]
        phasedx2_slope_90percentile = phasedx2_slope_percentiles[1]

        # check if nbrtimes, _mags, _errs are available
        if ((nbrtimes is not None) and
            (nbrmags is not None) and
            (nbrerrs is not None)):

            # get the finite values
            nfinind = (np.isfinite(nbrtimes) &
                       np.isfinite(nbrmags) &
                       np.isfinite(nbrerrs))
            nftimes, nfmags, nferrs = (nbrtimes[nfinind],
                                       nbrmags[nfinind],
                                       nbrerrs[nfinind])

            # get nonzero errors
            nnzind = np.nonzero(nferrs)
            nftimes, nfmags, nferrs = (nftimes[nnzind],
                                       nfmags[nnzind],
                                       nferrs[nnzind])

            # only operate on LC if enough points
            if nftimes.size > 49:

                # get the phased light curve using the same period and epoch as
                # the actual object
                nphasedlc = lcmath.phase_magseries_with_errs(
                    nftimes, nfmags, nferrs,
                    period, ftimes.min(),
                    wrap=False
                )

                # normalize the object and neighbor phased mags
                norm_pmags = pmags - np.median(pmags)
                norm_npmags = nphasedlc['mags'] - np.median(nphasedlc['mags'])

                # phase bin them both so we can compare LCs easily
                phabinned_objectlc = lcmath.phase_bin_magseries(phase,
                                                                norm_pmags,
                                                                minbinelems=1)
                phabinned_nbrlc = lcmath.phase_bin_magseries(nphasedlc['phase'],
                                                             norm_npmags,
                                                             minbinelems=1)

                absdiffs = []

                for pha, phamag in zip(phabinned_objectlc['binnedphases'],
                                       phabinned_objectlc['binnedmags']):

                    try:

                        # get the matching phase from the neighbor phased LC
                        phadiffs = np.abs(pha - phabinned_nbrlc['binnedphases'])
                        minphadiffind = np.where(
                            (phadiffs < 1.0e-4) &
                            (phadiffs == np.min(phadiffs))
                        )
                        absmagdiff = np.abs(
                            phamag - phabinned_nbrlc['binnedmags'][
                                minphadiffind
                            ]
                        )
                        if absmagdiff.size > 0:
                            absdiffs.append(absmagdiff.min())

                    except Exception as e:
                        continue

                # sum of absdiff between the normalized to 0.0 phased LC of this
                # object and that of the closest neighbor phased with the same
                # period and epoch
                if len(absdiffs) > 0:
                    sum_nbr_phasedlc_magdiff = sum(absdiffs)
                else:
                    sum_nbr_phasedlc_magdiff = np.nan

            else:

                sum_nbr_phasedlc_magdiff = np.nan

        else:
            sum_nbr_phasedlc_magdiff = np.nan

        return {
            'inveta_unphasedlc':inveta_unphasedlc,
            'inveta_bestperiod':inveta_bestperiod,
            'inveta_bestperiodx2':inveta_bestperiodx2,
            'inveta_ratio_phased_unphased':inveta_ratio_phased_unphased,
            'inveta_ratio_phasedx2_unphased':inveta_ratio_phasedx2_unphased,
            'phasedx2_magdiff_maxima':phasedx2_magdiff_maxval,
            'phasedx2_magdiff_minina':phasedx2_magdiff_minval,
            'phased_unphased_magdiff_mad_ratio':(
                phased_magdiffmad_unphased_mad_ratio
            ),
            'phasedx2_unphased_magdiff_mad_ratio':(
                phasedx2_magdiffmad_unphased_mad_ratio
            ),
            'phasedx2_slope_10percentile':phasedx2_slope_10percentile,
            'phasedx2_slope_90percentile':phasedx2_slope_90percentile,
            'sum_nbr_phasedlc_magdiff':sum_nbr_phasedlc_magdiff,
        }

    else:

        return {
            'inveta_unphasedlc':np.nan,
            'inveta_bestperiod':np.nan,
            'inveta_bestperiodx2':np.nan,
            'inveta_ratio_phased_unphased':np.nan,
            'inveta_ratio_phasedx2_unphased':np.nan,
            'phasedx2_magdiff_maxima':np.nan,
            'phasedx2_magdiff_minina':np.nan,
            'phased_unphased_magdiff_mad_ratio':np.nan,
            'phasedx2_unphased_magdiff_mad_ratio':np.nan,
            'phasedx2_slope_10percentile':np.nan,
            'phasedx2_slope_90percentile':np.nan,
            'sum_nbr_phasedlc_magdiff':np.nan,
        }
