#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''periodicfeatures - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This contains functions that calculate various light curve features using
information about periods and fits to phased light curves.

'''

import logging
from datetime import datetime
from traceback import format_exc
from time import time as unixtime
from itertools import combinations

import numpy as np


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.periodicfeatures' % parent_name)

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

from .. import lcmath
from ..varbase import lcfit
from ..lcmodels import sinusoidal, eclipses, transits
from ..periodbase import zgls

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

    planet_modelmags, _, _, ppmags, _ = transits.trapezoid_transit_func(
        planetfit_finalparams,
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

    eb_modelmags, _, _, ebpmags, _ = eclipses.invgauss_eclipses_func(
        ebfit_finalparams,
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

    ebx2_modelmags, _, _, ebx2pmags, _ = eclipses.invgauss_eclipses_func(
        ebfitx2_finalparams,
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



def periodogram_features(pgramlist,
                         times, mags, errs, period,
                         sigclip=10.0,
                         pdiffthreshold=1.0e-5,
                         sampling_startp=None,
                         sampling_endp=None,
                         verbose=True):
    '''
    This calculates various periodogram features (for each periodogram).

    '''
    # get the finite values
    finind = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes, fmags, ferrs = times[finind], mags[finind], errs[finind]

    # get nonzero errors
    nzind = np.nonzero(ferrs)
    ftimes, fmags, ferrs = ftimes[nzind], fmags[nzind], ferrs[nzind]

    # run the sampling peak periodogram
    sampling_lsp = zgls.specwindow_lsp(times,mags,errs,
                                       startp=sampling_startp,
                                       endp=sampling_endp,
                                       sigclip=sigclip,
                                       verbose=verbose)




    # freq_n_sidereal - number of top period estimates that are consistent with
    #                   a 1 day period (1.0027379 and 0.9972696 actually, for
    #                   sidereal day period) and 0.5x, 2x, and 3x multipliers

    # best_peak_height_over_sampling_height - ratio of best normalized
    #                                         periodogram peak height to that of
    #                                         the sampling periodogram at the
    #                                         same period

    # npeaks_above_5sig_sampling

    # smallest_nbestperiods_diff - the smallest cross-wise difference between
    #                              the best periods found by all the
    #                              period-finders used


def phasedlc_features(times, mags, errs, period):
    '''
    This calculates various phased LC features.

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


    # freq_model_max_delta_mags - absval of magdiff btw model phased LC maxima
    #                             using period x 2

    # freq_model_max_delta_mags - absval of magdiff btw model phased LC minima
    #                             using period x 2

    # freq_model_phi1_phi2 - ratio of the phase difference between the first
    #                        minimum and the first maximum to the phase
    #                        difference between first minimum and second maximum

    # scatter_res_raw - MAD of the GLS phased LC residuals divided by MAD of the
    #                   raw light curve (unphased). if this is close to 1.0 or
    #                   larger than 1.0, then the phased LC is no better than
    #                   the unphased light curve so the object may not be
    #                   periodic.

    # p2p_scatter_2praw - sum of the squared mag differences between pairs of
    #                     successive observations in the phased LC using best
    #                     period x 2 divided by that of the unphased light curve

    # p2p_scatter_pfold_over_mad - MAD of successive absolute mag diffs of the
    #                              phased LC using best period divided by the
    #                              MAD of the unphased LC

    # fold2P_slope_10percentile - 10th percentile of the slopes between adjacent
    #                             mags after the light curve is folded on best
    #                             period x 2

    # fold2P_slope_90percentile - 90th percentile of the slopes between adjacent
    #                             mags after the light curve is folded on best
    #                             period x 2



def neighbor_features(lclistpkl,
                      objectid,
                      fwhm_arcsec,
                      times,
                      mags,
                      errs,
                      period,
                      epoch):
    '''
    This calculates various features based on neighboring stars.

    '''

    # distance to closest neighbor in arcsec

    # number of neighbors within 2 x fwhm_arcsec

    # sum of absdiff between the phased LC of this object and that of the
    # closest neighbor phased with the same period and epoch
