#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''periodicfeatures - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This contains functions that calculate various light curve features using
information about periods and fits to phased light curves.

FIXME: add more interesting features from FATS and Upsilon.

'''

import logging
from datetime import datetime
from traceback import format_exc
from time import time as unixtime
from itertools import combinations

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


###################################
## FEATURE CALCULATION FUNCTIONS ##
###################################

def lcfit_features(times, mags, errs, period,
                   fourierorder=8,
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

    #
    # fourier fit
    #

    # we fit a Fourier series to the light curve using the best period and
    # extract the amplitudes and phases up to the 8th order to fit the LC. the
    # various ratios of the amplitudes A_ij and the differences in the phases
    # phi_ij are also used as periodic variability features

    # do the fit
    ffit = lcfit.fourier_fit_magseries(times, mags, errs, period,
                                       fourierorder=fourierorder,
                                       sigclip=sigclip,
                                       magsarefluxes=magsarefluxes,
                                       verbose=verbose)

    # get the coeffs and redchisq
    fourier_fitcoeffs = ffit['fitinfo']['finalparams']
    fourier_chisq = ffit['fitchisq']
    fourier_redchisq = ffit['fitredchisq']

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
    outdict = {'fourier_ampratios':fampratios,
               'fourier_phadiffs':fphadiffs,
               'fourier_fitparams':fourier_fitcoeffs,
               'fourier_redchisq':fourier_redchisq,
               'fourier_chisq':fourier_chisq}

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
    planet_fit = lcfit.traptransit_fit_magseries(times, mags, errs,
                                                 planetfitparams,
                                                 sigclip=sigclip,
                                                 magsarefluxes=magsarefluxes,
                                                 verbose=verbose)

    planetfit_finalparams = planet_fit['fitinfo']['finalparams']
    planetfit_chisq = planet_fit['fitchisq']
    planetfit_redchisq = planet_fit['fitredchisq']


    eb_fit = lcfit.gaussianeb_fit_magseries(times, mags, errs,
                                            ebfitparams,
                                            sigclip=sigclip,
                                            magsarefluxes=magsarefluxes,
                                            verbose=verbose)

    ebfit_finalparams = eb_fit['fitinfo']['finalparams']
    ebfit_chisq = eb_fit['fitchisq']
    ebfit_redchisq = eb_fit['fitredchisq']

    # do the EB fit with 2 x period
    ebfitparams[0] = ebfitparams[0]*2.0
    eb_fitx2 = lcfit.gaussianeb_fit_magseries(times, mags, errs,
                                              ebfitparams,
                                              sigclip=sigclip,
                                              magsarefluxes=magsarefluxes,
                                              verbose=verbose)

    ebfitx2_finalparams = eb_fitx2['fitinfo']['finalparams']
    ebfitx2_chisq = eb_fitx2['fitchisq']
    ebfitx2_redchisq = eb_fitx2['fitredchisq']

    # update the outdict
    outdict.update({
        'planet_fitparams':planetfit_finalparams,
        'planet_chisq':planetfit_chisq,
        'planet_redchisq':planetfit_redchisq,
        'eb_fitparams':ebfit_finalparams,
        'eb_chisq':ebfit_chisq,
        'eb_redchisq':ebfit_redchisq,
        'ebx2_fitparams':ebfitx2_finalparams,
        'ebx2_chisq':ebfitx2_chisq,
        'ebx2_redchisq':ebfitx2_redchisq,
    })


    return outdict



def periodogram_features(periodogramresults,
                         times, mags, errs, period,
                         pdiffthreshold=1.0e-5,
                         pgram_smoothwindow=5):
    '''
    This calculates various periodogram features.

    '''

    # freq_n_sidereal - number of top period estimates that are consistent with
    #                   a 1 day period (1.0027379 and 0.9972696 actually, for
    #                   sidereal day period) and 0.5x, 2x, and 3x multipliers

    # peak_height_over_background - ratio of best normalized periodogram peak
    #                               height to that of the periodogram background
    #                               near the same period peak

    # peak_height_over_sampling_peak_height - ratio of best normalized
    #                                         periodogram peak height to that of
    #                                         the sampling periodogram at the
    #                                         same period

    # smallest_nbestperiods_diff - the smallest cross-wise difference between
    #                              the best periods found by all the
    #                              period-finders used


def phasedlc_features(times, mags, errs, period):
    '''
    This calculates various phased LC features.

    '''


    # freq_model_max_delta_mags - absval of magdiff btw model phased LC maxima
    #                             using period x 2

    # freq_model_max_delta_mags - absval of magdiff btw model phased LC minima
    #                             using period x 2

    # freq_model_phi1_phi2 - ratio of the phase difference between the first
    #                        minimum and the first maximum to the phase
    #                        difference between first minimum and second maximum

    # scatter_res_raw - MAD of the GLS phased LC residuals divided by MAD of the
    #                   raw light curve (unphased)

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

    # skew, kurtosis - for the phased light curve



def neighbor_features(lclistpkl,
                      objectid,
                      fwhm_arcsec,
                      period,
                      epoch):
    '''
    This calculates various features based on neighboring stars.

    '''

    # distance to closest neighbor in arcsec

    # number of neighbors within 2 x fwhm_arcsec

    # sum of absdiff between the phased LC of this object and that of the
    # closest neighbor phased with the same period and epoch
