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

def fourier_features(times, mags, errs, period,
                     fourierorder=8,
                     sigclip=3.0,
                     magsarefluxes=False):
    '''
    This calculates various ratios of fourier-fit amplitudes and phase params.

    '''

    # freq_amplitude_ratio_21 - amp ratio of the 2nd to 1st Fourier component

    # freq_amplitude_ratio_31 - amp ratio of the 3rd to 1st Fourier component

    # freq_rrd - 1 if freq_frequency_ratio_21 or freq_frequency_ratio_31 are
    #            close to 0.746 (characteristic of RRc? -- double mode RRLyr), 0
    #            otherwise

    # in addition, we fit a Fourier series to the light curve using the best
    # period and extract the amplitudes and phases up to the 8th order to fit
    # the LC. the various ratios of the amplitudes A_ij and the differences in
    # the phases phi_ij are also used as periodic variability features


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



def lcfit_features(times, mags, errs, period):
    '''
    This calculates various features related to fits to the phased LC.

    '''

    # ebchisq, pltchisq, fourierchisq, splchisq - red-chisq-values for these
    #                                             fits to the phased LC

    # medperc90_2p_p - 90th percentile of the absolute residual values around
    #                  the light curve phased with best period x 2 divided by
    #                  the same quantity for the residuals using the phased
    #                  light curve with best period (to detect EBs)
