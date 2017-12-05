#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''periodicfeatures - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This contains functions that calculate various light curve features using
information about periods and fits to phased light curves.

'''

###################################
## PERIODIC VARIABILITY FEATURES ##
###################################

# features to calculate

# freq_amplitude_ratio_21 - amp ratio of the 2nd to 1st Fourier component
# freq_amplitude_ratio_31 - amp ratio of the 3rd to 1st Fourier component
# freq_model_max_delta_mags - absval of magdiff btw model phased LC maxima
#                             using period x 2
# freq_model_max_delta_mags - absval of magdiff btw model phased LC minima
#                             using period x 2
# freq_model_phi1_phi2 - ratio of the phase difference between the first minimum
#                        and the first maximum to the phase difference between
#                        first minimum and second maximum
# freq_n_alias - number of top period estimates that are consistent with a 1
#                day period
# freq_rrd - 1 if freq_frequency_ratio_21 or freq_frequency_ratio_31 are close
#            to 0.746 (characteristic of RRc? -- double mode RRLyr), 0 otherwise
# scatter_res_raw - MAD of the GLS phased LC residuals divided by MAD of the raw
#                   light curve (unphased)
# p2p_scatter_2praw - sum of the squared mag differences between pairs of
#                     successive observations in the phased LC using best period
#                     x 2 divided by that of the unphased light curve
# p2p_scatter_pfold_over_mad - MAD of successive absolute mag diffs of the
#                              phased LC using best period divided by the MAD of
#                              the unphased LC
# medperc90_2p_p - 90th percentile of the absolute residual values around the
#                  light curve phased with best period x 2 divided by the same
#                  quantity for the residuals using the phased light curve with
#                  best period (to detect EBs)
# fold2P_slope_10percentile - 10th percentile of the slopes between adjacent
#                             mags after the light curve is folded on best
#                             period x 2
# fold2P_slope_90percentile - 90th percentile of the slopes between adjacent
#                             mags after the light curve is folded on best
#                             period x 2
# splchisq_over_fourierchisq - the ratio of the chi-squared value of a cubic
#                              spline fit to the phased LC with the best period
#                              to the chi-squared value of an 8th order Fourier
#                              fit to the phased LC with the best period -- this
#                              might be a good way to figure out if something
#                              looks like a detached EB vs a sinusoidal LC.

# in addition, we fit a Fourier series to the light curve using the best period
# and extract the amplitudes and phases up to the 8th order to fit the LC. the
# various ratios of the amplitudes A_ij and the differences in the phases phi_ij
# are also used as periodic variabilty features
