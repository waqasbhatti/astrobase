#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''varbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2016

Contains functions to deal with light curve variability, fitting functions,
masking signals, autocorrelation, etc.

CURRENT SUBMODULES:

varbase.autocorr: calculating the autocorrelation function of light curves
varbase.lcfit: fitting Fourier series and splines to light curves
varbase.signals: masking periodic signals, pre-whitening light curves

'''

# there's nothing else here in this top-level module

def estimate_achievable_tmid_precision(snr, t_ingress_min=10,
                                       t_duration_hr=2.14):
    '''
    Using Carter et al. 2009's estimate, calculate the theoretical optimal
    precision on mid-transit time measurement possible given a transit of a
    particular SNR.

    sigma_tc = Q^{-1} * T * sqrt(θ/2)

    Q = SNR of the transit.
    T = transit duration, which is 2.14 hours from discovery paper.
    θ = τ/T = ratio of ingress to total duration
            ~= (few minutes [guess]) / 2.14 hours

    args:

        snr (float): measured signal-to-noise of transit, e.g., from
        `periodbase.get_snr_of_dip`

    kwargs:

        t_ingress_min (float): ingress duration in minutes, t_I to t_II in Winn
        (2010) nomenclature.

        t_duration_hr (float): total transit duration in hours, t_I to t_IV.
    '''

    t_ingress = t_ingress_min*u.minute
    t_duration = t_duration_hr*u.hour

    theta = t_ingress/t_duration

    sigma_tc = (1/snr * t_duration * np.sqrt(theta/2))

    print('assuming t_ingress = {:.1f}'.format(t_ingress))
    print('assuming t_duration = {:.1f}'.format(t_duration))
    print('measured SNR={:.2f}\n\t'.format(snr)+
          '-->theoretical sigma_tc = {:.2e} = {:.2e} = {:.2e}'.format(
          sigma_tc.to(u.minute), sigma_tc.to(u.hour), sigma_tc.to(u.day))
    )

    return sigma_tc.to(u.day).value
