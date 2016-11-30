#!/usr/bin/env python

'''glsp.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 02/2016

Implementation of the generalized Lomb-Scargle periodogram algorithm from
Zechmeister and Kurster (2008).

'''

import numpy as np

from numpy import sum as npsum, \
    cos as npcos, sin as npsin, \
    arctan as nparctan, zeros as npzeros

######################################################
## PERIODOGRAM VALUE EXPRESSIONS FOR A SINGLE OMEGA ##
######################################################

def generalized_lsp_value(times, mags, errs, omega):
    '''Generalized LSP value for a single omega.

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

    '''

    one_over_errs2 = 1.0/(errs*errs)

    W = npsum(one_over_errs2)
    wi = one_over_errs2/W

    sin_omegat = npsin(omega*times)
    cos_omegat = npcos(omega*times)

    sin2_omegat = sin_omegat*sin_omegat
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
    SS = 1 - CpC - S*S # use SpS = 1 - CpC
    CS = CpS - C*S

    # calculate tau
    tan_omega_tau_top = 2.0*CS
    tan_omega_tau_bottom = CC - SS
    tan_omega_tau = tan_omega_tau_top/tan_omega_tau_bottom
    tau = nparctan(tan_omega_tau/(2.0*omega))

    periodogramvalue = (YC*YC/CC + YS*YS/SS)/YY

    return periodogramvalue



def generalized_lsp_value_notau(times, mags, errs, omega):
    '''
    This is the simplified version not using tau.

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

    '''

    one_over_errs2 = 1.0/(errs*errs)

    W = npsum(one_over_errs2)
    wi = one_over_errs2/W

    sin_omegat = npsin(omega*times)
    cos_omegat = npcos(omega*times)

    sin2_omegat = sin_omegat*sin_omegat
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
    SS = 1 - CpC - S*S # use SpS = 1 - CpC
    CS = CpS - C*S

    # P(omega) = (SS*YC*YC + CC*YS*YS - 2.0*CS*YC*YS)/(YY*D)
    # D(omega) = CC*SS - CS*CS
    Domega = CC*SS - CS*CS
    lspval = (SS*YC*YC + CC*YS*YS - 2.0*CS*YC*YS)/(YY*Domega)

    return lspval


###################################
## BASIC PERIODOGRAM CALCULATION ##
###################################

def gen_lsp(times,
            mags,
            errs,
            omegas):
    '''
    This runs the loops for the LSP calculation.

    Requires cleaned times, mags, errs (no nans).

    '''

    ndet = times.size
    omegalen = omegas.size

    # the output array
    pvals = npzeros(omegalen, dtype=np.float64)

    for oind in range(omegalen):

        thisomega = omegas[oind]
        thispval = generalized_lsp_value(times, mags, errs, thisomega)
        pvals[oind] = thispval

    return pvals



def gen_lsp_notau(times,
                  mags,
                  errs,
                  omegas):
    '''
    This runs the loops for the LSP calculation.

    Requires cleaned times, mags, errs (no nans).

    '''

    ndet = times.size
    omegalen = omegas.size

    # the output array
    pvals = npzeros(omegalen, dtype=np.float64)

    for oind in range(omegalen):

        thisomega = omegas[oind]
        thispval = generalized_lsp_value_notau(times, mags, errs, thisomega)
        pvals[oind] = thispval

    return pvals
