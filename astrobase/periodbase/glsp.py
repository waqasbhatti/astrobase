#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''glsp.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 02/2016

Implementation of the generalized Lomb-Scargle periodogram algorithm from
Zechmeister and Kurster (2008).

'''

import numpy as np

from numpy import sum as npsum, \
    cos as npcos, sin as npsin, \
    arctan as nparctan, zeros as npzeros

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
