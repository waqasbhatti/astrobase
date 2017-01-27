#!/usr/bin/env python

'''
periodbase - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Contains various useful tools for period finding.

CURRENT:

periodbase.spdm -> Stellingwerf (1978) phase-dispersion minimization
periodbase.saov -> Schwarzenberg-Cerny (1989) analysis of variance
periodbase.zgls -> Zechmeister & Kurster (2009) generalized Lomb-Scargle
periodbase.kbls -> Kovacs et al. (2002) Box-Least-Squares search

TO BE IMPLEMENTED:

periodbase.smav -> Schwarzenberg-Cerny (1996) multi-harmonic AoV
periodbase.pfch -> Palmer (2002) fast-chi periodogram

'''


import logging
from datetime import datetime
from traceback import format_exc

import numpy as np

# import these to avoid lookup overhead
from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, \
    arctan as nparctan, nanargmax as npnanargmax, nanargmin as npnanargmin, \
    empty as npempty, ceil as npceil, mean as npmean, \
    digitize as npdigitize, unique as npunique, \
    argmax as npargmax, argmin as npargmin


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.periodbase' % parent_name)

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


#######################
## UTILITY FUNCTIONS ##
#######################

def get_frequency_grid(times,
                       samplesperpeak=5,
                       nyquistfactor=5,
                       minfreq=None,
                       maxfreq=None,
                       returnf0dfnf=False):
    '''This calculates a frequency grid for the period finding functions in this
    module.

    Based on the autofrequency function in astropy.stats.lombscargle.

    http://docs.astropy.org/en/stable/_modules/astropy/stats/lombscargle/core.html#LombScargle.autofrequency

    '''

    baseline = times.max() - times.min()
    nsamples = times.size

    df = 1. / baseline / samplesperpeak

    if minfreq is not None:
        f0 = minfreq
    else:
        f0 = 0.5 * df

    if maxfreq is not None:
        Nf = int(npceil((maxfreq - f0) / df))
    else:
        Nf = int(0.5 * samplesperpeak * nyquistfactor * nsamples)


    if returnf0dfnf:
        return f0, df, Nf, f0 + df * nparange(Nf)
    else:
        return f0 + df * nparange(Nf)


####################################################
## HOIST THE FINDER FUNCTIONS INTO THIS NAMESPACE ##
####################################################

from .zgls import pgen_lsp
from .spdm import stellingwerf_pdm
from .saov import aov_periodfind
from .kbls import bls_serial_pfind, bls_parallel_pfind
