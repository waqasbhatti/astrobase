#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
periodbase - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

Contains various useful tools for period finding.

CURRENT:

periodbase.spdm -> Stellingwerf (1978) phase-dispersion minimization
periodbase.saov -> Schwarzenberg-Czerny (1989) analysis of variance
periodbase.zgls -> Zechmeister & Kurster (2009) generalized Lomb-Scargle
periodbase.kbls -> Kovacs et al. (2002) Box-Least-Squares search
periodbase.macf -> McQuillan et al. (2013a, 2014) ACF period search
periodbase.smav -> Schwarzenberg-Czerny (1996) multi-harmonic AoV period search

TO BE IMPLEMENTED:

periodbase.gcep -> Graham et al. (2013) conditional entropy period search

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



########################
## SOME OTHER IMPORTS ##
########################

from ..lcmath import sigclip_magseries


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
from .smav import aovhm_periodfind
from .kbls import bls_serial_pfind, bls_parallel_pfind
from .macf import macf_period_find



#############################################################
## FUNCTIONS FOR TESTING SIGNIFICANCE OF PERIODOGRAM PEAKS ##
#############################################################

# used to figure out which function to run for bootstrap resampling
LSPMETHODS = {'bls':bls_parallel_pfind,
              'gls':pgen_lsp,
              'aov':aov_periodfind,
              'mav':aovhm_periodfind,
              'pdm':stellingwerf_pdm,
              'acf':macf_period_find}



def bootstrap_falsealarmprob(lspdict,
                             times,
                             mags,
                             errs,
                             nbootstrap=250,
                             magsarefluxes=False,
                             sigclip=10.0,
                             npeaks=None):
    '''Calculates the false alarm probabilities of periodogram peaks using
    bootstrap resampling of the magnitude time series.

    The false alarm probability here is defined as:

    (1.0 + sum(trialbestpeaks[i] > peak[j]))/(ntrialbestpeaks + 1)

    for each best periodogram peak j. The index i is for each bootstrap
    trial. This effectively gives us a significance for the peak. Smaller FAP
    means a better chance that the peak is real.

    The basic idea is to get the number of trial best peaks that are larger than
    the current best peak and divide this by the total number of trials. The
    distribution of these trial best peaks is obtained after scrambling the mag
    values and rerunning the specified periodogram method for a bunch of trials.

    The total number of trials is nbootstrap. This is set to 250 by default, but
    should probably be around 1000 for realistic results.

    lspdict is the output dict from a periodbase periodogram function and MUST
    contain a 'method' key that corresponds to one of the keys in the LSPMETHODS
    dict above. This will let this function know which periodogram function to
    run to generate the bootstrap samples. The lspdict SHOULD also have a
    'kwargs' key that corresponds to the input keyword arguments for the
    periodogram function as it was run originally, to keep everything the same
    during the bootstrap runs. If this is missing, default values will be used.

    FIXME: this may not be strictly correct; must look more into bootstrap
    significance testing. Also look into if we're doing resampling correctly for
    time series because the samples are not iid. Look into moving block
    bootstrap.

    '''

    # figure out how many periods to work on
    if (npeaks and (0 < npeaks < len(lspdict['nbestperiods']))):
        nperiods = npeaks
    else:
        LOGWARNING('npeaks not specified or invalid, '
                   'getting FAP for all %s periodogram peaks' %
                   len(lspdict['nbestperiods']))
        nperiods = len(lspdict['nbestperiods'])

    nbestperiods = lspdict['nbestperiods'][:nperiods]
    nbestpeaks = lspdict['nbestlspvals'][:nperiods]

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)

    allpeaks = []
    allperiods = []
    allfaps = []
    alltrialbestpeaks = []

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        for ind, period, peak in zip(range(len(nbestperiods)),
                                     nbestperiods,
                                     nbestpeaks):

            LOGINFO('peak %s: running %s trials...' % (ind+1, nbootstrap))

            trialbestpeaks = []

            for trial in range(nbootstrap):

                # get a scrambled index
                tindex = np.random.randint(0,
                                           high=mags.size,
                                           size=mags.size)


                # get the kwargs dict out of the lspdict
                if 'kwargs' in lspdict:

                    kwargs = lspdict['kwargs']

                    # update the kwargs with some local stuff
                    kwargs.update({'magsarefluxes':magsarefluxes,
                                   'sigclip':sigclip,
                                   'verbose':False})
                else:
                    kwargs = {'magsarefluxes':magsarefluxes,
                              'sigclip':sigclip,
                              'verbose':False}


                # run the periodogram with scrambled mags and errs
                # and the appropriate keyword arguments
                lspres = LSPMETHODS[lspdict['method']](
                    times, mags[tindex], errs[tindex],
                    **kwargs
                )
                trialbestpeaks.append(lspres['bestlspval'])

            trialbestpeaks = np.array(trialbestpeaks)
            alltrialbestpeaks.append(trialbestpeaks)

            # calculate the FAP for a trial peak j = FAP[j] =
            # (1.0 + sum(trialbestpeaks[i] > peak[j]))/(ntrialbestpeaks + 1)
            if lspdict['method'] != 'pdm':
                falsealarmprob = (
                    (1.0 + trialbestpeaks[trialbestpeaks > peak].size) /
                    (trialbestpeaks.size + 1.0)
                )
            # for PDM, we're looking for a peak smaller than the best peak
            # because values closer to 0.0 are more significant
            else:
                falsealarmprob = (
                    (1.0 + trialbestpeaks[trialbestpeaks < peak].size) /
                    (trialbestpeaks.size + 1.0)
                )

            LOGINFO('FAP for peak %s, period: %.6f = %.3g' % (ind+1,
                                                              period,
                                                              falsealarmprob))

            allpeaks.append(peak)
            allperiods.append(period)
            allfaps.append(falsealarmprob)

        return {'peaks':allpeaks,
                'periods':allperiods,
                'probabilities':allfaps,
                'alltrialbestpeaks':alltrialbestpeaks}

    else:
        LOGERROR('not enough mag series points to calculate periodogram')
        return None


############################################
## FUNCTIONS FOR COMPARING PERIOD-FINDERS ##
############################################


def make_combined_periodogram(pflist, outfile, addmethods=False):
    '''This just puts all of the period-finders on a single periodogram.

    This will renormalize all of the periodograms so their values lie between 0
    and 1, with values lying closer to 1 being more significant. Periodograms
    that give the same best periods will have their peaks line up together.

    Args
    ----

    pflist is a list of result dicts from any of the period-finders in
    periodbase.

    outfile is a file to write the output to. NOTE: EPS/PS won't work because we
    use alpha to better distinguish between the various periodograms.

    if addmethods = True, will add all of the normalized periodograms together,
    then renormalize them to between 0 and 1. In this way, if all of the
    period-finders agree on something, it'll stand out easily.

    '''

    import matplotlib.pyplot as plt

    for pf in pflist:

        if pf['method'] == 'pdm':

            plt.plot(pf['periods'],
                     np.max(pf['lspvals'])/pf['lspvals'] - 1.0,
                     label='%s P=%.5f' % (pf['method'], pf['bestperiod']),
                     alpha=0.5)

        else:

            plt.plot(pf['periods'],
                     pf['lspvals']/np.max(pf['lspvals']),
                     label='%s P=%.5f' % (pf['method'], pf['bestperiod']),
                     alpha=0.5)


    plt.xlabel('period [days]')
    plt.ylabel('normalized periodogram power')

    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close('all')

    return outfile
