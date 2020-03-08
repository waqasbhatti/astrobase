#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Apr 2019
# License: MIT - see the LICENSE file for the full text.

'''This contains some utilities for periodbase functions.

- :py:func:`.independent_freq_count`: gets the number of independent frequencies
  when calculating false alarm probabilities.

- :py:func:`.get_frequency_grid`: generates frequency grids automatically.

- :py:func:`.make_combined_periodogram`: makes a combined periodogram from the
  results of several period-finders

FIXME: add an iterative peak-removal and refit mode to all period-finders here.

'''

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


#############
## IMPORTS ##
#############

import numpy as np


#######################
## UTILITY FUNCTIONS ##
#######################

def resort_by_time(times, mags, errs):
    '''
    Resorts the input arrays so they're in time order.

    NOTE: the input arrays must not have nans in them.

    Parameters
    ----------

    times,mags,errs : np.arrays
        The times, mags, and errs arrays to resort by time. The times array is
        assumed to be the first one in the input args.

    Returns
    -------

    times,mags,errs : np.arrays
        The resorted times, mags, errs arrays.

    '''

    sort_order = np.argsort(times)
    times, mags, errs = times[sort_order], mags[sort_order], errs[sort_order]

    return times, mags, errs


def independent_freq_count(frequencies, times, conservative=True):
    '''This estimates the number of independent frequencies in a periodogram.

    This follows the terminology on page 3 of Zechmeister & Kurster (2009)::

        M = DELTA_f / delta_f

    where::

        DELTA_f = freq.max() - freq.min()
        delta_f = 1.0/(times.max() - times.min())

    Parameters
    ----------

    frequencies : np.array
        The frequencies array used for the calculation of the GLS periodogram.

    times : np.array
        The array of input times used for the calculation of the GLS
        periodogram.

    conservative : bool
        If True, will follow the prescription given in Schwarzenberg-Czerny
        (2003):

        http://adsabs.harvard.edu/abs/2003ASPC..292..383S

        and estimate the number of independent frequences as::

            min(N_obs, N_freq, DELTA_f/delta_f)

    Returns
    -------

    M : int
        The number of independent frequencies.

    '''

    M = frequencies.ptp()*times.ptp()

    if conservative:
        M_eff = min([times.size, frequencies.size, M])
    else:
        M_eff = M

    return M_eff


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

    Parameters
    ----------

    times : np.array
        The times to use to generate the frequency grid over.

    samplesperpeak : int
        The minimum sample coverage each frequency point in the grid will get.

    nyquistfactor : int
        The multiplier over the Nyquist rate to use.

    minfreq,maxfreq : float or None
        If not None, these will be the limits of the frequency grid generated.

    returnf0dfnf : bool
        If this is True, will return the values of `f0`, `df`, and `Nf`
        generated for this grid.

    Returns
    -------

    np.array
        A grid of frequencies.

    '''

    baseline = times.max() - times.min()
    nsamples = times.size

    df = 1. / baseline / samplesperpeak

    if minfreq is not None:
        f0 = minfreq
    else:
        f0 = 0.5 * df

    if maxfreq is not None:
        Nf = int(np.ceil((maxfreq - f0) / df))
    else:
        Nf = int(0.5 * samplesperpeak * nyquistfactor * nsamples)

    if returnf0dfnf:
        return f0, df, Nf, f0 + df * np.arange(Nf)
    else:
        return f0 + df * np.arange(Nf)


############################################
## FUNCTIONS FOR COMPARING PERIOD-FINDERS ##
############################################

def make_combined_periodogram(pflist, outfile, addmethods=False):
    '''This just puts all of the period-finders on a single periodogram.

    This will renormalize all of the periodograms so their values lie between 0
    and 1, with values lying closer to 1 being more significant. Periodograms
    that give the same best periods will have their peaks line up together.

    Parameters
    ----------

    pflist : list of dict
        This is a list of result dicts from any of the period-finders in
        periodbase. To use your own period-finders' results here, make sure the
        result dict is of the form and has at least the keys below::

            {'periods': np.array of all periods searched by the period-finder,
             'lspvals': np.array of periodogram power value for each period,
             'bestperiod': a float value that is the period with the highest
                           peak in the periodogram, i.e. the most-likely actual
                           period,
             'method': a three-letter code naming the period-finder used; must
                       be one of the keys in the
                       `astrobase.periodbase.METHODLABELS` dict,
             'nbestperiods': a list of the periods corresponding to periodogram
                             peaks (`nbestlspvals` below) to annotate on the
                             periodogram plot so they can be called out
                             visually,
             'nbestlspvals': a list of the power values associated with
                             periodogram peaks to annotate on the periodogram
                             plot so they can be called out visually; should be
                             the same length as `nbestperiods` above,
             'kwargs': dict of kwargs passed to your own period-finder function}

    outfile : str
        This is the output file to write the output to. NOTE: EPS/PS won't work
        because we use alpha transparency to better distinguish between the
        various periodograms.

    addmethods : bool
        If this is True, will add all of the normalized periodograms together,
        then renormalize them to between 0 and 1. In this way, if all of the
        period-finders agree on something, it'll stand out easily. FIXME:
        implement this kwarg.

    Returns
    -------

    str
        The name of the generated plot file.

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
