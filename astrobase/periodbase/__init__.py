#!/usr/bin/env python
# -*- coding: utf-8 -*-
# periodbase - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017

'''Contains various useful tools for period finding.

- :py:mod:`astrobase.periodbase.spdm`: Stellingwerf (1978) phase-dispersion
  minimization.

- :py:mod:`astrobase.periodbase.saov`: Schwarzenberg-Czerny (1989) analysis of
  variance.

- :py:mod:`astrobase.periodbase.smav`: Schwarzenberg-Czerny (1996)
  multi-harmonic AoV period search.

- :py:mod:`astrobase.periodbase.zgls`: Zechmeister & Kurster (2009) generalized
  Lomb-Scargle.

- :py:mod:`astrobase.periodbase.kbls`: Kovacs et al. (2002) Box-Least-Squares
  search using a wrapped `eebls.f` from G. Kovacs.

- :py:mod:`astrobase.periodbase.abls`: -> Kovacs et al. (2002) BLS using
  Astropy's implementation.

- :py:mod:`astrobase.periodbase.htls`: Hippke & Heller (2019) BLS, but with a
  nicer template.

- :py:mod:`astrobase.periodbase.macf`: -> McQuillan et al. (2013a, 2014) ACF
  period search.

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



def independent_freq_count(frequencies, times, conservative=True):
    '''This estimates M: the number of independent frequencies in the periodogram.

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



####################################################
## HOIST THE FINDER FUNCTIONS INTO THIS NAMESPACE ##
####################################################

from .zgls import pgen_lsp, specwindow_lsp
from .spdm import stellingwerf_pdm
from .saov import aov_periodfind
from .smav import aovhm_periodfind
from .macf import macf_period_find
from .kbls import bls_serial_pfind, bls_parallel_pfind

try:
    from .tls import tls_parallel_pfind
    HAVE_TLS = True
except Exception as e:
    HAVE_TLS = False

# used to figure out which function to run for bootstrap resampling
LSPMETHODS = {
    'bls':bls_parallel_pfind,
    'gls':pgen_lsp,
    'aov':aov_periodfind,
    'mav':aovhm_periodfind,
    'pdm':stellingwerf_pdm,
    'acf':macf_period_find,
    'win':specwindow_lsp
}
if HAVE_TLS:
    LSPMETHODS['tls'] = tls_parallel_pfind


# check if we have the astropy implementation of BLS available
import astropy
apversion = astropy.__version__
apversion = apversion.split('.')
apversion = [int(x) for x in apversion]

if len(apversion) == 2:
    apversion.append(0)

apversion = tuple(apversion)

if apversion >= (3,1,0):

    LOGINFO('An Astropy implementation of BLS is '
            'available because Astropy >= 3.1.')
    LOGINFO('If you want to use it as the default periodbase BLS runner, '
            'call the periodbase.use_astropy_bls() function.')

    def use_astropy_bls():
        '''This function can be used to switch from the default astrobase BLS
        implementation (kbls) to the Astropy version (abls).

        If this is called, subsequent calls to the BLS periodbase functions will
        use the Astropy versions instead::

            from astrobase import periodbase

            # initially points to periodbase.kbls.bls_serial_pfind
            periodbase.bls_serial_pfind(...)

            # initially points to periodbase.kbls.bls_parallel_pfind
            periodbase.bls_parallel_pfind(...)

            periodbase.use_astropy_bls()

            # now points to periodbase.abls.bls_serial_pfind
            periodbase.bls_serial_pfind(...)

            # now points to periodbase.abls.bls_parallel_pfind
            periodbase.bls_parallel_pfind(...)

        '''
        from .abls import bls_serial_pfind, bls_parallel_pfind
        globals()['bls_serial_pfind'] = bls_serial_pfind
        globals()['bls_parallel_pfind'] = bls_parallel_pfind
        globals()['LSPMETHODS']['bls'] = bls_parallel_pfind



#############################################################
## FUNCTIONS FOR TESTING SIGNIFICANCE OF PERIODOGRAM PEAKS ##
#############################################################

def bootstrap_falsealarmprob(lspinfo,
                             times,
                             mags,
                             errs,
                             nbootstrap=250,
                             magsarefluxes=False,
                             sigclip=10.0,
                             npeaks=None):
    '''Calculates the false alarm probabilities of periodogram peaks using
    bootstrap resampling of the magnitude time series.

    The false alarm probability here is defined as::

        (1.0 + sum(trialbestpeaks[i] > peak[j]))/(ntrialbestpeaks + 1)

    for each best periodogram peak j. The index i is for each bootstrap
    trial. This effectively gives us a significance for the peak. Smaller FAP
    means a better chance that the peak is real.

    The basic idea is to get the number of trial best peaks that are larger than
    the current best peak and divide this by the total number of trials. The
    distribution of these trial best peaks is obtained after scrambling the mag
    values and rerunning the specified periodogram method for a bunch of trials.

    `lspinfo` is the output dict from a periodbase periodogram function and MUST
    contain a 'method' key that corresponds to one of the keys in the LSPMETHODS
    dict above. This will let this function know which periodogram function to
    run to generate the bootstrap samples. The lspinfo SHOULD also have a
    'kwargs' key that corresponds to the input keyword arguments for the
    periodogram function as it was run originally, to keep everything the same
    during the bootstrap runs. If this is missing, default values will be used.

    FIXME: this may not be strictly correct; must look more into bootstrap
    significance testing. Also look into if we're doing resampling correctly for
    time series because the samples are not iid. Look into moving block
    bootstrap.

    Parameters
    ----------

    lspinfo : dict
        A dict of period-finder results from one of the period-finders in
        periodbase, or your own functions, provided it's of the form and
        contains at least the keys listed below::

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

        If you provide your own function's period-finder results, you should add
        a corresponding key for it to the LSPMETHODS dict above so the bootstrap
        function can use it correctly. Your period-finder function should take
        `times`, `mags`, errs and any extra parameters as kwargs and return a
        dict of the form described above. A small worked example::

            from your_module import your_periodfinder_func
            from astrobase import periodbase

            periodbase.LSPMETHODS['your-finder'] = your_periodfinder_func

            # run a period-finder session
            your_pfresults = your_periodfinder_func(times, mags, errs,
                                                    **extra_kwargs)

            # run bootstrap to find FAP
            falsealarm_info = periodbase.bootstrap_falsealarmprob(
                your_pfresults,
                times, mags, errs,
                nbootstrap=250,
                magsarefluxes=False,
            )

    times,mags,errs : np.arrays
        The magnitude/flux time-series to process along with their associated
        measurement errors.

    nbootstrap : int
        The total number of bootstrap trials to run. This is set to 250 by
        default, but should probably be around 1000 for realistic results.

    magsarefluxes : bool
        If True, indicates the input time-series is fluxes and not mags.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    npeaks : int or None
        The number of peaks from the list of 'nbestlspvals' in the period-finder
        result dict to run the bootstrap for. If None, all of the peaks in this
        list will have their FAP calculated.

    Returns
    -------

    dict
        Returns a dict of the form::

            {'peaks':allpeaks,
             'periods':allperiods,
             'probabilities':allfaps,
             'alltrialbestpeaks':alltrialbestpeaks}

    '''

    # figure out how many periods to work on
    if (npeaks and (0 < npeaks < len(lspinfo['nbestperiods']))):
        nperiods = npeaks
    else:
        LOGWARNING('npeaks not specified or invalid, '
                   'getting FAP for all %s periodogram peaks' %
                   len(lspinfo['nbestperiods']))
        nperiods = len(lspinfo['nbestperiods'])

    nbestperiods = lspinfo['nbestperiods'][:nperiods]
    nbestpeaks = lspinfo['nbestlspvals'][:nperiods]

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

            for _trial in range(nbootstrap):

                # get a scrambled index
                tindex = np.random.randint(0,
                                           high=mags.size,
                                           size=mags.size)


                # get the kwargs dict out of the lspinfo
                if 'kwargs' in lspinfo:

                    kwargs = lspinfo['kwargs']

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
                lspres = LSPMETHODS[lspinfo['method']](
                    times, mags[tindex], errs[tindex],
                    **kwargs
                )
                trialbestpeaks.append(lspres['bestlspval'])

            trialbestpeaks = np.array(trialbestpeaks)
            alltrialbestpeaks.append(trialbestpeaks)

            # calculate the FAP for a trial peak j = FAP[j] =
            # (1.0 + sum(trialbestpeaks[i] > peak[j]))/(ntrialbestpeaks + 1)
            if lspinfo['method'] != 'pdm':
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
