#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# falsealarm.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Apr 2019
# License: MIT - see the LICENSE file for the full text.

'''This contains functions useful for false-alarm probability calculation.

- :py:func:`.bootstrap_falsealarmprob`: calculates the false alarm probability
  for a period using bootstrap resampling.

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
import numpy.random as npr
npr.seed(0xdecaff)

from ..lcmath import sigclip_magseries


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

    from astrobase.periodbase import LSPMETHODS

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
                tindex = npr.randint(0,
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
