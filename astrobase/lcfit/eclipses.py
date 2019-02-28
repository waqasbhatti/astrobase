#!/usr/bin/env python
# -*- coding: utf-8 -*-
# eclipses.py
# Waqas Bhatti and Luke Bouma - Feb 2017
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''Light curve fitting routines for eclipsing binaries:

- :py:func:`astrobase.lcfit.eclipses.gaussianeb_fit_magseries`: fit a double
  inverted gaussian eclipsing binary model to the magnitude/flux time series

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

from numpy import (
    nan as npnan, sum as npsum, sqrt as npsqrt,
    nonzero as npnonzero, diag as npdiag
)

from scipy.optimize import leastsq as spleastsq

from ..lcmath import sigclip_magseries
from ..lcmodels import eclipses

from .utils import make_fit_plot
from .nonphysical import spline_fit_magseries, savgol_fit_magseries


############################################
## DOUBLE INVERTED GAUSSIAN ECLIPSE MODEL ##
############################################

def gaussianeb_fit_magseries(times, mags, errs,
                             ebparams,
                             sigclip=10.0,
                             plotfit=False,
                             magsarefluxes=False,
                             verbose=True):
    '''This fits a double inverted gaussian EB model to a magnitude time series.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to fit the EB model to.

    period : float
        The period to use for EB fit.

    ebparams : list of float
        This is a list containing the eclipsing binary parameters::

            ebparams = [period (time),
                        epoch (time),
                        pdepth (mags),
                        pduration (phase),
                        psdepthratio,
                        secondaryphase]

        `period` is the period in days.

        `epoch` is the time of primary minimum in JD.

        `pdepth` is the depth of the primary eclipse:

        - for magnitudes -> `pdepth` should be < 0
        - for fluxes     -> `pdepth` should be > 0

        `pduration` is the length of the primary eclipse in phase.

        `psdepthratio` is the ratio of the secondary eclipse depth to that of
        the primary eclipse.

        `secondaryphase` is the phase at which the minimum of the secondary
        eclipse is located. This effectively parameterizes eccentricity.

        If `epoch` is None, this function will do an initial spline fit to find
        an approximate minimum of the phased light curve using the given period.

        The `pdepth` provided is checked against the value of
        `magsarefluxes`. if `magsarefluxes = True`, the `ebdepth` is forced to
        be > 0; if `magsarefluxes = False`, the `ebdepth` is forced to be < 0.

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

    magsarefluxes : bool
        If True, will treat the input values of `mags` as fluxes for purposes of
        plotting the fit and sig-clipping.

    plotfit : str or False
        If this is a string, this function will make a plot for the fit to the
        mag/flux time-series and writes the plot to the path specified here.

    ignoreinitfail : bool
        If this is True, ignores the initial failure to find a set of optimized
        Fourier parameters using the global optimization function and proceeds
        to do a least-squares fit anyway.

    verbose : bool
        If True, will indicate progress and warn of any problems.

    Returns
    -------

    dict
        This function returns a dict containing the model fit parameters, the
        minimized chi-sq value and the reduced chi-sq value. The form of this
        dict is mostly standardized across all functions in this module::

            {
                'fittype':'gaussianeb',
                'fitinfo':{
                    'initialparams':the initial EB params provided,
                    'finalparams':the final model fit EB params,
                    'finalparamerrs':formal errors in the params,
                    'leastsqfit':the full tuple returned by scipy.leastsq,
                    'fitmags': the model fit mags,
                    'fitepoch': the epoch of minimum light for the fit,
                },
                'fitchisq': the minimized value of the fit's chi-sq,
                'fitredchisq':the reduced chi-sq value,
                'fitplotfile': the output fit plot if fitplot is not None,
                'magseries':{
                    'times':input times in phase order of the model,
                    'phase':the phases of the model mags,
                    'mags':input mags/fluxes in the phase order of the model,
                    'errs':errs in the phase order of the model,
                    'magsarefluxes':input value of magsarefluxes kwarg
                }
            }


    '''

    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

    # get rid of zero errs
    nzind = npnonzero(serrs)
    stimes, smags, serrs = stimes[nzind], smags[nzind], serrs[nzind]


    # check the ebparams
    ebperiod, ebepoch, ebdepth = ebparams[0:3]

    # check if we have a ebepoch to use
    if ebepoch is None:

        if verbose:
            LOGWARNING('no ebepoch given in ebparams, '
                       'trying to figure it out automatically...')
        # do a spline fit to figure out the approximate min of the LC
        try:
            spfit = spline_fit_magseries(times, mags, errs, ebperiod,
                                         sigclip=sigclip,
                                         magsarefluxes=magsarefluxes,
                                         verbose=verbose)
            ebepoch = spfit['fitinfo']['fitepoch']

        # if the spline-fit fails, try a savgol fit instead
        except Exception as e:
            sgfit = savgol_fit_magseries(times, mags, errs, ebperiod,
                                         sigclip=sigclip,
                                         magsarefluxes=magsarefluxes,
                                         verbose=verbose)
            ebepoch = sgfit['fitinfo']['fitepoch']

        # if everything failed, then bail out and ask for the ebepoch
        finally:

            if ebepoch is None:
                LOGERROR("couldn't automatically figure out the eb epoch, "
                         "can't continue. please provide it in ebparams.")

                # assemble the returndict
                returndict = {
                    'fittype':'gaussianeb',
                    'fitinfo':{
                        'initialparams':ebparams,
                        'finalparams':None,
                        'leastsqfit':None,
                        'fitmags':None,
                        'fitepoch':None,
                    },
                    'fitchisq':npnan,
                    'fitredchisq':npnan,
                    'fitplotfile':None,
                    'magseries':{
                        'phase':None,
                        'times':None,
                        'mags':None,
                        'errs':None,
                        'magsarefluxes':magsarefluxes,
                    },
                }

                return returndict

            else:

                if ebepoch.size > 1:
                    if verbose:
                        LOGWARNING('could not auto-find a single minimum '
                                   'for ebepoch, using the first one returned')
                    ebparams[1] = ebepoch[0]

                else:

                    if verbose:
                        LOGWARNING(
                            'using automatically determined ebepoch = %.5f'
                            % ebepoch
                        )
                    ebparams[1] = ebepoch.item()

    # next, check the ebdepth and fix it to the form required
    if magsarefluxes:
        if ebdepth < 0.0:
            ebparams[2] = -ebdepth[2]

    else:
        if ebdepth > 0.0:
            ebparams[2] = -ebdepth[2]

    # finally, do the fit
    try:
        leastsqfit = spleastsq(eclipses.invgauss_eclipses_residual,
                               ebparams,
                               args=(stimes, smags, serrs),
                               full_output=True)
    except Exception as e:
        leastsqfit = None

    # if the fit succeeded, then we can return the final parameters
    if leastsqfit and leastsqfit[-1] in (1,2,3,4):

        finalparams = leastsqfit[0]
        covxmatrix = leastsqfit[1]

        # calculate the chisq and reduced chisq
        fitmags, phase, ptimes, pmags, perrs = eclipses.invgauss_eclipses_func(
            finalparams,
            stimes, smags, serrs
        )
        fitchisq = npsum(
            ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
        )
        fitredchisq = fitchisq/(len(pmags) - len(finalparams) - 1)

        # get the residual variance and calculate the formal 1-sigma errs on the
        # final parameters
        residuals = leastsqfit[2]['fvec']
        residualvariance = (
            npsum(residuals*residuals)/(pmags.size - finalparams.size)
        )
        if covxmatrix is not None:
            covmatrix = residualvariance*covxmatrix
            stderrs = npsqrt(npdiag(covmatrix))
        else:
            LOGERROR('covxmatrix not available, fit probably failed!')
            stderrs = None

        if verbose:
            LOGINFO(
                'final fit done. chisq = %.5f, reduced chisq = %.5f' %
                (fitchisq, fitredchisq)
            )

        # get the fit epoch
        fperiod, fepoch = finalparams[:2]

        # assemble the returndict
        returndict = {
            'fittype':'gaussianeb',
            'fitinfo':{
                'initialparams':ebparams,
                'finalparams':finalparams,
                'finalparamerrs':stderrs,
                'leastsqfit':leastsqfit,
                'fitmags':fitmags,
                'fitepoch':fepoch,
            },
            'fitchisq':fitchisq,
            'fitredchisq':fitredchisq,
            'fitplotfile':None,
            'magseries':{
                'phase':phase,
                'times':ptimes,
                'mags':pmags,
                'errs':perrs,
                'magsarefluxes':magsarefluxes,
            },
        }

        # make the fit plot if required
        if plotfit and isinstance(plotfit, str):

            make_fit_plot(phase, pmags, perrs, fitmags,
                          fperiod, ptimes.min(), fepoch,
                          plotfit,
                          magsarefluxes=magsarefluxes)

            returndict['fitplotfile'] = plotfit

        return returndict

    # if the leastsq fit failed, return nothing
    else:

        LOGERROR('eb-fit: least-squared fit to the light curve failed!')

        # assemble the returndict
        returndict = {
            'fittype':'gaussianeb',
            'fitinfo':{
                'initialparams':ebparams,
                'finalparams':None,
                'finalparamerrs':None,
                'leastsqfit':leastsqfit,
                'fitmags':None,
                'fitepoch':None,
            },
            'fitchisq':npnan,
            'fitredchisq':npnan,
            'fitplotfile':None,
            'magseries':{
                'phase':None,
                'times':None,
                'mags':None,
                'errs':None,
                'magsarefluxes':magsarefluxes,
            },
        }

        return returndict
