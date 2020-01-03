#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sinusoidal.py
# Waqas Bhatti and Luke Bouma - Feb 2017
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''Light curve fitting routines for sinusoidal models:

- :py:func:`astrobase.lcfit.sinusoidal.fourier_fit_magseries`: fit an arbitrary
  order Fourier series to a magnitude/flux time series.

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

from functools import partial

from numpy import (
    nan as npnan, sum as npsum, median as npmedian, max as npmax,
    min as npmin, pi as pi_value, cos as npcos, where as npwhere,
    nonzero as npnonzero, array as nparray, concatenate as npconcatenate,
    diag as npdiag, sqrt as npsqrt, inf as npinf
)

from scipy.optimize import (
    minimize as spminimize,
    curve_fit
)

from ..lcmath import sigclip_magseries
from ..lcmodels import sinusoidal

from .utils import get_phased_quantities, make_fit_plot


#####################################################
## FOURIER FITTING TO PHASED MAGNITUDE TIME SERIES ##
#####################################################

def _fourier_func(fourierparams, phase, mags):
    '''This returns a summed Fourier cosine series.

    Parameters
    ----------

    fourierparams : list
        This MUST be a list of the following form like so::

            [period,
             epoch,
             [amplitude_1, amplitude_2, amplitude_3, ..., amplitude_X],
             [phase_1, phase_2, phase_3, ..., phase_X]]

        where X is the Fourier order.

    phase,mags : np.array
        The input phase and magnitude areas to use as the basis for the cosine
        series. The phases are used directly to generate the values of the
        function, while the mags array is used to generate the zeroth order
        amplitude coefficient.

    Returns
    -------

    np.array
        The Fourier cosine series function evaluated over `phase`.

    '''

    # figure out the order from the length of the Fourier param list
    order = int(len(fourierparams)/2)

    # get the amplitude and phase coefficients
    f_amp = fourierparams[:order]
    f_pha = fourierparams[order:]

    # calculate all the individual terms of the series
    f_orders = [f_amp[x]*npcos(2.0*pi_value*x*phase + f_pha[x])
                for x in range(order)]

    # this is the zeroth order coefficient - a constant equal to median mag
    total_f = npmedian(mags)

    # sum the series
    for fo in f_orders:
        total_f += fo

    return total_f


def _fourier_chisq(fourierparams,
                   phase,
                   mags,
                   errs):
    '''This is the chisq objective function to be minimized by `scipy.minimize`.

    The parameters are the same as `_fourier_func` above. `errs` is used to
    calculate the chisq value.

    '''

    f = _fourier_func(fourierparams, phase, mags)
    chisq = npsum(((mags - f)*(mags - f))/(errs*errs))

    return chisq


def fourier_fit_magseries(
        times, mags, errs, period,
        fourierorder=None,
        fourierparams=None,
        fix_period=True,
        scale_errs_redchisq_unity=True,
        sigclip=3.0,
        magsarefluxes=False,
        plotfit=False,
        ignoreinitfail=True,
        verbose=True,
        curve_fit_kwargs=None,
):
    '''This fits a Fourier series to a mag/flux time series.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to fit a Fourier cosine series to.

    period : float
        The period to use for the Fourier fit.

    fourierorder : None or int
        If this is an int, will be interpreted as the Fourier order of the
        series to fit to the input mag/flux times-series. If this is None and
        `fourierparams` is specified, `fourierparams` will be used directly to
        generate the fit Fourier series. If `fourierparams` is also None, this
        function will try to fit a Fourier cosine series of order 3 to the
        mag/flux time-series.

    fourierparams : list of floats or None
        If this is specified as a list of floats, it must be of the form below::

            [fourier_amp1, fourier_amp2, fourier_amp3,...,fourier_ampN,
             fourier_phase1, fourier_phase2, fourier_phase3,...,fourier_phaseN]

        to specify a Fourier cosine series of order N. If this is None and
        `fourierorder` is specified, the Fourier order specified there will be
        used to construct the Fourier cosine series used to fit the input
        mag/flux time-series. If both are None, this function will try to fit a
        Fourier cosine series of order 3 to the input mag/flux time-series.

    fix_period : bool
        If True, will fix the period with fitting the sinusoidal function to the
        phased light curve.

    scale_errs_redchisq_unity : bool
        If True, the standard errors on the fit parameters will be scaled to
        make the reduced chi-sq = 1.0. This sets the ``absolute_sigma`` kwarg
        for the ``scipy.optimize.curve_fit`` function to False.

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

    curve_fit_kwargs : dict or None
        If not None, this should be a dict containing extra kwargs to pass to
        the scipy.optimize.curve_fit function.

    Returns
    -------

    dict
        This function returns a dict containing the model fit parameters, the
        minimized chi-sq value and the reduced chi-sq value. The form of this
        dict is mostly standardized across all functions in this module::

            {
                'fittype':'fourier',
                'fitinfo':{
                    'finalparams': the list of final model fit params,
                    'finalparamerrs': list of errs for each model fit param,
                    'fitmags': the model fit mags,
                    'fitperiod': the fit period if this wasn't set to fixed,
                    'fitepoch': this is times.min() for this fit type,
                    'actual_fitepoch': time of minimum light from fit model
                    ... other fit function specific keys ...
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

        NOTE: the returned value of 'fitepoch' in the 'fitinfo' dict returned by
        this function is the time value of the first observation since this is
        where the LC is folded for the fit procedure. To get the actual time of
        minimum epoch as calculated by a spline fit to the phased LC, use the
        key 'actual_fitepoch' in the 'fitinfo' dict.

    '''

    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

    # get rid of zero errs
    nzind = npnonzero(serrs)
    stimes, smags, serrs = stimes[nzind], smags[nzind], serrs[nzind]

    phase, pmags, perrs, ptimes, mintime = (
        get_phased_quantities(stimes, smags, serrs, period)
    )

    # get the fourier order either from the scalar order kwarg...
    if fourierorder and fourierorder > 0 and not fourierparams:

        fourieramps = [0.6] + [0.2]*(fourierorder - 1)
        fourierphas = [0.1] + [0.1]*(fourierorder - 1)
        fourierparams = fourieramps + fourierphas

    # or from the fully specified coeffs vector
    elif not fourierorder and fourierparams:

        fourierorder = int(len(fourierparams)/2)

    else:
        LOGWARNING('specified both/neither Fourier order AND Fourier coeffs, '
                   'using default Fourier order of 3')
        fourierorder = 3
        fourieramps = [0.6] + [0.2]*(fourierorder - 1)
        fourierphas = [0.1] + [0.1]*(fourierorder - 1)
        fourierparams = fourieramps + fourierphas

    if verbose:
        LOGINFO('fitting Fourier series of order %s to '
                'mag series with %s observations, '
                'using period %.6f, folded at %.6f' % (fourierorder,
                                                       len(phase),
                                                       period,
                                                       mintime))

    # initial minimize call to find global minimum in chi-sq
    initialfit = spminimize(_fourier_chisq,
                            fourierparams,
                            args=(phase, pmags, perrs))

    # make sure this initial fit succeeds before proceeding
    if initialfit.success or ignoreinitfail:

        if verbose:
            LOGINFO('initial fit done, refining...')

        leastsqparams = initialfit.x

        try:

            curvefit_params = npconcatenate((
                nparray([period]),
                leastsqparams
            ))

            # set up the bounds for the fit parameters
            if fix_period:
                curvefit_bounds = (
                    [period - 1.0e-7] +
                    [-npinf]*fourierorder +
                    [-npinf]*fourierorder,
                    [period + 1.0e-7] +
                    [npinf]*fourierorder +
                    [npinf]*fourierorder
                )
            else:
                curvefit_bounds = (
                    [0.0] +
                    [-npinf]*fourierorder +
                    [-npinf]*fourierorder,
                    [npinf] +
                    [npinf]*fourierorder +
                    [npinf]*fourierorder
                )

            curvefit_func = partial(
                sinusoidal.fourier_curvefit_func,
                zerolevel=npmedian(smags),
                epoch=mintime,
                fixed_period=period if fix_period else None,
            )

            if curve_fit_kwargs is not None:

                finalparams, covmatrix = curve_fit(
                    curvefit_func,
                    stimes, smags,
                    p0=curvefit_params,
                    sigma=serrs,
                    bounds=curvefit_bounds,
                    absolute_sigma=(not scale_errs_redchisq_unity),
                    **curve_fit_kwargs
                )

            else:

                finalparams, covmatrix = curve_fit(
                    curvefit_func,
                    stimes, smags,
                    p0=curvefit_params,
                    sigma=serrs,
                    bounds=curvefit_bounds,
                    absolute_sigma=(not scale_errs_redchisq_unity),
                )

        except Exception:
            LOGEXCEPTION("curve_fit returned an exception")
            finalparams, covmatrix = None, None

        # if the fit succeeded, then we can return the final parameters
        if finalparams is not None and covmatrix is not None:

            # this is the fit period
            fperiod = finalparams[0]

            phase, pmags, perrs, ptimes, mintime = (
                get_phased_quantities(stimes, smags, serrs, fperiod)
            )

            # calculate the chisq and reduced chisq
            fitmags = _fourier_func(finalparams[1:], phase, pmags)

            fitchisq = npsum(
                ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
            )

            n_free_params = len(pmags) - len(finalparams)
            if fix_period:
                n_free_params -= 1

            fitredchisq = fitchisq/n_free_params
            stderrs = npsqrt(npdiag(covmatrix))

            if verbose:
                LOGINFO(
                    'final fit done. chisq = %.5f, reduced chisq = %.5f' %
                    (fitchisq,fitredchisq)
                )

            # figure out the time of light curve minimum (i.e. the fit epoch)
            # this is when the fit mag is maximum (i.e. the faintest)
            # or if magsarefluxes = True, then this is when fit flux is minimum
            if not magsarefluxes:
                fitmagminind = npwhere(fitmags == npmax(fitmags))
            else:
                fitmagminind = npwhere(fitmags == npmin(fitmags))
            if len(fitmagminind[0]) > 1:
                fitmagminind = (fitmagminind[0][0],)

            # assemble the returndict
            returndict = {
                'fittype':'fourier',
                'fitinfo':{
                    'fourierorder':fourierorder,
                    # return coeffs only for backwards compatibility with
                    # existing functions that use the returned value of
                    # fourier_fit_magseries
                    'finalparams':finalparams[1:],
                    'finalparamerrs':stderrs,
                    'initialfit':initialfit,
                    'fitmags':fitmags,
                    'fitperiod':finalparams[0],
                    # the 'fitepoch' is just the minimum time here
                    'fitepoch':mintime,
                    # the actual fit epoch is calculated as the time of minimum
                    # light OF the fit model light curve
                    'actual_fitepoch':ptimes[fitmagminind]
                },
                'fitchisq':fitchisq,
                'fitredchisq':fitredchisq,
                'fitplotfile':None,
                'magseries':{
                    'times':ptimes,
                    'phase':phase,
                    'mags':pmags,
                    'errs':perrs,
                    'magsarefluxes':magsarefluxes
                },
            }

            # make the fit plot if required
            if plotfit and isinstance(plotfit, str):

                make_fit_plot(phase, pmags, perrs, fitmags,
                              fperiod, mintime, mintime,
                              plotfit,
                              magsarefluxes=magsarefluxes)

                returndict['fitplotfile'] = plotfit

            return returndict

        # if the leastsq fit did not succeed, return Nothing
        else:
            LOGERROR('fourier-fit: least-squared fit to the light curve failed')
            return {
                'fittype':'fourier',
                'fitinfo':{
                    'fourierorder':fourierorder,
                    'finalparams':None,
                    'finalparamerrs':None,
                    'initialfit':initialfit,
                    'fitmags':None,
                    'fitperiod':None,
                    'fitepoch':None,
                    'actual_fitepoch':None,
                },
                'fitchisq':npnan,
                'fitredchisq':npnan,
                'fitplotfile':None,
                'magseries':{
                    'times':ptimes,
                    'phase':phase,
                    'mags':pmags,
                    'errs':perrs,
                    'magsarefluxes':magsarefluxes
                }
            }

    # if the fit didn't succeed, we can't proceed
    else:

        LOGERROR('initial Fourier fit did not succeed, '
                 'reason: %s, returning scipy OptimizeResult'
                 % initialfit.message)

        return {
            'fittype':'fourier',
            'fitinfo':{
                'fourierorder':fourierorder,
                'finalparams':None,
                'finalparamerrs':None,
                'initialfit':initialfit,
                'fitmags':None,
                'fitperiod':None,
                'fitepoch':None,
                'actual_fitepoch':None,
            },
            'fitchisq':npnan,
            'fitredchisq':npnan,
            'fitplotfile':None,
            'magseries':{
                'times':ptimes,
                'phase':phase,
                'mags':pmags,
                'errs':perrs,
                'magsarefluxes':magsarefluxes
            }
        }
