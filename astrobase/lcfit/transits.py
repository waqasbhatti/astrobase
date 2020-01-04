#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# transits.py
# Waqas Bhatti and Luke Bouma - Feb 2017
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''Fitting routines for planetary transits:

- :py:func:`astrobase.lcfit.transits.traptransit_fit_magseries`: fit a
  trapezoid-shaped transit signal to the magnitude/flux time series

- :py:func:`astrobase.lcfit.transits.mandelagol_fit_magseries`: fit a Mandel &
  Agol (2002) planet transit model to the flux time series, fixing some
  parameters (e.g., eccentricity) and varying other parameters (e.g., t0,
  period, a/Rstar). Priors must be passed by user.

- :py:func:`astrobase.lcfit.transits.mandelagol_and_line_fit_magseries`: fit a
  Mandel & Agol 2002 model, + a local line to the flux time series. Priors must
  be passed by user.

- :py:func:`astrobase.lcfit.transits.fivetransitparam_fit_magseries`: fit out a
  line around each transit window in the given light curve, and then fit all the
  transits in the light curve for t0, period, a/Rstar, Rp/Rstar, and
  inclination. Fixes e to 0, and uses theoretical quadratic limb darkening
  coefficients in the bandpass given by the user. Figures out the priors, user
  only needs to pass stellar parameters instead.
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

import os.path
import os
import pickle
from functools import partial

import numpy as np
from scipy.optimize import minimize as spminimize, curve_fit

from astropy import units as units
from numpy.polynomial.legendre import Legendre

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import batman
    import emcee
    import corner
    import h5py

    if int(emcee.__version__[0]) >= 3:
        mandel_agol_dependencies = True
    else:
        mandel_agol_dependencies = False

except Exception:
    mandel_agol_dependencies = False

try:
    from astroquery.vizier import Vizier
    vizier_dependency = True
    from astrobase.services.limbdarkening import get_tess_limb_darkening_guesses
except Exception:
    vizier_dependency = False

from ..lcmodels import transits
from ..lcmath import sigclip_magseries
from .utils import make_fit_plot
from .nonphysical import savgol_fit_magseries, spline_fit_magseries


###############################################
## TRAPEZOID TRANSIT MODEL FIT TO MAG SERIES ##
###############################################

def traptransit_fit_magseries(
        times, mags, errs,
        transitparams,
        param_bounds=None,
        scale_errs_redchisq_unity=True,
        sigclip=10.0,
        plotfit=False,
        magsarefluxes=False,
        verbose=True,
        curve_fit_kwargs=None,
):
    '''This fits a trapezoid transit model to a magnitude time series.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to fit a trapezoid planet-transit model
        to.

    period : float
        The period to use for the model fit.

    transitparams : list of floats
        These are initial parameters for the transit model fit. A list of the
        following form is required::

            transitparams = [transit_period (time),
                             transit_epoch (time),
                             transit_depth (flux or mags),
                             transit_duration (phase),
                             ingress_duration (phase)]

        - for magnitudes -> `transit_depth` should be < 0
        - for fluxes     -> `transit_depth` should be > 0

        If `transitepoch` is None, this function will do an initial spline fit
        to find an approximate minimum of the phased light curve using the given
        period.

        The `transitdepth` provided is checked against the value of
        `magsarefluxes`. if `magsarefluxes = True`, the `transitdepth` is forced
        to be > 0; if `magsarefluxes` = False, the `transitdepth` is forced to
        be < 0.

    param_bounds : dict or None
        This is a dict of the upper and lower bounds on each fit
        parameter. Should be of the form::

            {'period':          (lower_bound_period, upper_bound_period),
             'epoch':           (lower_bound_epoch, upper_bound_epoch),
             'depth':           (lower_bound_depth, upper_bound_depth),
             'duration':        (lower_bound_duration, upper_bound_duration),
             'ingressduration': (lower_bound_ingressduration,
                                 upper_bound_ingressduration)}

        - To indicate that a parameter is fixed, use 'fixed' instead of a tuple
          providing its lower and upper bounds as tuple.

        - To indicate that a parameter has no bounds, don't include it in the
          param_bounds dict.

        If this is None, the default value of this kwarg will be::

            {'period':(0.0,np.inf),       # period is between 0 and inf
             'epoch':(0.0, np.inf),       # epoch is between 0 and inf
             'depth':(-np.inf,np.inf),    # depth is between -np.inf and np.inf
             'duration':(0.0,1.0),        # duration is between 0.0 and 1.0
             'ingressduration':(0.0,0.5)} # ingress duration between 0.0 and 0.5

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
                'fittype':'traptransit',
                'fitinfo':{
                    'initialparams':the initial transit params provided,
                    'finalparams':the final model fit transit params ,
                    'finalparamerrs':formal errors in the params,
                    'fitmags': the model fit mags,
                    'fitepoch': the epoch of minimum light for the fit,
                    'ntransitpoints': the number of LC points in transit phase
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
    nzind = np.nonzero(serrs)
    stimes, smags, serrs = stimes[nzind], smags[nzind], serrs[nzind]

    # check the transitparams
    transitperiod, transitepoch, transitdepth = transitparams[0:3]

    # check if we have a transitepoch to use
    if transitepoch is None:

        if verbose:
            LOGWARNING('no transitepoch given in transitparams, '
                       'trying to figure it out automatically...')
        # do a spline fit to figure out the approximate min of the LC
        try:
            spfit = spline_fit_magseries(times, mags, errs, transitperiod,
                                         sigclip=sigclip,
                                         magsarefluxes=magsarefluxes,
                                         verbose=verbose)
            transitepoch = spfit['fitinfo']['fitepoch']

        # if the spline-fit fails, try a savgol fit instead
        except Exception:
            sgfit = savgol_fit_magseries(times, mags, errs, transitperiod,
                                         sigclip=sigclip,
                                         magsarefluxes=magsarefluxes,
                                         verbose=verbose)
            transitepoch = sgfit['fitinfo']['fitepoch']

        # if everything failed, then bail out and ask for the transitepoch
        finally:

            if transitepoch is None:
                LOGERROR("couldn't automatically figure out the transit epoch, "
                         "can't continue. please provide it in transitparams.")

                # assemble the returndict
                returndict = {
                    'fittype':'traptransit',
                    'fitinfo':{
                        'initialparams':transitparams,
                        'finalparams':None,
                        'finalparamerrs':None,
                        'fitmags':None,
                        'fitepoch':None,
                    },
                    'fitchisq':np.nan,
                    'fitredchisq':np.nan,
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

                # check the case when there are more than one transitepochs
                # returned
                if transitepoch.size > 1:
                    if verbose:
                        LOGWARNING(
                            "could not auto-find a single minimum in LC for "
                            "transitepoch, using the first one returned"
                        )
                    transitparams[1] = transitepoch[0]

                else:

                    if verbose:
                        LOGWARNING(
                            'using automatically determined transitepoch = %.5f'
                            % transitepoch
                        )
                    transitparams[1] = transitepoch.item()

    # next, check the transitdepth and fix it to the form required
    if magsarefluxes:
        if transitdepth < 0.0:
            transitparams[2] = -transitdepth

    else:
        if transitdepth > 0.0:
            transitparams[2] = -transitdepth

    # finally, do the fit
    try:

        # set up the fit parameter bounds
        if param_bounds is None:

            curvefit_bounds = (
                np.array([0.0, 0.0, -np.inf, 0.0, 0.0]),
                np.array([np.inf, np.inf, np.inf, 1.0, 0.5])
            )
            fitfunc_fixed = {}

        else:

            # figure out the bounds
            lower_bounds = []
            upper_bounds = []
            fitfunc_fixed = {}

            for ind, key in enumerate(('period','epoch','depth',
                                       'duration','ingressduration')):

                # handle fixed parameters
                if (key in param_bounds and
                    isinstance(param_bounds[key], str) and
                    param_bounds[key] == 'fixed'):

                    lower_bounds.append(transitparams[ind]-1.0e-7)
                    upper_bounds.append(transitparams[ind]+1.0e-7)
                    fitfunc_fixed[key] = transitparams[ind]

                # handle parameters with lower and upper bounds
                elif key in param_bounds and isinstance(param_bounds[key],
                                                        (tuple,list)):

                    lower_bounds.append(param_bounds[key][0])
                    upper_bounds.append(param_bounds[key][1])

                # handle no parameter bounds
                else:

                    lower_bounds.append(-np.inf)
                    upper_bounds.append(np.inf)

            # generate the bounds sequence in the required format
            curvefit_bounds = (
                np.array(lower_bounds),
                np.array(upper_bounds)
            )

        #
        # set up the curve fit function
        #
        curvefit_func = partial(transits.trapezoid_transit_curvefit_func,
                                zerolevel=np.median(smags),
                                fixed_params=fitfunc_fixed)

        #
        # run the fit
        #
        if curve_fit_kwargs is not None:

            finalparams, covmatrix = curve_fit(
                curvefit_func,
                stimes, smags,
                p0=transitparams,
                sigma=serrs,
                bounds=curvefit_bounds,
                absolute_sigma=(not scale_errs_redchisq_unity),
                **curve_fit_kwargs
            )

        else:

            finalparams, covmatrix = curve_fit(
                curvefit_func,
                stimes, smags,
                p0=transitparams,
                sigma=serrs,
                absolute_sigma=(not scale_errs_redchisq_unity),
                bounds=curvefit_bounds,
            )

    except Exception:
        LOGEXCEPTION("curve_fit returned an exception")
        finalparams, covmatrix = None, None

    # if the fit succeeded, then we can return the final parameters
    if finalparams is not None and covmatrix is not None:

        # calculate the chisq and reduced chisq
        fitmags, phase, ptimes, pmags, perrs, n_transitpoints = (
            transits.trapezoid_transit_func(
                finalparams,
                stimes, smags, serrs,
                get_ntransitpoints=True
            )
        )
        fitchisq = np.sum(
            ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
        )
        fitredchisq = fitchisq/(len(pmags) -
                                len(finalparams) - len(fitfunc_fixed))

        stderrs = np.sqrt(np.diag(covmatrix))

        if verbose:
            LOGINFO(
                'final fit done. chisq = %.5f, reduced chisq = %.5f' %
                (fitchisq, fitredchisq)
            )

        # get the fit epoch
        fperiod, fepoch = finalparams[:2]

        # assemble the returndict
        returndict = {
            'fittype':'traptransit',
            'fitinfo':{
                'initialparams':transitparams,
                'finalparams':finalparams,
                'finalparamerrs':stderrs,
                'fitmags':fitmags,
                'fitepoch':fepoch,
                'ntransitpoints':n_transitpoints
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

        LOGERROR('trapezoid-fit: least-squared fit to the light curve failed!')

        # assemble the returndict
        returndict = {
            'fittype':'traptransit',
            'fitinfo':{
                'initialparams':transitparams,
                'finalparams':None,
                'finalparamerrs':None,
                'fitmags':None,
                'fitepoch':None,
                'ntransitpoints':0
            },
            'fitchisq':np.nan,
            'fitredchisq':np.nan,
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


###########################################################
# helper functions for interfacing between emcee & BATMAN #
###########################################################

def _get_value(quantitystr, fitparams, fixedparams):
    """This decides if a value is to be fit for or is fixed in a model fit.

    When you want to get the value of some parameter, but you're not sure if
    it's being fit or if it is fixed. then, e.g. for `period`::

        period_value = _get_value('period', fitparams, fixedparams)

    """

    # for Mandel-Agol fitting, sometimes we want to fix some parameters,
    # and fit others. this function allows that flexibility.
    fitparamskeys, fixedparamskeys = fitparams.keys(), fixedparams.keys()
    if quantitystr in fitparamskeys:
        quantity = fitparams[quantitystr]
    elif quantitystr in fixedparamskeys:
        quantity = fixedparams[quantitystr]
    return quantity


def _transit_model(times, t0, per, rp, a, inc, ecc, w, u, limb_dark,
                   exp_time_minutes=2, supersample_factor=7):
    '''This returns a BATMAN planetary transit model.

    Parameters
    ----------

    times : np.array
        The times at which the model will be evaluated.

    t0 : float
        The time of periastron for the transit.

    per : float
        The orbital period of the planet.

    rp : float
        The stellar radius of the planet's star (in Rsun).

    a : float
        The semi-major axis of the planet's orbit (in Rsun).

    inc : float
        The orbital inclination (in degrees).

    ecc : float
        The eccentricity of the orbit.

    w : float
        The longitude of periastron (in degrees).

    u : list of floats
        The limb darkening coefficients specific to the limb darkening model
        used.

    limb_dark : {"uniform", "linear", "quadratic", "square-root", "logarithmic", "exponential", "power2", "custom"}
        The type of limb darkening model to use. See the full list here:

        https://www.cfa.harvard.edu/~lkreidberg/batman/tutorial.html#limb-darkening-options

    exp_time_minutes : float
        The amount of time to 'smear' the transit LC points over to simulate a
        long exposure time.

    supersample_factor: int
        The number of supersampled time data points to average the lightcurve
        model over.

    Returns
    -------

    (params, batman_model) : tuple
        The returned tuple contains the params list and the generated
        `batman.TransitModel` object.

    '''
    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = t0                   # time of periastron
    params.per = per                 # orbital period
    params.rp = rp                   # planet radius (in stellar radii)
    params.a = a                     # semi-major axis (in stellar radii)
    params.inc = inc                 # orbital inclination (in degrees)
    params.ecc = ecc                 # the eccentricity of the orbit
    params.w = w                     # longitude of periastron (in degrees)
    params.u = u                     # limb darkening coefficient list
    params.limb_dark = limb_dark     # limb darkening model to use

    t = times
    m = batman.TransitModel(params, t, exp_time=exp_time_minutes/60./24.,
                            supersample_factor=supersample_factor)

    return params, m


def _log_prior_transit(theta, priorbounds):
    '''
    Assume priors on all parameters have uniform probability.
    '''
    # priorbounds contains the input priors, and because of how we previously
    # sorted theta, its sorted keys tell us which parts of theta correspond to
    # which physical quantities.

    allowed = True
    for ix, key in enumerate(np.sort(list(priorbounds.keys()))):
        if priorbounds[key][0] < theta[ix] < priorbounds[key][1]:
            allowed = True and allowed
        else:
            allowed = False

    if allowed:
        return 0.

    return -np.inf


def _log_prior_transit_plus_line(theta, priorbounds):
    return _log_prior_transit(theta, priorbounds)


def _log_likelihood_transit(theta, params, model, t, flux, err_flux,
                            priorbounds):
    '''
    Given a batman TransitModel and its proposed parameters (theta), update the
    batman params object with the proposed parameters and evaluate the gaussian
    likelihood.

    Note: the priorbounds are only needed to parse theta.
    '''

    u = []

    for ix, key in enumerate(sorted(priorbounds.keys())):

        if key == 'rp':
            params.rp = theta[ix]
        elif key == 't0':
            params.t0 = theta[ix]
        elif key == 'sma':
            params.a = theta[ix]
        elif key == 'incl':
            params.inc = theta[ix]
        elif key == 'period':
            params.per = theta[ix]
        elif key == 'ecc':
            params.per = theta[ix]
        elif key == 'omega':
            params.w = theta[ix]
        elif key == 'u_linear':
            u.append(theta[ix])
        elif key == 'u_quadratic':
            u.append(theta[ix])
            params.u = u

    lc = model.light_curve(params)
    residuals = flux - lc
    log_likelihood = -0.5*(
        np.sum((residuals/err_flux)**2 + np.log(2*np.pi*(err_flux)**2))
    )

    return log_likelihood


def _log_likelihood_transit_plus_line(theta, params, model, t, data_flux,
                                      err_flux, priorbounds):
    '''
    Given a batman TransitModel and its proposed parameters (theta), update the
    batman params object with the proposed parameters and evaluate the gaussian
    likelihood.

    Note: the priorbounds are only needed to parse theta.
    '''

    u = []
    for ix, key in enumerate(sorted(priorbounds.keys())):

        if key == 'rp':
            params.rp = theta[ix]
        elif key == 't0':
            params.t0 = theta[ix]
        elif key == 'sma':
            params.a = theta[ix]
        elif key == 'incl':
            params.inc = theta[ix]
        elif key == 'period':
            params.per = theta[ix]
        elif key == 'ecc':
            params.per = theta[ix]
        elif key == 'omega':
            params.w = theta[ix]
        elif key == 'u_linear':
            u.append(theta[ix])
        elif key == 'u_quadratic':
            u.append(theta[ix])
            params.u = u
        elif key == 'poly_order0':
            poly_order0 = theta[ix]
        elif key == 'poly_order1':
            poly_order1 = theta[ix]

    try:
        poly_order0
    except Exception:
        poly_order0 = 0
    else:
        pass

    transit = model.light_curve(params)
    line = poly_order0 + t*poly_order1
    model = transit + line

    residuals = data_flux - model

    log_likelihood = -0.5*(
        np.sum((residuals/err_flux)**2 + np.log(2*np.pi*(err_flux)**2))
    )

    return log_likelihood


def log_posterior_transit(theta, params, model, t, flux, err_flux, priorbounds):
    '''
    Evaluate posterior probability given proposed model parameters and
    the observed flux timeseries.
    '''
    lp = _log_prior_transit(theta, priorbounds)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + _log_likelihood_transit(theta, params, model, t, flux,
                                            err_flux, priorbounds)


def log_posterior_transit_plus_line(theta, params, model, t, flux, err_flux,
                                    priorbounds):
    '''
    Evaluate posterior probability given proposed model parameters and
    the observed flux timeseries.
    '''
    lp = _log_prior_transit_plus_line(theta, priorbounds)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return (
            lp + _log_likelihood_transit_plus_line(
                theta, params, model, t, flux, err_flux, priorbounds)
        )


###################################################
## MANDEL & AGOL TRANSIT MODEL FIT TO MAG SERIES ##
###################################################

def mandelagol_fit_magseries(
        times, mags, errs,
        fitparams,
        priorbounds,
        fixedparams,
        trueparams=None,
        burninpercent=0.3,
        plotcorner=False,
        samplesavpath=False,
        n_walkers=50,
        n_mcmc_steps=400,
        exp_time_minutes=2,
        eps=1e-4,
        skipsampling=False,
        overwriteexistingsamples=False,
        mcmcprogressbar=False,
        plotfit=False,
        magsarefluxes=False,
        sigclip=10.0,
        verbose=True,
        nworkers=4
):
    '''
    This fits a Mandel & Agol (2002) planetary transit model to a flux time
    series. You can fit and fix whatever parameters you want.

    It relies on Kreidberg (2015)'s BATMAN implementation for the transit model,
    emcee to sample the posterior (Foreman-Mackey et al 2013), `corner` to plot
    it, and `h5py` to save the samples. See e.g., Claret's work for good guesses
    of star-appropriate limb-darkening parameters.

    NOTE: this only works for flux time-series at the moment.

    NOTE: Between the `fitparams`, `priorbounds`, and `fixedparams` dicts, you
    must specify all of the planetary transit parameters required by BATMAN:
    `['t0', 'rp', 'sma', 'incl', 'u', 'rp', 'ecc', 'omega', 'period']`, or the
    BATMAN model will fail to initialize.

    Parameters
    ----------

    times,mags,errs : np.array
        The input flux time-series to fit a Fourier cosine series to.

    fitparams : dict
        This is the initial parameter guesses for MCMC, found e.g., by
        BLS. The key string format must not be changed, but any parameter can be
        either "fit" or "fixed". If it is "fit", it must have a corresponding
        prior. For example::

            fitparams = {'t0':1325.9, 'rp':np.sqrt(fitd['transitdepth']),
                         'sma':6.17, 'incl':85, 'u':[0.3, 0.2]}

        where 'u' is a list of the limb darkening parameters, Linear first, then
        quadratic. Quadratic limb darkening is the only form implemented.

    priorbounds : dict
        This sets the lower & upper bounds on uniform prior, e.g.::

            priorbounds = {'rp':(0.135, 0.145), 'u_linear':(0.3-1, 0.3+1),
                           'u_quad':(0.2-1, 0.2+1), 't0':(np.min(time),
                           np.max(time)), 'sma':(6,6.4), 'incl':(80,90)}

    fixedparams : dict
        This sets which parameters are fixed, and their values. For example::

            fixedparams = {'ecc':0.,
                           'omega':90.,
                           'limb_dark':'quadratic',
                           'period':fitd['period'] }

        `limb_dark` must be "quadratic".  It's "fixed", because once you
        choose your limb-darkening model, it's fixed.

    trueparams : list of floats
        The true parameter values you're fitting for, if they're known (e.g., a
        known planet, or fake data). Only for plotting purposes.

    burninpercent : float
        The percent of MCMC samples to discard as burn-in.

    plotcorner : str or False
        If this is a str, points to the path of output corner plot that will be
        generated for this MCMC run.

    samplesavpath : str
        This must be provided so `emcee` can save its MCMC samples to disk as
        HDF5 files. This will set the path of the output HDF5file written.

    n_walkers : int
        The number of MCMC walkers to use.

    n_mcmc_steps : int
        The number of MCMC steps to take.

    exp_time_minutes : int
        Exposure time, in minutes, passed to transit model to smear
        observations.

    eps : float
        The radius of the `n_walkers-dimensional` Gaussian ball used to
        initialize the MCMC.

    skipsampling : bool
        If you've already collected MCMC samples, and you do not want any more
        sampling (e.g., just make the plots), set this to be True.

    overwriteexistingsamples : bool
        If you've collected samples, but you want to overwrite them, set this to
        True. Usually, it should be False, which appends samples to
        `samplesavpath` HDF5 file.

    mcmcprogressbar : bool
        If True, will show a progress bar for the MCMC process.

    plotfit: str or bool
        If a str, indicates the path of the output fit plot file. If False, no
        fit plot will be made.

    magsarefluxes : bool
        This indicates if the input measurements in `mags` are actually fluxes.

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

    verbose : bool
        If True, will indicate MCMC progress.

    nworkers : int
        The number of parallel workers to launch for MCMC.

    Returns
    -------

    dict
        This function returns a dict containing the model fit parameters and
        other fit information. The form of this dict is mostly standardized
        across all functions in this module::

            {
                'fittype':'mandelagol',
                'fitinfo':{
                    'initialparams':the initial transit params provided,
                    'fixedparams':the fixed transit params provided,
                    'finalparams':the final model fit transit params,
                    'finalparamerrs':formal errors in the params,
                    'fitmags': the model fit mags,
                    'fitepoch': the epoch of minimum light for the fit,
                    'acceptancefraction': fraction of MCMC ensemble. low=bad.
                    'autocorrtime': if autocorrtime ~= n_mcmc_steps, not good.
                },
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

    from multiprocessing import Pool
    fittype = 'mandelagol'

    if not magsarefluxes:
        raise NotImplementedError('magsarefluxes is not implemented yet.')
    if not samplesavpath:
        raise ValueError(
            'This function requires that you save the samples somewhere'
        )
    if not mandel_agol_dependencies:
        raise ImportError(
            'This function depends on BATMAN, emcee>3.0, corner, and h5py.'
        )

    # sigma clip and get rid of zero errs
    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)
    nzind = np.nonzero(serrs)
    stimes, smags, serrs = stimes[nzind], smags[nzind], serrs[nzind]

    init_period = _get_value('period', fitparams, fixedparams)
    init_epoch = _get_value('t0', fitparams, fixedparams)
    init_rp = _get_value('rp', fitparams, fixedparams)
    init_sma = _get_value('sma', fitparams, fixedparams)
    init_incl = _get_value('incl', fitparams, fixedparams)
    init_ecc = _get_value('ecc', fitparams, fixedparams)
    init_omega = _get_value('omega', fitparams, fixedparams)
    limb_dark = _get_value('limb_dark', fitparams, fixedparams)
    init_u = _get_value('u', fitparams, fixedparams)

    if not limb_dark == 'quadratic':
        raise ValueError(
            'only quadratic limb-darkening is supported at the moment'
        )

    # initialize the model and calculate the initial model light-curve
    init_params, init_m = _transit_model(stimes, init_epoch, init_period,
                                         init_rp, init_sma, init_incl, init_ecc,
                                         init_omega, init_u, limb_dark,
                                         exp_time_minutes=exp_time_minutes)
    init_flux = init_m.light_curve(init_params)

    # guessed initial params. give nice guesses, or else emcee struggles.
    theta, fitparamnames = [], []
    for k in np.sort(list(fitparams.keys())):
        if isinstance(fitparams[k], float) or isinstance(fitparams[k], int):
            theta.append(fitparams[k])
            fitparamnames.append(fitparams[k])
        elif isinstance(fitparams[k], list):
            if not len(fitparams[k]) == 2:
                raise ValueError('should only be quadratic LD coeffs')
            theta.append(fitparams[k][0])
            theta.append(fitparams[k][1])
            fitparamnames.append(fitparams[k][0])
            fitparamnames.append(fitparams[k][1])

    # initialize sampler
    n_dim = len(theta)
    initial_position_vec = [theta + eps*np.random.randn(n_dim)
                            for i in range(n_walkers)]

    # run the MCMC, unless you just want to load the available samples
    if not skipsampling:

        backend = emcee.backends.HDFBackend(samplesavpath)
        if overwriteexistingsamples:
            LOGWARNING(
                'erased samples previously at {:s}'.format(samplesavpath)
            )
            backend.reset(n_walkers, n_dim)

        # if this is the first run, then start from a gaussian ball.
        # otherwise, resume from the previous samples.
        starting_positions = initial_position_vec
        isfirstrun = True
        if os.path.exists(backend.filename):
            if backend.iteration > 1:
                starting_positions = None
                isfirstrun = False

        if verbose and isfirstrun:
            LOGINFO(
                'start {:s} MCMC with {:d} dims, {:d} steps, {:d} walkers,'.
                format(fittype, n_dim, n_mcmc_steps, n_walkers) +
                ' {:d} threads'.format(nworkers)
            )
        elif verbose and not isfirstrun:
            LOGINFO(
                'continue {:s} with {:d} dims, {:d} steps, {:d} walkers, '.
                format(fittype, n_dim, n_mcmc_steps, n_walkers) +
                '{:d} threads'.format(nworkers)
            )

        with Pool(nworkers) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, n_dim, log_posterior_transit,
                args=(init_params, init_m, stimes,
                      smags, serrs, priorbounds),
                pool=pool,
                backend=backend
            )
            sampler.run_mcmc(starting_positions, n_mcmc_steps,
                             progress=mcmcprogressbar)

        if verbose:
            LOGINFO(
                'ended {:s} MCMC run with {:d} steps, {:d} walkers, '.format(
                    fittype, n_mcmc_steps, n_walkers
                ) + '{:d} threads'.format(nworkers)
            )

    reader = emcee.backends.HDFBackend(samplesavpath)

    n_to_discard = int(burninpercent*n_mcmc_steps)

    samples = reader.get_chain(discard=n_to_discard, flat=True)

    # Get best-fit parameters and their 1-sigma error bars
    fit_statistics = list(
        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            list(zip( *np.percentile(samples, [15.85, 50, 84.15], axis=0))))
    )

    medianparams, std_perrs, std_merrs = {}, {}, {}
    for ix, k in enumerate(np.sort(list(priorbounds.keys()))):
        medianparams[k] = fit_statistics[ix][0]
        std_perrs[k] = fit_statistics[ix][1]
        std_merrs[k] = fit_statistics[ix][2]

    stderrs = {'std_perrs':std_perrs, 'std_merrs':std_merrs}

    per = _get_value('period', medianparams, fixedparams)
    t0 = _get_value('t0', medianparams, fixedparams)
    rp = _get_value('rp', medianparams, fixedparams)
    sma = _get_value('sma', medianparams, fixedparams)
    incl = _get_value('incl', medianparams, fixedparams)
    ecc = _get_value('ecc', medianparams, fixedparams)
    omega = _get_value('omega', medianparams, fixedparams)
    limb_dark = _get_value('limb_dark', medianparams, fixedparams)
    try:
        u = fixedparams['u']
    except Exception:
        u = [medianparams['u_linear'], medianparams['u_quad']]

    fit_params, fit_m = _transit_model(stimes, t0, per, rp, sma, incl, ecc,
                                       omega, u, limb_dark,
                                       exp_time_minutes=exp_time_minutes)
    fitmags = fit_m.light_curve(fit_params)
    fepoch = t0

    # assemble the return dictionary. for the autocorrelation time, the "c"
    # input parameter indicates the number of auto-correlation lengths (ACLs)
    # needed to "reliably" calculate the ACL itself. set it to 1 because
    # oftentimes, the sampler has not collected enough samples to satsify the
    # default value of 10.
    returndict = {
        'fittype':fittype,
        'fitinfo':{
            'initialparams':fitparams,
            'initialmags':init_flux,
            'fixedparams':fixedparams,
            'finalparams':medianparams,
            'finalparamerrs':stderrs,
            'fitmags':fitmags,
            'fitepoch':fepoch,
            'acceptancefraction':np.mean(sampler.acceptance_fraction),
            'autocorrtime':np.mean(sampler.get_autocorr_time(c=1, quiet=True))
        },
        'fitplotfile':None,
        'magseries':{
            'times':stimes,
            'mags':smags,
            'errs':serrs,
            'magsarefluxes':magsarefluxes,
        },
    }

    # make the output corner plot, and lightcurve plot if desired
    if plotcorner:
        if isinstance(trueparams,dict):
            trueparamkeys = np.sort(list(trueparams.keys()))
            truelist = [trueparams[k] for k in trueparamkeys]
            fig = corner.corner(
                samples,
                labels=trueparamkeys,
                truths=truelist,
                quantiles=[0.1585, 0.5, .8415], show_titles=True
            )
        else:
            fig = corner.corner(samples,
                                labels=fitparamnames,
                                quantiles=[0.1585, 0.5, .8415],
                                show_titles=True)

        plt.savefig(plotcorner, dpi=300)
        if verbose:
            LOGINFO('saved {:s}'.format(plotcorner))

    if plotfit and isinstance(plotfit, str):

        f, ax = plt.subplots(figsize=(8,6))
        ax.scatter(stimes, smags, c='k', alpha=0.5, label='observed',
                   zorder=1, s=1.5, rasterized=True, linewidths=0)
        ax.scatter(stimes, init_flux, c='r', alpha=1,
                   s=3.5, zorder=2, rasterized=True, linewidths=0,
                   label='initial guess')
        ax.scatter(
            stimes, fitmags, c='b', alpha=1,
            s=1.5, zorder=3, rasterized=True, linewidths=0,
            label='fit {:d} dims'.format(
                len(fitparamnames))
        )
        ax.legend(loc='best')
        ax.set(xlabel='time [days]', ylabel='relative flux')
        f.savefig(plotfit, dpi=300, bbox_inches='tight')
        if verbose:
            LOGINFO('saved {:s}'.format(plotfit))

        returndict['fitplotfile'] = plotfit

    return returndict


def mandelagol_and_line_fit_magseries(
        times, mags, errs,
        fitparams,
        priorbounds,
        fixedparams,
        trueparams=None,
        burninpercent=0.3,
        plotcorner=False,
        timeoffset=0,
        samplesavpath=False,
        n_walkers=50,
        n_mcmc_steps=400,
        exp_time_minutes=2,
        eps=1e-4,
        skipsampling=False,
        overwriteexistingsamples=False,
        mcmcprogressbar=False,
        plotfit=False,
        scatterxdata=None,
        scatteryaxes=None,
        magsarefluxes=True,
        sigclip=10.0,
        verbose=True,
        nworkers=4
):
    '''The model fit by this function is: a Mandel & Agol (2002) transit, PLUS a
    line. You can fit and fix whatever parameters you want.

    Typical use case: you want to measure transit times of individual SNR >~
    50 transits. You fix all the transit parameters except for the mid-time,
    and also fit for a line locally.

    NOTE: this only works for flux time-series at the moment.

    NOTE: Between the `fitparams`, `priorbounds`, and `fixedparams` dicts, you
    must specify all of the planetary transit parameters required by BATMAN and
    the parameters for the line fit: `['t0', 'rp', 'sma', 'incl', 'u', 'rp',
    'ecc', 'omega', 'period', 'poly_order0', poly_order1']`, or the BATMAN model
    will fail to initialize.

    Parameters
    ----------

    times,mags,errs : np.array
        The input flux time-series to fit a Fourier cosine series to.

    fitparams : dict
        This is the initial parameter guesses for MCMC, found e.g., by
        BLS. The key string format must not be changed, but any parameter can be
        either "fit" or "fixed". If it is "fit", it must have a corresponding
        prior. For example::

            fitparams = {'t0':1325.9,
                         'poly_order0':1,
                         'poly_order1':0.}

        where `t0` is the time of transit-center for a reference transit.
        `poly_order0` corresponds to the intercept of the line, `poly_order1` is
        the slope.

    priorbounds : dict
        This sets the lower & upper bounds on uniform prior, e.g.::

            priorbounds = {'t0':(np.min(time), np.max(time)),
                            'poly_order0':(0.5,1.5),
                            'poly_order1':(-0.5,0.5) }

    fixedparams : dict
        This sets which parameters are fixed, and their values. For example::

            fixedparams = {'ecc':0.,
                           'omega':90.,
                           'limb_dark':'quadratic',
                           'period':fitd['period'],
                           'rp':np.sqrt(fitd['transitdepth']),
                           'sma':6.17, 'incl':85, 'u':[0.3, 0.2]}

        `limb_dark` must be "quadratic".  It's "fixed", because once you
        choose your limb-darkening model, it's fixed.

    trueparams : list of floats
        The true parameter values you're fitting for, if they're known (e.g., a
        known planet, or fake data). Only for plotting purposes.

    burninpercent : float
        The percent of MCMC samples to discard as burn-in.

    plotcorner : str or False
        If this is a str, points to the path of output corner plot that will be
        generated for this MCMC run.

    timeoffset : float
        If input times are offset by some constant, and you want saved pickles
        to fix that.

    samplesavpath : str
        This must be provided so `emcee` can save its MCMC samples to disk as
        HDF5 files. This will set the path of the output HDF5file written.

    n_walkers : int
        The number of MCMC walkers to use.

    n_mcmc_steps : int
        The number of MCMC steps to take.

    exp_time_minutes : int
        Exposure time, in minutes, passed to transit model to smear
        observations.

    eps : float
        The radius of the `n_walkers-dimensional` Gaussian ball used to
        initialize the MCMC.

    skipsampling : bool
        If you've already collected MCMC samples, and you do not want any more
        sampling (e.g., just make the plots), set this to be True.

    overwriteexistingsamples : bool
        If you've collected samples, but you want to overwrite them, set this to
        True. Usually, it should be False, which appends samples to
        `samplesavpath` HDF5 file.

    mcmcprogressbar : bool
        If True, will show a progress bar for the MCMC process.

    plotfit: str or bool
        If a str, indicates the path of the output fit plot file. If False, no
        fit plot will be made.

    scatterxdata : np.array or None
        Use this to overplot x,y scatter points on the output model/data
        lightcurve (e.g., to highlight bad data, or to indicate an ephemeris),
        this can take a `np.ndarray` with the same units as `times`.

    scatteryaxes : np.array or None
        Use this to provide the y-values for scatterxdata, in units of fraction
        of an axis.

    magsarefluxes : bool
        This indicates if the input measurements in `mags` are actually fluxes.

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

    verbose : bool
        If True, will indicate MCMC progress.

    nworkers : int
        The number of parallel workers to launch for MCMC.

    Returns
    -------

    dict
        This function returns a dict containing the model fit parameters and
        other fit information. The form of this dict is mostly standardized
        across all functions in this module::

            {
                'fittype':'mandelagol_and_line',
                'fitinfo':{
                    'initialparams':the initial transit params provided,
                    'fixedparams':the fixed transit params provided,
                    'finalparams':the final model fit transit params,
                    'finalparamerrs':formal errors in the params,
                    'fitmags': the model fit mags,
                    'fitepoch': the epoch of minimum light for the fit,
                    'acceptancefraction': fraction of MCMC ensemble. low=bad.
                    'autocorrtime': if autocorrtime ~= n_mcmc_steps, not good.
                },
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

    from multiprocessing import Pool
    fittype = 'mandelagol_and_line'

    if not magsarefluxes:
        raise NotImplementedError('magsarefluxes is not implemented yet.')
    if not samplesavpath:
        raise ValueError(
            'This function requires that you save the samples somewhere'
        )
    if not mandel_agol_dependencies:
        raise ImportError(
            'This function depends on BATMAN, emcee>3.0, corner, and h5py.'
        )

    # sigma clip and get rid of zero errs
    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)
    nzind = np.nonzero(serrs)
    stimes, smags, serrs = stimes[nzind], smags[nzind], serrs[nzind]

    init_period = _get_value('period', fitparams, fixedparams)
    init_epoch = _get_value('t0', fitparams, fixedparams)
    init_rp = _get_value('rp', fitparams, fixedparams)
    init_sma = _get_value('sma', fitparams, fixedparams)
    init_incl = _get_value('incl', fitparams, fixedparams)
    init_ecc = _get_value('ecc', fitparams, fixedparams)
    init_omega = _get_value('omega', fitparams, fixedparams)
    limb_dark = _get_value('limb_dark', fitparams, fixedparams)
    init_u = _get_value('u', fitparams, fixedparams)

    init_poly_order0 = _get_value('poly_order0', fitparams, fixedparams)
    init_poly_order1 = _get_value('poly_order1', fitparams, fixedparams)

    if not limb_dark == 'quadratic':
        raise ValueError(
            'only quadratic limb-darkening is supported at the moment'
        )

    # initialize the model and calculate the initial model light-curve
    init_params, init_m = _transit_model(
        stimes, init_epoch, init_period, init_rp, init_sma, init_incl,
        init_ecc, init_omega, init_u, limb_dark,
        exp_time_minutes=exp_time_minutes)

    init_flux = (
        init_m.light_curve(init_params) +
        init_poly_order0 + init_poly_order1*stimes
    )

    # guessed initial params. give nice guesses, or else emcee struggles.
    theta, fitparamnames = [], []
    for k in np.sort(list(fitparams.keys())):
        if isinstance(fitparams[k], float) or isinstance(fitparams[k], int):
            theta.append(fitparams[k])
            fitparamnames.append(fitparams[k])
        elif isinstance(fitparams[k], list):
            if not len(fitparams[k]) == 2:
                raise ValueError('should only be quadratic LD coeffs')
            theta.append(fitparams[k][0])
            theta.append(fitparams[k][1])
            fitparamnames.append(fitparams[k][0])
            fitparamnames.append(fitparams[k][1])

    # initialize sampler
    n_dim = len(theta)

    # run the MCMC, unless you just want to load the available samples
    if not skipsampling:

        backend = emcee.backends.HDFBackend(samplesavpath)
        if overwriteexistingsamples:
            LOGWARNING(
                'erased samples previously at {:s}'.format(samplesavpath)
            )
            backend.reset(n_walkers, n_dim)

        # if this is the first run, then start from a gaussian ball, centered
        # on the maximum likelihood solution.  otherwise, resume from the
        # previous samples.
        def nll(*args):
            return -_log_likelihood_transit_plus_line(*args)

        soln = spminimize(
            nll, theta, method='BFGS',
            args=(init_params, init_m, stimes, smags, serrs, priorbounds)
        )
        theta_ml = soln.x
        ml_poly_order0 = theta_ml[0]
        ml_poly_order1 = theta_ml[1]
        ml_rp = theta_ml[2]
        ml_t0 = theta_ml[3]

        ml_params, ml_m = _transit_model(stimes, ml_t0, init_period,
                                         ml_rp, init_sma, init_incl,
                                         init_ecc, init_omega, init_u,
                                         limb_dark,
                                         exp_time_minutes=exp_time_minutes)
        ml_mags = (
            ml_m.light_curve(ml_params) +
            ml_poly_order0 + ml_poly_order1*stimes
        )

        initial_position_vec = [theta_ml + eps*np.random.randn(n_dim)
                                for i in range(n_walkers)]
        starting_positions = initial_position_vec
        isfirstrun = True
        if os.path.exists(backend.filename):
            if backend.iteration > 1:
                starting_positions = None
                isfirstrun = False

        if verbose and isfirstrun:
            LOGINFO(
                'start {:s} MCMC with {:d} dims, {:d} steps, {:d} walkers,'.
                format(fittype, n_dim, n_mcmc_steps, n_walkers) +
                ' {:d} threads'.format(nworkers)
            )
        elif verbose and not isfirstrun:
            LOGINFO(
                'continue {:s} with {:d} dims, {:d} steps, {:d} walkers, '.
                format(fittype, n_dim, n_mcmc_steps, n_walkers) +
                '{:d} threads'.format(nworkers)
            )

        with Pool(nworkers) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, n_dim, log_posterior_transit_plus_line,
                args=(init_params, init_m, stimes, smags, serrs, priorbounds),
                pool=pool,
                backend=backend
            )
            sampler.run_mcmc(starting_positions, n_mcmc_steps,
                             progress=mcmcprogressbar)

        if verbose:
            LOGINFO(
                'ended {:s} MCMC run with {:d} steps, {:d} walkers, '.format(
                    fittype, n_mcmc_steps, n_walkers
                ) + '{:d} threads'.format(nworkers)
            )

    reader = emcee.backends.HDFBackend(samplesavpath)

    n_to_discard = int(burninpercent*n_mcmc_steps)

    samples = reader.get_chain(discard=n_to_discard, flat=True)

    # Get best-fit parameters and their 1-sigma error bars
    fit_statistics = list(
        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            list(zip( *np.percentile(samples, [15.85, 50, 84.15], axis=0))))
    )

    medianparams, std_perrs, std_merrs = {}, {}, {}
    for ix, k in enumerate(np.sort(list(priorbounds.keys()))):
        medianparams[k] = fit_statistics[ix][0]
        std_perrs[k] = fit_statistics[ix][1]
        std_merrs[k] = fit_statistics[ix][2]

    stderrs = {'std_perrs':std_perrs, 'std_merrs':std_merrs}

    per = _get_value('period', medianparams, fixedparams)
    t0 = _get_value('t0', medianparams, fixedparams)
    rp = _get_value('rp', medianparams, fixedparams)
    sma = _get_value('sma', medianparams, fixedparams)
    incl = _get_value('incl', medianparams, fixedparams)
    ecc = _get_value('ecc', medianparams, fixedparams)
    omega = _get_value('omega', medianparams, fixedparams)
    limb_dark = _get_value('limb_dark', medianparams, fixedparams)
    try:
        u = fixedparams['u']
    except Exception:
        u = [medianparams['u_linear'], medianparams['u_quad']]

    poly_order0 = _get_value('poly_order0', medianparams, fixedparams)
    poly_order1 = _get_value('poly_order1', medianparams, fixedparams)

    # initialize the model and calculate the initial model light-curve
    fit_params, fit_m = _transit_model(stimes, t0, per, rp, sma, incl, ecc,
                                       omega, u, limb_dark,
                                       exp_time_minutes=exp_time_minutes)

    fitmags = (
        fit_m.light_curve(fit_params) +
        poly_order0 + poly_order1*stimes
    )
    fepoch = t0

    # assemble the return dictionary. for the autocorrelation time, the "c"
    # input parameter indicates the number of auto-correlation lengths (ACLs)
    # needed to "reliably" calculate the ACL itself. set it to 1 because
    # oftentimes, the sampler has not collected enough samples to satsify the
    # default value of 10.
    medianparams['t0'] += timeoffset
    returndict = {
        'fittype':fittype,
        'fitinfo':{
            'initialparams':fitparams,
            'initialmags':init_flux,
            'fixedparams':fixedparams,
            'finalparams':medianparams,
            'finalparamerrs':stderrs,
            'fitmags':fitmags,
            'fitepoch':fepoch+timeoffset,
            'acceptancefraction':np.mean(sampler.acceptance_fraction),
            'autocorrtime':np.mean(sampler.get_autocorr_time(c=1, quiet=True))
        },
        'fitplotfile':None,
        'magseries':{
            'times':stimes+timeoffset,
            'mags':smags,
            'errs':serrs,
            'magsarefluxes':magsarefluxes,
        },
    }

    # make the output corner plot, and lightcurve plot if desired
    if plotcorner:
        fig = corner.corner(
            samples,
            labels=['line intercept-1', 'line slope',
                    'rp','t0-{:.4f}'.format(timeoffset)],
            truths=[ml_poly_order0, ml_poly_order1, ml_rp, ml_t0],
            quantiles=[0.1585, 0.5, .8415], show_titles=True
        )
        plt.savefig(plotcorner, dpi=300)
        if verbose:
            LOGINFO('saved {:s}'.format(plotcorner))

    if plotfit and isinstance(plotfit, str):

        plt.close('all')
        f, (a0, a1) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                   figsize=(8,5),
                                   gridspec_kw={'height_ratios':[3, 1]})

        a0.scatter(stimes, smags, c='k', alpha=0.9, label='data', zorder=1,
                   s=10, rasterized=True, linewidths=0)

        DEBUGGING = False
        if DEBUGGING:
            a0.scatter(stimes, init_flux, c='r', alpha=1, s=3.5, zorder=2,
                       rasterized=True, linewidths=0,
                       label='initial guess for ml')
            a0.scatter(stimes, ml_mags, c='g', alpha=1, s=3.5, zorder=2,
                       rasterized=True, linewidths=0, label='max likelihood')

        a0.plot(
            stimes, fitmags, c='b',
            zorder=0, rasterized=True, lw=2, alpha=0.4,
            label='{:s} fit, {:d} dims'.format(fittype, len(fitparamnames))
        )

        a1.scatter(
            stimes, smags-fitmags, c='k', alpha=0.9,
            rasterized=True, s=10, linewidths=0
        )

        if scatterxdata and scatteryaxes:
            import matplotlib.transforms as transforms
            for a in [a0, a1]:
                transform = transforms.blended_transform_factory(
                    a.transData, a.transAxes
                )
                a.scatter(scatterxdata, scatteryaxes, c='r', alpha=0.9,
                          zorder=2, s=10, rasterized=True, linewidths=0,
                          marker="^", transform=transform)

        a1.set_xlabel('time-t0 [days]')
        a0.set_ylabel('relative flux')
        a1.set_ylabel('residual')
        a0.legend(loc='best', fontsize='x-small')
        for a in [a0, a1]:
            a.get_yaxis().set_tick_params(which='both', direction='in')
            a.get_xaxis().set_tick_params(which='both', direction='in')

        f.tight_layout(h_pad=0, w_pad=0)
        f.savefig(plotfit, dpi=300, bbox_inches='tight')
        if verbose:
            LOGINFO('saved {:s}'.format(plotfit))

        returndict['fitplotfile'] = plotfit

    return returndict


if vizier_dependency:

    def _fivetransitparam_fit_magseries(
            time, flux, err,
            tlsr,
            teff, rstar, logg,
            identifier,
            fit_savdir,
            chain_savdir,
            exp_time_minutes=30,
            n_transit_durations=5,
            nworkers=40,
            n_mcmc_steps=1,
            burninpercent=0.3,
            overwriteexistingsamples=True,
            mcmcprogressbar=True
    ):
        '''Helper to ``lcfit.transits.fivetransitparam_fit_magseries``.

        Procedure implemented does the following.

            1. Defines initial parameter guesses to fit light curve.

            2. Isolate each transit to within +/- n_transit_durations.

            3. Fit out a line within each transit window (having masked out the
               transit), to get a clean phase-folded light curve.

            4. Make plots, in fit_savdir, showing the result of the above
               fitting procedure.

            5. Fit for (t0, period, a/Rstar, Rp/Rstar, inclination). Fixes e to
               0, and uses theoretical quadratic limb darkening coefficients in
               the bandpass given by the user, as found with the stellar
               parameters.  6. Fit happens in two stages. First, it is done with
               the error bars passed to the function in ``err``. The best-fit
               model is then subtracted, and the errors are set to equal the RMS
               of the OOT points in the subtracted light curve. Then the fit is
               redone.  7. Fit outputs: corner plots, phase-folded light curve,
               light curve with fit overplotted.

        '''
        # import inside function to avoid circular imports
        from astrobase.varbase.transits import get_snr_of_dip
        from astrobase.varbase.transits import (
            estimate_achievable_tmid_precision
        )
        from astrobase.plotbase import plot_phased_magseries

        #
        # Define initial parameter guesses from TLS results. Inclination and
        # impact parameter guesses don't really matter.
        #
        incl = 85
        b = 0.2
        period = tlsr['period']
        T_dur = tlsr['duration']
        rp_by_rstar = tlsr['rp_rs']
        a_by_rstar = (
            (period*units.day)/(np.pi*T_dur*units.day) * (1-b**2)**(1/2)
        ).cgs.value
        t0 = tlsr['T0']
        u_linear, u_quad = get_tess_limb_darkening_guesses(teff, logg)

        #
        # Isolate each transit to within +/- n_transit_durations.  Then fit only
        # +/- n_transit_durations near the transit data. Don't try to fit OOT or
        # occultation data.
        #
        tmids_obsd = tlsr['transit_times']

        t_Is = tmids_obsd - T_dur/2
        t_IVs = tmids_obsd + T_dur/2

        t_starts = t_Is - n_transit_durations*T_dur
        t_ends = t_IVs + n_transit_durations*T_dur

        sel_inds = np.zeros_like(time).astype(bool)
        for t_start,t_end in zip(t_starts, t_ends):
            these_inds = (time > t_start) & (time < t_end)
            if np.any(these_inds):
                sel_inds |= these_inds

        #
        # To construct the phase-folded light curve, fit a line to the OOT flux
        # data, and use the parameters of the best-fitting line to "rectify"
        # each lightcurve. Note that an order 1 legendre polynomial == a line,
        # so we'll use that implementation.
        #
        out_fluxs, in_fluxs, fit_fluxs, time_list, intra_inds_list, err_list = (
            [], [], [], [], [], []
        )
        for t_start,t_end in zip(t_starts, t_ends):
            this_window_inds = (time > t_start) & (time < t_end)
            tmid = t_start + (t_end-t_start)/2
            # flag out slightly more than expected "in transit" points
            prefactor = 1.05
            transit_start = tmid - prefactor*T_dur/2
            transit_end = tmid + prefactor*T_dur/2

            this_window_intra = (
                (time[this_window_inds] > transit_start) &
                (time[this_window_inds] < transit_end)
            )
            this_window_oot = ~this_window_intra

            this_oot_time = time[this_window_inds][this_window_oot]
            this_oot_flux = flux[this_window_inds][this_window_oot]

            if len(this_oot_flux) == len(this_oot_time) == 0:
                continue
            try:
                p = Legendre.fit(this_oot_time, this_oot_flux, 1)
                this_window_fit_flux = p(time[this_window_inds])

                time_list.append( time[this_window_inds] )
                out_fluxs.append(
                    flux[this_window_inds] / this_window_fit_flux
                )
                fit_fluxs.append( this_window_fit_flux )
                in_fluxs.append( flux[this_window_inds] )
                intra_inds_list.append(
                    (time[this_window_inds] > transit_start) &
                    (time[this_window_inds] < transit_end)
                )
                err_list.append( err[this_window_inds] )
            except np.linalg.LinAlgError:
                LOGWARNING(
                    'Legendre.fit failed, b/c bad data for this transit. '
                    'Continue.'
                )
                continue

        #
        # Make plots to verify that this line-rectifying procedure is working.
        #
        ix = 0
        for _time, _flux, _fit_flux, _out_flux, _intra in zip(
            time_list, in_fluxs, fit_fluxs, out_fluxs, intra_inds_list
        ):

            savpath = os.path.join(
                fit_savdir,
                '{}_transitnum{}.png'.format(identifier, str(ix).zfill(3))
            )

            if os.path.exists(savpath):
                LOGINFO('found & skipped making {}'.format(savpath))
                ix += 1
                continue

            plt.close('all')
            fig, (a0,a1) = plt.subplots(nrows=2, sharex=True, figsize=(6,6))

            a0.scatter(_time, _flux, c='k', alpha=0.9, label='data', zorder=1,
                       s=10, rasterized=True, linewidths=0)
            a0.scatter(
                _time[_intra],
                _flux[_intra],
                c='r',
                alpha=1,
                label='in-transit (for fit)',
                zorder=2,
                s=10,
                rasterized=True,
                linewidths=0
            )

            a0.plot(_time, _fit_flux, c='b', zorder=0, rasterized=True, lw=2,
                    alpha=0.4, label='linear fit to OOT')

            a1.scatter(_time, _out_flux, c='k', alpha=0.9, rasterized=True,
                       s=10, linewidths=0)
            a1.plot(_time, _fit_flux/_fit_flux, c='b', zorder=0,
                    rasterized=True,
                    lw=2, alpha=0.4, label='linear fit to OOT')

            xlim = a1.get_xlim()

            for a in [a0,a1]:
                a.hlines(1, np.min(_time)-10, np.max(_time)+10, color='gray',
                         zorder=-2, rasterized=True, alpha=0.2, lw=1,
                         label='flux=1')

            a1.set_xlabel('time-t0 [days]')
            a0.set_ylabel('relative flux')
            a1.set_ylabel('residual')
            a0.legend(loc='best', fontsize='x-small')
            for a in [a0, a1]:
                a.get_yaxis().set_tick_params(which='both', direction='in')
                a.get_xaxis().set_tick_params(which='both', direction='in')
                a.set_xlim(xlim)

            fig.tight_layout(h_pad=0, w_pad=0)
            fig.savefig(savpath, dpi=300, bbox_inches='tight')
            LOGINFO('saved {:s}'.format(savpath))
            ix += 1

        sel_time = np.concatenate(time_list)
        sel_flux = np.concatenate(out_fluxs)
        sel_err = np.concatenate(err_list)
        if not len(sel_flux) == len(sel_time) == len(sel_err):
            raise ValueError(
                'failed to properly concatenate '
                'light curve after rectifying each '
                'transit window'
            )

        # model = transit only (no line). "transit" as defined by BATMAN has
        # flux=1 out of transit.
        fittype = 'fivetransitparam'
        initfitparams = {
            't0':t0,
            'period':period,
            'sma':a_by_rstar,
            'rp':rp_by_rstar,
            'incl':incl
        }
        fixedparams = {
            'ecc':0.,
            'omega':90.,
            'limb_dark':'quadratic',
            'u':[u_linear,u_quad]
        }
        priorbounds = {
            't0':(t0 - period/10, t0 + period/10),
            'period':(period-1e-1, period+1e-1),
            'sma':(a_by_rstar/5, 5*a_by_rstar),
            'rp':(rp_by_rstar/3, np.min([3*rp_by_rstar,0.5])),
            'incl':(65, 90)
        }
        cornerparams = {
            't0':t0,
            'period':period,
            'sma':a_by_rstar,
            'rp':rp_by_rstar,
            'incl':incl
        }

        ndims = len(initfitparams)

        ##########################################################
        # FIRST: run the fit using the errors given in the data. #
        ##########################################################
        phasedlcsavpath = os.path.join(
            fit_savdir,
            '{}_phased_{}_fit_{}d_dataerrs.png'.
            format(identifier, fittype, ndims)
        )
        cornersavpath = os.path.join(
            fit_savdir,
            '{}_phased_corner_{}_fit_{}d_dataerrs.png'.
            format(identifier, fittype, ndims)
        )
        samplesavpath = os.path.join(
            chain_savdir,
            '{}_phased_{}_fit_samples_{}d_dataerrs.h5'.
            format(identifier, fittype, ndims)
        )

        if not os.path.exists(chain_savdir):
            try:
                os.mkdir(chain_savdir)
            except Exception:
                raise IOError('you need to save chains')

        LOGINFO('beginning {:s}'.format(samplesavpath))

        plt.close('all')

        fitparamdir = fit_savdir
        if not os.path.exists(fitparamdir):
            os.mkdir(fitparamdir)
        fitpklsavpath = os.path.join(
            fitparamdir,
            '{}_phased_{}_fit_dataerrs.pickle'.
            format(identifier, fittype)
        )

        if os.path.exists(fitpklsavpath) and not overwriteexistingsamples:
            maf_data_errs = pickle.load(open(fitpklsavpath, 'rb'))

        else:
            maf_data_errs = mandelagol_fit_magseries(
                sel_time, sel_flux, sel_err,
                initfitparams, priorbounds, fixedparams,
                trueparams=cornerparams, magsarefluxes=True,
                sigclip=None, plotfit=phasedlcsavpath,
                plotcorner=cornersavpath,
                samplesavpath=samplesavpath, nworkers=nworkers,
                n_mcmc_steps=n_mcmc_steps,
                exp_time_minutes=exp_time_minutes,
                burninpercent=burninpercent,
                eps=1e-6, n_walkers=500, skipsampling=False,
                overwriteexistingsamples=overwriteexistingsamples,
                mcmcprogressbar=mcmcprogressbar
            )
            with open(fitpklsavpath, 'wb') as f:
                pickle.dump(maf_data_errs, f, pickle.HIGHEST_PROTOCOL)
                LOGINFO('saved {:s}'.format(fitpklsavpath))

        fitfluxs = maf_data_errs['fitinfo']['fitmags']
        fitepoch = maf_data_errs['fitinfo']['fitepoch']
        fiterrors = maf_data_errs['fitinfo']['finalparamerrs']
        fitepoch_perr = fiterrors['std_perrs']['t0']
        fitepoch_merr = fiterrors['std_merrs']['t0']

        # Winn (2010) eq 14 gives the transit duration
        k = maf_data_errs['fitinfo']['finalparams']['rp']
        t_dur_day = (
            (period*units.day)/np.pi * np.arcsin(
                1/a_by_rstar * np.sqrt(
                    (1 + k)**2 - b**2
                ) / np.sin((incl*units.deg))
            )
        ).to(units.day*units.rad).value

        per_point_cadence = exp_time_minutes*units.min
        npoints_in_transit = (
            int(np.floor(((t_dur_day*units.day)/per_point_cadence).cgs.value))
        )

        # use the whole LC's RMS as the "noise". check how precisely you are
        # determining the ephemeris vs. what is theoretically expected.
        snr, _, empirical_errs = get_snr_of_dip(
            sel_time, sel_flux, sel_time, fitfluxs,
            magsarefluxes=True, atol_normalization=1e-2,
            transitdepth=k**2, npoints_in_transit=npoints_in_transit)

        sigma_tc_theory = estimate_achievable_tmid_precision(
            snr, t_ingress_min=0.05*t_dur_day*24*60,
            t_duration_hr=t_dur_day*24)

        LOGINFO(
            'mean fitepoch err: {:.2e}'.format(
                np.mean([fitepoch_merr, fitepoch_perr])
            )
        )
        LOGINFO(
            'mean fitepoch err / theory err = {:.2e}'.format(
                np.mean([fitepoch_merr, fitepoch_perr]) / sigma_tc_theory
            )
        )
        LOGINFO(
            'mean error from data lightcurve =' +
            '{:.2e}'.format(np.mean(sel_err)) +
            '\nmeasured empirical RMS = {:.2e}'.format(empirical_errs)
        )

        empirical_err = np.ones_like(sel_err)*empirical_errs

        ###################################################################
        # THEN: rerun the fit using the empirically determined errors     #
        # (measured from RMS of the transit-model subtracted lightcurve). #
        ###################################################################
        phasedlcsavpath = os.path.join(
            fit_savdir,
            '{}_phased_{}_fit_{}d_empiricalerrs.png'.
            format(identifier, fittype, ndims)
        )
        cornersavpath = os.path.join(
            fit_savdir,
            '{}_phased_corner_{}_fit_{}d_empiricalerrs.png'.
            format(identifier, fittype, ndims)
        )
        samplesavpath = os.path.join(
            chain_savdir,
            '{}_phased_{}_fit_samples_{}d_empiricalerrs.h5'.
            format(identifier, fittype, ndims)
        )

        plt.close('all')

        LOGINFO('beginning {}'.format(samplesavpath))

        fitpklsavpath = os.path.join(
            fitparamdir,
            '{}_phased_{}_fit_empiricalerrs.pickle'.
            format(identifier, fittype)
        )

        if os.path.exists(fitpklsavpath) and not overwriteexistingsamples:
            maf_empc_errs = pickle.load(open(fitpklsavpath, 'rb'))

        else:
            maf_empc_errs = mandelagol_fit_magseries(
                sel_time, sel_flux, empirical_err,
                initfitparams, priorbounds, fixedparams,
                trueparams=cornerparams, magsarefluxes=True,
                sigclip=None, plotfit=phasedlcsavpath,
                plotcorner=cornersavpath,
                samplesavpath=samplesavpath, nworkers=nworkers,
                n_mcmc_steps=n_mcmc_steps, exp_time_minutes=30,
                eps=1e-6, n_walkers=500, skipsampling=False,
                overwriteexistingsamples=overwriteexistingsamples,
                mcmcprogressbar=mcmcprogressbar
            )

            with open(fitpklsavpath, 'wb') as f:
                pickle.dump(maf_empc_errs, f, pickle.HIGHEST_PROTOCOL)
                LOGINFO('saved {:s}'.format(fitpklsavpath))

        # fitfluxs, fittimes = _get_interp_fitfluxs(maf_empc_errs, sel_time)
        # now plot the phased lightcurve
        fitfluxs = maf_empc_errs['fitinfo']['fitmags']
        fittimes = maf_empc_errs['magseries']['times']
        fitperiod = maf_empc_errs['fitinfo']['finalparams']['period']
        fitepoch = maf_empc_errs['fitinfo']['finalparams']['t0']

        outfile = os.path.join(
            fit_savdir,
            "fancy_{}_phased_{}_fit_empiricalerrs.png".
            format(identifier, fittype)
        )

        tdur_phase = t_dur_day/fitperiod

        plot_phased_magseries(
            sel_time, sel_flux, fitperiod, magsarefluxes=True, errs=None,
            normto=False, epoch=fitepoch, outfile=outfile, sigclip=False,
            phasebin=0.01, phasewrap=True, phasesort=True,
            plotphaselim=[-n_transit_durations*tdur_phase,
                          n_transit_durations*tdur_phase],
            plotdpi=400, modelmags=fitfluxs, modeltimes=fittimes,
            xaxlabel='Time from mid-transit [days]', yaxlabel='Relative flux',
            xtimenotphase=True)

        LOGINFO('made {}'.format(outfile))

        #
        # check if we have converged chains.
        #
        autocorrtime = maf_empc_errs['fitinfo']['autocorrtime']

        if n_mcmc_steps < autocorrtime*50:
            # autocorrtime estimate can't be trusted if above condition isn't
            # true.  past that condition, you're probably "converged" (though
            # really each parameter takes different lengths of time to
            # converge).  see https://dfm.io/posts/autocorr/, or the emcee docs.
            # might actually be converged if shorter, but to be safe, say it is
            # not.
            is_converged = False

        else:
            is_converged = True

        return maf_empc_errs, is_converged


def fivetransitparam_fit_magseries(
        times, mags, errs,
        teff, rstar, logg,
        identifier,
        fit_savdir,
        chain_savdir,
        n_mcmc_steps=1,
        overwriteexistingsamples=False,
        burninpercent=0.3,
        n_transit_durations=5,
        make_tlsfit_plot=True,
        exp_time_minutes=30,
        bandpass='tess',
        magsarefluxes=True,
        nworkers=32,
):
    '''
    Wrapper to `mandelagol_fit_magseries` that fits out a line around each
    transit window in the given light curve, and then fits the entire light
    curve for (t0, period, a/Rstar, Rp/Rstar, inclination). Fixes e to 0, and
    uses theoretical quadratic limb darkening coefficients in the bandpass
    given by the user, as found with the stellar parameters. Figures out the
    priors for you.

    Typical use case: you have a light curve with >=2 transits in it.  You want
    to fit the entire light curve for the parameters noted above, but you don't
    want to need to manually determine all the priors.

    Parameters
    ----------

    times,mags,errs : np.array
        The input flux time-series to fit.

    teff,rstar,logg : float
        Stellar parameters [K, Rsun, cgs] used to get limb darkening
        coefficients.

    identifier : str
        String that goes into file names to identify the object being fit.
        E.g., fit CSV file will be at
        `{fit_savdir}/{identifier}_fivetransitparam_fitresults.csv`

    fit_savdir : str
        Path to directory where CSV results of fits, fit status files, and
        diagnostic plots are saved. If it doesn't exist, this function
        tries to make it.

    chain_savdir : str
        Path to directory where MCMC chains are saved.

    n_mcmc_steps : int
        Number of steps to run MCMC. (Note: convergence not guaranteed).

    overwriteexistingsamples : bool
        If False, and finds pickle file with saved parameters (in
        `fit_savdir`), no additoinal MCMC sampling is done.

    exp_time_minutes : int
        Exposure time in minutes. Used for the model fitting.

    n_transit_durations : int
        The points used in the fit are only those within +/- N transit
        durations of each transit mid-point. This is to prevent excessive
        out-of-transit data being used in the fit (these points do not
        inform the model's parameters).

    Returns
    -------

    (mafr, tlsr, is_converged) : tuple
        ``mafr`` is the Mandel-Agol fit result dictionary, which contains the
        same information as from ``mandelagol_and_line_fit_magseries``.  Fit
        parameters are accessed like
        ``maf_empc_errs['fitinfo']['finalparams']['sma']``,

        ``tlsr`` is the TLS result dictionary, containing keys documented in
        ``periodbase/htls.tls_parallel_pfind``.

        is_converged : boolean for whether the fitting converged, according
        to the chain autocorrelation time.

    '''

    if bandpass != 'tess':
        raise NotImplementedError(
            'currently only call Claret coefficients for '
            'tess bandpass. the others exist, just need to implement'
        )

    if not magsarefluxes:
        raise ValueError(
            'batman & TLS require mags to be fluxes'
        )

    #
    # Run TLS to get parameters that will be fed into transit model priors.
    # Note: the htls import needs to be in this function, not in the header, to
    # avoid recursive module imports upon initialization.
    #
    from astrobase.periodbase import htls

    tlsp = htls.tls_parallel_pfind(times, mags, errs,
                                   magsarefluxes=magsarefluxes,
                                   tls_rstar_min=0.1, tls_rstar_max=10,
                                   tls_mstar_min=0.1, tls_mstar_max=5.0,
                                   tls_oversample=8, tls_mintransits=1,
                                   tls_transit_template='default',
                                   nbestpeaks=5, sigclip=None,
                                   nworkers=nworkers)

    tlsr = tlsp['tlsresult']
    t0, per = tlsr.T0, tlsr.period

    #
    # Optionally make plot of TLS fit.
    #
    if make_tlsfit_plot:

        if not os.path.exists(fit_savdir):
            os.mkdir(fit_savdir)

        tlsfit_savfile = os.path.join(
            fit_savdir, 'tlsfit_{}_quickplot.png'.format(identifier)
        )

        make_fit_plot(tlsr['folded_phase'], tlsr['folded_y'], None,
                      tlsr['model_folded_model'], per, t0, t0, tlsfit_savfile,
                      model_over_lc=False, magsarefluxes=True,
                      fitphase=tlsr['model_folded_phase'])

        LOGINFO('made {}'.format(tlsfit_savfile))

    #
    # Fit the phased transit, within N durations of the transit itself, to
    # determine (t0, period, a/Rstar, Rp/Rstar, inclination). Most of the work
    # happens in this helper function. Returns Mandel-Agol fit results dict,
    # and whether we converged.
    #
    mafr, is_converged = (
        _fivetransitparam_fit_magseries(
            times, mags, errs,
            tlsr,
            teff, rstar, logg,
            identifier,
            fit_savdir,
            chain_savdir,
            exp_time_minutes=exp_time_minutes,
            n_transit_durations=n_transit_durations,
            nworkers=nworkers,
            n_mcmc_steps=n_mcmc_steps,
            burninpercent=burninpercent,
            overwriteexistingsamples=overwriteexistingsamples,
            mcmcprogressbar=True
        )
    )

    return mafr, tlsr, is_converged
