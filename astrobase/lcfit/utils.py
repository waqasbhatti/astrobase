#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py
# Waqas Bhatti and Luke Bouma - Feb 2017
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''This contains utilities for fitting routines in the rest of this subpackage.

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

import copy
from functools import partial

import numpy as np
from scipy.optimize import least_squares

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


########################################
## FUNCTIONS FOR SIMPLE LC OPERATIONS ##
########################################

def get_phased_quantities(stimes, smags, serrs, period):
    '''Does phase-folding for the mag/flux time-series given a period.

    Given finite and sigma-clipped times, magnitudes, and errors, along with the
    period at which to phase-fold the data, perform the phase-folding and
    return the phase-folded values.

    Parameters
    ----------

    stimes,smags,serrs : np.array
        The sigma-clipped and finite input mag/flux time-series arrays to
        operate on.

    period : float
        The period to phase the mag/flux time-series at. stimes.min() is used as
        the epoch value to fold the times-series around.

    Returns
    -------

    (phase, pmags, perrs, ptimes, mintime) : tuple
        The tuple returned contains the following items:

        - `phase`: phase-sorted values of phase at each of stimes
        - `pmags`: phase-sorted magnitudes at each phase
        - `perrs`: phase-sorted errors
        - `ptimes`: phase-sorted times
        - `mintime`: earliest time in stimes.

    '''

    # phase the mag series using the given period and faintest mag time
    # mintime = stimes[npwhere(smags == npmax(smags))]

    # phase the mag series using the given period and epoch = min(stimes)
    mintime = np.min(stimes)

    # calculate the unsorted phase, then sort it
    iphase = (stimes - mintime)/period - np.floor((stimes - mintime)/period)
    phasesortind = np.argsort(iphase)

    # these are the final quantities to use for the Fourier fits
    phase = iphase[phasesortind]
    pmags = smags[phasesortind]
    perrs = serrs[phasesortind]

    # get the times sorted in phase order (useful to get the fit mag minimum
    # with respect to phase -- the light curve minimum)
    ptimes = stimes[phasesortind]

    return phase, pmags, perrs, ptimes, mintime


########################
## PLOTTING UTILITIES ##
########################

def make_fit_plot(phase, pmags, perrs, fitmags,
                  period, mintime, magseriesepoch,
                  plotfit,
                  magsarefluxes=False,
                  wrap=False,
                  model_over_lc=True,
                  fitphase=None):
    '''This makes a plot of the LC model fit.

    Parameters
    ----------

    phase,pmags,perrs : np.array
        The actual mag/flux time-series.

    fitmags : np.array
        The model fit time-series.

    period : float
        The period at which the phased LC was generated.

    mintime : float
        The minimum time value.

    magseriesepoch : float
        The value of time around which the phased LC was folded.

    plotfit : str
        The name of a file to write the plot to.

    magsarefluxes : bool
        Set this to True if the values in `pmags` and `fitmags` are actually
        fluxes.

    wrap : bool
        If True, will wrap the phased LC around 0.0 to make some phased LCs
        easier to look at.

    model_over_lc : bool
        Usually, this function will plot the actual LC over the model LC. Set
        this to True to plot the model over the actual LC; this is most useful
        when you have a very dense light curve and want to be able to see how it
        follows the model.

    fitphase : optional np.array
        If passed, use this as x values for fitmags

    Returns
    -------

    Nothing.

    '''

    # set up the figure
    plt.close('all')
    plt.figure(figsize=(8,4.8))

    if model_over_lc:
        model_z = 100
        lc_z = 0
    else:
        model_z = 0
        lc_z = 100

    if not wrap:

        if isinstance(fitphase,np.ndarray):
            plt.plot(fitphase, fitmags, linewidth=3.0,
                     color='red',zorder=model_z)
        else:
            plt.plot(phase, fitmags, linewidth=3.0, color='red',zorder=model_z)
        plt.plot(phase,pmags,
                 marker='o',
                 markersize=1.0,
                 linestyle='none',
                 rasterized=True, color='k',zorder=lc_z)

        # set the x axis ticks and label
        plt.gca().set_xticks(
            [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        )

    else:
        plt.plot(np.concatenate([phase-1.0,phase]),
                 np.concatenate([fitmags,fitmags]),
                 linewidth=3.0,
                 color='red',zorder=model_z)
        plt.plot(np.concatenate([phase-1.0,phase]),
                 np.concatenate([pmags,pmags]),
                 marker='o',
                 markersize=1.0,
                 linestyle='none',
                 rasterized=True, color='k',zorder=lc_z)

        plt.gca().set_xlim((-0.8,0.8))
        # set the x axis ticks and label
        plt.gca().set_xticks(
            [-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,
             0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        )

    # set the y axis limit and label
    ymin, ymax = plt.ylim()
    if not magsarefluxes:
        plt.gca().invert_yaxis()
        plt.ylabel('magnitude')
    else:
        plt.ylabel('flux')

    plt.xlabel('phase')
    plt.title('period: %.6f, folded at %.6f, fit epoch: %.6f' %
              (period, mintime, magseriesepoch))
    plt.savefig(plotfit)
    plt.close()


#######################
## ITERATIVE FITTING ##
#######################

def iterative_fit(data_x,
                  data_y,
                  init_coeffs,
                  objective_func,
                  objective_args=None,
                  objective_kwargs=None,
                  optimizer_func=least_squares,
                  optimizer_kwargs=None,
                  optimizer_needs_scalar=False,
                  objective_residualarr_func=None,
                  fit_iterations=5,
                  fit_reject_sigma=3.0,
                  verbose=True,
                  full_output=False):
    '''This is a function to run iterative fitting based on repeated
    sigma-clipping of fit outliers.

    Parameters
    ----------

    data_x : np.array
        Array of the independent variable.

    data_y : np.array
        Array of the dependent variable.

    init_coeffs:
        The initial values of the fit function coefficients.

    objective_func : Python function
        A function that is used to calculate residuals between the model and the
        `data_y` array. This should have a signature similar to::

            def objective_func(fit_coeffs, data_x, data_y,
                               *objective_args, **objective_kwargs)

        and return an array of residuals or a scalar value indicating some sort
        of sum of residuals (depending on what the optimizer function
        requires).

        If this function returns a scalar value, you must set
        `optimizer_needs_scalar` to True, and provide a Python function in
        `objective_residualarr_func` that returns an array of residuals for each
        value of `data_x` and `data_y` given an array of fit coefficients.

    objective_args : tuple or None
        A tuple of arguments to pass into the `objective_func`.

    objective_kwargs : dict or None
        A dict of keyword arguments to pass into the `objective_func`.

    optimizer_func : Python function
        The function that minimizes the residual between the model and the
        `data_y` array using the `objective_func`. This should have a
        signature similar to one of the optimizer functions in `scipy.optimize
        <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_, i.e.::

            def optimizer_func(objective_func,
                               initial_coeffs,
                               args=(),
                               kwargs={},
                               ...)

        and return a `scipy.optimize.OptimizeResult
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_. We'll
        rely on the ``.success`` attribute to determine if the EPD fit was
        successful, and the ``.x`` attribute to get the values of the fit
        coefficients.

    optimizer_kwargs : dict or None
        A dict of kwargs to pass into the `optimizer_func` function.

    optimizer_needs_scalar : bool
        If True, this indicates that the optimizer requires a scalar value to be
        returned from the `objective_func`. This is the case for
        `scipy.optimize.minimize`. If this is True, you must also provide a
        function in `objective_residual_func`.

    objective_residualarr_func : Python function
        This is used in conjunction with `optimizer_needs_scalar`. The function
        provided here must return an array of residuals for each value of
        `data_x` and `data_y` given an array of fit coefficients. This is then
        used to calculate which points are outliers after a fit iteration. The
        function here must have the following signature::

            def objective_residualarr_func(coeffs, data_x, data_y,
                                           *objective_args, **objective_kwargs)

    fit_iterations : int
        The number of iterations of the fit to perform while throwing out
        outliers to the fit.

    fit_reject_sigma : float
        The maximum deviation allowed to consider a `data_y` item as an outlier
        to the fit and to remove it from consideration in a successive iteration
        of the fit.

    verbose : bool
        If True, reports per iteration on the cost function value and the number
        of items remaining in `data_x` and `data_y` after sigma-clipping
        outliers.

    full_output : bool
        If True, returns the full output from the `optimizer_func` along with
        the resulting fit function coefficients.

    Returns
    -------

    result : np.array or tuple
        If `full_output` was True, will return the fit coefficients np.array as
        the first element and the optimizer function fit output from the last
        iteration as the second element of a tuple. If `full_output` was False,
        will only return the final fit coefficients as an np.array.

    '''

    iteration_count = 0

    # paranoid copying for the input --- probably unnecessary but just in case
    coeffs = copy.deepcopy(init_coeffs)
    fit_data_x = copy.deepcopy(data_x)
    fit_data_y = copy.deepcopy(data_y)

    while iteration_count < fit_iterations:

        if not optimizer_kwargs:
            optimizer_kwargs = {}

        if not objective_args:
            obj_args = (fit_data_x, fit_data_y)
        else:
            obj_args = (fit_data_x, fit_data_y, *objective_args)

        if not objective_kwargs:
            obj_func = objective_func
        else:
            obj_func = partial(objective_func, **objective_kwargs)

        # set up the residualarr function if provided
        if objective_residualarr_func is not None and optimizer_needs_scalar:

            if not objective_kwargs:
                objective_resarr_func = objective_residualarr_func
            else:
                objective_resarr_func = partial(objective_residualarr_func,
                                                **objective_kwargs)

        fit_info = optimizer_func(
            obj_func,
            coeffs,
            args=obj_args,
            **optimizer_kwargs
        )

        # this handles the case where the optimizer is
        # scipy.optimize.least_squares
        if 'cost' in fit_info.keys():

            residual = fit_info.fun
            residual_median = np.nanmedian(residual)
            residual_mad = np.nanmedian(np.abs(residual - residual_median))
            residual_stdev = residual_mad*1.4826
            keep_ind = np.abs(residual) < residual_stdev*fit_reject_sigma

            fit_data_x = fit_data_x[keep_ind]
            fit_data_y = fit_data_y[keep_ind]
            coeffs = fit_info.x

            if verbose:
                LOGINFO(
                    "Fit success: %s for iteration: %s, "
                    "remaining items after sigma-clip: %s, "
                    "cost function value: %s" % (fit_info.success,
                                                 iteration_count,
                                                 keep_ind.sum(),
                                                 fit_info.cost)
                )

        # this handles the case where the optimizer is scipy.optimize.minimize
        # or similar
        elif ('cost' not in fit_info.keys() and
              (optimizer_needs_scalar and
               objective_residualarr_func is not None)):

            residual = objective_resarr_func(
                fit_info.x,
                *obj_args
            )
            residual_median = np.nanmedian(residual)
            residual_mad = np.nanmedian(np.abs(residual - residual_median))
            residual_stdev = residual_mad*1.4826
            keep_ind = np.abs(residual) < residual_stdev*fit_reject_sigma

            fit_data_x = fit_data_x[keep_ind]
            fit_data_y = fit_data_y[keep_ind]
            coeffs = fit_info.x

            if verbose:
                LOGINFO(
                    "Fit success: %s, for iteration: %s, "
                    "remaining items after sigma-clip: %s, "
                    "residual scalar value: %s" % (fit_info.success,
                                                   iteration_count,
                                                   keep_ind.sum(),
                                                   fit_info.fun)
                )

        else:

            LOGERROR("Fit did not succeed on iteration: %s" % iteration_count)

        iteration_count = iteration_count + 1

    # at the end, return the fit coeffs
    if not full_output:
        return fit_info.x
    else:
        return fit_info.x, fit_info
