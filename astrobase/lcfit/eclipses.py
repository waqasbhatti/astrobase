#!/usr/bin/env python3
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

from functools import partial

from numpy import (
    nan as npnan, sum as npsum, sqrt as npsqrt,
    nonzero as npnonzero, diag as npdiag, median as npmedian,
    inf as npinf, array as nparray
)

from scipy.optimize import curve_fit

from ..lcmath import sigclip_magseries
from ..lcmodels import eclipses

from .utils import make_fit_plot
from .nonphysical import spline_fit_magseries, savgol_fit_magseries


############################################
## DOUBLE INVERTED GAUSSIAN ECLIPSE MODEL ##
############################################

def gaussianeb_fit_magseries(
        times, mags, errs,
        ebparams,
        param_bounds=None,
        scale_errs_redchisq_unity=True,
        sigclip=10.0,
        plotfit=False,
        magsarefluxes=False,
        verbose=True,
        curve_fit_kwargs=None,
):
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

    param_bounds : dict or None
        This is a dict of the upper and lower bounds on each fit
        parameter. Should be of the form::

            {'period':         (lower_bound_period, upper_bound_period),
             'epoch':          (lower_bound_epoch, upper_bound_epoch),
             'pdepth':         (lower_bound_pdepth, upper_bound_pdepth),
             'pduration':      (lower_bound_pduration, upper_bound_pduration),
             'psdepthratio':   (lower_bound_psdepthratio,
                                upper_bound_psdepthratio),
             'secondaryphase': (lower_bound_secondaryphase,
                                upper_bound_secondaryphase)}

        - To indicate that a parameter is fixed, use 'fixed' instead of a tuple
          providing its lower and upper bounds as tuple.

        - To indicate that a parameter has no bounds, don't include it in the
          param_bounds dict.

        If this is None, the default value of this kwarg will be::

            {'period':(0.0,np.inf),      # period is between 0 and inf
             'epoch':(0.0, np.inf),      # epoch is between 0 and inf
             'pdepth':(-np.inf,np.inf),  # pdepth is between -np.inf and np.inf
             'pduration':(0.0,1.0),      # pduration is between 0.0 and 1.0
             'psdepthratio':(0.0,1.0),   # psdepthratio is between 0.0 and 1.0
             'secondaryphase':(0.0,1.0), # secondaryphase is between 0.0 and 1.0

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
                'fittype':'gaussianeb',
                'fitinfo':{
                    'initialparams':the initial EB params provided,
                    'finalparams':the final model fit EB params,
                    'finalparamerrs':formal errors in the params,
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
        except Exception:
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
                        'finalparamerrs':None,
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

        # set up the fit parameter bounds
        if param_bounds is None:

            curvefit_bounds = (
                nparray([0.0, 0.0, -npinf, 0.0, 0.0, 0.0]),
                nparray([npinf, npinf, npinf, 1.0, 1.0, 1.0])
            )
            fitfunc_fixed = {}

        else:

            # figure out the bounds
            lower_bounds = []
            upper_bounds = []
            fitfunc_fixed = {}

            for ind, key in enumerate(('period',
                                       'epoch',
                                       'pdepth',
                                       'pduration',
                                       'psdepthratio',
                                       'secondaryphase')):

                # handle fixed parameters
                if (key in param_bounds and
                    isinstance(param_bounds[key], str) and
                    param_bounds[key] == 'fixed'):

                    lower_bounds.append(ebparams[ind]-1.0e-7)
                    upper_bounds.append(ebparams[ind]+1.0e-7)
                    fitfunc_fixed[key] = ebparams[ind]

                # handle parameters with lower and upper bounds
                elif key in param_bounds and isinstance(param_bounds[key],
                                                        (tuple,list)):

                    lower_bounds.append(param_bounds[key][0])
                    upper_bounds.append(param_bounds[key][1])

                # handle no parameter bounds
                else:

                    lower_bounds.append(-npinf)
                    upper_bounds.append(npinf)

            # generate the bounds sequence in the required format
            curvefit_bounds = (
                nparray(lower_bounds),
                nparray(upper_bounds)
            )

        #
        # set up the curve fit function
        #
        curvefit_func = partial(eclipses.invgauss_eclipses_curvefit_func,
                                zerolevel=npmedian(smags),
                                fixed_params=fitfunc_fixed)

        #
        # run the fit
        #
        if curve_fit_kwargs is not None:

            finalparams, covmatrix = curve_fit(
                curvefit_func,
                stimes, smags,
                p0=ebparams,
                sigma=serrs,
                bounds=curvefit_bounds,
                absolute_sigma=(not scale_errs_redchisq_unity),
                **curve_fit_kwargs
            )

        else:

            finalparams, covmatrix = curve_fit(
                curvefit_func,
                stimes, smags,
                p0=ebparams,
                sigma=serrs,
                bounds=curvefit_bounds,
                absolute_sigma=(not scale_errs_redchisq_unity),
            )

    except Exception:
        LOGEXCEPTION("curve_fit returned an exception")
        finalparams, covmatrix = None, None

    # if the fit succeeded, then we can return the final parameters
    if finalparams is not None and covmatrix is not None:

        # calculate the chisq and reduced chisq
        fitmags, phase, ptimes, pmags, perrs = eclipses.invgauss_eclipses_func(
            finalparams,
            stimes, smags, serrs
        )
        fitchisq = npsum(
            ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
        )
        fitredchisq = fitchisq/(len(pmags) -
                                len(finalparams) -
                                len(fitfunc_fixed))

        stderrs = npsqrt(npdiag(covmatrix))

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
