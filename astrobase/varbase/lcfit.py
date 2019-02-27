#!/usr/bin/env python
# -*- coding: utf-8 -*-
# varbase/lcfit.py
# Waqas Bhatti and Luke Bouma - Feb 2017
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''Fitting routines for light curves. Includes:

- :py:func:`astrobase.varbase.lcfit.fourier_fit_magseries`: fit an arbitrary
  order Fourier series to a magnitude/flux time series.

- :py:func:`astrobase.varbase.lcfit.spline_fit_magseries`: fit a univariate
  cubic spline to a magnitude/flux time series with a specified spline knot
  fraction.

- :py:func:`astrobase.varbase.lcfit.savgol_fit_magseries`: apply a
  Savitzky-Golay smoothing filter to a magnitude/flux time series, returning the
  resulting smoothed function as a "fit".

- :py:func:`astrobase.varbase.lcfit.legendre_fit_magseries`: fit a Legendre
  function of the specified order to the magnitude/flux time series.

- :py:func:`astrobase.varbase.lcfit.traptransit_fit_magseries`: fit a
  trapezoid-shaped transit signal to the magnitude/flux time series

- :py:func:`astrobase.varbase.lcfit.gaussianeb_fit_magseries`: fit a double
  inverted gaussian eclipsing binary model to the magnitude/flux time series

- :py:func:`astrobase.varbase.lcfit.mandelagol_fit_magseries`: fit a Mandel &
  Agol (2002) planet transit model to the flux time series.

- :py:func:`astrobase.varbase.lcfit.mandelagol_and_line_fit_magseries`: fit a
  Mandel & Agol 2002 model, + a local line to the flux time series.

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

import os

import numpy as np
from numpy import (
    nan as npnan, sum as npsum, sqrt as npsqrt, median as npmedian,
    array as nparray, max as npmax, min as npmin, pi as pi_value,
    floor as npfloor, argsort as npargsort, cos as npcos, where as npwhere,
    linspace as nplinspace, full_like as npfull_like, nonzero as npnonzero,
    diag as npdiag, diff as npdiff, concatenate as npconcatenate
)

from scipy.optimize import leastsq as spleastsq, minimize as spminimize
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter
from numpy.polynomial.legendre import Legendre

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import batman
    import emcee
    import corner

    if int(emcee.__version__[0]) >= 3:
        mandel_agol_dependencies = True
    else:
        mandel_agol_dependencies = False

except Exception as e:
    mandel_agol_dependencies = False

from ..lcmath import sigclip_magseries

# import the models
from ..lcmodels import eclipses, transits


########################################
## FUNCTIONS FOR SIMPLE LC OPERATIONS ##
########################################

def _get_phased_quantities(stimes, smags, serrs, period):
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
    mintime = npmin(stimes)

    # calculate the unsorted phase, then sort it
    iphase = (stimes - mintime)/period - npfloor((stimes - mintime)/period)
    phasesortind = npargsort(iphase)

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
                  model_over_lc=False):
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



def _fourier_residual(fourierparams,
                      phase,
                      mags):
    '''
    This is the residual objective function to be minimized by `scipy.leastsq`.

    The parameters are the same as `_fourier_func` above.

    '''

    f = _fourier_func(fourierparams, phase, mags)
    residual = mags - f

    return residual



def fourier_fit_magseries(times, mags, errs, period,
                          fourierorder=None,
                          fourierparams=None,
                          sigclip=3.0,
                          magsarefluxes=False,
                          plotfit=False,
                          ignoreinitfail=True,
                          verbose=True):
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
                'fittype':'fourier',
                'fitinfo':{
                    'finalparams': the list of final model fit params,
                    'leastsqfit':the full tuple returned by scipy.leastsq,
                    'fitmags': the model fit mags,
                    'fitepoch': the epoch of minimum light for the fit,
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
        _get_phased_quantities(stimes, smags, serrs, period)
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
                            method='BFGS',
                            args=(phase, pmags, perrs))

    # make sure this initial fit succeeds before proceeding
    if initialfit.success or ignoreinitfail:

        if verbose:
            LOGINFO('initial fit done, refining...')

        leastsqparams = initialfit.x

        try:
            leastsqfit = spleastsq(_fourier_residual,
                                   leastsqparams,
                                   args=(phase, pmags))
        except Exception as e:
            leastsqfit = None

        # if the fit succeeded, then we can return the final parameters
        if leastsqfit and leastsqfit[-1] in (1,2,3,4):

            finalparams = leastsqfit[0]

            # calculate the chisq and reduced chisq
            fitmags = _fourier_func(finalparams, phase, pmags)

            fitchisq = npsum(
                ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
            )

            fitredchisq = fitchisq/(len(pmags) - len(finalparams) - 1)

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
                    'finalparams':finalparams,
                    'initialfit':initialfit,
                    'leastsqfit':leastsqfit,
                    'fitmags':fitmags,
                    'fitepoch':mintime,
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
                              period, mintime, mintime,
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
                    'initialfit':initialfit,
                    'leastsqfit':None,
                    'fitmags':None,
                    'fitepoch':None
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
                'initialfit':initialfit,
                'leastsqfit':None,
                'fitmags':None,
                'fitepoch':None
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


#################################################################
## SPLINE FITTING TO PHASED AND UNPHASED MAGNITUDE TIME SERIES ##
#################################################################

def spline_fit_magseries(times, mags, errs, period,
                         knotfraction=0.01,
                         maxknots=30,
                         sigclip=30.0,
                         plotfit=False,
                         ignoreinitfail=False,
                         magsarefluxes=False,
                         verbose=True):

    '''This fits a univariate cubic spline to the phased light curve.

    This fit may be better than the Fourier fit for sharply variable objects,
    like EBs, so can be used to distinguish them from other types of variables.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to fit a spline to.

    period : float
        The period to use for the spline fit.

    knotfraction : float
        The knot fraction is the number of internal knots to use for the
        spline. A value of 0.01 (or 1%) of the total number of non-nan
        observations appears to work quite well, without over-fitting. maxknots
        controls the maximum number of knots that will be allowed.

    maxknots : int
        The maximum number of knots that will be used even if `knotfraction`
        gives a value to use larger than `maxknots`. This helps dealing with
        over-fitting to short time-scale variations.

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
                'fittype':'spline',
                'fitinfo':{
                    'nknots': the number of knots used for the fit
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

    # this is required to fit the spline correctly
    if errs is None:
        errs = npfull_like(mags, 0.005)

    # sigclip the magnitude time series
    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)
    # get rid of zero errs
    nzind = npnonzero(serrs)
    stimes, smags, serrs = stimes[nzind], smags[nzind], serrs[nzind]

    # phase the mag series
    phase, pmags, perrs, ptimes, mintime = (
        _get_phased_quantities(stimes, smags, serrs, period)
    )

    # now figure out the number of knots up to max knots (=100)
    nobs = len(phase)
    nknots = int(npfloor(knotfraction*nobs))
    nknots = maxknots if nknots > maxknots else nknots
    splineknots = nplinspace(phase[0] + 0.01,
                             phase[-1] - 0.01,
                             num=nknots)

    # NOTE: newer scipy needs x to be strictly increasing. this means we should
    # filter out anything that doesn't have np.diff(phase) > 0.0
    # FIXME: this needs to be tested
    phase_diffs_ind = npdiff(phase) > 0.0
    incphase_ind = npconcatenate((nparray([True]), phase_diffs_ind))
    phase, pmags, perrs = (phase[incphase_ind],
                           pmags[incphase_ind],
                           perrs[incphase_ind])

    # generate and fit the spline
    spl = LSQUnivariateSpline(phase, pmags, t=splineknots, w=1.0/perrs)

    # calculate the spline fit to the actual phases, the chisq and red-chisq
    fitmags = spl(phase)

    fitchisq = npsum(
        ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
    )

    fitredchisq = fitchisq/(len(pmags) - nknots - 1)

    if verbose:
        LOGINFO(
            'spline fit done. nknots = %s,  '
            'chisq = %.5f, reduced chisq = %.5f' %
            (nknots, fitchisq, fitredchisq)
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
    magseriesepoch = ptimes[fitmagminind]

    # assemble the returndict
    returndict = {
        'fittype':'spline',
        'fitinfo':{
            'nknots':nknots,
            'fitmags':fitmags,
            'fitepoch':magseriesepoch
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
                      period, mintime, magseriesepoch,
                      plotfit,
                      magsarefluxes=magsarefluxes)

        returndict['fitplotfile'] = plotfit

    return returndict



#####################################################
## SAVITZKY-GOLAY FITTING TO MAGNITUDE TIME SERIES ##
#####################################################

def savgol_fit_magseries(times, mags, errs, period,
                         windowlength=None,
                         polydeg=2,
                         sigclip=30.0,
                         plotfit=False,
                         magsarefluxes=False,
                         verbose=True):

    '''Fit a Savitzky-Golay filter to the magnitude/flux time series.

    SG fits successive sub-sets (windows) of adjacent data points with a
    low-order polynomial via least squares. At each point (magnitude), it
    returns the value of the polynomial at that magnitude's time.  This is made
    significantly cheaper than *actually* performing least squares for each
    window through linear algebra tricks that are possible when specifying the
    window size and polynomial order beforehand.  Numerical Recipes Ch 14.8
    gives an overview, Eq. 14.8.6 is what Scipy has implemented.

    The idea behind Savitzky-Golay is to preserve higher moments (>=2) of the
    input data series than would be done by a simple moving window average.

    Note that the filter assumes evenly spaced data, which magnitude time series
    are not. By *pretending* the data points are evenly spaced, we introduce an
    additional noise source in the function values. This is a relatively small
    noise source provided that the changes in the magnitude values across the
    full width of the N=windowlength point window is < sqrt(N/2) times the
    measurement noise on a single point.

    TODO:
    - Find correct dof for reduced chi squared in savgol_fit_magseries

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to fit the Savitsky-Golay model to.

    period : float
        The period to use for the model fit.

    windowlength : None or int
        The length of the filter window (the number of coefficients). Must be
        either positive and odd, or None. (The window is the number of points to
        the left, and to the right, of whatever point is having a polynomial fit
        to it locally). Bigger windows at fixed polynomial order risk lowering
        the amplitude of sharp features. If None, this routine (arbitrarily)
        sets the `windowlength` for phased LCs to be either the number of finite
        data points divided by 300, or polydeg+3, whichever is bigger.

    polydeg : int
        This is the order of the polynomial used to fit the samples.  Must be
        less than `windowlength`. "Higher-order filters do better at preserving
        feature heights and widths, but do less smoothing on broader features."
        (Numerical Recipes).

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
                'fittype':'savgol',
                'fitinfo':{
                    'windowlength': the window length used for the fit,
                    'polydeg':the polynomial degree used for the fit,
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

    phase, pmags, perrs, ptimes, mintime = (
        _get_phased_quantities(stimes, smags, serrs, period)
    )

    if not isinstance(windowlength, int):
        windowlength = max(
            polydeg + 3,
            int(len(phase)/300)
        )
        if windowlength % 2 == 0:
            windowlength += 1

    if verbose:
        LOGINFO('applying Savitzky-Golay filter with '
                'window length %s and polynomial degree %s to '
                'mag series with %s observations, '
                'using period %.6f, folded at %.6f' % (windowlength,
                                                       polydeg,
                                                       len(pmags),
                                                       period,
                                                       mintime))

    # generate the function values obtained by applying the SG filter. The
    # "wrap" option is best for phase-folded LCs.
    sgf = savgol_filter(pmags, windowlength, polydeg, mode='wrap')

    # here the "fit" to the phases is the function produced by the
    # Savitzky-Golay filter. then compute the chisq and red-chisq.
    fitmags = sgf

    fitchisq = npsum(
        ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
    )

    # TODO: quantify dof for SG filter.
    nparams = int(len(pmags)/windowlength) * polydeg
    fitredchisq = fitchisq/(len(pmags) - nparams - 1)
    fitredchisq = -99.

    if verbose:
        LOGINFO(
            'SG filter applied. chisq = %.5f, reduced chisq = %.5f' %
            (fitchisq, fitredchisq)
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
    magseriesepoch = ptimes[fitmagminind]

    # assemble the returndict
    returndict = {
        'fittype':'savgol',
        'fitinfo':{
            'windowlength':windowlength,
            'polydeg':polydeg,
            'fitmags':fitmags,
            'fitepoch':magseriesepoch
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
        }
    }

    # make the fit plot if required
    if plotfit and isinstance(plotfit, str):

        make_fit_plot(phase, pmags, perrs, fitmags,
                      period, mintime, magseriesepoch,
                      plotfit,
                      magsarefluxes=magsarefluxes)

        returndict['fitplotfile'] = plotfit

    return returndict


##########################################################
## LEGENDRE-POLYNOMIAL FITTING TO MAGNITUDE TIME SERIES ##
##########################################################

def legendre_fit_magseries(times, mags, errs, period,
                           legendredeg=10,
                           sigclip=30.0,
                           plotfit=False,
                           magsarefluxes=False,
                           verbose=True):

    '''Fit an arbitrary-order Legendre series, via least squares, to the
    magnitude/flux time series.

    This is a series of the form::

        p(x) = c_0*L_0(x) + c_1*L_1(x) + c_2*L_2(x) + ... + c_n*L_n(x)

    where L_i's are Legendre polynomials (also called "Legendre functions of the
    first kind") and c_i's are the coefficients being fit.

    This function is mainly just a wrapper to
    `numpy.polynomial.legendre.Legendre.fit`.

    Parameters
    ----------

    times,mags,errs : np.array
        The input mag/flux time-series to fit a Legendre series polynomial to.

    period : float
        The period to use for the Legendre fit.

    legendredeg : int
        This is `n` in the equation above, e.g. if you give `n=5`, you will
        get 6 coefficients. This number should be much less than the number of
        data points you are fitting.

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
                'fittype':'legendre',
                'fitinfo':{
                    'legendredeg': the Legendre polynomial degree used,
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


    phase, pmags, perrs, ptimes, mintime = (
        _get_phased_quantities(stimes, smags, serrs, period)
    )

    if verbose:
        LOGINFO('fitting Legendre series with '
                'maximum Legendre polynomial order %s to '
                'mag series with %s observations, '
                'using period %.6f, folded at %.6f' % (legendredeg,
                                                       len(pmags),
                                                       period,
                                                       mintime))

    # Least squares fit of Legendre polynomial series to the data. The window
    # and domain (see "Using the Convenience Classes" in the numpy
    # documentation) are handled automatically, scaling the times to a minimal
    # domain in [-1,1], in which Legendre polynomials are a complete basis.

    p = Legendre.fit(phase, pmags, legendredeg)
    coeffs = p.coef
    fitmags = p(phase)

    # Now compute the chisq and red-chisq.

    fitchisq = npsum(
        ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
    )

    nparams = legendredeg + 1
    fitredchisq = fitchisq/(len(pmags) - nparams - 1)

    if verbose:
        LOGINFO(
            'Legendre fit done. chisq = %.5f, reduced chisq = %.5f' %
            (fitchisq, fitredchisq)
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
    magseriesepoch = ptimes[fitmagminind]

    # assemble the returndict
    returndict = {
        'fittype':'legendre',
        'fitinfo':{
            'legendredeg':legendredeg,
            'fitmags':fitmags,
            'fitepoch':magseriesepoch,
            'finalparams':coeffs,
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
        }
    }

    # make the fit plot if required
    if plotfit and isinstance(plotfit, str):

        make_fit_plot(phase, pmags, perrs, fitmags,
                      period, mintime, magseriesepoch,
                      plotfit,
                      magsarefluxes=magsarefluxes)

        returndict['fitplotfile'] = plotfit

    return returndict


###############################################
## TRAPEZOID TRANSIT MODEL FIT TO MAG SERIES ##
###############################################

def traptransit_fit_magseries(times, mags, errs,
                              transitparams,
                              sigclip=10.0,
                              plotfit=False,
                              magsarefluxes=False,
                              verbose=True):
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

            transitparams = [transitperiod (time),
                             transitepoch (time),
                             transitdepth (flux or mags),
                             transitduration (phase),
                             ingressduration (phase)]

        - for magnitudes -> `transitdepth` should be < 0
        - for fluxes     -> `transitdepth` should be > 0

        If `transitepoch` is None, this function will do an initial spline fit
        to find an approximate minimum of the phased light curve using the given
        period.

        The `transitdepth` provided is checked against the value of
        `magsarefluxes`. if `magsarefluxes = True`, the `transitdepth` is forced
        to be > 0; if `magsarefluxes` = False, the `transitdepth` is forced to
        be < 0.

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
                'fittype':'traptransit',
                'fitinfo':{
                    'initialparams':the initial transit params provided,
                    'finalparams':the final model fit transit params ,
                    'finalparamerrs':formal errors in the params,
                    'leastsqfit':the full tuple returned by scipy.leastsq,
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
    nzind = npnonzero(serrs)
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
        except Exception as e:
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
        leastsqfit = spleastsq(transits.trapezoid_transit_residual,
                               transitparams,
                               args=(stimes, smags, serrs),
                               full_output=True)
    except Exception as e:
        leastsqfit = None

    # if the fit succeeded, then we can return the final parameters
    if leastsqfit and leastsqfit[-1] in (1,2,3,4):

        finalparams = leastsqfit[0]
        covxmatrix = leastsqfit[1]

        # calculate the chisq and reduced chisq
        fitmags, phase, ptimes, pmags, perrs, n_transitpoints = (
            transits.trapezoid_transit_func(
                finalparams,
                stimes, smags, serrs,
                get_ntransitpoints=True
            )
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
            'fittype':'traptransit',
            'fitinfo':{
                'initialparams':transitparams,
                'finalparams':finalparams,
                'finalparamerrs':stderrs,
                'leastsqfit':leastsqfit,
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
                'leastsqfit':leastsqfit,
                'fitmags':None,
                'fitepoch':None,
                'ntransitpoints':0
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
    except Exception as e:
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
    '''This fits a Mandel & Agol (2002) planetary transit model to a flux time
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
    nzind = npnonzero(serrs)
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
                                         init_omega, init_u, limb_dark)
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

        import sys

        if sys.version_info >= (3, 3):
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

        elif sys.version_info < (3, 3):

            sampler = emcee.EnsembleSampler(
                n_walkers, n_dim, log_posterior_transit,
                args=(init_params, init_m, stimes, smags, serrs, priorbounds),
                threads=nworkers,
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
    log_prob_samples = reader.get_log_prob(discard=n_to_discard, flat=True)
    log_prior_samples = reader.get_blobs(discard=n_to_discard, flat=True)

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
    except Exception as e:
        u = [medianparams['u_linear'], medianparams['u_quad']]

    fit_params, fit_m = _transit_model(stimes, t0, per, rp, sma, incl, ecc,
                                       omega, u, limb_dark)
    fitmags = fit_m.light_curve(fit_params)
    fepoch = t0

    # assemble the return dictionary
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

    A typical use case: you want to measure transit times of individual SNR >~
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
    nzind = npnonzero(serrs)
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
        init_ecc, init_omega, init_u, limb_dark)

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
                                         limb_dark)
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
    log_prob_samples = reader.get_log_prob(discard=n_to_discard, flat=True)
    log_prior_samples = reader.get_blobs(discard=n_to_discard, flat=True)

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
    except Exception as e:
        u = [medianparams['u_linear'], medianparams['u_quad']]

    poly_order0 = _get_value('poly_order0', medianparams, fixedparams)
    poly_order1 = _get_value('poly_order1', medianparams, fixedparams)

    # initialize the model and calculate the initial model light-curve
    fit_params, fit_m = _transit_model(stimes, t0, per, rp, sma, incl, ecc,
                                       omega, u, limb_dark)
    fitmags = (
        fit_m.light_curve(fit_params) +
        poly_order0 + poly_order1*stimes
    )
    fepoch = t0

    # assemble the return dictionary
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
