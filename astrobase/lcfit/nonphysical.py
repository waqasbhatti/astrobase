#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nonphysical.py
# Waqas Bhatti and Luke Bouma - Feb 2017
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''Light curve fitting routines for 'non-physical' models:

- :py:func:`astrobase.lcfit.nonphysical.spline_fit_magseries`: fit a univariate
  cubic spline to a magnitude/flux time series with a specified spline knot
  fraction.

- :py:func:`astrobase.lcfit.nonphysical.savgol_fit_magseries`: apply a
  Savitzky-Golay smoothing filter to a magnitude/flux time series, returning the
  resulting smoothed function as a "fit".

- :py:func:`astrobase.lcfit.nonphysical.legendre_fit_magseries`: fit a Legendre
  function of the specified order to the magnitude/flux time series.

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
    sum as npsum, array as nparray, max as npmax, min as npmin,
    floor as npfloor, where as npwhere, linspace as nplinspace,
    full_like as npfull_like, nonzero as npnonzero, diff as npdiff,
    concatenate as npconcatenate
)

from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter
from numpy.polynomial.legendre import Legendre

from ..lcmath import sigclip_magseries
from .utils import get_phased_quantities, make_fit_plot


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
        get_phased_quantities(stimes, smags, serrs, period)
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
        get_phased_quantities(stimes, smags, serrs, period)
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
        get_phased_quantities(stimes, smags, serrs, period)
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
