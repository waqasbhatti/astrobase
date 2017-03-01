#!/usr/bin/env python

'''
varbase/lcfit.py
Waqas Bhatti and Luke Bouma - Feb 2017
(wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

Fitting routines for light curves. Includes:

- fourier_fit_magseries: fit an arbitrary order Fourier series to a
                         magnitude/flux time series.

- spline_fit_magseries: fit a univariate cubic spline to a magnitude/flux time
                        series with a specified spline knot fraction.

- savgol_fit_magseries: apply a Savitzky-Golay smoothing filter to a
                        magnitude/flux time series, returning the resulting
                        smoothed function as a "fit".

- legendre_fit_magseries: fit a Legendre function of the specified order to the
                          magnitude/flux time series.

TODO:
- Find correct dof for reduced chi squared in spline_fit_magseries
- Find correct dof for reduced chi squared in savgol_fit_magseries
'''

import logging
from datetime import datetime
from traceback import format_exc
from time import time as unixtime

from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, all as npall, \
    correlate as npcorrelate

from scipy.optimize import leastsq as spleastsq, minimize as spminimize
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter
from numpy.polynomial.legendre import Legendre, legval
import matplotlib.pyplot as plt


from ..lcmath import sigclip_magseries


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.lcfit' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


########################################
## FUNCTIONS FOR SIMPLE LC OPERATIONS ##
########################################

def _get_phased_quantities(stimes, smags, serrs, period):
    '''
    Given finite and sigma-clipped times, magnitudes, and errors (i.e. the
    output of lcfit.get_finite_and_sigclipped_data), along with the period at
    which to phase-fold the data, perform the phase-folding and return:
        1) phase: phase-sorted values of phase at each of stimes
        2) pmags: phase-sorted magnitudes at each phase
        3) perrs: phase-sorted errors
        4) ptimes: phase-sorted times
        5) mintime: earliest time in stimes.
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

def _make_fit_plot(phase, pmags, perrs, fitmags,
                   period, mintime, magseriesepoch,
                   plotfit,
                   magsarefluxes=False):

    # set up the figure
    plt.close('all')
    plt.figure(figsize=(8,6))
    plt.axvline(0.5,color='g',linestyle='--')

    # plot the light curve and the fit
    plt.errorbar(phase,pmags,fmt='bo',yerr=perrs,
                 markersize=2.0,capsize=0)
    plt.plot(phase,fitmags, 'r-',linewidth=2.0)

    # set the y axis limit and label
    ymin, ymax = plt.ylim()
    if not magsarefluxes:
        plt.ylim(ymax,ymin)
        plt.ylabel('magnitude')
    else:
        plt.ylim(ymin, ymax)
        plt.ylabel('flux')

    # set the x axis ticks and label
    plt.gca().set_xticks(
        [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    )
    plt.xlabel('phase')

    plt.title('period: %.6f, folded at %.6f, fit epoch: %.6f' %
              (period, mintime, magseriesepoch))
    plt.savefig(plotfit)
    plt.close()




#####################################################
## FOURIER FITTING TO PHASED MAGNITUDE TIME SERIES ##
#####################################################

def _fourier_func(fourierparams, phase, mags):
    '''
    This returns a summed Fourier series generated using fourierparams.

    fourierparams is a sequence like so:

    [ampl_1, ampl_2, ampl_3, ..., ampl_X,
     pha_1, pha_2, pha_3, ..., pha_X]

    where X is the Fourier order.

    mags and phase MUST NOT have any nans.

    '''

    # figure out the order from the length of the Fourier param list
    order = int(len(fourierparams)/2)

    # get the amplitude and phase coefficients
    f_amp = fourierparams[:order]
    f_pha = fourierparams[order:]

    # calculate all the individual terms of the series
    f_orders = [f_amp[x]*npcos(2.0*MPI*x*phase + f_pha[x])
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
    '''
    This is the chisq objective function to be minimized by scipy.minimize.

    The parameters are the same as _fourier_func above.

    '''

    f = _fourier_func(fourierparams, phase, mags)
    chisq = npsum(((mags - f)*(mags - f))/(errs*errs))

    return chisq



def _fourier_residual(fourierparams,
                      phase,
                      mags):
    '''
    This is the residual objective function to be minimized by scipy.leastsq.

    The parameters are the same as _fourier_func above.

    '''

    f = _fourier_func(fourierparams, phase, mags)
    residual = mags - f

    return residual



def fourier_fit_magseries(times, mags, errs, period,
                          initfourierparams=[0.6,0.2,0.2,0.2,0.2,0.2,0.2,0.2,
                                             0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                          sigclip=3.0,
                          magsarefluxes=False,
                          plotfit=False,
                          ignoreinitfail=True):
    '''This fits a Fourier series to a magnitude time series.

    This uses an 8th-order Fourier series by default. This is good for light
    curves with many thousands of observations (HAT light curves have ~10k
    observations). Lower the order accordingly if you have fewer observations in
    your light curves to avoid over-fitting.

    Returns the Fourier fit parameters, the minimum chisq and reduced
    chisq. Makes a plot for the fit to the mag series if plotfit is a string
    containing a filename to write the plot to.

    This folds the time series using the given period and at the first
    observation. Can optionally sigma-clip observations.

    if ignoreinitfail is True, ignores the initial failure to find a set of
    optimized Fourier parameters and proceeds to do a least-squares fit anyway.

    magsarefluxes is a boolean value for setting the ylabel and ylimits of
    plots for either magnitudes (False) or flux units (i.e. normalized to 1, in
    which case magsarefluxes should be set to True).

    '''

    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

    phase, pmags, perrs, ptimes, mintime = (
            _get_phased_quantities(stimes, smags, serrs, period)
        )


    fourierorder = int(len(initfourierparams)/2)

    LOGINFO('fitting Fourier series of order %s to '
            'mag series with %s observations, '
            'using period %.6f, folded at %.6f' % (fourierorder,
                                                   len(phase),
                                                   period,
                                                   mintime))

    # initial minimize call to find global minimum in chi-sq
    initialfit = spminimize(_fourier_chisq,
                            initfourierparams,
                            method='BFGS',
                            args=(phase, pmags, perrs))

    # make sure this initial fit succeeds before proceeding
    if initialfit.success or ignoreinitfail:

        LOGINFO('initial fit done, refining...')

        leastsqparams = initialfit.x

        leastsqfit = spleastsq(_fourier_residual,
                               leastsqparams,
                               args=(phase, pmags))

        # if the fit succeeded, then we can return the final parameters
        if leastsqfit[-1] in (1,2,3,4):

            finalparams = leastsqfit[0]

            # calculate the chisq and reduced chisq
            fitmags = _fourier_func(finalparams, phase, pmags)

            fitchisq = npsum(
                ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
            )

            fitredchisq = fitchisq/(len(pmags) - len(finalparams) - 1)

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
            magseriesepoch = ptimes[fitmagminind]

            # assemble the returndict
            returndict =  {
                'fittype':'fourier',
                'fitinfo':{
                    'fourierorder':fourierorder,
                    'finalparams':finalparams,
                    'initialfit':initialfit,
                    'leastsqfit':leastsqfit,
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

                _make_fit_plot(phase, pmags, perrs, fitmags,
                               period, mintime, magseriesepoch,
                               plotfit,
                               magsarefluxes=magsarefluxes)

                returndict['fitplotfile'] = plotfit

            return returndict


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
            'fitchisq':None,
            'fitredchisq':None,
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
                         maxknots=100,
                         sigclip=30.0,
                         plotfit=False,
                         ignoreinitfail=False,
                         magsarefluxes=False):

    '''This fits a univariate cubic spline to the phased light curve.

    This fit may be better than the Fourier fit for sharply variable objects,
    like EBs, so can be used to distinguish them from other types of variables.

    The knot fraction is the number of internal knots to use for the spline. A
    value of 0.01 (or 1%) of the total number of non-nan observations appears to
    work quite well, without over-fitting.

    magsarefluxes is a boolean value for setting the ylabel and ylimits of
    plots for either magnitudes (False) or flux units (i.e. normalized to 1, in
    which case magsarefluxes should be set to True).

    Returns the chisq of the fit, as well as the reduced chisq. FIXME: check
    this equation below to see if it's right.

    reduced_chisq = fit_chisq/(len(pmags) - len(knots) - 1)

    '''

    # this is required to fit the spline correctly
    if errs is None:
        errs = npfull_like(mags, 0.005)

    # sigclip the magnitude time series
    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

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

    # generate and fit the spline
    spl = LSQUnivariateSpline(phase,pmags,t=splineknots,w=1.0/perrs)

    # calculate the spline fit to the actual phases, the chisq and red-chisq
    fitmags = spl(phase)

    fitchisq = npsum(
        ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
    )

    fitredchisq = fitchisq/(len(pmags) - nknots - 1)

    LOGINFO(
        'spline fit done. nknots = %s,  chisq = %.5f, reduced chisq = %.5f' %
        (nknots, fitchisq, fitredchisq)
    )

    # figure out the time of light curve minimum (i.e. the fit epoch)
    # this is when the fit mag is maximum (i.e. the faintest)
    # or if magsarefluxes = True, then this is when fit flux is minimum
    if not magsarefluxes:
        fitmagminind = npwhere(fitmags == npmax(fitmags))
    else:
        fitmagminind = npwhere(fitmags == npmin(fitmags))
    magseriesepoch = ptimes[fitmagminind]

    # assemble the returndict
    returndict =  {
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

        _make_fit_plot(phase, pmags, perrs, fitmags,
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
                         magsarefluxes=False):

    '''
    Fit a Savitzky-Golay filter to the magnitude/flux time series.
    SG fits successive sub-sets (windows) of adjacent data points with a
    low-order polynomial via least squares. At each point (magnitude),
    it returns the value of the polynomial at that magnitude's time.
    This is made significantly cheaper than *actually* performing least squares
    for each window through linear algebra tricks that are possible when
    specifying the window size and polynomial order beforehand.
    Numerical Recipes Ch 14.8 gives an overview, Eq. 14.8.6 is what Scipy has
    implemented.

    The idea behind Savitzky-Golay is to preserve higher moments (>=2) of the
    input data series than would be done by a simple moving window average.

    Note that the filter assumes evenly spaced data, which magnitude time
    series are not. By *pretending* the data points are evenly spaced, we
    introduce an additional noise source in the function values. This is a
    relatively small noise source provided that the changes in the magnitude
    values across the full width of the N=windowlength point window is <
    sqrt(N/2) times the measurement noise on a single point.

    Args:

    windowlength (int): length of the filter window (the number of
    coefficients). Must be either positive and odd, or None. (The window is
    the number of points to the left, and to the right, of whatever point is
    having a polynomial fit to it locally). Bigger windows at fixed polynomial
    order risk lowering the amplitude of sharp features. If None, this routine
    (arbitrarily) sets the windowlength for phased LCs to be either the number
    of finite data points divided by 300, or polydeg+3, whichever is bigger.

    polydeg (int): the order of the polynomial used to fit the samples. Must
    be less than windowlength. "Higher-order filters do better at preserving
    feature heights and widths, but do less smoothing on broader features."
    (NumRec).

    magsarefluxes (bool): sets the ylabel and ylimits of plots for either
    magnitudes (False) or flux units (i.e. normalized to 1, in which case
    magsarefluxes should be set to True).

    '''
    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

    phase, pmags, perrs, ptimes, mintime = (
            _get_phased_quantities(stimes, smags, serrs, period)
        )

    if not isinstance(windowlength, int):
        windowlength = max(
                polydeg+3,
                int(len(phase)/300)
                )
        if windowlength % 2 == 0:
            windowlength += 1

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

        _make_fit_plot(phase, pmags, perrs, fitmags,
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
                           magsarefluxes=False):

    '''
    Fit an arbitrary-order Legendre series, via least squares, to the
    magnitude/flux time series. This is a series of the form:

        p(x) = c_0*L_0(x) + c_1*L_1(x) + c_2*L_2(x) + ... + c_n*L_n(x)

    where L_i's are Legendre polynomials (also caleld "Legendre functions of
    the first kind") and c_i's are the coefficients being fit.

    Args:

    legendredeg (int): n in the above equation. (I.e., if you give n=5, you
    will get 6 coefficients). This number should be much less than the number
    of data points you are fitting.

    sigclip (float): number of standard deviations away from the mean of the
    magnitude time-series from which to "clip" data points.

    magsarefluxes (bool): sets the ylabel and ylimits of plots for either
    magnitudes (False) or flux units (i.e. normalized to 1, in which case
    magsarefluxes should be set to True).

    Returns:

    returndict:
    {
        'fittype':'legendre',
        'fitinfo':{
            'legendredeg':legendredeg,
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
            'magsarefluxes':magsarefluxes},
    }

    where `fitmags` is the values of the fit function interpolated onto
    magseries' `phase`.

    This function is mainly just a wrapper to
    numpy.polynomial.legendre.Legendre.fit.

    '''
    stimes, smags, serrs = sigclip_magseries(times, mags, errs,
                                             sigclip=sigclip,
                                             magsarefluxes=magsarefluxes)

    phase, pmags, perrs, ptimes, mintime = (
            _get_phased_quantities(stimes, smags, serrs, period)
        )


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
    fitmags = p(phase)

    # Now compute the chisq and red-chisq.

    fitchisq = npsum(
        ((fitmags - pmags)*(fitmags - pmags)) / (perrs*perrs)
    )

    nparams = legendredeg + 1
    fitredchisq = fitchisq/(len(pmags) - nparams - 1)

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
    magseriesepoch = ptimes[fitmagminind]

    # assemble the returndict
    returndict = {
        'fittype':'legendre',
        'fitinfo':{
            'legendredeg':legendredeg,
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

        _make_fit_plot(phase, pmags, perrs, fitmags,
                       period, mintime, magseriesepoch,
                       plotfit,
                       magsarefluxes=magsarefluxes)

        returndict['fitplotfile'] = plotfit

    return returndict
