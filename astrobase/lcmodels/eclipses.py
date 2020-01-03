#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# eclipses.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017

'''
This contains a double gaussian model for first order modeling of eclipsing
binaries.

'''

import numpy as np


##################################
## MODEL AND RESIDUAL FUNCTIONS ##
##################################

def _gaussian(x, amp, loc, std):
    '''This is a simple gaussian.

    Parameters
    ----------

    x : np.array
        The items at which the Gaussian is evaluated.

    amp : float
        The amplitude of the Gaussian.

    loc : float
        The central value of the Gaussian.

    std : float
        The standard deviation of the Gaussian.

    Returns
    -------

    np.array
        Returns the Gaussian evaluated at the items in `x`, using the provided
        parameters of `amp`, `loc`, and `std`.

    '''

    return amp * np.exp(-((x - loc)*(x - loc))/(2.0*std*std))


def _double_inverted_gaussian(x,
                              amp1, loc1, std1,
                              amp2, loc2, std2):
    '''This is a double inverted gaussian.

    Parameters
    ----------

    x : np.array
        The items at which the Gaussian is evaluated.

    amp1,amp2 : float
        The amplitude of Gaussian 1 and Gaussian 2.

    loc1,loc2 : float
        The central value of Gaussian 1 and Gaussian 2.

    std1,std2 : float
        The standard deviation of Gaussian 1 and Gaussian 2.

    Returns
    -------

    np.array
        Returns a double inverted Gaussian function evaluated at the items in
        `x`, using the provided parameters of `amp`, `loc`, and `std` for two
        component Gaussians 1 and 2.

    '''

    gaussian1 = -_gaussian(x,amp1,loc1,std1)
    gaussian2 = -_gaussian(x,amp2,loc2,std2)
    return gaussian1 + gaussian2


def invgauss_eclipses_func(ebparams, times, mags, errs):
    '''This returns a double eclipse shaped function.

    Suitable for first order modeling of eclipsing binaries.

    Parameters
    ----------

    ebparams : list of float
        This contains the parameters for the eclipsing binary::

            ebparams = [period (time),
                        epoch (time),
                        pdepth: primary eclipse depth (mags),
                        pduration: primary eclipse duration (phase),
                        psdepthratio: primary-secondary eclipse depth ratio,
                        secondaryphase: center phase of the secondary eclipse]

        `period` is the period in days.

        `epoch` is the time of minimum in JD.

        `pdepth` is the depth of the primary eclipse.

        - for magnitudes -> pdepth should be < 0
        - for fluxes     -> pdepth should be > 0

        `pduration` is the length of the primary eclipse in phase.

        `psdepthratio` is the ratio in the eclipse depths:
        `depth_secondary/depth_primary`. This is generally the same as the ratio
        of the `T_effs` of the two stars.

        `secondaryphase` is the phase at which the minimum of the secondary
        eclipse is located. This effectively parameterizes eccentricity.

        All of these will then have fitted values after the fit is done.

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the eclipse model will be generated. The times will be used to generate
        model mags, and the input `times`, `mags`, and `errs` will be resorted
        by model phase and returned.

    Returns
    -------

    (modelmags, phase, ptimes, pmags, perrs) : tuple
        Returns the model mags and phase values. Also returns the input `times`,
        `mags`, and `errs` sorted by the model's phase.

    '''

    (period, epoch, pdepth, pduration, depthratio, secondaryphase) = ebparams

    # generate the phases
    iphase = (times - epoch)/period
    iphase = iphase - np.floor(iphase)

    phasesortind = np.argsort(iphase)
    phase = iphase[phasesortind]
    ptimes = times[phasesortind]
    pmags = mags[phasesortind]
    perrs = errs[phasesortind]

    zerolevel = np.median(pmags)
    modelmags = np.full_like(phase, zerolevel)

    primaryecl_amp = -pdepth
    secondaryecl_amp = -pdepth * depthratio

    primaryecl_std = pduration/5.0    # we use 5-sigma as full-width -> duration
    secondaryecl_std = pduration/5.0  # secondary eclipse has the same duration

    halfduration = pduration/2.0

    # phase indices
    primary_eclipse_ingress = (
        (phase >= (1.0 - halfduration)) & (phase <= 1.0)
    )
    primary_eclipse_egress = (
        (phase >= 0.0) & (phase <= halfduration)
    )

    secondary_eclipse_phase = (
        (phase >= (secondaryphase - halfduration)) &
        (phase <= (secondaryphase + halfduration))
    )

    # put in the eclipses
    modelmags[primary_eclipse_ingress] = (
        zerolevel + _gaussian(phase[primary_eclipse_ingress],
                              primaryecl_amp,
                              1.0,
                              primaryecl_std)
    )
    modelmags[primary_eclipse_egress] = (
        zerolevel + _gaussian(phase[primary_eclipse_egress],
                              primaryecl_amp,
                              0.0,
                              primaryecl_std)
    )
    modelmags[secondary_eclipse_phase] = (
        zerolevel + _gaussian(phase[secondary_eclipse_phase],
                              secondaryecl_amp,
                              secondaryphase,
                              secondaryecl_std)
    )

    return modelmags, phase, ptimes, pmags, perrs


def invgauss_eclipses_curvefit_func(
        times,
        period,
        epoch,
        pdepth,
        pduration,
        psdepthratio,
        secondaryphase,
        zerolevel=0.0,
        fixed_params=None,
):
    '''This is the inv-gauss eclipses function used with scipy.optimize.curve_fit.

    Parameters
    ----------

    times : np.array
        The array of times at which the model will be evaluated.

    period : float
        The period of the eclipsing binary.

    epoch : float
        The mid eclipse time of the primary eclipse. In the same units as times.

    pdepth : float
        The depth of the primary eclipse.

    pduration : float
        The duration of the primary eclipse. In units of phase.

    psdepthratio : float
        The ratio between the depths of the primary and secondary eclipse.

    secondaryphase : float
        The phase of the secondary eclipse.

    zerolevel : float
        The out of eclipse value of the model.

    fixed_params : dict or None
        If this is provided, must be a dict containing the parameters to fix and
        their values. Should be of the form below::

            {'period': fixed value,
             'epoch': fixed value,
             'pdepth': fixed value,
             'pduration': fixed value,
             'psdepthratio': fixed value,
             'secondaryphase': fixed value}

        Any parameter in the dict provided will have its parameter fixed to the
        provided value. This is best done with an application of
        functools.partial before passing the function to the
        scipy.optimize.curve_fit function, e.g.::

            curvefit_func = functools.partial(
                                eclipses.invgauss_eclipses_curvefit_func,
                                zerolevel=np.median(mags),
                                fixed_params={'secondaryphase':0.5})

            fit_params, fit_cov = scipy.optimize.curve_fit(
                                    curvefit_func,
                                    times, mags,
                                    p0=initial_params,
                                    sigma=errs,
                                    ...)

    Returns
    -------

    model : np.array
        Returns the transit model as an np.array. This is in the same order as
        the times input array.

    '''

    if fixed_params is not None and len(fixed_params) > 0:

        if 'period' in fixed_params:
            period = fixed_params['period']
        if 'epoch' in fixed_params:
            epoch = fixed_params['epoch']
        if 'pdepth' in fixed_params:
            pdepth = fixed_params['pdepth']
        if 'pduration' in fixed_params:
            pduration = fixed_params['pduration']
        if 'psdepthratio' in fixed_params:
            psdepthratio = fixed_params['psdepthratio']
        if 'secondaryphase' in fixed_params:
            secondaryphase = fixed_params['secondaryphase']

    # generate the phases
    phase = (times - epoch)/period
    phase = phase - np.floor(phase)

    eclipsemodel = np.full_like(phase, zerolevel)

    primaryecl_amp = -pdepth
    secondaryecl_amp = -pdepth * psdepthratio

    primaryecl_std = pduration/5.0    # we use 5-sigma as full-width -> duration
    secondaryecl_std = pduration/5.0  # secondary eclipse has the same duration

    halfduration = pduration/2.0

    # phase indices
    primary_eclipse_ingress = (
        (phase >= (1.0 - halfduration)) & (phase <= 1.0)
    )
    primary_eclipse_egress = (
        (phase >= 0.0) & (phase <= halfduration)
    )

    secondary_eclipse_phase = (
        (phase >= (secondaryphase - halfduration)) &
        (phase <= (secondaryphase + halfduration))
    )

    # put in the eclipses
    eclipsemodel[primary_eclipse_ingress] = (
        zerolevel + _gaussian(phase[primary_eclipse_ingress],
                              primaryecl_amp,
                              1.0,
                              primaryecl_std)
    )
    eclipsemodel[primary_eclipse_egress] = (
        zerolevel + _gaussian(phase[primary_eclipse_egress],
                              primaryecl_amp,
                              0.0,
                              primaryecl_std)
    )
    eclipsemodel[secondary_eclipse_phase] = (
        zerolevel + _gaussian(phase[secondary_eclipse_phase],
                              secondaryecl_amp,
                              secondaryphase,
                              secondaryecl_std)
    )

    return eclipsemodel


def invgauss_eclipses_residual(ebparams, times, mags, errs):
    '''This returns the residual between the modelmags and the actual mags.

    Parameters
    ----------

    ebparams : list of float
        This contains the parameters for the eclipsing binary::

            ebparams = [period (time),
                        epoch (time),
                        pdepth: primary eclipse depth (mags),
                        pduration: primary eclipse duration (phase),
                        psdepthratio: primary-secondary eclipse depth ratio,
                        secondaryphase: center phase of the secondary eclipse]

        `period` is the period in days.

        `epoch` is the time of minimum in JD.

        `pdepth` is the depth of the primary eclipse.

        - for magnitudes -> `pdepth` should be < 0
        - for fluxes     -> `pdepth` should be > 0

        `pduration` is the length of the primary eclipse in phase.

        `psdepthratio` is the ratio in the eclipse depths:
        `depth_secondary/depth_primary`. This is generally the same as the ratio
        of the `T_effs` of the two stars.

        `secondaryphase` is the phase at which the minimum of the secondary
        eclipse is located. This effectively parameterizes eccentricity.

        All of these will then have fitted values after the fit is done.

    times,mags,errs : np.array
        The input time-series of measurements and associated errors for which
        the eclipse model will be generated. The times will be used to generate
        model mags, and the input `times`, `mags`, and `errs` will be resorted
        by model phase and returned.

    Returns
    -------

    np.array
        The residuals between the input `mags` and generated `modelmags`,
        weighted by the measurement errors in `errs`.

    '''

    modelmags, phase, ptimes, pmags, perrs = (
        invgauss_eclipses_func(ebparams, times, mags, errs)
    )

    # this is now a weighted residual taking into account the measurement err
    return (pmags - modelmags)/perrs
