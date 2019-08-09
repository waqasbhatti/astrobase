"""
test_transitfit.py - Luke Bouma (luke@astro.princeton.edu) - Aug 2019
License: MIT - see the LICENSE file for details.

This tests the following:

- downloads a light curve from the github repository notebooks/nb-data dir
(TODO)

- fits an easy light curve using fivetransitparam_fit_magseries
    - and derives uncertainties

- attempts fitting a harder light curve using same model
    - and should return a useful indication that it failed
"""

###########
# imports #
###########
import os
from glob import glob

import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits
import astrobase.imageutils as iu

from astrobase.lcfit.transits import fivetransitparam_fit_magseries

#########
# tests #
#########

def test_fivetransitparam_fit_magseries_easy():
    """
    Fit one TESS sector of data, with a HJ (candidate) in it, for a transit
    model (t0, period, incl, sma, rp/star) with believable error bars.
    """

    identifier = '0003007171311355035136'
    lcpath = (
        'hlsp_cdips_tess_ffi_gaiatwo{}-0006_tess_v01_llc.fits'.format(identifier)
    )

    hdul = fits.open(lcpath)
    hdr, lc = hdul[0].header, hdul[1].data
    hdul.close()

    time = lc['TMID_BJD']
    mag = lc['TFA2']
    mag_0, f_0 = 12, 1e4
    flux = f_0 * 10**( -0.4 * (mag - mag_0) )
    flux /= np.nanmedian(flux)
    err = np.ones_like(flux)*1e-4

    teff = hdr['TICTEFF']
    rstar = hdr['TICRAD']
    logg = hdr['TICLOGG']

    fit_savdir = os.path.join(os.getcwd(), 'fivetransitparam_results')
    chain_savdir = os.path.join(os.getcwd(), 'fivetransitparam_chains')

    mafr, tlsr, is_converged = fivetransitparam_fit_magseries(
                time, flux, err,
                teff, rstar, logg,
                identifier,
                fit_savdir,
                chain_savdir,
                n_mcmc_steps=4000,
                overwriteexistingsamples=False,
                n_transit_durations=5,
                make_tlsfit_plot=True,
                exp_time_minutes=30,
                bandpass='tess',
                magsarefluxes=True,
                nworkers=16
    )

    assert is_converged
    assert_allclose(tlsr['period'], 3.495, atol=1e-2)
    assert_allclose(mafr['fitinfo']['finalparams']['period'], 3.495, atol=1e-2)

    # theoretical t0 for this data (like 5 or 6 transits, over 1 TESS sector)
    # is 3.99e+00 min = 6.64e-02 h = 2.77e-03 days.
    assert mafr['fitinfo']['finalparamerrs']['std_perrs']['t0'] < 5e-3
    assert mafr['fitinfo']['finalparamerrs']['std_merrs']['t0'] < 5e-3

    # guess-timate period should be better than 3 minutes too.
    assert mafr['fitinfo']['finalparamerrs']['std_perrs']['period'] < 3/(24*60)
    assert mafr['fitinfo']['finalparamerrs']['std_merrs']['period'] < 3/(24*60)
