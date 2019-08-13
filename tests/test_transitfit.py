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
import os, multiprocessing
from glob import glob
try:
    from urllib import urlretrieve
except Exception:
    from urllib.request import urlretrieve

import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits
import astrobase.imageutils as iu

from astrobase.lcfit.transits import fivetransitparam_fit_magseries
from test_periodbase import on_download_chunk

##########
# config #
##########

# download the light curves used for tests if they do not exist. first is a
# nice easy hot jupiter, 6 transits.  second is a tricky two-transit warm
# jupiter.
LCURLS = [
    (
    "https://github.com/waqasbhatti/astrobase-notebooks/raw/master/nb-data/"
    "hlsp_cdips_tess_ffi_gaiatwo0003007171311355035136-0006_tess_v01_llc.fits"
    ),
    (
    "https://github.com/waqasbhatti/astrobase-notebooks/raw/master/nb-data/"
    "hlsp_cdips_tess_ffi_gaiatwo0004827527233363019776-0006_tess_v01_llc.fits"
    )
]

modpath = os.path.abspath(__file__)
LCPATHS = [
    os.path.abspath(os.path.join(
        os.getcwd(),
        'hlsp_cdips_tess_ffi_gaiatwo0003007171311355035136-0006_tess_v01_llc.fits')
    ),
    os.path.abspath(os.path.join(
        os.getcwd(),
        'hlsp_cdips_tess_ffi_gaiatwo0004827527233363019776-0006_tess_v01_llc.fits')
    )
]

for LCPATH, LCURL in zip(LCPATHS, LCURLS):
    if not os.path.exists(LCPATH):
        localf, headerr = urlretrieve(
            LCURL,LCPATH,reporthook=on_download_chunk)


#########
# tests #
#########

def test_fivetransitparam_fit_magseries_easy():
    """
    Fit one TESS sector of data, with a HJ (candidate) in it, for a transit
    model (t0, period, incl, sma, rp/star) with believable error bars.
    """

    # path and identifier for GaiaDR2 3007171311355035136
    lcpath = LCPATHS[0]
    identifier = str(lcpath.split('gaiatwo')[1].split('-')[0].lstrip('0'))

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
                nworkers=multiprocessing.cpu_count()
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
