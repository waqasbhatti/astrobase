"""test_transitfit.py - Luke Bouma (luke@astro.princeton.edu) - Aug 2019
License: MIT - see the LICENSE file for details.

NOTE: the tests in this module take a very long time. To enable and actually run
them, you must set the environmental variable RUN_LONG_TESTS=1 before running
pytest, like so: ``RUN_LONG_TESTS=1 pytest test_transitfit.py``, or set this
environmental variable in your CI runner.

When run, this file downloads two TESS light curves from the
astrobase-notebooks/nb-data directory.

Implemented tests include:

- fit an easy single-sector light curve (6 big HJ transits) using
fivetransitparam_fit_magseries, derive uncertainties, check them against
theoretical predictions.

- fit a trickier single-sector light curve (2 transits), do as above.

- test_multithread_speed: ensure that increasing nworkers speeds up the MCMC
sampling. 2019/08/15: this test fails.

"""

###########
# imports #
###########

from pytest import mark

import os
import multiprocessing
import time as _time
try:
    from urllib import urlretrieve
except Exception:
    from urllib.request import urlretrieve

import numpy as np
from numpy.testing import assert_allclose

from astropy.io import fits
import astrobase.imageutils as iu

try:
    import transitleastsquares
    from astrobase.lcfit.transits import fivetransitparam_fit_magseries
    test_ok = True
except Exception:
    test_ok = False


##########
# config #
##########

# this function is used to check progress of the download
def on_download_chunk(transferred,blocksize,totalsize):
    progress = transferred*blocksize/float(totalsize)*100.0
    print('downloading test LC: {progress:.1f}%'.format(progress=progress),
          end='\r')


# download the light curves used for tests if they do not exist. first is a
# nice easy hot jupiter, 6 transits.  second is a tricky two-transit warm
# jupiter. (TOI-450).
LCURLS = [
    ("https://github.com/waqasbhatti/astrobase-notebooks/raw/master/nb-data/"
     "hlsp_cdips_tess_ffi_gaiatwo0003007171311355035136-0006_tess_v01_llc.fits"),
    ("https://github.com/waqasbhatti/astrobase-notebooks/raw/master/nb-data/"
     "hlsp_cdips_tess_ffi_gaiatwo0004827527233363019776-0006_tess_v01_llc.fits")
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


if not os.environ.get('RUN_LONG_TESTS'):
    test_ok = False


#########
# tests #
#########

if test_ok:

    def test_fivetransitparam_fit_magseries_easy():
        """
        Fit one TESS sector of data, with a HJ (candidate) in it, for a transit
        model (t0, period, incl, sma, rp/star) with believable error bars.
        """

        # path and identifier for GaiaDR2 3007171311355035136
        lcpath = LCPATHS[0]
        identifier = str(lcpath.split('gaiatwo')[1].split('-')[0].lstrip('0'))

        lc = iu.get_data_keyword_list(lcpath, ['TMID_BJD', 'TFA2'])
        hdr = iu.get_header_keyword_list(lcpath,
                                         ['TICTEFF', 'TICRAD', 'TICLOGG'])

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
        assert_allclose(mafr['fitinfo']['finalparams']['period'],
                        3.495,
                        atol=1e-2)

        # theoretical t0 for this data (like 5 or 6 transits, over 1 TESS
        # sector) is 3.99e+00 min = 6.64e-02 h = 2.77e-03 days.
        assert mafr['fitinfo']['finalparamerrs']['std_perrs']['t0'] < 5e-3
        assert mafr['fitinfo']['finalparamerrs']['std_merrs']['t0'] < 5e-3

        # guess-timate period should be better than 3 minutes too.
        assert (
            mafr['fitinfo']['finalparamerrs']['std_perrs']['period'] < 3/(24*60)
        )
        assert (
            mafr['fitinfo']['finalparamerrs']['std_merrs']['period'] < 3/(24*60)
        )

    def test_fivetransitparam_fit_magseries_hard():
        """
        Fit one TESS sector of data, with a hard WJ (candidate) in it, for a
        transit model (t0, period, incl, sma, rp/star) with believable error bars.
        """

        # path and identifier for GaiaDR2 4827527233363019776
        lcpath = LCPATHS[1]
        identifier = str(lcpath.split('gaiatwo')[1].split('-')[0].lstrip('0'))

        lc = iu.get_data_keyword_list(lcpath, ['TMID_BJD', 'TFA2'])
        hdr = iu.get_header_keyword_list(lcpath,
                                         ['TICTEFF', 'TICRAD', 'TICLOGG'])

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

        # Autocorrelation time is like 400 steps for this case. But it does
        # converge.
        mafr, tlsr, is_converged = fivetransitparam_fit_magseries(
            time, flux, err,
            teff, rstar, logg,
            identifier,
            fit_savdir,
            chain_savdir,
            n_mcmc_steps=25000,
            overwriteexistingsamples=False,
            n_transit_durations=5,
            make_tlsfit_plot=True,
            exp_time_minutes=30,
            bandpass='tess',
            magsarefluxes=True,
            nworkers=multiprocessing.cpu_count()
        )

        print(is_converged)
        print(tlsr['period'])
        print(mafr['fitinfo']['finalparams']['period'])

        assert is_converged
        assert_allclose(tlsr['period'], 10.714, atol=1e-2)
        assert_allclose(mafr['fitinfo']['finalparams']['period'], 10.714, atol=1e-2)

        # exofopTESS quotes t0 for TOI-450 to <1 minute (multisector). < 10 minutes
        # required.
        assert mafr['fitinfo']['finalparamerrs']['std_perrs']['t0'] < 10/(24*60)
        assert mafr['fitinfo']['finalparamerrs']['std_merrs']['t0'] < 10/(24*60)

        # guess-timate period should be better than say 10 minutes.
        assert mafr['fitinfo']['finalparamerrs']['std_perrs']['period'] < 10/(24*60)
        assert mafr['fitinfo']['finalparamerrs']['std_merrs']['period'] < 10/(24*60)


    @mark.skip(reason="2019/08/15 fails, b/c MCMC multithreading broken(?)")
    def test_multithread_speed():
        """Ensure that increasing nworkers speeds up the MCMC sampling.

        2019/08/15: this test fails.

        Assumption is ideally that run time goes as 1/nworkers. We are a bit
        nicer here, and take out a factor of two for overhead. Even this fails,
        because the emcee multithread scaling in
        lcfit/transits.mandelagol_fit_magseries is non-existent.

        """

        # NOTE: this test fails, because something is wrong with the emcee
        # multithreading in lcfit/transits.py. (This is an issue that would be
        # nice to resolve -- though for the time being not "mission-critical")

        # path and identifier for GaiaDR2 3007171311355035136, the nice HJ.
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

        fit_savdir = os.path.join(os.getcwd(),
                                  'fivetransitparam_results_single_thread')
        chain_savdir = os.path.join(os.getcwd(),
                                    'fivetransitparam_chains_single_thread')

        start = _time.time()
        mafr, tlsr, is_converged = fivetransitparam_fit_magseries(
            time, flux, err,
            teff, rstar, logg,
            identifier,
            fit_savdir,
            chain_savdir,
            n_mcmc_steps=1000,
            overwriteexistingsamples=True,
            n_transit_durations=5,
            make_tlsfit_plot=True,
            exp_time_minutes=30,
            bandpass='tess',
            magsarefluxes=True,
            nworkers=1
        )
        end_singlethread = _time.time()

        fit_savdir = os.path.join(os.getcwd(),
                                  'fivetransitparam_results_manythread')
        chain_savdir = os.path.join(os.getcwd(),
                                    'fivetransitparam_chains_manythread')

        mafr, tlsr, is_converged = fivetransitparam_fit_magseries(
            time, flux, err,
            teff, rstar, logg,
            identifier,
            fit_savdir,
            chain_savdir,
            n_mcmc_steps=1000,
            overwriteexistingsamples=True,
            n_transit_durations=5,
            make_tlsfit_plot=True,
            exp_time_minutes=30,
            bandpass='tess',
            magsarefluxes=True,
            nworkers=multiprocessing.cpu_count()
        )
        end_multithread = _time.time()

        multithread_time = end_multithread - end_singlethread
        singlethread_time = end_singlethread - start

        print("Singlethread took {0:.1f} seconds".
              format(singlethread_time))
        print("Multithreaded took {0:.1f} seconds with {} workers".
              format(multithread_time, multiprocessing.cpu_count()))

        print("{0:.1f} times faster than serial".
              format(singlethread_time / multithread_time))

        # passes, but mainly b/c of the overhead from TLS
        assert multithread_time < singlethread_time

        # fails
        assert (
            multithread_time <
            (singlethread_time/(0.5*multiprocessing.cpu_count()))
        )
