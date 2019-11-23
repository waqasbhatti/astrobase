'''test_lcfit.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2018
License: MIT - see the LICENSE file for details.

This tests the following:

- downloads a light curve from the github repository notebooks/nb-data dir
- fits the light curve using Fourier, SavGol, Legendre, transit model, and
  eclipsing binary models

'''
from __future__ import print_function
import os
import os.path

try:
    from urllib import urlretrieve
except Exception:
    from urllib.request import urlretrieve

import numpy as np
from numpy.testing import assert_allclose

from astrobase.hatsurveys import hatlc
from astrobase import lcfit


############
## CONFIG ##
############

# this is the light curve used for tests
LCURL = ("https://github.com/waqasbhatti/astrobase-notebooks/raw/master/"
         "nb-data/HAT-772-0554686-V0-DR0-hatlc.sqlite.gz")


# this function is used to check progress of the download
def on_download_chunk(transferred,blocksize,totalsize):
    progress = transferred*blocksize/float(totalsize)*100.0
    print('downloading test LC: {progress:.1f}%'.format(progress=progress),
          end='\r')


# get the light curve if it's not there
modpath = os.path.abspath(__file__)
LCPATH = os.path.abspath(os.path.join(os.getcwd(),
                                      'HAT-772-0554686-V0-DR0-hatlc.sqlite.gz'))
if not os.path.exists(LCPATH):
    localf, headerr = urlretrieve(
        LCURL,LCPATH,reporthook=on_download_chunk
    )

PERIOD = 3.08578956


###########
## TESTS ##
###########

def test_fourierfit():
    '''
    Tests lcfit.fourier_fit_magseries.

    '''

    EXPECTED_FOURIERPARAMS = np.array(
        [-0.15680457,
         -0.02936829,
         0.0609173,
         0.02692699,
         0.05093689,
         -0.02880862,
         -0.0418917,
         1.64850372,
         -5.02156053,
         -0.26441503,
         -3.15644339,
         -0.28243494,
         -0.23831426,
         -3.35074058]
    )

    EXPECTED_REDCHISQ = 2.756479193621726

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    fit = lcfit.fourier_fit_magseries(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'],
                                      PERIOD,
                                      fourierorder=7,
                                      sigclip=10.0,
                                      plotfit='test-fourierfit.png')

    assert isinstance(fit, dict)
    assert os.path.exists('test-fourierfit.png')

    assert_allclose(fit['fitredchisq'],
                    EXPECTED_REDCHISQ,
                    rtol=1.0e-3)
    assert_allclose(fit['fitinfo']['fitepoch'],
                    np.array([56092.640558]),
                    rtol=1.0e-5)
    assert_allclose(fit['fitinfo']['finalparams'],
                    EXPECTED_FOURIERPARAMS,
                    rtol=1.0e-5,
                    atol=0.01)


def test_splinefit():
    '''
    Tests lcfit.spline_fit_magseries.

    '''

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    fit = lcfit.spline_fit_magseries(lcd['rjd'],
                                     lcd['aep_000'],
                                     lcd['aie_000'],
                                     PERIOD,
                                     sigclip=10.0,
                                     plotfit='test-splinefit.png')

    assert isinstance(fit, dict)
    assert os.path.exists('test-splinefit.png')

    assert_allclose(fit['fitredchisq'], 2.0388607985104708)
    assert_allclose(fit['fitinfo']['fitepoch'], np.array([56794.6838758]))


def test_savgolfit():
    '''
    Tests lcfit.savgol_fit_magseries.

    '''

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    fit = lcfit.savgol_fit_magseries(lcd['rjd'],
                                     lcd['aep_000'],
                                     lcd['aie_000'],
                                     PERIOD,
                                     sigclip=10.0,
                                     plotfit='test-savgolfit.png')

    assert isinstance(fit, dict)
    assert os.path.exists('test-savgolfit.png')

    assert_allclose(fit['fitchisq'], 22198.963657474334)
    assert_allclose(fit['fitinfo']['fitepoch'], np.array([56856.3795333]))


def test_legendrefit():
    '''
    Tests lcfit.legendre_fit_magseries.

    '''

    LEGCOEFFS = np.array([
        1.51913612e+01, -3.76241414e-02, -4.75784157e-03, 6.41476816e-03,
        1.24890424e-01, 1.50597767e-02, -7.58171649e-02, -2.21526213e-02,
        1.23594509e-01, 2.95992178e-02, -1.33913179e-01
    ])

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    fit = lcfit.legendre_fit_magseries(lcd['rjd'],
                                       lcd['aep_000'],
                                       lcd['aie_000'],
                                       PERIOD,
                                       sigclip=10.0,
                                       plotfit='test-legendrefit.png')

    assert isinstance(fit, dict)
    assert os.path.exists('test-legendrefit.png')

    assert_allclose(fit['fitredchisq'], 4.0964492873537495,)
    assert_allclose(fit['fitinfo']['fitepoch'], np.array([56297.8650305]))
    assert_allclose(fit['fitinfo']['finalparams'], LEGCOEFFS, rtol=1.0e-6)


def test_transitfit():
    '''
    Tests lcfit.traptransit_fit_magseries.

    '''

    FITPARAMS = np.array([3.08554939e+00,
                          5.67946743e+04,
                          -3.48312233e-01,
                          8.94843762e-02,
                          2.36164894e-02])

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    fit = lcfit.traptransit_fit_magseries(lcd['rjd'],
                                          lcd['aep_000'],
                                          lcd['aie_000'],
                                          [PERIOD, None,
                                           -0.1, 0.1, 0.05],
                                          sigclip=10.0,
                                          plotfit='test-transitfit.png')

    assert isinstance(fit, dict)
    assert os.path.exists('test-transitfit.png')

    assert_allclose(fit['fitredchisq'], 2.9804742789196497)
    assert_allclose(fit['fitinfo']['fitepoch'], np.array([56794.67427438755]))
    assert_allclose(fit['fitinfo']['finalparams'], FITPARAMS, rtol=1.0e-6)


def test_gaussianebfit():
    '''
    This tests lcfit.gaussianeb_fit_magseries.

    '''

    FITPARAMS = np.array([
        3.08556129e+00,
        5.67946748e+04,
        -4.11857480e-01,
        1.13484014e-01,
        3.35893325e-01,
        5.11450964e-01
    ])

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    fit = lcfit.gaussianeb_fit_magseries(lcd['rjd'],
                                         lcd['aep_000'],
                                         lcd['aie_000'],
                                         [PERIOD, None, -0.3, 0.1, 0.3, 0.5],
                                         sigclip=10.0,
                                         plotfit='test-ebfit.png')

    assert isinstance(fit, dict)
    assert os.path.exists('test-ebfit.png')

    assert_allclose(fit['fitredchisq'], 2.6358311084545702)
    assert_allclose(fit['fitinfo']['fitepoch'], np.array([56794.67481399149]))
    assert_allclose(fit['fitinfo']['finalparams'], FITPARAMS, rtol=1.0e-6)
