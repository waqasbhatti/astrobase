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
except:
    from urllib.request import urlretrieve

import numpy as np
from numpy.testing import assert_allclose

from astrobase.hatsurveys import hatlc
from astrobase.varbase import lcfit


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

    FOURIERPARAMS = np.array(
        [-0.20722085, -0.03259998,  0.06764817,  0.03392241,  0.06109763,
         -0.03714063, -0.0492521 ,  1.6819149 , -5.27918097, -0.23971138,
         -3.23509529, -0.24212238, -0.20960648, -3.36844061]
    )

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

    assert_allclose(fit['fitredchisq'], 2.892135304465803, rtol=1.0e-6)
    assert_allclose(fit['fitinfo']['fitepoch'], np.array([56853.3082622]))
    assert_allclose(fit['fitinfo']['finalparams'], FOURIERPARAMS, rtol=1.0e-5)



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
        1.51913612e+01, -3.76241414e-02, -4.75784157e-03,  6.41476816e-03,
        1.24890424e-01,  1.50597767e-02, -7.58171649e-02, -2.21526213e-02,
        1.23594509e-01,  2.95992178e-02, -1.33913179e-01
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

    FITPARAMS = np.array([3.08578957e+00,
                          5.67946843e+04,
                          -4.32689876e-01,
                          1.00188320e-01,
                          4.99058419e-02])

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

    assert_allclose(fit['fitredchisq'], 3.145980551134323)
    assert_allclose(fit['fitinfo']['fitepoch'], np.array([56794.68428345605]))
    assert_allclose(fit['fitinfo']['finalparams'], FITPARAMS, rtol=1.0e-6)



def test_gaussianebfit():
    '''
    This tests lcfit.gaussianeb_fit_magseries.

    '''

    FITPARAMS = np.array([
        3.08578957e+00,  5.67946843e+04,
        -3.76719042e-01,  1.30328332e-01,
        3.27081266e-01,  5.09789096e-01
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

    assert_allclose(fit['fitredchisq'], 2.7854070457947366)
    assert_allclose(fit['fitinfo']['fitepoch'], np.array([56794.68429695685,]))
    assert_allclose(fit['fitinfo']['finalparams'], FITPARAMS, rtol=1.0e-6)
