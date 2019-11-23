'''test_periodbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2018
License: MIT - see the LICENSE file for details.

This tests the following:

- downloads a light curve from the github repository notebooks/nb-data dir
- reads the light curve using astrobase.hatlc
- runs the GLS, WIN, PDM, AoV, BLS, AoVMH, and ACF period finders on the LC

'''
from __future__ import print_function
import os
import os.path
try:
    from urllib import urlretrieve
except Exception:
    from urllib.request import urlretrieve

from numpy.testing import assert_allclose

from astrobase.hatsurveys import hatlc
from astrobase import periodbase

# separate testing for kbls and abls from now on
from astrobase.periodbase import kbls
from astrobase.periodbase import abls

try:
    import transitleastsquares
    from astrobase.periodbase import htls
    htls_ok = True
except Exception:
    htls_ok = False


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


###########
## TESTS ##
###########

def test_gls():
    '''
    Tests periodbase.pgen_lsp.

    '''

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)


def test_win():
    '''
    Tests periodbase.specwindow_lsp

    '''

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    win = periodbase.specwindow_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(win, dict)
    assert_allclose(win['bestperiod'], 592.0307682142864)


def test_pdm():
    '''
    Tests periodbase.stellingwerf_pdm.

    '''
    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    pdm = periodbase.stellingwerf_pdm(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(pdm, dict)
    assert_allclose(pdm['bestperiod'], 3.08578956)


def test_aov():
    '''
    Tests periodbase.aov_periodfind.

    '''
    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    aov = periodbase.aov_periodfind(lcd['rjd'],
                                    lcd['aep_000'],
                                    lcd['aie_000'])

    assert isinstance(aov, dict)
    assert_allclose(aov['bestperiod'], 3.08578956)


def test_aovhm():
    '''
    Tests periodbase.aov_periodfind.

    '''
    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    mav = periodbase.aovhm_periodfind(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(mav, dict)
    assert_allclose(mav['bestperiod'], 3.08578956)


def test_acf():
    '''
    Tests periodbase.macf_period_find.

    '''
    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    acf = periodbase.macf_period_find(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'],
                                      smoothacf=721)

    assert isinstance(acf, dict)
    assert_allclose(acf['bestperiod'], 3.0750854011348565)


def test_kbls_serial():
    '''
    Tests periodbase.kbls.bls_serial_pfind.

    '''
    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    bls = kbls.bls_serial_pfind(lcd['rjd'],
                                lcd['aep_000'],
                                lcd['aie_000'],
                                startp=1.0)

    assert isinstance(bls, dict)
    assert_allclose(bls['bestperiod'], 3.08560655)


def test_kbls_parallel():
    '''
    Tests periodbase.kbls.bls_parallel_pfind.

    '''
    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    bls = kbls.bls_parallel_pfind(lcd['rjd'],
                                  lcd['aep_000'],
                                  lcd['aie_000'],
                                  startp=1.0)

    assert isinstance(bls, dict)
    assert_allclose(bls['bestperiod'], 3.08560655)


def test_abls_serial():
    '''
    This tests periodbase.abls.bls_serial_pfind.

    '''

    EXPECTED_PERIOD = 3.0873018

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    bls = abls.bls_serial_pfind(lcd['rjd'],
                                lcd['aep_000'],
                                lcd['aie_000'],
                                startp=1.0,
                                ndurations=50)

    assert isinstance(bls, dict)
    assert_allclose(bls['bestperiod'], EXPECTED_PERIOD)


def test_abls_parallel():
    '''
    This tests periodbase.abls.bls_parallel_pfind.

    '''

    EXPECTED_PERIOD = 3.0848887

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    bls = abls.bls_parallel_pfind(lcd['rjd'],
                                  lcd['aep_000'],
                                  lcd['aie_000'],
                                  startp=1.0,
                                  ndurations=50)

    assert isinstance(bls, dict)
    assert_allclose(bls['bestperiod'], EXPECTED_PERIOD, atol=1.0e-4)


if htls_ok:

    def test_tls_parallel():
        '''
        This tests periodbase.htls.tls_parallel_pfind.
        '''

        EXPECTED_PERIOD = 3.0848887

        lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

        tlsdict = htls.tls_parallel_pfind(
            lcd['rjd'],
            lcd['aep_000'],
            lcd['aie_000'],
            startp=2.0,
            endp=5.0
        )

        tlsresult = tlsdict['tlsresult']

        assert isinstance(tlsresult, dict)

        # ensure period is within 2 sigma of what's expected.
        assert_allclose(tlsdict['bestperiod'], EXPECTED_PERIOD,
                        atol=2.0*tlsresult['period_uncertainty'])
