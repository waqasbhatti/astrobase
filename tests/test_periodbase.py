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
except:
    from urllib.request import urlretrieve
from numpy.testing import assert_allclose

from astrobase.hatsurveys import hatlc
from astrobase import periodbase


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



def test_bls_serial():
    '''
    Tests periodbase.bls_serial_pfind.

    '''
    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    bls = periodbase.bls_serial_pfind(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'],
                                      startp=1.0)

    assert isinstance(bls, dict)
    assert_allclose(bls['bestperiod'], 3.08560655)



def test_bls_parallel():
    '''
    Tests periodbase.bls_parallel_pfind.

    '''
    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    bls = periodbase.bls_parallel_pfind(lcd['rjd'],
                                        lcd['aep_000'],
                                        lcd['aie_000'],
                                        startp=1.0)

    assert isinstance(bls, dict)
    assert_allclose(bls['bestperiod'], 3.08560655)
