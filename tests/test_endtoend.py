'''
This is a basic end-to-end test of astrobase.

This tests the following:

- downloads a light curve from the HAT data server
- reads the light curve using astrobase.hatlc
- runs the GLS, PDM, AoV, and BLS functions on the LC
- creates a checkplot pickle using these results

'''
from __future__ import print_function

import os.path
try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve

from astrobase import hatlc, periodbase, checkplot

# this is the light curve used for tests
LCURL = ("https://github.com/waqasbhatti/astrobase/raw/master/"
         "notebooks/nb-data/HAT-772-0554686-V0-DR0-hatlc.sqlite.gz")

# this function is used to check progress of the download
def on_download_chunk(transferred,blocksize,totalsize):
    progress = transferred*blocksize/float(totalsize)*100.0
    print('downloading test LC: {progress:.1f}%'.format(progress=progress),
          end='\r')

# get the light curve if it's not there
modpath = os.path.abspath(__file__)
dlpath = os.path.join(os.path.dirname(modpath),
                      '../notebooks/nb-data/',
                      'HAT-772-0554686-V0-DR0-hatlc.sqlite.gz')
if not os.path.exists(dlpath):
    localf, headerr = urlretrieve(
        LCURL,dlpath,reporthook=on_download_chunk
    )



def test_hatlc():
    '''
    Tests that a HAT light curve can be loaded.

    '''


def test_normalize():
    '''
    Tests that a HAT light curve can be normalized.

    '''


def test_gls():
    '''
    Tests periodbase.pgen_lsp.

    '''


def test_pdm():
    '''
    Tests periodbase.stellingwerf_pdm.

    '''


def test_aov():
    '''
    Tests periodbase.aov_periodfind.

    '''


def test_bls_serial():
    '''
    Tests periodbase.bls_serial_pfind.

    '''


def test_bls_parallel():
    '''
    Tests periodbase.bls_parallel_pfind.

    '''


def test_checkplot_png():
    '''
    Tests if a checkplot PNG can be made.

    '''


def test_checkplot_twolsp_png():
    '''
    Tests if a two-LSP checkplot PNG can be made.

    '''


def test_checkplot_pickle_make():
    '''
    Tests if a checkplot pickle can be made.

    '''


def test_checkplot_pickle_read():
    '''
    Tests if a checkplot pickle can be read.

    '''
