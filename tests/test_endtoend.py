'''test_endtoend.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Apr 2017
License: MIT - see the LICENSE file for details.

This is a basic end-to-end test of astrobase.

This tests the following:

- downloads a light curve from the github repository notebooks/nb-data dir
- reads the light curve using astrobase.hatlc
- runs the GLS, PDM, AoV, and BLS functions on the LC
- creates a checkplot PNG, twolsp PNG, and pickle using these results

These will take 5-7 minutes to run, depending on your CPU speed and number of
cores.

To run all the tests from the base directory of the git repository, make sure a
virtualenv is active with all the needed requirements, and then run setup.py:

$ virtualenv astrobase-testing (or use: python3 -m venv astrobase-testing)
$ source astrobase-testing/bin/activate
$ git clone https://github.com/waqasbhatti/astrobase.git
$ cd astrobase

# run these to hopefully get wheels for faster install (can skip this step)
$ pip install numpy  # do this first for f2py stuff
$ pip install -r requirements.txt

# finally run the tests (this will get and compile requirements automatically)
$ python setup.py test

'''
from __future__ import print_function
import os.path
try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve
from numpy.testing import assert_allclose

from astrobase import hatlc, periodbase, checkplot


############
## CONFIG ##
############

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
LCPATH = os.path.abspath(os.path.join(os.path.dirname(modpath),
                                      '../notebooks/nb-data/',
                                      'HAT-772-0554686-V0-DR0-hatlc.sqlite.gz'))
if not os.path.exists(LCPATH):
    localf, headerr = urlretrieve(
        LCURL,LCPATH,reporthook=on_download_chunk
    )


###########
## TESTS ##
###########

def test_hatlc():
    '''
    Tests that a HAT light curve can be loaded.

    '''

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    assert isinstance(lcd, dict)
    assert msg == 'no SQL filters, LC OK'



def test_gls():
    '''
    Tests periodbase.pgen_lsp.

    '''

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)



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



def test_checkplot_png():
    '''
    Tests if a checkplot PNG can be made.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.png')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)

    cpf = checkplot.checkplot_png(gls,
                                  lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
                                  outfile=outpath,
                                  objectinfo=lcd['objectinfo'])

    assert os.path.exists(outpath)



def test_checkplot_twolsp_png():
    '''
    Tests if a two-LSP checkplot PNG can be made.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-twolsp-checkplot.png')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)

    pdm = periodbase.stellingwerf_pdm(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(pdm, dict)
    assert_allclose(pdm['bestperiod'], 3.08578956)

    cpf = checkplot.twolsp_checkplot_png(
        gls, pdm,
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo']
    )

    assert os.path.exists(outpath)



def test_checkplot_pickle_make():
    '''
    Tests if a checkplot pickle can be made.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.pkl')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)

    pdm = periodbase.stellingwerf_pdm(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(pdm, dict)
    assert_allclose(pdm['bestperiod'], 3.08578956)

    cpf = checkplot.checkplot_pickle(
        [gls, pdm],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo']
    )

    assert os.path.exists(outpath)



def test_checkplot_pickle_read():
    '''
    Tests if a checkplot pickle can be made and read back.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.pkl')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)

    pdm = periodbase.stellingwerf_pdm(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(pdm, dict)
    assert_allclose(pdm['bestperiod'], 3.08578956)

    cpf = checkplot.checkplot_pickle(
        [gls, pdm],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo']
    )

    assert os.path.exists(outpath)

    cpd = checkplot._read_checkplot_picklefile(cpf)

    assert isinstance(cpd, dict)

    cpdkeys = set(list(cpd.keys()))
    testset = {'comments', 'finderchart', 'sigclip', 'objectid',
               'pdm', 'gls', 'objectinfo', 'status', 'varinfo',
               'normto', 'magseries', 'normmingap'}
    assert (testset - cpdkeys) == set()

    assert_allclose(cpd['gls']['bestperiod'], 1.54289477)
    assert_allclose(cpd['pdm']['bestperiod'], 3.08578956)



def test_checkplot_pickle_update():
    '''
    Tests if a checkplot pickle can be made, read back, and updated.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.pkl')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)

    pdm = periodbase.stellingwerf_pdm(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(pdm, dict)
    assert_allclose(pdm['bestperiod'], 3.08578956)

    # test write
    cpf = checkplot.checkplot_pickle(
        [gls, pdm],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo']
    )

    assert os.path.exists(outpath)

    # test read back
    cpd = checkplot._read_checkplot_picklefile(cpf)

    assert isinstance(cpd, dict)

    cpdkeys = set(list(cpd.keys()))
    testset = {'comments', 'finderchart', 'sigclip', 'objectid',
               'pdm', 'gls', 'objectinfo', 'status', 'varinfo',
               'normto', 'magseries', 'normmingap'}
    assert (testset - cpdkeys) == set()

    assert_allclose(cpd['gls']['bestperiod'], 1.54289477)
    assert_allclose(cpd['pdm']['bestperiod'], 3.08578956)

    # test update write to pickle
    cpd['comments'] = ('this is a test of the checkplot pickle '
                       'update mechanism. this is only a test.')
    cpfupdated = checkplot.checkplot_pickle_update(cpf, cpd)

    cpdupdated = checkplot._read_checkplot_picklefile(cpfupdated)

    assert cpdupdated['comments'] == cpd['comments']



def test_checkplot_pickle_topng():
    '''Tests if a checkplot pickle can be made, read, updated, exported to PNG.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.pkl')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)

    pdm = periodbase.stellingwerf_pdm(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(pdm, dict)
    assert_allclose(pdm['bestperiod'], 3.08578956)

    # test write
    cpf = checkplot.checkplot_pickle(
        [gls, pdm],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo']
    )

    assert os.path.exists(outpath)

    # test read back
    cpd = checkplot._read_checkplot_picklefile(cpf)

    assert isinstance(cpd, dict)

    cpdkeys = set(list(cpd.keys()))
    testset = {'comments', 'finderchart', 'sigclip', 'objectid',
               'pdm', 'gls', 'objectinfo', 'status', 'varinfo',
               'normto', 'magseries', 'normmingap'}
    assert (testset - cpdkeys) == set()

    assert_allclose(cpd['gls']['bestperiod'], 1.54289477)
    assert_allclose(cpd['pdm']['bestperiod'], 3.08578956)

    # test update write to pickle
    cpd['comments'] = ('this is a test of the checkplot pickle '
                       'update mechanism. this is only a test.')
    cpfupdated = checkplot.checkplot_pickle_update(cpf, cpd)

    cpdupdated = checkplot._read_checkplot_picklefile(cpfupdated)

    assert cpdupdated['comments'] == cpd['comments']

    # export to PNG
    cpd['varinfo']['objectisvar'] = "1"
    cpd['varinfo']['varperiod'] = cpd['pdm']['bestperiod']

    exportedpng = checkplot.checkplot_pickle_to_png(cpd, 'exported-checkplot.png')
    assert (exportedpng and os.path.exists(exportedpng))
