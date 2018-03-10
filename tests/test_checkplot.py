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
import os
import os.path
try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve
from numpy.testing import assert_allclose

from astrobase.hatsurveys import hatlc
from astrobase import periodbase, checkplot


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

def test_hatlc():
    '''
    Tests that a HAT light curve can be loaded.

    '''

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    assert isinstance(lcd, dict)
    assert msg == 'no SQL filters, LC OK'



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

    testset = {'0-gls', '1-pdm', 'comments', 'externalplots',
               'finderchart', 'magseries', 'neighbors', 'normmingap',
               'normto', 'objectid', 'objectinfo', 'pfmethods', 'sigclip',
               'signals', 'status', 'varinfo'}
    assert (testset - cpdkeys) == set()

    assert_allclose(cpd['0-gls']['bestperiod'], 1.54289477)
    assert_allclose(cpd['1-pdm']['bestperiod'], 3.08578956)



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
    testset = {'0-gls', '1-pdm', 'comments', 'externalplots',
               'finderchart', 'magseries', 'neighbors', 'normmingap',
               'normto', 'objectid', 'objectinfo', 'pfmethods', 'sigclip',
               'signals', 'status', 'varinfo'}
    assert (testset - cpdkeys) == set()

    assert_allclose(cpd['0-gls']['bestperiod'], 1.54289477)
    assert_allclose(cpd['1-pdm']['bestperiod'], 3.08578956)

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
    testset = {'0-gls', '1-pdm', 'comments', 'externalplots',
               'finderchart', 'magseries', 'neighbors', 'normmingap',
               'normto', 'objectid', 'objectinfo', 'pfmethods', 'sigclip',
               'signals', 'status', 'varinfo'}
    assert (testset - cpdkeys) == set()

    assert_allclose(cpd['0-gls']['bestperiod'], 1.54289477)
    assert_allclose(cpd['1-pdm']['bestperiod'], 3.08578956)

    # test update write to pickle
    cpd['comments'] = ('this is a test of the checkplot pickle '
                       'update mechanism. this is only a test.')
    cpfupdated = checkplot.checkplot_pickle_update(cpf, cpd)

    cpdupdated = checkplot._read_checkplot_picklefile(cpfupdated)

    assert cpdupdated['comments'] == cpd['comments']

    # export to PNG
    cpd['varinfo']['objectisvar'] = "1"
    cpd['varinfo']['varperiod'] = cpd['1-pdm']['bestperiod']

    exportedpng = checkplot.checkplot_pickle_to_png(cpd, 'exported-checkplot.png')
    assert (exportedpng and os.path.exists(exportedpng))



def test_checkplot_with_multiple_same_pfmethods():
    '''
    This tests running the same period-finder for different period ranges.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.pkl')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)

    gls_1 = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
                                startp=0.01,endp=0.1)
    gls_2 = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
                                startp=0.1,endp=300.0)
    pdm_1 = periodbase.stellingwerf_pdm(lcd['rjd'],
                                        lcd['aep_000'],
                                        lcd['aie_000'],
                                        startp=0.01,endp=0.1)
    pdm_2 = periodbase.stellingwerf_pdm(lcd['rjd'],
                                        lcd['aep_000'],
                                        lcd['aie_000'],
                                        startp=0.1,endp=300.0)

    assert isinstance(gls_1, dict)
    assert isinstance(gls_2, dict)
    assert isinstance(pdm_1, dict)
    assert isinstance(pdm_2, dict)

    cpf = checkplot.checkplot_pickle(
        [gls_1, gls_2, pdm_1, pdm_2],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo']
    )

    assert os.path.exists(cpf)
    assert os.path.abspath(cpf) == os.path.abspath(outpath)

    cpd = checkplot._read_checkplot_picklefile(cpf)
    pfmethods = list(cpd['pfmethods'])

    assert len(pfmethods) == 4

    assert '0-gls' in pfmethods
    assert '1-gls' in pfmethods
    assert '2-pdm' in pfmethods
    assert '3-pdm' in pfmethods


def test_checkplot_png_varepoch_handling(capsys):
    '''This tests the various different ways to give varepoch to checkplot_png.

    Uses the py.test capsys fixture to capture stdout and see if we reported the
    correct epoch being used.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.png')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)


    # 1. usual handling where epoch is None
    # should use min(times) as epoch all the time
    cpf = checkplot.checkplot_png(gls,
                                  lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
                                  outfile=outpath,
                                  objectinfo=lcd['objectinfo'],
                                  varepoch=None)

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'plotting phased LC' in x]

    # these are the periods and epochs to match per line
    lookfor = ['period 1.542895, epoch 56092.64056',
               'period 0.771447, epoch 56092.64056',
               'period 3.085790, epoch 56092.64056',
               'period 0.771304, epoch 56092.64056',
               'period 0.514234, epoch 56092.64056',
               'period 3.085790, epoch 56092.64056',
               'period 3.029397, epoch 56092.64056']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 2. handle varepoch = 'min'
    cpf = checkplot.checkplot_png(gls,
                                  lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
                                  outfile=outpath,
                                  objectinfo=lcd['objectinfo'],
                                  varepoch='min')

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'plotting phased LC' in x]

    # these are the periods and epochs to match per line
    lookfor = ['period 1.542895, epoch 56341.11968',
               'period 0.771447, epoch 56848.65894',
               'period 3.085790, epoch 56828.62835',
               'period 0.771304, epoch 56856.46618',
               'period 0.514234, epoch 56889.88470',
               'period 3.085790, epoch 56828.62835',
               'period 3.029397, epoch 56799.51745']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 3. handle varepoch = some float
    cpf = checkplot.checkplot_png(gls,
                                  lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
                                  outfile=outpath,
                                  objectinfo=lcd['objectinfo'],
                                  varepoch=56000.5)

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'plotting phased LC' in x]

    # these are the periods and epochs to match per line
    lookfor = ['period 1.542895, epoch 56000.50000',
               'period 0.771447, epoch 56000.50000',
               'period 3.085790, epoch 56000.50000',
               'period 0.771304, epoch 56000.50000',
               'period 0.514234, epoch 56000.50000',
               'period 3.085790, epoch 56000.50000',
               'period 3.029397, epoch 56000.50000']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 4. handle varepoch = list of floats
    cpf = checkplot.checkplot_png(gls,
                                  lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
                                  outfile=outpath,
                                  objectinfo=lcd['objectinfo'],
                                  varepoch=[56000.1,56000.2,
                                            56000.3,56000.4,
                                            56000.5,56000.6,
                                            56000.7])

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'plotting phased LC' in x]

    # these are the periods and epochs to match per line
    lookfor = ['period 1.542895, epoch 56000.10000',
               'period 0.771447, epoch 56000.20000',
               'period 3.085790, epoch 56000.30000',
               'period 0.771304, epoch 56000.40000',
               'period 0.514234, epoch 56000.50000',
               'period 3.085790, epoch 56000.60000',
               'period 3.029397, epoch 56000.70000']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline



def test_twolsp_checkplot_png_varepoch_handling(capsys):
    '''This tests the various different ways to give varepoch
    to twolsp_checkplot_png.

    Uses the py.test capsys fixture to capture stdout and see if we reported the
    correct epoch being used.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.png')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])
    pdm = periodbase.stellingwerf_pdm(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)

    # 1. usual handling where epoch is None
    # should use min(times) as epoch all the time
    cpf = checkplot.twolsp_checkplot_png(
        gls, pdm,
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo'],
        varepoch=None
    )

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'plotting phased LC' in x]

    # these are the periods and epochs to match per line
    lookfor = ['period 1.542895, epoch 56092.64056',
               'period 0.771304, epoch 56092.64056',
               'period 0.514234, epoch 56092.64056',
               'period 3.085790, epoch 56092.64056',
               'period 1.542895, epoch 56092.64056',
               'period 6.157824, epoch 56092.64056']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 2. handle varepoch = 'min'
    cpf = checkplot.twolsp_checkplot_png(
        gls, pdm,
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo'],
        varepoch='min'
    )

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'plotting phased LC' in x]

    # these are the periods and epochs to match per line
    lookfor = ['period 1.542895, epoch 56341.11968',
               'period 0.771304, epoch 56856.46618',
               'period 0.514234, epoch 56889.88470',
               'period 3.085790, epoch 56828.62835',
               'period 1.542895, epoch 56341.11968',
               'period 6.157824, epoch 56154.33367']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 3. handle varepoch = some float
    cpf = checkplot.twolsp_checkplot_png(
        gls, pdm,
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo'],
        varepoch=56000.5
    )

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'plotting phased LC' in x]

    # these are the periods and epochs to match per line
    lookfor = ['period 1.542895, epoch 56000.50000',
               'period 0.771304, epoch 56000.50000',
               'period 0.514234, epoch 56000.50000',
               'period 3.085790, epoch 56000.50000',
               'period 1.542895, epoch 56000.50000',
               'period 6.157824, epoch 56000.50000']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 4. handle varepoch = list of floats
    cpf = checkplot.twolsp_checkplot_png(
        gls, pdm,
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo'],
        varepoch=[[56000.1,56000.2,56000.3],
                  [56000.4,56000.5,56000.6]]
    )

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'plotting phased LC' in x]

    # these are the periods and epochs to match per line
    lookfor = ['period 1.542895, epoch 56000.10000',
               'period 0.771304, epoch 56000.20000',
               'period 0.514234, epoch 56000.30000',
               'period 3.085790, epoch 56000.40000',
               'period 1.542895, epoch 56000.50000',
               'period 6.157824, epoch 56000.60000']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline



def test_checkplot_pickle_varepoch_handling(capsys):
    '''This tests the various different ways to give varepoch
    to checkplot_pickle.

    Uses the py.test capsys fixture to capture stdout and see if we reported the
    correct epoch being used.

    '''

    outpath = os.path.join(os.path.dirname(LCPATH),
                           'test-checkplot.png')

    lcd, msg = hatlc.read_and_filter_sqlitecurve(LCPATH)
    gls = periodbase.pgen_lsp(lcd['rjd'], lcd['aep_000'], lcd['aie_000'])
    pdm = periodbase.stellingwerf_pdm(lcd['rjd'],
                                      lcd['aep_000'],
                                      lcd['aie_000'])

    assert isinstance(gls, dict)
    assert_allclose(gls['bestperiod'], 1.54289477)

    # 1. usual handling where epoch is None
    # should use min(times) as epoch all the time
    cpf = checkplot.checkplot_pickle(
        [gls, pdm],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo'],
        varepoch=None
    )

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'phased LC with period' in x]

    # these are the periods and epochs to match per line
    lookfor = ['gls phased LC with period 0: 1.542895, epoch: 56092.64056',
               'gls phased LC with period 1: 0.771304, epoch: 56092.64056',
               'gls phased LC with period 2: 0.514234, epoch: 56092.64056',
               'pdm phased LC with period 0: 3.085790, epoch: 56092.64056',
               'pdm phased LC with period 1: 1.542895, epoch: 56092.64056',
               'pdm phased LC with period 2: 6.157824, epoch: 56092.64056']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 2. handle varepoch = 'min'
    cpf = checkplot.checkplot_pickle(
        [gls, pdm],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo'],
        varepoch='min'
    )

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'phased LC with period' in x]

    # these are the periods and epochs to match per line
    lookfor = ['gls phased LC with period 0: 1.542895, epoch: 56341.11968',
               'gls phased LC with period 1: 0.771304, epoch: 56856.46618',
               'gls phased LC with period 2: 0.514234, epoch: 56889.88470',
               'pdm phased LC with period 0: 3.085790, epoch: 56828.62835',
               'pdm phased LC with period 1: 1.542895, epoch: 56341.11968',
               'pdm phased LC with period 2: 6.157824, epoch: 56154.33367']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 3. handle varepoch = some float
    cpf = checkplot.checkplot_pickle(
        [gls, pdm],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo'],
        varepoch=56000.5
    )

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'phased LC with period' in x]

    # these are the periods and epochs to match per line
    lookfor = ['gls phased LC with period 0: 1.542895, epoch: 56000.50000',
               'gls phased LC with period 1: 0.771304, epoch: 56000.50000',
               'gls phased LC with period 2: 0.514234, epoch: 56000.50000',
               'pdm phased LC with period 0: 3.085790, epoch: 56000.50000',
               'pdm phased LC with period 1: 1.542895, epoch: 56000.50000',
               'pdm phased LC with period 2: 6.157824, epoch: 56000.50000']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline


    # 4. handle varepoch = list of floats
    cpf = checkplot.checkplot_pickle(
        [gls, pdm],
        lcd['rjd'], lcd['aep_000'], lcd['aie_000'],
        outfile=outpath,
        objectinfo=lcd['objectinfo'],
        varepoch=[[56000.1,56000.2,56000.3],
                  [56000.4,56000.5,56000.6]]
    )

    assert os.path.exists(outpath)

    # capture the stdout
    captured = capsys.readouterr()

    # see if we used the correct periods and epochs
    splitout = captured.out.split('\n')

    # plot output lines
    plotoutlines = [x for x in splitout if 'phased LC with period' in x]

    # these are the periods and epochs to match per line
    lookfor = ['gls phased LC with period 0: 1.542895, epoch: 56000.10000',
               'gls phased LC with period 1: 0.771304, epoch: 56000.20000',
               'gls phased LC with period 2: 0.514234, epoch: 56000.30000',
               'pdm phased LC with period 0: 3.085790, epoch: 56000.40000',
               'pdm phased LC with period 1: 1.542895, epoch: 56000.50000',
               'pdm phased LC with period 2: 6.157824, epoch: 56000.60000']

    for expected, plotline in zip(lookfor,plotoutlines):
        assert expected in plotline
