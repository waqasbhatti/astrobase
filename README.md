[![DOI](https://zenodo.org/badge/75150575.svg)](https://zenodo.org/badge/latestdoi/75150575) [![Documentation Status](https://readthedocs.org/projects/astrobase/badge/?version=latest)](https://astrobase.readthedocs.io/en/latest/?badge=latest)

Astrobase is a Python package for analyzing light curves and finding variable
stars. It includes implementations of several period-finding algorithms, batch
work drivers for working on large collections of light curves, and a small
web-app useful for reviewing and classifying light curves by stellar variability
type.

Most functions in this package that deal with light curves usually require three
Numpy ndarrays as input: `times`, `mags`, and `errs`, so they should work with
any time-series data that can be represented in this form. If you have flux time
series measurements, most functions also take a `magsarefluxes` keyword argument
that makes them handle flux light curves correctly.

- Read the docs: https://astrobase.readthedocs.io/en/latest/
- Jupyter notebooks that demonstrate some of the functionality are available in
  the [astrobase-notebooks](https://github.com/waqasbhatti/astrobase-notebooks)
  repository.
- A [overview](#contents) of the modules and subpackages is provided below.

Install **[astrobase](https://pypi.org/project/astrobase/)** from the
Python Package Index (PyPI):

```bash
$ pip install numpy # needed to set up Fortran wrappers
$ pip install astrobase
```

See the [installation instructions](#installation) below for details. This
package requires Python >= 3.5 as of version 0.5.0. Use `pip install
astrobase<0.5.0` for older Python versions.

Python 3.6: [![Python
3.6](https://ci.wbhatti.org/buildStatus/icon?job=astrobase-py3)](https://ci.wbhatti.org/job/astrobase-py3)
Python 3.7: [![Python
3.7](https://ci.wbhatti.org/buildStatus/icon?job=astrobase-py37)](https://ci.wbhatti.org/job/astrobase-py37)
Python 3.8: [![Python 3.8](https://ci.wbhatti.org/buildStatus/icon?job=astrobase-py38)](https://ci.wbhatti.org/job/astrobase-py38)

# Contents

- **[astrokep](https://astrobase.readthedocs.io/en/latest/astrobase.astrokep.html)**:
  contains functions for dealing with Kepler and K2 Mission light curves from
  STScI MAST (reading the FITS files, consolidating light curves for objects
  over quarters), and some basic operations (converting fluxes to mags,
  decorrelation of light curves, filtering light curves, and fitting object
  centroids for eclipse analysis, etc.)

- **[astrotess](https://astrobase.readthedocs.io/en/latest/astrobase.astrotess.html)**:
  contains functions for dealing with TESS 2-minute cadence light curves from
  STScI MAST (reading the FITS files, consolidating light curves for objects
  over sectors), and some basic operations (converting fluxes to mags, filtering
  light curves, etc.)

- **[checkplot](https://astrobase.readthedocs.io/en/latest/astrobase.checkplot.html)**:
  contains functions to make checkplots: a grid of plots used to quickly decide
  if a period search for a possibly variable object was successful. Checkplots
  come in two forms:

  Python pickles: If you want to interactively browse through large numbers of
  checkplots (e.g., as part of a large variable star classification project),
  you can use the `checkplotserver` webapp that works on checkplot pickle
  files. This interface allows you to review all phased light curves from all
  period-finder methods applied, set and save variability tags, object type
  tags, best periods and epochs, and comments for each object using a
  browser-based UI (see below). The information entered can then be exported as
  CSV or JSON for the next stage of a variable star classification pipeline.

  The
  [lightcurves-and-checkplots](https://nbviewer.jupyter.org/github/waqasbhatti/astrobase-notebooks/blob/master/lightcurves-and-checkplots.ipynb)
  Jupyter notebook outlines how to do this. A more detailed example using light
  curves of an arbitrary format is available in the
  [lc-collection-work](https://nbviewer.jupyter.org/github/waqasbhatti/astrobase-notebooks/blob/master/lc-collection-work.ipynb)
  notebook, which shows how to add in support for a custom LC format, add
  neighbor, cross-match, and color-mag diagram info to checkplots, and visualize
  these with the checkplotserver.

  ![Checkplot Server](https://raw.githubusercontent.com/waqasbhatti/astrobase/master/astrobase/data/checkplotserver.png)

  PNG images: Alternatively, if you want to simply glance through lots of
  checkplots (e.g. for an initial look at a collection of light curves), there's
  a `checkplot-viewer` webapp available that operates on checkplot PNG
  images. The
  [lightcurve-work](https://nbviewer.jupyter.org/github/waqasbhatti/astrobase-notebooks/blob/master/lightcurve-work.ipynb)
  Jupyter notebook goes through an example of generating these checkplot PNGs
  for light curves. See the
  [checkplot-viewer.js](https://github.com/waqasbhatti/astrobase/blob/master/astrobase/cpserver/checkplot-viewer.js) file for more
  instructions and [checkplot-viewer.png](https://raw.githubusercontent.com/waqasbhatti/astrobase/master/astrobase/data/checkplot-viewer.png)
  for a screenshot.

- **[coordutils](https://astrobase.readthedocs.io/en/latest/astrobase.coordutils.html)**:
  functions for dealing with coordinates (conversions, distances, proper motion)

- **[fakelcs](https://astrobase.readthedocs.io/en/latest/astrobase.fakelcs.html)**:
  modules and functions to conduct an end-to-end variable star recovery
  simulation.

- **[hatsurveys](https://astrobase.readthedocs.io/en/latest/astrobase.hatsurveys.html)**:
  modules to read, filter, and normalize light curves from various HAT surveys.

- **[lcfit](https://astrobase.readthedocs.io/en/latest/astrobase.lcfit.html)**:
  functions for fitting light curve models to observations, including
  sinusoidal, trapezoidal and full Mandel-Agol planet transits, eclipses, and
  splines.

- **[lcmath](https://astrobase.readthedocs.io/en/latest/astrobase.lcmath.html)**: functions for light curve operations such
  as phasing, normalization, binning (in time and phase), sigma-clipping,
  external parameter decorrelation (EPD), etc.

- **[lcmodels](https://astrobase.readthedocs.io/en/latest/astrobase.lcmodels.html)**:
  first order models for fast fitting (for the purposes of variable
  classification) to various periodic variable types, including sinusoidal
  variables, eclipsing binaries, transiting planets, and flares.

- **[lcproc](https://astrobase.readthedocs.io/en/latest/astrobase.lcproc.html)**:
    driver functions for running an end-to-end pipeline including: (i) object
    selection from a collection of light curves by position, cross-matching to
    external catalogs, or light curve objectinfo keys, (ii) running variability
    feature calculation and detection, (iii) running period-finding, and (iv)
    object review using the checkplotserver webapp for variability
    classification.

- **[periodbase](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.html)**: parallelized functions (using
  `multiprocessing.map`) to run fast period searches on light curves, including:
  the generalized Lomb-Scargle algorithm from Zechmeister & Kurster
  ([2008](http://adsabs.harvard.edu/abs/2009A%26A...496..577Z);
  **[periodbase.zgls](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.zgls.html)**), the phase dispersion
  minimization algorithm from Stellingwerf
  ([1978](http://adsabs.harvard.edu/abs/1978ApJ...224..953S),
  [2011](http://adsabs.harvard.edu/abs/2011rrls.conf...47S);
  **[periodbase.spdm](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.spdm.html)**), the AoV and
  AoV-multiharmonic algorithms from Schwarzenberg-Czerny
  ([1989](http://adsabs.harvard.edu/abs/1989MNRAS.241..153S),
  [1996](http://adsabs.harvard.edu/abs/1996ApJ...460L.107S);
  **[periodbase.saov](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.saov.html)**,
  **[periodbase.smav](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.smav.html)**), the BLS algorithm from
  Kovacs et al. ([2002](http://adsabs.harvard.edu/abs/2002A%26A...391..369K);
  **[periodbase.kbls](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.kbls.html)**
  and **[periodbase.abls](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.abls.html)**),
  the similar TLS algorithm from Hippke & Heller
  ([2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract);
  **[periodbase.htls](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.htls.html)**),
  and the ACF period-finding algorithm from McQuillan et al.
  ([2013a](http://adsabs.harvard.edu/abs/2013MNRAS.432.1203M),
  [2014](http://adsabs.harvard.edu/abs/2014ApJS..211...24M);
  **[periodbase.macf](https://astrobase.readthedocs.io/en/latest/astrobase.periodbase.macf.html)**).

- **[plotbase](https://astrobase.readthedocs.io/en/latest/astrobase.plotbase.html)**: functions to plot light curves, phased
  light curves, periodograms, and download Digitized Sky Survey cutouts from the
  NASA SkyView service.

- **[services](https://astrobase.readthedocs.io/en/latest/astrobase.services.html)**: modules and functions to query various
  astronomical catalogs and data services, including GAIA, SIMBAD, TRILEGAL,
  NASA SkyView, and 2MASS DUST.

- **[timeutils](https://astrobase.readthedocs.io/en/latest/astrobase.timeutils.html)**: functions for converting from
  Julian dates to Baryocentric Julian dates, and precessing coordinates between
  equinoxes and due to proper motion; this will automatically download and save
  the JPL ephemerides **de430.bsp** from JPL upon first import.

- **[varbase](https://astrobase.readthedocs.io/en/latest/astrobase.varbase.html)**:
  functions for calculating auto-correlation features, masking and pre-whitening
  periodic signals in light curves, and planet transit specific tools.

- **[varclass](https://astrobase.readthedocs.io/en/latest/astrobase.varclass.html)**: functions for calculating various
  variability, stellar color and motion, and neighbor proximity features, along
  with a Random Forest based classifier.


# Changelog

Please see https://github.com/waqasbhatti/astrobase/blob/master/CHANGELOG.md for
a list of changes applicable to tagged release versions.


# Installation

## Requirements

This package requires the following other packages:

- numpy
- scipy
- astropy
- matplotlib
- Pillow
- jplephem
- requests
- tornado
- pyeebls
- tqdm
- scikit-learn

For optional functionality, some additional packages are required:

- for `astrobase.lcdb` to work, you'll need `psycopg2-binary`.
- for `lcfit.transits.mandelagol_fit_magseries`, you'll need `batman-package`,
  `emcee`, `corner`, and `h5py`.
- for `lcproc.awsrun`, you'll need `paramiko`, `boto3`, and `awscli`.
- for `periodbase.tls`, you'll need `transitleastsquares`

## Installing with pip

If you're using:

- 64-bit Linux and Python 2.7, 3.4, 3.5, 3.6, 3.7
- 64-bit Mac OSX 10.12+ with Python 2.7 or 3.6
- 64-bit Windows with Python 2.7 and 3.6

You can simply install astrobase with:

```bash

(venv)$ pip install astrobase
```

Otherwise, you'll need to make sure that a Fortran compiler and numpy are
installed beforehand to compile the pyeebls package that astrobase depends on:

```bash
## you'll need a Fortran compiler.                              ##
## on Linux: dnf/yum/apt install gcc gfortran                   ##
## on OSX (using homebrew): brew install gcc && brew link gcc   ##

## make sure numpy is installed as well!                        ##
## this is required for the pyeebls module installation         ##

(venv)$ pip install numpy # in a virtualenv
# or use dnf/yum/apt install numpy to install systemwide
```

Once that's done, install astrobase.

```bash
(venv)$ pip install astrobase
```

### Other installation methods

Install all the optional dependencies as well:

```bash
(venv)$ pip install astrobase[all]
```

Install the latest version (may be unstable at times):

```bash
$ git clone https://github.com/waqasbhatti/astrobase
$ cd astrobase
$ python setup.py install
$ # or use pip install . to install requirements automatically
$ # or use pip install -e . to install in develop mode along with requirements
$ # or use pip install -e .[all] to install in develop mode along with all requirements
```

# License

`astrobase` is provided under the MIT License. See the LICENSE file for the full
text.
