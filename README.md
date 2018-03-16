[![DOI](https://zenodo.org/badge/75150575.svg)](https://zenodo.org/badge/latestdoi/75150575)

This is a bunch of Python modules I wrote for my astronomy work with the HAT
surveys, mostly focused on handling light curves and characterizing variable
stars. Module functions that deal with light curves (e.g. in the modules
`astrobase.lcmath`, `astrobase.periodbase`, `astrobase.varbase`,
`astrobase.plotbase`, and `astrobase.checkplot`) usually just require three
numpy ndarrays as input: `times`, `mags`, and `errs`, so they should work with
any time-series data that can be represented in this form. If you have flux time
series measurements, most functions take a `magsarefluxes` keyword argument that
makes them handle flux light curves correctly.

Full documentation is still a work in progress (as soon as I figure out how
Sphinx works), but the docstrings are fairly good and an [overview](#contents)
is provided below, along with Jupyter notebooks that demonstrate some of the
functionality in a [companion repository](https://github.com/waqasbhatti/astrobase-notebooks).

To install **[astrobase](https://pypi.python.org/pypi/astrobase)** from the
Python Package Index (PyPI):

```bash
$ pip install numpy # needed to set up Fortran wrappers
$ pip install astrobase
```

The package should work with Python >= 3.4 and Python 2.7. Using the newest
Python 3 version available is recommended. See the [installation
instructions](#installation) below for details.

Python 2.7: [![Python 2.7](https://ci.wbhatti.org/buildStatus/icon?job=astrobase)](https://ci.wbhatti.org/job/astrobase) Python 3.6: [![Python 3.6](https://ci.wbhatti.org/buildStatus/icon?job=astrobase-py3)](https://ci.wbhatti.org/job/astrobase-py3)

# Contents

## notebooks

These are now located over at
[astrobase-notebooks](https://github.com/waqasbhatti/astrobase-notebooks).

## astrobase

Most of the modules with useful external functions live in here. The
`astrobase.conf` file contains module-wide settings that may need to be tweaked
for your purposes.

- **[astrokep](astrobase/astrokep.py)**: contains functions for dealing with
  Kepler and K2 Mission light curves from STScI MAST (reading the FITS files,
  consolidating light curves for objects over quarters), and some basic
  operations (converting fluxes to mags, decorrelation of light curves,
  filtering light curves, and fitting object centroids for eclipse analysis,
  etc.)

- **[checkplot](astrobase/checkplot.py)**: contains functions to make
  checkplots: a grid of plots used to quickly decide if a period search for a
  possibly variable object was successful. Checkplots come in two forms:

  Python pickles: If you want to interactively browse through large numbers of
  checkplots (e.g., as part of a large variable star classification project),
  you can use the `checkplotserver` webapp that works on checkplot pickle
  files. This interface (see below for an example) allows you to set and save
  variability tags, object type tags, best periods and epochs, and comments for
  each object using a browser-based UI (see below). The information entered can
  then be exported as CSV or JSON for the next stage of work. The
  [lightcurves-and-checkplots](https://github.com/waqasbhatti/astrobase-notebooks/blob/master/lightcurves-and-checkplots.ipynb)
  Jupyter notebook outlines how to do this. A more detailed example using light curves of an arbitrary format is available
  in the
  [lc-collection-work](https://nbviewer.jupyter.org/github/waqasbhatti/astrobase-notebooks/blob/master/lc-collection-work.ipynb)
  notebook, which shows how to add in support for a custom LC format, add
  neighbor, cross-match, and color-mag diagram info to checkplots, and visualize
  these with the checkplotserver.

  ![Checkplot Server](astrobase/data/checkplotserver.png?raw=true)

  PNG images: Alternatively, if you want to simply glance through lots of
  checkplots (e.g. for an initial look at a collection of light curves), there's
  a tiny `checkplot-viewer` webapp available (see below for an example) that
  operates on checkplot PNG images. The
  [lightcurve-work](https://github.com/waqasbhatti/astrobase-notebooks/blob/master/lightcurve-work.ipynb) Jupyter notebook goes
  through an example of generating these checkplot PNGs for light curves. See
  the [checkplot-viewer.js](astrobase/cpserver/checkplot-viewer.js) file for more
  instructions and [checkplot-viewer.png](astrobase/data/checkplot-viewer.png)
  for a screenshot.

- **[coordutils](astrobase/coordutils.py)**: functions for dealing with
  coordinates (conversions, distances, proper motion)

- **[emailutils](astrobase/emailutils.py)**: contains a simple emailer
  function suitable for use in long-running scripts and the like; this uses the
  provided credentials and server to send messages

- **[fortney2k7](astrobase/services/fortney2k7.py)**: giant planet models from Fortney
  et al. 2007, ApJ, 2659, 1661 made importable as Python dicts

- **[hatsurveys](astrobase/hatsurveys)**: modules to read, filter, and normalize
  light curves from various HAT surveys

- **[lcdb](astrobase/lcdb.py)**: a lightweight wrapper around the
  `psycopg2` library to talk to PostgreSQL database servers

- **[lcmath](astrobase/lcmath.py)**: functions for light curve operations such
  as phasing, normalization, binning (in time and phase), sigma-clipping,
  external parameter decorrelation (EPD), etc.

- **[lcproc](astrobase/lcproc.py)**: driver functions for running an end-to-end
    pipeline including: (i) object selection from a collection of light curves
    by position, cross-matching to external catalogs, or light curve objectinfo
    keys, (ii) running variability feature calculation and detection, (iii)
    running period-finding, and (iv) object review using the checkplotserver
    webapp for variability classification.

- **[periodbase](astrobase/periodbase)**: parallelized functions (using
  `multiprocessing.map`) to run fast period searches on light curves, including:
  the generalized Lomb-Scargle algorithm from Zechmeister & Kurster
  ([2008](http://adsabs.harvard.edu/abs/2009A%26A...496..577Z);
  **[periodbase.zgls](astrobase/periodbase/zgls.py)**), the phase dispersion
  minimization algorithm from Stellingwerf
  ([1978](http://adsabs.harvard.edu/abs/1978ApJ...224..953S),
  [2011](http://adsabs.harvard.edu/abs/2011rrls.conf...47S);
  **[periodbase.spdm](astrobase/periodbase/spdm.py)**), the AoV and
  AoV-multiharmonic algorithms from Schwarzenberg-Czerny
  ([1989](http://adsabs.harvard.edu/abs/1989MNRAS.241..153S),
  [1996](http://adsabs.harvard.edu/abs/1996ApJ...460L.107S);
  **[periodbase.saov](astrobase/periodbase/saov.py)**,
  **[periodbase.smav](astrobase/periodbase/smav.py)**), the BLS algorithm from
  Kovacs et al. ([2002](http://adsabs.harvard.edu/abs/2002A%26A...391..369K);
  **[periodbase.kbls](astrobase/periodbase/kbls.py)**), and the ACF
  period-finding algorithm from McQuillan et
  al. ([2013a](http://adsabs.harvard.edu/abs/2013MNRAS.432.1203M),
  [2014](http://adsabs.harvard.edu/abs/2014ApJS..211...24M);
  **[periodbase.macf](astrobase/periodbase/macf.py)**).

- **[plotbase](astrobase/plotbase.py)**: functions to plot light curves, phased
  light curves, periodograms, and download Digitized Sky Survey cutouts from the
  NASA SkyView service.

- **[services](astrobase/services)**: modules and functions to query various
  astronomical catalogs and data services, including GAIA, TRILEGAL, NASA
  SkyView, and 2MASS DUST.

- **[timeutils](astrobase/timeutils.py)**: functions for converting from
  Julian dates to Baryocentric Julian dates, and precessing coordinates between
  equinoxes and due to proper motion; this will automatically download and save
  the JPL ephemerides **de430.bsp** from JPL upon first import

- **[varbase](astrobase/varbase)**: functions for calculating variability
  indices for light curves, fitting and obtaining Fourier coefficients for use
  in classifications, and other variability features

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

For some extra functionality:

- for `astrobase.lcdb` to work, you'll also need psycopg2

## Installing with pip

If you're using:

- 64-bit Linux and Python 2.7, 3.4, 3.5, 3.6
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

Or if you want to install optional dependencies as well:

```bash
(venv)$ pip install astrobase[all]
```

Finally, if you want the latest version:

```bash
$ git clone https://github.com/waqasbhatti/astrobase
$ cd astrobase
$ python setup.py install
$ # or use pip install . to install requirements automatically
$ # or use pip install -e . to install in develop mode along with requirements
```

# License

`astrobase` is provided under the MIT License. See the LICENSE file for the full
text.
