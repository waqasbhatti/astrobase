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
functionality.

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

This contains Jupyter notebooks that demonstrate various functions from this
package. Also contains other useful notes-to-self.

- **[lightcurve-work](notebooks/lightcurve-work.ipynb)**: demonstrates usage of
    the [hatlc](astrobase/hatlc.py), [periodbase](astrobase/periodbase.py), and
    [checkplot](astrobase/checkplot.py) modules for reading HAT light curves,
    finding periods, and plotting phased light curves.

- **[lightcurves-and-checkplots](notebooks/lightcurves-and-checkplots.ipynb)**:
    demonstrates usage of the [hatlc](astrobase/hatlc.py),
    [periodbase](astrobase/periodbase.py), [checkplot](astrobase/checkplot.py)
    modules, and the [checkplotserver](astrobase/checkplotserver.py) for doing
    period-finding and variability-classification work on a collection of light
    curves.

- **[parallel-ipython](notebooks/parallel-ipython.ipynb)**: shows examples of
    how to map `astrobase` functions across an
    [ipyparallel](http://ipyparallel.readthedocs.io/en/stable/) cluster to speed
    up light curve processing.

## astrobase

Most of the modules with useful external functions live in here. The
`astrobase.conf` file contains module-wide settings that may need to be tweaked
for your purposes.

- **[astrokep](astrobase/astrokep.py)**: contains functions for dealing with
  Kepler light curves (reading and converting) and some basic operations
  (converting fluxes to mags, decorrelation of light curves, etc.)

- **[checkplot](astrobase/checkplot.py)**: contains functions to make
  checkplots: a grid of plots used to quickly decide if a period search
  was successful. Checkplots come in two forms:

  Python pickles: If you want to interactively browse through large numbers of
  checkplots (e.g., as part of a large variable star classification project),
  you can use the `checkplotserver` webapp that works on checkplot pickle
  files. This interface (see below for an example) allows you to set and save
  variability tags, object type tags, best periods and epochs, and comments for
  each object using a browser-based UI (see below). The information entered can
  then be exported as CSV or JSON for the next stage of work. The
  [lightcurves-and-checkplots](notebooks/lightcurves-and-checkplots.ipynb)
  Jupyter notebook details how to do this and goes through a full example.

  ![Checkplot Server](astrobase/data/checkplotserver-th.png?raw=true)

  PNG images: Alternatively, if you want to simply glance through lots of
  checkplots (e.g. for an initial look at a collection of light curves), there's
  a tiny `checkplot-viewer` webapp available (see below for an example) that
  operates on checkplot PNG images. The
  [lightcurve-work](notebooks/lightcurve-work.ipynb) Jupyter notebook goes
  through an example of generating these checkplot PNGs for light curves. See
  the [checkplot-viewer.js](astrobase/checkplot-viewer.js) file for more
  instructions.

  ![Checkplot Viewer](astrobase/data/checkplot-viewer.png?raw=true)

- **[coordutils](astrobase/coordutils.py)**: functions for dealing with
  coordinates (conversions, distances, proper motion)

- **[emailutils](astrobase/emailutils.py)**: contains a simple emailer
  function suitable for use in long-running scripts and the like; this uses the
  provided credentials and server to send messages

- **[fortney2k7](astrobase/fortney2k7.py)**: giant planet models from Fortney
  et al. 2007, ApJ, 2659, 1661 made importable as Python dicts

- **[hatlc](astrobase/hatlc.py)**: functions to read, filter, and normalize
  new generation light curves from the HAT data server; the format is described
  here: http://data.hatsurveys.org/docs/lcformat

- **[imageutils](astrobase/imageutils.py)**: various functions to deal with
  FITS images: reading headers, generating postage stamps, converting to JPEGs,
  and checking for warps

- **[lcdb](astrobase/lcdb.py)**: a lightweight wrapper around the
  `psycopg2` library to talk to PostgreSQL database servers

- **[lcmath](astrobase/lcmath.py)**: functions for light curve operations such
  as phasing, normalization, binning (in time and phase), sigma-clipping,
  external parameter decorrelation (EPD), etc.

- **[oldhatlc](astrobase/oldhatlc.py)**: functions to read light curves in the
  older HAT light curve format (i.e. those from http://hatnet.org and
  http://hatsouth.org); the format is described here:
  http://hatnet.org/planets/discovery-hatlcs.html#lightcurve-schema

- **[periodbase](astrobase/periodbase)**: parallelized functions (using
  `multiprocessing.map`) to run fast period searches on light curves, including:
  the generalized Lomb-Scargle algorithm from Zechmeister & Kurster (2008;
  **[periodbase.zgls](astrobase/periodbase/zgls.py)**), the phase dispersion
  minimization algorithm from Stellingwerf (1978, 2011;
  **[periodbase.spdm](astrobase/periodbase/spdm.py)**), the AoV algorithm from
  Schwarzenberg-Cerny (1989;
  **[periodbase.saov](astrobase/periodbase/saov.py)**), and the BLS algorithm
  from Kovacs et al. (2002;
  **[periodbase.kbls](astrobase/periodbase/kbls.py)**).

- **[plotbase](astrobase/plotbase.py)**: functions to plot light curves, phased
  light curves, periodograms, and download cutouts using `astroquery` and the
  NASA SkyView service.

- **[texthatlc](astrobase/texthatlc.py)**: contains a function to read some
    original HAT text light curves (.epdlc, .tfalc). These are produced by the
    HAT pipeline, are most useful for internal HAT work, and may not contain all
    measurements from overlapping observations or any object metadata. Using the
    public HAT data server light curves (-hatlc.sqlite.gz, -hatlc.csv.gz) and
    reading these using **[hatlc](astrobase/hatlc.py)** is recommended instead.

- **[timeutils](astrobase/timeutils.py)**: functions for converting from
  Julian dates to Baryocentric Julian dates, and precessing coordinates between
  equinoxes and due to proper motion; this will automatically download and save
  the JPL ephemerides **de430.bsp** from JPL upon first import

- **[varbase](astrobase/varbase)**: functions for calculating variability
  indices for light curves, fitting and obtaining Fourier coefficients for use
  in classifications, and other variability features

# Installation

This package requires the following other packages:

- numpy
- scipy
- astropy
- matplotlib
- Pillow
- jplephem
- astroquery
- tornado
- pyeebls

You might need to install `openssl-devel` or a similar RPM/DEB package for the
`python-cryptography` module that gets pulled in as a dependency for
`astroquery`. For some extra functionality, you'll need the following modules:

- for `astrobase.lcdb` to work, you'll also need psycopg2

First, make sure numpy and a Fortran compiler are installed:

```bash
## you'll need a Fortran compiler.                      ##
## on Linux: dnf/yum/apt install gcc gcc-gfortran       ##
## on OSX (using homebrew): brew install gcc            ##

## make sure numpy is installed first!                  ##
## this is required for the pyeebls module installation ##
(venv)$ pip install numpy # in a virtualenv
# or use dnf/yum/apt install numpy to install systemwide
```

Next, install astrobase.

```bash
(venv)$ pip install astrobase
```

Or if you want the latest version:

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
