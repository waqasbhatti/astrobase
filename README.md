This is a bunch of Python modules I wrote for my astronomy work with the HAT
surveys, mostly focused on variable stars.

Full documentation is still a work in progress (as soon as I figure out how
Sphinx works), but the docstrings are fairly good and an [overview](#contents)
is provided below. See the instructions for [installation](#installation) to get
started.

astrobase should work with Python >= 3.4 and Python 2.7. Using the newest Python
3 version available is recommended.

# Contents

## notebooks

This contains Jupyter notebooks that demonstrate various functions from this
package. Also contains other useful notes-to-self.

- **[lightcurve-work](notebooks/lightcurve-work.ipynb)**: demonstrates usage of
    the [hatlc](astrobase/hatlc.py), [periodbase](astrobase/periodbase.py), and
    [checkplot](astrobase/checkplot.py) modules for reading HAT light curves,
    finding periods, and plotting phased light curves

- **[parallel-ipython](notebooks/parallel-ipython.ipynb)**: shows examples of
    how to map `astrobase` functions across an
    [ipyparallel](http://ipyparallel.readthedocs.io/en/stable/) cluster to speed
    up light curve processing

## astrobase

Most of the modules with useful external functions live in here. The
`astrobase.conf` file contains module-wide settings that may need to be
tweaked for your purposes.

- **[astrokep](astrobase/astrokep.py)**: contains functions for dealing with
  Kepler light curves (reading and converting) and some basic operations
  (converting fluxes to mags, decorrelation of light curves, etc.)

- **[checkplot](astrobase/checkplot.py)**: contains functions to make
  checkplots: a grid of plots used to quickly decide if a period search
  was successful. Checkplots come in two forms:

  PNG images: If you want to simply glance through lots of checkplots, e.g. for
  an initial look at a collection of light curves; there's a tiny
  checkplot-viewer webapp available. An example of using this is shown
  below. See the [checkplot-viewer.js](astrobase/checkplot-viewer.js) file for
  instructions.

  ![Checkplot viewer](astrobase/data/checkplot-viewer.png?raw=true)

  Python pickles: Alternatively, if you want to interactively browse through
  checkplots prepared as part of a large variable star classification project,
  for example, you can use the `checkplotserver` webapp that works on checkplot
  pickle files. This interface (see below for an example) allows you to set and
  save variability tags, object type tags, and best periods and epochs for each
  object using a browser-based UI (see below). The information entered can then
  be exported as CSV or JSON for the next stage of work. The
  [lightcurves-and-checkplots](https://github.com/waqasbhatti/astrobase/blob/master/notebooks/lightcurves-and-checkplots.ipynb)
  Jupyter notebook details how to do this and goes through a full example.

  ![Checkplot Server](astrobase/data/checkplotserver.png?raw=true)

- **[coordutils](astrobase/coordutils.py)**: functions for dealing with
  coordinates (conversions, distances, proper motion)

- **[emailutils](astrobase/emailutils.py)**: contains a simple emailer
  function suitable for use in long-running scripts and the like; this uses the
  provided credentials and server to send messages

- **[fortney2k7](astrobase/fortney2k7.py)**: giant planet models from Fortney
  et al. 2007, ApJ, 2659, 1661 made importable as Python dicts

- **[glsp](astrobase/glsp.py)**: simple implementation of the Generalized
  Lomb-Scargle periodogram from Zechmeister and Kurster (2008); use the more
  optimized functions in `periodbase` for actual work

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

- **[periodbase](astrobase/periodbase.py)**: parallelized functions (using
  `multiprocessing.map`) to run fast period searches on light curves, including:
  the generalized Lomb-Scargle algorithm from Zechmeister & Kurster (2008), the
  phase dispersion minimization algorithm from Stellingwerf (1978, 2011), the
  AoV algorithm from Schwarzenberg-Cerny (1989), and the BLS algorithm from
  Kovacs et al. (2002)

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

- **[varbase](astrobase/varbase.py)**: functions for calculating variability
  indices for light curves, fitting and obtaining Fourier coefficients for use
  in classifications, and other variability features

## bls

This wraps `eebls.f` from Geza Kovacs. Extracted from
[python-bls](http://github.com/dfm/python-bls) by Daniel Foreman-Mackey, Ruth
Angus, and others. Used as the BLS implementation by `astrobase.periodbase`
functions. See its [README](bls/README.md) for details.

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

You might need to install `openssl-devel` or a similar RPM/DEB package for the
`python-cryptography` module that gets pulled in as a dependency for
`astroquery`. For some extra functionality, you'll need the following modules:

- for `astrobase.lcdb` to work, you'll also need psycogp2

First, make sure numpy and a Fortran compiler are installed:

```bash
## make sure numpy is installed first!                ##
## this is required for the bls module installation   ##

$ pip install numpy # in a virtualenv
# or use dnf/yum/apt install numpy to install systemwide

## you'll need a Fortran compiler for the bls module! ##
## on Linux: dnf/yum/apt install gcc gcc-gfortran     ##
## on OSX (using homebrew): brew install gcc          ##
```

Next, install astrobase.

```bash
$ git clone https://github.com/waqasbhatti/astrobase
$ cd astrobase
$ python setup.py install
$ # or use pip install . to install requirements automatically
$ # or use pip install -e . to install in develop mode along with requirements
```

This package isn't yet available from PyPI, but will be as soon as it becomes
more stable.


# License

`astrobase` is provided under the MIT License. See the LICENSE file for the full
text.
