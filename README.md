This is a bunch of Python modules I wrote for my astronomy work with the HAT
surveys, mostly focused on variable stars. Full documentation is still a work in
progress (as soon as I figure out how Sphinx works), but the docstrings are
fairly good and an overview is provided below.

# Installation

This package requires the following other packages:

- numpy
- scipy
- astropy
- matplotlib
- Pillow
- jplephem
- simplejson
- astroquery

For `lcdb` to work, you'll also need:

- psycopg2

To install:

```bash
$ git clone https://github.com/waqasbhatti/astrobase
$ cd astrobase
$ python setup.py install # or use pip install -e . to get requirements as well
```


# License

`astrobase` uses the MIT License. See the LICENSE file for the full text.


# Modules

## `astrobase`

Most of the modules with useful external functions live in here. The
`astrobase.conf` file contains module-wide settings that may need to be
tweaked for your purposes.

- ``[astrokep](astrobase/astrokep.py)`: contains functions for dealing with
  Kepler light curves (reading and converting) and some basic operations
  (converting fluxes to mags, decorrelation of light curves, etc.)

- `[coordutils](astrobase/coordutils.py)`: functions for dealing with
  coordinates (conversions, distances, proper motion)

- `[emailutils](astrobase/emailutils.py)`: contains a simple emailer
  function suitable for use in long-running scripts and the like; this uses the
  provided credentials and server to send messages

- `[epd](astrobase/epd.py)`: a simple implementation of External Parameter
  Decorrelation (Bakos et al. 2010:
  http://adsabs.harvard.edu/abs/2010ApJ...710.1724B) for magnitude time series

- `[fortney2k7](astrobase/fortney2k7.py)`: giant planet models from Fortney
  et al. 2007, ApJ, 2659, 1661 made importable as Python dicts

- `[glsp](astrobase/glsp.py)`: simple implementation of the Generalized
  Lomb-Scargle periodogram from Zechmeister and Kurster (2008); use the more
  optimized functions in `periodbase` for actual work

- `[hatlc](astrobase/hatlc.py)`: functions to read, filter, and normalize
  new generation light curves from the HAT data server; the format is described
  here: http://data.hatsurveys.org/docs/lcformat

- `[imageutils](astrobase/imageutils.py)`: various functions to deal with
  FITS images: reading headers, generating postage stamps, converting to JPEGs,
  and checking for warps

- `[lcdb](astrobase/lcdb.py)`: a lightweight wrapper around the
  `psycopg2` library to talk to PostgreSQL database servers

- `[lcmath](astrobase/lcmath.py)`: functions for light curve operations such
  as phasing, normalization, binning, sigma-clipping ,etc.

- `[oldhatlc](astrobase/oldhatlc.py)`: functions to read light curves in the
  older HAT light curve format (i.e. those from http://hatnet.org and
  http://hatsouth.org); the format is described here:
  http://hatnet.org/planets/discovery-hatlcs.html#lightcurve-schema

- `[periodbase](astrobase/periodbase.py)`: parellized functions to run
  period searches on light curves, including: the generalized Lomb-Scargle
  algorithm from Zechmeister & Kurster (2008), the string length algorithm from
  Dworetsky (1983), the phase dispersion minimization algorithm from
  Stellingwerf (1978, 2011), the AoV algorithm from Schwarzenberg-Cerny (1989),
  and the BLS algorithm from Kovacs et al. (2002)

- `[plotbase](astrobase/plotbase.py)`: functions to plot light curves,
  phased light curves, and make checkplots (a 3 x 3 grid of plots used to
  quickly decide if a period search was successful; see example below)

  ![Voting mode image](astrobase/data/checkplot-example.png?raw=true)

- `[timeutils](astrobase/timeutils.py)`: functions for converting from
  Julian dates to Baryocentric Julian dates, and precessing coordinates between
  equinoxes and due to proper motion; this will automatically download and save
  the JPL ephemerides `de430.bsp` from JPL upon first import

- `[varbase](astrobase/varbase.py)`: functions for calculating variability
  indices for light curves, fitting and obtaining Fourier coefficients for use
  in classifications, and other variability features

## ``bls``

This wraps `eebls.f` from Geza Kovacs. Extracted from
[python-bls](http://github.com/dfm/python-bls) by Daniel Foreman-Mackey, Ruth
Angus, and others. Used as the BLS implementation by `astrobase.periodbase`
functions.

## `notebooks`

This contains notebooks that demonstrate various functions from this
package. Also contains other useful notes-to-self.
