astrobase
=========

This is a bunch of Python modules I wrote for my astronomy work with the HAT
surveys, mostly focused on handling light curves and characterizing variable
stars. Module functions that deal with light curves (e.g. in the modules
``astrobase.lcmath``, ``astrobase.periodbase``, ``astrobase.varbase``,
``astrobase.plotbase``, and ``astrobase.checkplot``) usually just require three
numpy ndarrays as input: ``times``, ``mags``, and ``errs``, so they should work
with any time-series data that can be represented in this form. If you have flux
time series measurements, most functions take a ``magsarefluxes`` keyword
argument that makes them handle flux light curves correctly.

Full documentation is still a work in progress (as soon as I figure out how
Sphinx works), but the docstrings are fairly good and an overview is provided at
https://github.com/waqasbhatti/astrobase, along with Jupyter notebooks that
demonstrate some of the functionality.

Changelog
---------

Please see https://github.com/waqasbhatti/astrobase/blob/master/CHANGELOG.md for
the latest changelog for tagged versions.

Installation
------------

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

For some extra functionality, you'll need the following modules:

- for ``astrobase.lcdb`` to work, you'll also need psycopg2

First, make sure numpy and a Fortran compiler are installed ::

  ## you'll need a Fortran compiler.                      ##
  ## on Linux: dnf/yum/apt install gcc gcc-gfortran       ##
  ## on OSX (using homebrew): brew install gcc            ##
  ## make sure numpy is installed first!                  ##
  ## this is required for the pyeebls module installation ##

  (venv)$ pip install numpy # in a virtualenv
  # or use dnf/yum/apt install numpy to install systemwide

Next, install astrobase ::

  (venv)$ pip install astrobase

If you want to install all optional dependencies as well ::

  (venv)$ pip install astrobase[all]

Or if you want the latest version ::

  $ git clone https://github.com/waqasbhatti/astrobase
  $ cd astrobase
  $ python setup.py install
  $ # or use pip install . to install requirements automatically
  $ # or use pip install -e . to install in develop mode along with requirements
