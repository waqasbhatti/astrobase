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
demonstrate some of the functionality at
https://github.com/waqasbhatti/astrobase-notebooks.

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

If you're using:

- 64-bit Linux and Python 2.7, 3.4, 3.5, 3.6
- 64-bit Mac OSX 10.12+ with Python 2.7 or 3.6
- 64-bit Windows with Python 2.7 and 3.6

You can simply install astrobase with ::

  (venv)$ pip install astrobase

Otherwise, you'll need to make sure that a Fortran compiler and numpy are
installed beforehand to compile the pyeebls package that astrobase depends on ::

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
