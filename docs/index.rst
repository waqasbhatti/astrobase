.. Astrobase documentation master file, created by
   sphinx-quickstart on Tue Feb 19 23:51:53 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Astrobase
=========

Astrobase is a Python package for analyzing light curves and finding variable
stars. It includes implementations of several period-finding algorithms, batch
work drivers for working on large collections of light curves, as well as an
interactive webapp useful for reviewing and classifying light curves by stellar
variability type.

This package was spun out of a bunch of Python modules I wrote and maintain for
my work with the `HAT Exoplanet Surveys <https://hatsurveys.org>`_. It's
applicable to many other astronomical time-series observations, and includes
support for the light curves produced by Kepler and TESS in particular. Most
functions in this package that deal with light curves (e.g. in the modules
:py:mod:`astrobase.lcmath`, :py:mod:`astrobase.periodbase`,
:py:mod:`astrobase.varbase`, :py:mod:`astrobase.plotbase`,
:py:mod:`astrobase.checkplot`) usually just require three Numpy ndarrays as
input: `times`, `mags`, and `errs`, so they should work with any time-series
data that can be represented in this form. If you have flux time series
measurements, most functions take a `magsarefluxes` keyword argument that makes
them handle flux light curves correctly.

The :py:mod:`astrobase.lcproc` subpackage implements drivers for working on
large collections of light curve files, and includes functions to register your
own light curve format so that it gets recognized and can be worked on by other
Astrobase functions transparently.

These docs are a work in progress, but the docstrings for the Python modules
themselves are fairly decent at this stage.

- Guides for specific tasks are available as Jupyter notebooks at Github:
  `astrobase-notebooks <https://github.com/waqasbhatti/astrobase-notebooks/>`_.
- Automatically generated `API documentation`_ is available below.


Package overview
----------------



API documentation
-----------------

.. toctree::
   :maxdepth: 4

   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
