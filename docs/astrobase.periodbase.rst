astrobase.periodbase package
============================

This package contains various useful tools for finding periods in astronomical
time-series observations.

- :py:mod:`astrobase.periodbase.spdm`: Stellingwerf (1978) phase-dispersion
  minimization period search algorithm.

- :py:mod:`astrobase.periodbase.saov`: Schwarzenberg-Czerny (1989) analysis of
  variance period search algorithm.

- :py:mod:`astrobase.periodbase.smav`: Schwarzenberg-Czerny (1996)
  multi-harmonic AoV period search algorithm.

- :py:mod:`astrobase.periodbase.zgls`: Zechmeister & Kurster (2009) generalized
  Lomb-Scargle period search algorithm.

- :py:mod:`astrobase.periodbase.kbls`: Kovacs et al. (2002) Box-Least-Squares
  search using a wrapped `eebls.f` from G. Kovacs.

- :py:mod:`astrobase.periodbase.abls`: Kovacs et al. (2002) BLS using Astropy's
  implementation.

- :py:mod:`astrobase.periodbase.htls`: Hippke & Heller (2019)
  Transit-Least-Squares period search algorithm.

- :py:mod:`astrobase.periodbase.macf`: McQuillan et al. (2013a, 2014) ACF period
  search algorithm.


Some utility functions are present in:

- :py:mod:`astrobase.periodbase.utils`: Functions to generate frequency grids
  and other useful bits.

- :py:mod:`astrobase.periodbase.falsealarm`: Functions to calculate false-alarm
  probabilities.


This module
-----------

.. automodule:: astrobase.periodbase
    :members:
    :undoc-members:
    :show-inheritance:


Submodules
----------

.. toctree::

   astrobase.periodbase.utils
   astrobase.periodbase.falsealarm
   astrobase.periodbase.abls
   astrobase.periodbase.kbls
   astrobase.periodbase.htls
   astrobase.periodbase.macf
   astrobase.periodbase.saov
   astrobase.periodbase.smav
   astrobase.periodbase.spdm
   astrobase.periodbase.zgls
