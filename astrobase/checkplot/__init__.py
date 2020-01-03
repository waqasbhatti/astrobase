#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# checkplot.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2017
# License: MIT.

'''Contains functions to make checkplots: quick views for determining periodic
variability for light curves and sanity-checking results from period-finding
functions (e.g., from periodbase).

The :py:func:`astrobase.checkplot.pkl.checkplot_pickle` function takes, for a
single object, an arbitrary number of results from independent period-finding
functions (e.g. BLS, PDM, AoV, GLS, etc.) in periodbase, and generates a pickle
file that contains object and variability information, finder chart, mag series
plot, and for each period-finding result: a periodogram and phased mag series
plots for an arbitrary number of 'best periods'.

This is intended for use with an external checkplot viewer: the Tornado webapp
`checkplotserver.py`, but you can also use the
:py:func:`astrobase.checkplot.pkl_png.checkplot_pickle_to_png` function to
render this to a PNG that will look something like::

    [    finder    ] [  objectinfo  ] [ variableinfo ] [ unphased LC  ]
    [ periodogram1 ] [ phased LC P1 ] [ phased LC P2 ] [ phased LC P3 ]
    [ periodogram2 ] [ phased LC P1 ] [ phased LC P2 ] [ phased LC P3 ]
                                     .
                                     .
    [ periodogramN ] [ phased LC P1 ] [ phased LC P2 ] [ phased LC P3 ]

for N independent period-finding methods producing:

- periodogram1,2,3...N: the periodograms from each method
- phased LC P1,P2,P3: the phased lightcurves using the best 3 peaks in each
  periodogram

The :py:func:`astrobase.checkplot.png.checkplot_png` function takes a single
period-finding result and makes the following 3 x 3 grid and writes to a PNG::

    [LSP plot + objectinfo] [     unphased LC     ] [ period 1 phased LC ]
    [period 1 phased LC /2] [period 1 phased LC x2] [ period 2 phased LC ]
    [ period 3 phased LC  ] [period 4 phased LC   ] [ period 5 phased LC ]

The :py:func:`astrobase.checkplot.png.twolsp_checkplot_png` function makes a
similar plot for two independent period-finding routines and writes to a PNG::

    [ pgram1 + objectinfo ] [        pgram2       ] [     unphased LC     ]
    [ pgram1 P1 phased LC ] [ pgram1 P2 phased LC ] [ pgram1 P3 phased LC ]
    [ pgram2 P1 phased LC ] [ pgram2 P2 phased LC ] [ pgram2 P3 phased LC ]

where:

- pgram1 is the plot for the periodogram in the lspinfo1 dict
- pgram1 P1, P2, and P3 are the best three periods from lspinfo1
- pgram2 is the plot for the periodogram in the lspinfo2 dict
- pgram2 P1, P2, and P3 are the best three periods from lspinfo2

'''

# import our publicly visible functions from the other modules
from .png import checkplot_png, twolsp_checkplot_png
from .pkl_png import checkplot_pickle_to_png, cp2png
from .pkl import checkplot_dict, checkplot_pickle
