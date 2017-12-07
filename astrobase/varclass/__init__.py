#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''varclass - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This contains various modules that run variable star classification and
characterize its reliability and completeness via simulating LCs.

features.py           - non-periodic light curve features
periodicfeatures.py   - light curve features for phased light curves
starfeatures.py       - features related to color, proper motion, etc.

rfclass.py          - random forest classifier and support functions for
                      variability classification
nnclass.py          - optional RNN classifier (will require Keras) and support
                      functions for variability classification

How to use other astrobase modules with this one:

- use varclass/fakelcgen to generate light curves corresponding to the
  brightness distribution and time-sampling of your actual light curves.

- use varclass/fakelcrecovery and varclass/rfclass to run variable/non-variable
  classification on your fake light curves.

- use varclass/fakelcrecovery to run period-finding on fake light curves,
  generate periodic light curve features, and characterize how well
  period-finding methods work on the fake light curves.

- use varclass/rfclass to classify recovered variable stars based on their
  periodic light curve features.

- use varclass/fakelcrecovery to characterize recovery rates of all variables
  and just the periodic variables.

'''
