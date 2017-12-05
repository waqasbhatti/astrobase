#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''varclass - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This contains various modules that run variable star classification and
characterize its reliability and completeness via simulating LCs.

fakelcgen.py        - fake light curve generation and injection of variability
fakelcrecovery.py   - recovery of fake light curve variability and periodic vars

features.py           - non-periodic light curve features
periodicfeatures.py   - light curve features for phased light curves
starfeatures.py       - features related to color, proper motion, etc.

rfclass.py          - random forest classifier and support functions for
                      variability classification
nnclass.py          - optional RNN classifier (will require Keras) and support
                      functions for variability classification

'''
