#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''recoverysim - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This generates light curves of variable stars using the astrobase.lcmodels
package, adds noise and observation sampling to them based on given parameters
(or example light curves) and then runs them through variable star detection and
classification to see how well they are recovered.

'''
