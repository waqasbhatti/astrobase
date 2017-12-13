#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''services - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2017
License: MIT. See the LICENSE file for more details.

This contains various modules to query online data services. These are not
exhaustive and are meant to support other astrobase modules.

dust.py       - interface to the 2MASS DUST extinction/emission service
gaia.py       - interface to the GAIA TAP+ ADQL query service
hatds.py      - interface to the new-generation HAT data server
skyview.py    - interface to the NASA GSFC SkyView cutout service
trilegal.py   - interface to the TRILEGAL galaxy model service

For a much broader interface to online data services, use the astroquery package
by A. Ginsburg, B. Sipocz, et al.:

http://astroquery.readthedocs.io

'''
