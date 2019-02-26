#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tic - Luke Bouma (luke@astro.princeton.edu) - Sep 2018
# License: MIT. See the LICENSE file for more details.

'''This interacts with the TESS Input Catalog (TIC) hosted on MAST.  The code
was almost entirely pilfered from the tutorial at:

https://mast.stsci.edu/api/v0/MastApiTutorial.html

If you use this, please cite the TIC paper (Stassun et al 2018).

For further documentation, see:

- https://mast.stsci.edu/api/v0/MastApiTutorial.html
- https://mast.stsci.edu/api/v0/pyex.html
- https://mast.stsci.edu/api/v0/_t_i_cfields.html

.. deprecated:: 0.3.20
   This function will be removed in astrobase v0.4.2. The
   :py:func:`astrobase.services.mast.tic_conesearch` and
   :py:func:`astrobase.services.mast.tic_xmatch` functions are preferred over
   using functions in this module.

'''

#########################
## DEPRECATION WARNING ##
#########################

import warnings
warnings.warn(
    "The services.mast.tic_conesearch and services.mast.tic_xmatch "
    "functions are preferred over using functions in this module. "
    "This module will be removed in astrobase v0.4.2.",
    FutureWarning
)

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


#############
## IMPORTS ##
#############

import sys
import json

# Python 3.x
try:
    from urllib.parse import quote as urlencode
# Python 2.x
except ImportError:
    from urllib import pathname2url as urlencode

# Python 3.x
try:
    import http.client as httplib
# Python 2.x
except ImportError:
    import httplib

###################
## MAST QUERIES ##
###################

def _mast_query(request):

    server = 'mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "Connection": "close",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    del conn  # needed for tricky memory management reasons

    return head,content


def tic_single_object_crossmatch(ra, dec, radius):
    '''This does a cross-match against the TIC catalog on MAST.

    Speed tests: about 10 crossmatches per second. (-> 3 hours for 10^5 objects
    to crossmatch).

    Parameters
    ----------

    ra,dec : np.array
        The coordinates to cross match against, all in decimal degrees.

    radius : float
        The cross-match radius to use, in decimal degrees.

    Returns
    -------

    dict
        Returns the match results JSON from MAST loaded into a dict.

    '''
    for val in ra,dec,radius:
        if not isinstance(val, float):
            raise AssertionError('plz input ra,dec,radius in decimal degrees')

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":ra,"dec":dec}]}

    request = {"service":"Mast.Tic.Crossmatch",
               "data":crossmatchInput,
               "params":{
                   "raColumn":"ra",
                   "decColumn":"dec",
                   "radius":radius
               },
               "format":"json",
               'removecache':True}

    headers,out_string = _mast_query(request)

    out_data = json.loads(out_string)

    return out_data
