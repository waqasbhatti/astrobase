#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
tic - Luke Bouma (luke@astro.princeton.edu) - Sep 2018
License: MIT. See the LICENSE file for more details.

This interacts with the TESS Input Catalog (TIC) hosted on MAST.  The code was
almost entirely pilfered from the tutorial at
    https://mast.stsci.edu/api/v0/MastApiTutorial.html

If you use this, please cite the TIC paper (Stassun et al 2018).

For further documentation, see:
    https://mast.stsci.edu/api/v0/MastApiTutorial.html
and
    https://mast.stsci.edu/api/v0/pyex.html
and
    https://mast.stsci.edu/api/v0/_t_i_cfields.html
'''

#############
## LOGGING ##
#############

import logging
from datetime import datetime
from traceback import format_exc

# setup a logger
LOGGER = None
LOGMOD = __name__
DEBUG = False

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.%s' % (parent_name, LOGMOD))

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('[%s - DBUG] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('[%s - INFO] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('[%s - ERR!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('[%s - WRN!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '[%s - EXC!] %s\nexception was: %s' % (
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                message, format_exc()
                )
            )


#############
## IMPORTS ##
#############

import sys, os, time, re, json

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib

###################
## MAST QUERIES ##
###################

def mast_query(request):

    server='mast.stsci.edu'

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

    del conn # needed for tricky memory management reasons

    return head,content


def tic_single_object_crossmatch(ra, dec, radius):
    '''
    ra, dec, radius: all in decimal degrees

    speed tests: about 10 crossmatches per second.
        (-> 3 hours for 10^5 objects to crossmatch).
    '''
    for val in ra,dec,radius:
        if not isinstance(val, float):
            raise AssertionError('plz input ra,dec,radius in decimal degrees')

    # This is a json object
    crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                                 {"name":"dec","type":"float"}],
                       "data":[{"ra":ra,"dec":dec}]}

    request =  {"service":"Mast.Tic.Crossmatch",
                "data":crossmatchInput,
                "params":{
                    "raColumn":"ra",
                    "decColumn":"dec",
                    "radius":radius
                },
                "format":"json",
                'removecache':True}

    headers,out_string = mast_query(request)

    out_data = json.loads(out_string)

    return out_data
