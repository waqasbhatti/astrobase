#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''mast - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2018
License: MIT. See the LICENSE file for more details.

This interfaces with the MAST API. The main use for this (for now) is to fill in
TIC information for checkplots.

For a more general and useful interface to MAST, see the astroquery
package by A. Ginsburg, B. Sipocz, et al.:

http://astroquery.readthedocs.io

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

import os
import os.path
import gzip
import hashlib
import time
import pickle
import json

import numpy as np

import random

# to do the queries
import requests
import requests.exceptions



###################
## FORM SETTINGS ##
###################

MAST_URLS = {
    'v0':{'url':'https://mast.stsci.edu/api/v0/invoke'},
}


# valid return formats
RETURN_FORMATS = {
    'json':'json.gz',
    'csv':'csv.gz',
    'votable':'vot',
}


#####################
## QUERY FUNCTIONS ##
#####################


def mast_query(service,
               params,
               apiversion='v0',
               returnformat='json',
               forcefetch=False,
               cachedir='~/.astrobase/mast-cache',
               verbose=True,
               timeout=10.0,
               refresh=10.0,
               maxtimeout=90.0,
               maxtries=3):
    '''This queries the STScI MAST service.

    service is the name of the service to use:

    https://mast.stsci.edu/api/v0/_services.html

    params is a dict containing the input params to the service as described on
    its details page linked off the page above.

    returnformat is one of 'csv', 'votable', or 'json':

    https://mast.stsci.edu/api/v0/md_result_formats.html

    If forcefetch is True, the query will be retried even if cached results for
    it exist.

    cachedir points to the directory where results will be downloaded.

    timeout sets the amount of time in seconds to wait for the service to
    respond.

    refresh sets the amount of time in seconds to wait before checking if the
    result file is available. If the results file isn't available after refresh
    seconds have elapsed, the function will wait for refresh continuously, until
    maxtimeout is reached or the results file becomes available.

    '''

    # this matches:
    # https://mast.stsci.edu/api/v0/class_mashup_1_1_mashup_request.html
    inputparams = {
        'format':returnformat,
        'params':params,
        'service':service,
        'timeout':timeout,
    }

    # see if the cachedir exists
    if '~' in cachedir:
        cachedir = os.path.expanduser(cachedir)
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    # generate the cachefname and look for it
    xcachekey = '-'.join([repr(inputparams[x])
                          for x in sorted(inputparams.keys())])
    cachekey = hashlib.sha256(xcachekey.encode()).hexdigest()

    cachefname = os.path.join(
        cachedir,
        '%s.%s' % (cachekey, RETURN_FORMATS[returnformat])
    )
    provenance = 'cache'


    #####################
    ## RUN A NEW QUERY ##
    #####################

    # otherwise, we check the cache if it's done already, or run it again if not
    if forcefetch or (not os.path.exists(cachefname)):

        provenance = 'new download'
        waitdone = False
        timeelapsed = 0.0
        ntries = 1

        while (not waitdone) or (ntries < maxtries):

            try:

                url = MAST_URLS[apiversion]['url']

                resp = requests.post(url,
                                     data=inputparams,
                                     # we'll let the service time us out first
                                     # if that fails, we'll timeout ourselves
                                     timeout=timeout+1.0)
                resp.raise_for_status()

                respjson = resp.json()

                if respjson['status'] == 'COMPLETE':

                    data = respjson['data']
                    nrows = len(data)

                    if nrows > 0:

                        with gzip.open(cachefname, 'wb') as outfd:
                            json.dump(respjson, outfd)

                        retdict = {
                            'params':inputparams,
                            'provenance':provenance,
                            'cachefname':cachefname
                        }
                        waitdone = True

                    else:

                        LOGERROR('no matching objects found for inputparams: %r' %
                                 inputparams)
                        retdict = None
                        break

                # if we're still executing after the initial timeout is done
                elif respjson['status'] == 'EXECUTING':

                    LOGINFO('query is still executing, '
                            'waiting %s seconds to retry' % refresh)
                    waitdone = False
                    time.sleep(refresh)
                    timeelapsed = timeelapsed + refresh

                else:

                    LOGERROR('query failed! message from service: %s' %
                             respjson['msg'])
                    retdict = None
                    waitdone = True


            except Exception as e:
                pass

            ntries = ntries + 1
