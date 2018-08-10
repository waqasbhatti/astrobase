#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

'''lccs.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Aug 2018
License: MIT - see LICENSE for the full text.

This contains functions to search for objects and get light curves from an LCC
server (https://github.com/waqasbhatti/lcc-server).

This currently supports the following LCC server services:

- conesearch   -> use function cone_search
- ftsquery     -> use function fulltext_search
- columnsearch -> use function column_search
- xmatch       -> use function xmatch_search

'''

# put this in here because hatds can be used as a standalone module
__version__ = '0.3.16'


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


####################
## SYSTEM IMPORTS ##
####################

import os
import os.path
import stat
import multiprocessing as mp
import json
import argparse
from datetime import datetime, timezone

# import url methods here.  we use built-ins because we want this module to be
# usable as a single file. otherwise, we'd use something sane like Requests.

# Python 2
try:
    from urllib import urlretrieve, urlencode
    from urllib2 import urlopen
# Python 3
except:
    from urllib.request import urlretrieve, urlopen
    from urllib.parse import urlencode


####################
## API KEY CONFIG ##
####################

def check_existing_apikey(lcc_server):
    '''This validates if an API key for lcc_server is available in
    ~/.astrobase/lccs.

    API keys are stored using the following file scheme:

    ~/.astrobase/lccs/apikey-domain.of.lccserver.org

    e.g. for the HAT LCC server at https://data.hatsurveys.org:

    ~/.astrobase/lccs/apikey-https-data.hatsurveys.org

    '''

    USERHOME = os.path.expanduser('~')
    APIKEYFILE = os.path.join(USERHOME,
                              '.astrobase',
                              'lccs-apikeys',
                              lcc_server.replace(
                                  'https://',
                                  'https-'
                              ).replace(
                                  'http://',
                                  'http-'
                              ))

    if os.path.exists(APIKEYFILE):

        # check if this file is readable/writeable by user only
        fileperm = oct(os.stat(APIKEYFILE)[stat.ST_MODE])

        if fileperm == '0100600' or fileperm == '0o100600':

            with open(APIKEYFILE) as infd:
                apikey, expires = infd.read().strip('\n').split()

            # get today's datetime
            now = datetime.now(timezone.utc)
            expdt = datetime.fromisoformat(expires.replace('Z','+00:00'))
            if now > expdt:
                LOGERROR('API key has expired. expiry was on: %s' % expires)
                return False, apikey, expires
            else:
                return True, apikey, expires

        else:
            LOGWARNING('The API key file %s has bad permissions '
                       'and is insecure, not reading it.\n'
                       '(you need to chmod 600 this file)'
                       % APIKEYFILE)

            return False, None, None
    else:
        LOGWARNING('No LCC server API key '
                   'found in: {apikeyfile}'.format(apikeyfile=APIKEYFILE))

        return False, None, None



def get_new_apikey(lcc_server):
    '''
    This gets a new API key from the LCC server at lcc_server.

    '''

    USERHOME = os.path.expanduser('~')
    APIKEYFILE = os.path.join(USERHOME,
                              '.astrobase',
                              'lccs-apikeys',
                              lcc_server.replace(
                                  'https://',
                                  'https-'
                              ).replace(
                                  'http://',
                                  'http-'
                              ))

    # url for getting an API key
    url = '%s/api/key' % lcc_server

    # get the API key
    resp = urlopen(url)

    if resp.status == 200:

        respdict = json.loads(resp.read())

    else:

        LOGERROR('could not fetch the API key from LCC server at: %s' %
                 lcc_server)
        LOGERROR('the HTTP status code was: %s' % resp.status_code)
        return None

    #
    # now that we have an API key dict, get the API key out of it and write it
    # to the APIKEYFILE
    #
    apikey = respdict['result']['key']
    expires = respdict['result']['expires']

    # write this to the apikey file

    if not os.path.exists(os.path.dirname(APIKEYFILE)):
        os.makedirs(os.path.dirname(APIKEYFILE))

    with open(APIKEYFILE,'w'):
        outfd.write('%s %sZ\n' % (apikey, expires))

    LOGINFO('key fetched successfully from: %s. expires on: %s' % (lcc_server,
                                                                   expires))
    LOGINFO('written to: %s' % APIKEYFILE)

    return apikey, expires



########################
## DOWNLOAD UTILITIES ##
########################

# this function is used to check progress of the download
def on_download_chunk(transferred, blocksize, totalsize):
    progress = transferred*blocksize/float(totalsize)*100.0
    print('Downloading: {progress:.1f}%'.format(progress=progress),end='\r')



##############################
## QUERY HANDLING FUNCTIONS ##
##############################


def submit_get_query(url, params, apikey=None):
    '''This submits a GET query to an LCC server API endpoint.

    Handles streaming of the results, and returns the final JSON stream. Also
    handles results that time out.

    apikey is not currently required for GET requests, but may be in the future,
    so it's handled here anyway.

    '''



def submit_post_query(url, data, apikey):
    '''This submits a POST query to an LCC server API endpoint.

    Handles streaming of the results, and returns the final JSON stream. Also
    handles results that time out.

    apikey is currently required for any POST requests.

    '''



def retrieve_dataset_files(searchresult, outdir=None):
    '''This retrieves the dataset's CSV, pickle, and any LC zip files.

    Takes the resultdict from submit_*_query functions above or a pickle file
    generated from these if the query timed out.

    Puts the files in outdir. If it's None, they will be placed in the current
    directory.

    '''



######################
## SEARCH FUNCTIONS ##
######################

def cone_search(lcc_server,
                center_ra,
                center_decl,
                radiusarcmin=5.0,
                collections=None,
                columns=None,
                filters=None,
                download_data=True,
                outdir=None,
                result_ispublic=True):
    '''This runs a cone-search query.

    lcc_server is the base URL of the LCC server to talk to.
    (e.g. for HAT, use: https://data.hatsurveys.org)

    center_ra, center_decl are the central coordinates of the search to
    conduct. These can be either decimal degrees of type float, or sexagesimal
    coordinates of type str:

    OK: 290.0, 45.0
    OK: 15:00:00 +45:00:00
    OK: 15 00 00.0 -45 00 00.0
    NOT OK: 290.0 +45:00:00
    NOT OK: 15:00:00 45.0

    radiusarcmin is the search radius. This is in arcminutes. The maximum radius
    you can use is 60 arcminutes.

    collections is a list of LC collections to search in. If this is None, all
    collections will be searched.

    columns is a list of columns to return in the results. Matching objects'
    object IDs, RAs, DECs, and links to light curve files will always be
    returned so there is no need to specify these columns.

    filters is a filter string to use to filtering the objects that match the
    initial search parameters.

    download_data sets if the accompanying data from the search results will be
    downloaded automatically. This includes the data table CSV, the dataset
    pickle file, and a light curve ZIP file. Note that if the search service
    indicates that your query is still in progress, this function will block
    until the light curve ZIP file becomes available. To avoid this, set
    download_data to False and the function will write a pickle file to
    ~/.astrobase/lccs/query-[setid].pkl containing all the information necessary
    to retrieve these data files later when the query is done. To do so, call
    the retrieve_dataset_files with the path to this pickle file (it will be
    returned).

    outdir if not None, sets the output directory of the downloaded dataset
    files. If None, they will be downloaded to the current directory.

    result_ispublic sets if you want your dataset to be publicly visible on the
    Recent Datasets and /datasets page of the LCC server you're talking to. If
    False, only people who know the unique dataset URL can view and fetch data
    files from it later.

    '''
