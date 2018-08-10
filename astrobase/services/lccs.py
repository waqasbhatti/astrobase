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



######################
## SEARCH FUNCTIONS ##
######################










#######################################
## DATA SERVER LIGHT CURVE RETRIEVAL ##
#######################################

def hatlc_for_object(objectid,
                     hatproject,
                     datarelease=None,
                     anonmode=True,
                     outdir=None,
                     lcformat='sqlite'):
    '''This gets the light curve for the specified objectid.

    outdir is where to put the downloaded file. If not provided, will download
    to the current directory.

    hatproject is one of:

    'hatnet'   -> The HATNet Exoplanet Survey
    'hatsouth' -> The HATSouth Exoplanet Survey
    'hatpi'    -> The HATPI Survey

    datarelease is a string starting with 'DR' and ending with a number,
    indicating the data release to use for the light curve. By default, this is
    None, meaning that the latest data release light curve will be fetched.

    if anonmode is True, will not use apikey.

    lcformat is one of the following:

    'sqlite' -> HAT sqlitecurve format: sqlite database file (readable by
                astrobase.hatlc)
    'csv'    -> HAT CSV light curve format: text CSV (astrobase.hatlc can read
                this too)
    'check'    -> this just returns a JSON string indicating if you have access
                  to the light curve based on your access privilege level.

    '''

    if not anonmode:
        apikey_avail, apikey_user, apikey_key = check_apikey_settings()
        if not apikey_avail:
            LOGERROR("no API key available, can't continue")
            return None




def hatlcs_for_objectlist(objectidlist,
                          hatproject,
                          datarelease=None,
                          anonmode=True,
                          outdir=None,
                          lcformat='sqlite',
                          nworkers=None):
    '''This gets light curves for all objectids in objectidlist.

    All args and kwargs are the same as for get_hatlc, except for:

    nworkers: the total number of parallel workers to use when getting the light
    curves. By default, this is equal to the number of visible CPUs on your
    machine. There will be a random delay enforced for each worker's download
    job.

    '''

    if not anonmode:
        apikey_avail, apikey_user, apikey_key = check_apikey_settings()
        if not apikey_avail:
            LOGERROR("no API key available, can't continue")
            return None



def hatlcs_at_radec(coordstring,
                    hatproject,
                    datarelease=None,
                    anonmode=False,
                    outdir=None,
                    lcformat='sqlite',
                    nworkers=None):
    '''This gets light curves for all objectids at coordstring.

    All args and kwargs are the same as for get_hatlc, except for:

    coordstring: this is a string of the following form:

    '<ra> <decl> <search radius in arcmin>'

    <ra> is the right ascension of the center coordinate in decimal degrees or
    HH:MM:SS.ssss... format.

    <decl> is the declination of the center coordinate in decimal degrees or
    [+|-]DD:MM:SS.sss... format. Make sure to include the + sign if the
    declination is positive (for both decimal degrees or sexagesimal format).

    <search radius in arcmin> is the search radius used for the cone
    search. This will be restricted by your access privileges. Anonymous users
    can search up to a 1.0 arcminute radius. If you have an API key with
    sufficient privileges, you may be able to search a wider radius.

    This function will return a path to a ZIP archive containing all the
    accessible light curves for objects found using the coordstring
    specification.

    '''

    if not anonmode:
        apikey_avail, apikey_user, apikey_key = check_apikey_settings()
        if not apikey_avail:
            LOGERROR("no API key available, can't continue")
            return None




def main():
    '''
    This enables execution as a commandline script.

    '''


if __name__ == '__main__':
    main()
