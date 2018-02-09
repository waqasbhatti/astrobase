#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''hatds.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Apr 2017
License: MIT - see LICENSE for the full text.

This contains functions to search for objects and get HAT sqlite
("sqlitecurves") or CSV gzipped text light curves from the new HAT data
server. These can be read by the hatlc module.

To get a single light curve for an object with objectid, belonging to hatproject
and in datarelease:

hatds.hatlc_for_object(objectid,
                       hatproject,
                       datarelease=None,
                       apikey=None,
                       outdir=None,
                       lcformat='sqlite')

To get multiple light curves for objects in parallel:

hatds.hatlcs_for_objectlist(objectidlist,
                            hatproject,
                            datarelease=None,
                            apikey=None,
                            outdir=None,
                            lcformat='sqlite',
                            nworkers=None)

To get all light curves for a specified location and search radius:

hatds.hatlcs_at_radec(coordstring,
                      hatproject,
                      datarelease=None,
                      apikey=None,
                      outdir=None,
                      lcformat='sqlite')

For all of these functions:

    hatproject is one of:

    'hatnet'   -> The HATNet Exoplanet Survey
    'hatsouth' -> The HATSouth Exoplanet Survey
    'hatpi'    -> The HATPI Survey

    datarelease is a string starting with 'DR' and ending with a number,
    indicating the data release to use for the light curve. By default, this is
    None, meaning that the latest data release light curve will be fetched.

    apikey is your HAT Data Server apikey. If not provided, this will search for
    a ~/.hatdsrc file and get it from there. If that file is not available,
    will use anonymous mode.

    lcformat is one of the following:

    'sqlite' -> HAT sqlitecurve format: sqlite database file (readable by
                astrobase.hatlc)
    'csv'    -> HAT CSV light curve format: text CSV (astrobase.hatlc can read
                this too)
    'check'    -> this just returns a JSON string indicating if you have access
                  to the light curve based on your access privilege level.

See the docstrings for each function and the APIKEYHELP string below for
details.

'''

# put this in here because hatds can be used as a standalone module
__version__ = '0.3.5'


#############
## LOGGING ##
#############

from __future__ import print_function
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


#####################
## THE DATA SERVER ##
#####################

# the light curve API
LCAPI = 'https://data.hatsurveys.org/api/lc'

APIKEYHELP = '''\
The HAT Data Server requires an API key to access data that is not open for
anonymous public access. This module will search for a key credentials file in
your home directory. If not found, only anonymous access to the Data Server will
be available.

If you don't have a HAT Data Server account
-------------------------------------------

Please email wbhatti@astro.princeton.edu for an API key. We'll automate this
procedure once the HAT Data Server is out of testing. Follow the instructions
below to create a API key credentials file.

If you have a HAT Data Server account
-------------------------------------

Create a file called .hatdsrc in your home directory and put your email address
and API key into it using the format below:

<your email address>:<API key string>

Make sure that this file is only readable/writeable by your user:

$ chmod 600 ~/.hatdsrc

Then import this module as usual; it should pick up the file automatically.
'''


####################
## SYSTEM IMPORTS ##
####################

import os
import os.path
import stat
import multiprocessing as mp
import json
import argparse

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

def check_apikey_settings():
    '''
    This checks if an API key is available.

    '''

    USERHOME = os.path.expanduser('~')
    APIKEYFILE = os.path.join(USERHOME, '.hatdsrc')

    if os.path.exists(APIKEYFILE):

        # check if this file is readable/writeable by user only
        fileperm = oct(os.stat(APIKEYFILE)[stat.ST_MODE])

        if fileperm == '0100600' or fileperm == '0o100600':

            with open(APIKEYFILE) as infd:
                creds = infd.read().strip('\n')
            APIUSER, APIKEY = creds.split(':')
            HAVEAPIKEY = True

            return HAVEAPIKEY, APIUSER, APIKEY

        else:
            LOGWARNING('The API key credentials file %s has bad permissions '
                       'and is insecure, not reading it.\n'
                       '(you need to chmod 600 this file)'
                       % APIKEYFILE)
            HAVEAPIKEY = False

            return HAVEAPIKEY, None, None
    else:
        LOGWARNING('No HAT Data Server API credentials found in: %s\n'
                   'Only anonymous access is available.\n\n
                   {apikeyhelp}'.format(apikeyhelp=APIKEYHELP))
        HAVEAPIKEY = False

        return HAVEAPIKEY, None, None



########################
## DOWNLOAD UTILITIES ##
########################

# this function is used to check progress of the download
def on_download_chunk(transferred, blocksize, totalsize):
    progress = transferred*blocksize/float(totalsize)*100.0
    print('Downloading: {progress:.1f}%'.format(progress=progress),end='\r')


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
