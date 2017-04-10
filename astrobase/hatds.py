#!/usr/bin/env python
'''hatds.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Apr 2017
License: MIT - see LICENSE for the full text.

This contains functions to search for objects and get HAT sqlite
("sqlitecurves") from the new HAT data server. These can be read by the
astrobase.hatlc module.

TODO:

- get HAT CSV light curves from the new HAT data server

'''

#####################
## THE DATA SERVER ##
#####################

# the light curve API
LCAPI = 'https://data.hatsurveys.org/api/lc'

# the quicksearch API
QSAPI = 'https://data.hatsurveys.org/api/qs'

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
from __future__ import print_function

import os
import os.path
import stat
import logging
from datetime import datetime
from traceback import format_exc
import multiprocessing as mp
import json

try:
    from urllib import urlretrieve
except:
    from urllib.request import urlretrieve


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.hatds' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


####################
## API KEY CONFIG ##
####################

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

    else:
        LOGWARNING('The API key credentials file %s has bad permissions '
                   'and is insecure, not reading it.\n'
                   '(you need to chmod 600 this file)'
                   % APIKEYFILE)
        HAVEAPIKEY = False


else:
    LOGWARNING('No HAT Data Server API credentials found in: %s\n'
               'Only anonymous access is available.\n\n
               {apikeyhelp}'.format(apikeyhelp=APIKEYHELP))
    HAVEAPIKEY = False


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

def get_hatlc(objectid,
              hatproject,
              datarelease=None,
              apiuser=None,
              apikey=None,
              outdir=None,
              lcformat='sqlite'):
    '''This gets the light curve for the specified objectid.

    hatproject is one of:

    'hatnet'   -> The HATNet Exoplanet Survey
    'hatsouth' -> The HATSouth Exoplanet Survey
    'hatpi'    -> The HATPI Survey

    datarelease is a string starting with 'DR' and ending with a number,
    indicating the data release to use for the light curve. By default, this is
    None, meaning that the latest data release light curve will be fetched.

    apiuser and apikey are your HAT Data Server API user email and key. If not
    provided, will search for a ~/.hatdsrc file and get these from there. If
    that file is not available, will use anonymous mode.

    outdir is where to put the downloaded file. If not provided, will download
    to the current directory.

    lcformat is one of the following:

    'sqlite' -> HAT sqlitecurve format: sqlite database file (readable by
                astrobase.hatlc)
    'csv'    -> HAT CSV light curve format: text CSV (astrobase.hatlc can read
                this too)

    '''




def get_hatlcs(objectidlist,
               hatproject,
               datarelease=None,
               apiuser=None,
               apikey=None,
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
