#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''skyview - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
License: MIT. See the LICENSE file for more details.

This gets cutout images from the Digitized Sky Survey using the NASA GSFC
SkyView server.

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
import re
import json

import numpy as np

# to do the queries
import requests
import requests.exceptions
try:
    from urllib.parse import urljoin
except:
    from urlparse import urljoin


###################
## FORM SETTINGS ##
###################

SKYVIEW_URL = 'https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl'

SKYVIEW_PARAMS = {
    'CatalogIDs': ['on'],
    'Deedger': ['_skip_'],
    'Position': ['0.0, 0.0'],
    'Sampler': ['_skip_'],
    'coordinates': ['J2000'],
    'ebins': ['null'],
    'float': ['on'],
    'grid': ['_skip_'],
    'gridlabels': ['1'],
    'lut': ['colortables/b-w-linear.bin'],
    'pixels': ['300'],
    'projection': ['Tan'],
    'resolver': ['SIMBAD-NED'],
    'scaling': ['Linear'],
    'survey': ['DSS2 Red', '_skip_', '_skip_', '_skip_']
}

FITS_REGEX = re.compile(r'(tempspace\/fits\/skv\d{8,20}\.fits)')
FITS_BASEURL = 'https://skyview.gsfc.nasa.gov'

#####################
## QUERY FUNCTIONS ##
#####################

def get_stamp(ra, decl,
              survey='DSS2 Red',
              scaling='Linear',
              sizepix=300,
              forcefetch=False,
              cachedir='~/.astrobase/stamp-cache',
              timeout=10.0,
              verbose=True):
    '''This gets a FITS cutout from the NASA GSFC SkyView service.

    ra, decl are decimal equatorial coordinates for the cutout center.

    survey is the name of the survey to get the stamp for. This is 'DSS2 Red' by
    default.

    scaling is the type of pixel value scaling to apply to the cutout. This is
    'Linear' by default.

    sizepix is the size of the cutout in pixels.

    cachedir points to the astrobase stamp-cache directory.

    timeout is the amount of time in seconds to wait for a response from the
    service.

    '''

    # parse the given params into the correct format for the form
    formposition = ['%.4f, %.4f' % (ra, decl)]
    formscaling = [scaling]

    formparams = SKYVIEW_PARAMS.copy()
    formparams['Position'] = formposition
    formparams['survey'][0] = survey
    formparams['scaling'] = formscaling
    formparams['pixels'] = ['%s' % sizepix]

    # see if the cachedir exists
    if '~' in cachedir:
        cachedir = os.path.expanduser(cachedir)
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    # figure out if we can get this image from the cache
    cachekey = '%s-%s-%s-%s' % (formposition[0], survey, scaling, sizepix)
    cachekey = hashlib.sha256(cachekey.encode()).hexdigest()
    cachefname = os.path.join(cachedir, '%s.fits.gz' % cachekey)
    provenance = 'cache'

    # this is to handle older cached stamps that didn't include the sizepix
    # parameter
    if sizepix == 300:

        oldcachekey = '%s-%s-%s' % (formposition[0], survey, scaling)
        oldcachekey = hashlib.sha256(oldcachekey.encode()).hexdigest()
        oldcachefname = os.path.join(cachedir, '%s.fits.gz' % oldcachekey)

        if os.path.exists(oldcachefname):
            cachefname = oldcachefname

    # if this exists in the cache and we're not refetching, get the frame
    if forcefetch or (not os.path.exists(cachefname)):

        provenance= 'new download'

        # fire the request
        try:

            if verbose:
                LOGINFO('submitting stamp request for %s, %s, %s, %s' % (
                    formposition[0],
                    survey,
                    scaling,
                    sizepix)
                )
            req = requests.get(SKYVIEW_URL, params=formparams, timeout=timeout)
            req.raise_for_status()

            # get the text of the response, this includes the locations of the
            # generated FITS on the server
            resp = req.text

            # find the URLS of the FITS
            fitsurls = FITS_REGEX.findall(resp)

            # download the URLs
            if fitsurls:

                for fitsurl in fitsurls:

                    fullfitsurl = urljoin(FITS_BASEURL, fitsurl)

                    if verbose:
                        LOGINFO('getting %s' % fullfitsurl)

                    fitsreq = requests.get(fullfitsurl, timeout=timeout)

                    with gzip.open(cachefname,'wb') as outfd:
                        outfd.write(fitsreq.content)

            else:
                LOGERROR('no FITS URLs found in query results for %s' %
                         formposition)
                return None

        except requests.exceptions.HTTPError as e:
            LOGEXCEPTION('SkyView stamp request for '
                         'coordinates %s failed' % repr(formposition))
            return None

        except requests.exceptions.Timeout as e:
            LOGERROR('SkyView stamp request for '
                     'coordinates %s did not complete within %s seconds' %
                     (repr(formposition), timeout))
            return None

        except Exception as e:
            LOGEXCEPTION('SkyView stamp request for '
                         'coordinates %s failed' % repr(formposition))
            return None

    #
    # DONE WITH FETCHING STUFF
    #

    retdict = {
        'params':{'ra':ra,
                  'decl':decl,
                  'survey':survey,
                  'scaling':scaling,
                  'sizepix':sizepix},
        'provenance':provenance,
        'fitsfile':cachefname
    }

    return retdict
