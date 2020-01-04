#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# skyview - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
# License: MIT. See the LICENSE file for more details.

'''
This gets cutout images from the Digitized Sky Survey using the NASA GSFC
SkyView server.

'''

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

import os
import os.path
import gzip
import hashlib
import re
import random
import time
import copy

# to do the queries
import requests
import requests.exceptions
from urllib.parse import urljoin

import astropy.io.fits as pyfits

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
              retry_failed=True,
              verbose=True,
              jitter=5.0):
    '''This gets a FITS cutout from the NASA GSFC SkyView service.

    This downloads stamps in FITS format from the NASA SkyView service:

    https://skyview.gsfc.nasa.gov/current/cgi/query.pl


    Parameters
    ----------

    ra,decl : float
        These are decimal equatorial coordinates for the cutout center.

    survey : str
        The survey name to get the stamp from. This is one of the
        values in the 'SkyView Surveys' option boxes on the SkyView
        webpage. Currently, we've only tested using 'DSS2 Red' as the value for
        this kwarg, but the other ones should work in principle.

    scaling : str
        This is the pixel value scaling function to use.

    sizepix : int
        The width and height of the cutout are specified by this value.

    forcefetch : bool
        If True, will disregard any existing cached copies of the stamp already
        downloaded corresponding to the requested center coordinates and
        redownload the FITS from the SkyView service.

    cachedir : str
        This is the path to the astrobase cache directory. All downloaded FITS
        stamps are stored here as .fits.gz files so we can immediately respond
        with the cached copy when a request is made for a coordinate center
        that's already been downloaded.

    timeout : float
        Sets the timeout in seconds to wait for a response from the NASA SkyView
        service.

    retry_failed : bool
        If the initial request to SkyView fails, and this is True, will retry
        until it succeeds.

    verbose : bool
        If True, indicates progress.

    jitter : float
        This is used to control the scale of the random wait in seconds before
        starting the query. Useful in parallelized situations.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {
                'params':{input ra, decl and kwargs used},
                'provenance':'cached' or 'new download',
                'fitsfile':FITS file to which the cutout was saved on disk
            }

    '''

    # parse the given params into the correct format for the form
    formposition = ['%.4f, %.4f' % (ra, decl)]
    formscaling = [scaling]

    formparams = copy.deepcopy(SKYVIEW_PARAMS)
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

        provenance = 'new download'
        time.sleep(random.randint(1,jitter))

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

        except requests.exceptions.HTTPError:
            LOGEXCEPTION('SkyView stamp request for '
                         'coordinates %s failed' % repr(formposition))
            return None

        except requests.exceptions.Timeout:
            LOGERROR('SkyView stamp request for '
                     'coordinates %s did not complete within %s seconds' %
                     (repr(formposition), timeout))
            return None

        except Exception:
            LOGEXCEPTION('SkyView stamp request for '
                         'coordinates %s failed' % repr(formposition))
            return None

    #
    # DONE WITH FETCHING STUFF
    #

    # make sure the returned file is OK
    try:

        stampfits = pyfits.open(cachefname)
        stampfits.close()

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

    except Exception:

        LOGERROR('could not open cached FITS from Skyview download: %r' %
                 {'ra':ra,
                  'decl':decl,
                  'survey':survey,
                  'scaling': scaling,
                  'sizepix': sizepix})

        if retry_failed:

            return get_stamp(ra, decl,
                             survey=survey,
                             scaling=scaling,
                             sizepix=sizepix,
                             forcefetch=True,
                             cachedir=cachedir,
                             timeout=timeout,
                             verbose=verbose)

        else:

            return None
