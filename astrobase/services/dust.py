#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# dust - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
# License: MIT. See the LICENSE file for more details.

'''
This gets extinction tables from the the 2MASS DUST service at:

http://irsa.ipac.caltech.edu/applications/DUST/

If you use this, please cite the SF11 and SFD98 papers and acknowledge the use
of 2MASS/IPAC services.

- http://www.adsabs.harvard.edu/abs/1998ApJ...500..525S
- http://www.adsabs.harvard.edu/abs/2011ApJ...737..103S

Also see:

http://irsa.ipac.caltech.edu/applications/DUST/docs/background.html

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
import hashlib
import re
import random
import time

import numpy as np

# to do the queries
import requests
import requests.exceptions

# to read IPAC tables
from astropy.table import Table


##############################
## 2MASS DUST FORM SETTINGS ##
##############################

DUST_URL = 'https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust'

DUST_PARAMS = {'locstr':'',
               'regSize':'5.0'}

DUST_REGEX = re.compile(r'http[s|]://\S*extinction\.tbl')


################################
## 2MASS DUST QUERY FUNCTIONS ##
################################

def extinction_query(lon, lat,
                     coordtype='equatorial',
                     sizedeg=5.0,
                     forcefetch=False,
                     cachedir='~/.astrobase/dust-cache',
                     verbose=True,
                     timeout=10.0,
                     jitter=5.0):
    '''This queries the 2MASS DUST service to find the extinction parameters
    for the given `lon`, `lat`.

    Parameters
    ----------

    lon,lat: float
        These are decimal right ascension and declination if `coordtype =
        'equatorial'`. These are are decimal Galactic longitude and latitude if
        `coordtype = 'galactic'`.

    coordtype : {'equatorial','galactic'}
        Sets the type of coordinates passed in as `lon`, `lat`.

    sizedeg : float
        This is the width of the image returned by the DUST service. This can
        usually be left as-is if you're interested in the extinction only.

    forcefetch : bool
        If this is True, the query will be retried even if cached results for
        it exist.

    cachedir : str
        This points to the directory where results will be downloaded.

    verbose : bool
        If True, will indicate progress and warn of any issues.

    timeout : float
        This sets the amount of time in seconds to wait for the service to
        respond to our request.

    jitter : float
        This is used to control the scale of the random wait in seconds before
        starting the query. Useful in parallelized situations.

    Returns
    -------

    dict
        A dict of the following form is returned::

            {'Amag':{dict of extinction A_v values for several mag systems},
             'table': array containing the full extinction table,
             'tablefile': the path to the full extinction table file on disk,
             'provenance': 'cached' or 'new download',
             'request': string repr of the request made to 2MASS DUST}

    '''

    dustparams = DUST_PARAMS.copy()

    # convert the lon, lat to the required format
    # and generate the param dict
    if coordtype == 'equatorial':
        locstr = '%.3f %.3f Equ J2000' % (lon, lat)
    elif coordtype == 'galactic':
        locstr = '%.3f %.3f gal' % (lon, lat)
    else:
        LOGERROR('unknown coordinate type: %s' % coordtype)
        return None

    dustparams['locstr'] = locstr
    dustparams['regSize'] = '%.3f' % sizedeg

    # see if the cachedir exists
    if '~' in cachedir:
        cachedir = os.path.expanduser(cachedir)
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    # generate the cachekey and cache filename
    cachekey = '%s - %.1f' % (locstr, sizedeg)
    cachekey = hashlib.sha256(cachekey.encode()).hexdigest()
    cachefname = os.path.join(cachedir, '%s.txt' % cachekey)
    provenance = 'cache'

    # if this does not exist in cache or if we're forcefetching, do the query
    if forcefetch or (not os.path.exists(cachefname)):

        time.sleep(random.randint(1,jitter))
        provenance = 'new download'

        try:

            if verbose:
                LOGINFO('submitting 2MASS DUST request for '
                        'lon = %.3f, lat = %.3f, type = %s, size = %.1f' %
                        (lon, lat, coordtype, sizedeg))

            req = requests.get(DUST_URL, dustparams, timeout=timeout)
            req.raise_for_status()

            resp = req.text

            # see if we got an extinction table URL in the response
            tableurl = DUST_REGEX.search(resp)

            # if we did, download it to the cache directory
            if tableurl:

                tableurl = tableurl.group(0)

                req2 = requests.get(tableurl, timeout=timeout)

                # write the table to the cache directory
                with open(cachefname,'wb') as outfd:
                    outfd.write(req2.content)

                tablefname = cachefname

            else:
                LOGERROR('could not get extinction parameters for '
                         '%s (%.3f, %.3f) with size = %.1f' % (coordtype,
                                                               lon,lat,sizedeg))
                LOGERROR('error from DUST service follows:\n%s' % resp)
                return None

        except requests.exceptions.Timeout:
            LOGERROR('DUST request timed out for '
                     '%s (%.3f, %.3f) with size = %.1f' % (coordtype,
                                                           lon,lat,sizedeg))
            return None

        except Exception:
            LOGEXCEPTION('DUST request failed for '
                         '%s (%.3f, %.3f) with size = %.1f' % (coordtype,
                                                               lon,lat,sizedeg))
            return None

    # if this result is available in the cache, get it from there
    else:

        if verbose:
            LOGINFO('getting cached 2MASS DUST result for '
                    'lon = %.3f, lat = %.3f, coordtype = %s, size = %.1f' %
                    (lon, lat, coordtype, sizedeg))

        tablefname = cachefname

    #
    # now we should have the extinction table in some form
    #
    # read and parse the extinction table using astropy.Table
    extinction_table = Table.read(tablefname, format='ascii.ipac')

    # get the columns we need
    filters = np.array(extinction_table['Filter_name'])
    a_sf11_byfilter = np.array(extinction_table['A_SandF'])
    a_sfd98_byfilter = np.array(extinction_table['A_SFD'])

    # generate the output dict
    extdict = {'Amag':{x:{'sf11':y, 'sfd98':z} for
                       x,y,z in zip(filters,a_sf11_byfilter,a_sfd98_byfilter)},
               'table':np.array(extinction_table),
               'tablefile':os.path.abspath(cachefname),
               'provenance':provenance,
               'request':'%s (%.3f, %.3f) with size = %.1f' % (coordtype,
                                                               lon,lat,
                                                               sizedeg)}

    return extdict
