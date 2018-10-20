#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''mast - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2018
License: MIT. See the LICENSE file for more details.

This interfaces with the MAST API. The main use for this (for now) is to fill in
TIC information for checkplots.

The MAST API service documentation is at:

https://mast.stsci.edu/api/v0/index.html

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
import hashlib
import time
import json

# to do the queries
import requests
import requests.exceptions



###################
## FORM SETTINGS ##
###################

MAST_URLS = {
    'v0':{'url':'https://mast.stsci.edu/api/v0/invoke'},
}


#####################
## QUERY FUNCTIONS ##
#####################


def mast_query(service,
               params,
               apiversion='v0',
               forcefetch=False,
               cachedir='~/.astrobase/mast-cache',
               verbose=True,
               timeout=10.0,
               refresh=5.0,
               maxtimeout=90.0,
               maxtries=3,
               raiseonfail=False):
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
        'format':'json',
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
        '%s.json' % (cachekey,)
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

        url = MAST_URLS[apiversion]['url']
        formdata = {'request':json.dumps(inputparams)}

        while (not waitdone) or (ntries < maxtries):

            if timeelapsed > maxtimeout:
                retdict = None
                break

            try:

                resp = requests.post(url,
                                     data=formdata,
                                     # we'll let the service time us out first
                                     # if that fails, we'll timeout ourselves
                                     timeout=timeout+1.0)
                resp.raise_for_status()

                respjson = resp.json()

                if respjson['status'] == 'COMPLETE':

                    data = respjson['data']
                    nrows = len(data)

                    if nrows > 0:

                        with open(cachefname, 'w') as outfd:
                            json.dump(respjson, outfd)

                        retdict = {
                            'params':inputparams,
                            'provenance':provenance,
                            'cachefname':cachefname
                        }
                        waitdone = True

                        if verbose:
                            LOGINFO('query successful. nmatches: %s' % nrows)

                        break

                    else:

                        LOGERROR(
                            'no matching objects found for inputparams: %r' %
                            inputparams
                        )
                        retdict = None
                        waitdone = True
                        break

                # if we're still executing after the initial timeout is done
                elif respjson['status'] == 'EXECUTING':

                    if verbose:
                        LOGINFO('query is still executing, '
                                'waiting %s seconds to retry...' % refresh)
                    waitdone = False
                    time.sleep(refresh)
                    timeelapsed = timeelapsed + refresh
                    retdict = None

                else:

                    LOGERROR('Query failed! Message from service: %s' %
                             respjson['msg'])
                    retdict = None
                    waitdone = True
                    break

            except requests.exceptions.Timeout as e:

                if verbose:
                    LOGWARNING('MAST query try timed out, '
                               'site is probably down. '
                               'Waiting for %s seconds to try again...' %
                               refresh)
                waitdone = False
                time.sleep(refresh)
                timeelapsed = timeelapsed + refresh
                retdict = None

            except KeyboardInterrupt as e:

                LOGERROR('MAST request wait aborted for '
                         '%s' % repr(inputparams))
                return None

            except Exception as e:

                LOGEXCEPTION('MAST query failed!')

                if raiseonfail:
                    raise

                return None

            #
            # increment number of tries at the bottom of the loop
            #
            ntries = ntries + 1

        #
        # done with waiting for completion
        #
        if retdict is None:

            LOGERROR('Timed out, errored out, or reached maximum number '
                     'of tries with no response. Query was: %r' % inputparams)
            return None

        else:

            return retdict

    # otherwise, get the file from the cache
    else:

        if verbose:
            LOGINFO('getting cached MAST query result for '
                    'request: %s' %
                    (repr(inputparams)))

        retdict = {
            'params':inputparams,
            'provenance':provenance,
            'cachefname':cachefname
        }

        return retdict



def tic_conesearch(ra,
                   decl,
                   radius_arcmin=5.0,
                   apiversion='v0',
                   forcefetch=False,
                   cachedir='~/.astrobase/mast-cache',
                   verbose=True,
                   timeout=10.0,
                   refresh=5.0,
                   maxtimeout=90.0,
                   maxtries=3,
                   raiseonfail=False):
    '''This runs a TESS Input Catalog cone search.

    If successful, the result is written to a JSON text file in the specified
    cachedir.

    If you use this, please cite the TIC paper (Stassun et al 2018;
    http://adsabs.harvard.edu/abs/2018AJ....156..102S). Also see the "living"
    TESS input catalog docs:

    https://docs.google.com/document/d/1zdiKMs4Ld4cXZ2DW4lMX-fuxAF6hPHTjqjIwGqnfjqI

    Also see: https://mast.stsci.edu/api/v0/_t_i_cfields.html for the fields
    returned by the service and present in the result JSON file.

    '''

    params = {'ra':ra,
              'dec':decl,
              'radius':radius_arcmin/60.0}
    service = 'Mast.Catalogs.Tic.Cone'

    return mast_query(service,
                      params,
                      apiversion=apiversion,
                      forcefetch=forcefetch,
                      cachedir=cachedir,
                      verbose=verbose,
                      timeout=timeout,
                      refresh=refresh,
                      maxtimeout=maxtimeout,
                      maxtries=maxtries,
                      raiseonfail=raiseonfail)
