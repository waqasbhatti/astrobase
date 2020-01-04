#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mast - Waqas Bhatti (wbhatti@astro.princeton.edu) - Oct 2018
# License: MIT. See the LICENSE file for more details.

'''
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
import time
import json
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


#####################
## QUERY FUNCTIONS ##
#####################

def mast_query(service,
               params,
               data=None,
               apiversion='v0',
               forcefetch=False,
               cachedir='~/.astrobase/mast-cache',
               verbose=True,
               timeout=10.0,
               refresh=5.0,
               maxtimeout=90.0,
               maxtries=3,
               raiseonfail=False,
               jitter=5.0):
    '''This queries the STScI MAST service for catalog data.

    All results are downloaded as JSON files that are written to `cachedir`.

    Parameters
    ----------

    service : str
        This is the name of the service to use. See
        https://mast.stsci.edu/api/v0/_services.html for a list of all available
        services.

    params : dict
        This is a dict containing the input params to the service as described
        on its details page linked in the `service description page on MAST
        <https://mast.stsci.edu/api/v0/_services.html>`_.

    data : dict or None
        This contains optional data to upload to the service.

    apiversion : str
        The API version of the MAST service to use. This sets the URL that this
        function will call, using `apiversion` as key into the `MAST_URLS` dict
        above.

    forcefetch : bool
        If this is True, the query will be retried even if cached results for
        it exist.

    cachedir : str
        This points to the directory where results will be downloaded.

    verbose : bool
        If True, will indicate progress and warn of any issues.

    timeout : float
        This sets the amount of time in seconds to wait for the service to
        respond to our initial request.

    refresh : float
        This sets the amount of time in seconds to wait before checking if the
        result file is available. If the results file isn't available after
        `refresh` seconds have elapsed, the function will wait for `refresh`
        seconds continuously, until `maxtimeout` is reached or the results file
        becomes available.

    maxtimeout : float
        The maximum amount of time in seconds to wait for a result to become
        available after submitting our query request.

    maxtries : int
        The maximum number of tries (across all mirrors tried) to make to either
        submit the request or download the results, before giving up.

    raiseonfail : bool
        If this is True, the function will raise an Exception if something goes
        wrong, instead of returning None.

    jitter : float
        This is used to control the scale of the random wait in seconds before
        starting the query. Useful in parallelized situations.

    Returns
    -------

    dict
        This returns a dict of the following form::

            {'params':dict of the input params used for the query,
             'provenance':'cache' or 'new download',
             'result':path to the file on disk with the downloaded data table}

    '''

    # this matches:
    # https://mast.stsci.edu/api/v0/class_mashup_1_1_mashup_request.html
    inputparams = {
        'format':'json',
        'params':params,
        'service':service,
        'timeout':timeout,
    }

    if data is not None:
        inputparams['data'] = data

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

        time.sleep(random.randint(1,jitter))

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

            except requests.exceptions.Timeout:

                if verbose:
                    LOGWARNING('MAST query try timed out, '
                               'site is probably down. '
                               'Waiting for %s seconds to try again...' %
                               refresh)
                waitdone = False
                time.sleep(refresh)
                timeelapsed = timeelapsed + refresh
                retdict = None

            except KeyboardInterrupt:

                LOGERROR('MAST request wait aborted for '
                         '%s' % repr(inputparams))
                return None

            except Exception:

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


def tic_conesearch(
        ra,
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
        jitter=5.0,
        raiseonfail=False
):
    '''This runs a TESS Input Catalog cone search on MAST.

    If you use this, please cite the TIC paper (Stassun et al 2018;
    http://adsabs.harvard.edu/abs/2018AJ....156..102S). Also see the "living"
    TESS input catalog docs:

    https://docs.google.com/document/d/1zdiKMs4Ld4cXZ2DW4lMX-fuxAF6hPHTjqjIwGqnfjqI

    Also see: https://mast.stsci.edu/api/v0/_t_i_cfields.html for the fields
    returned by the service and present in the result JSON file.

    Parameters
    ----------

    ra,decl : float
        The center coordinates of the cone-search in decimal degrees.

    radius_arcmin : float
        The cone-search radius in arcminutes.

    apiversion : str
        The API version of the MAST service to use. This sets the URL that this
        function will call, using `apiversion` as key into the `MAST_URLS` dict
        above.

    forcefetch : bool
        If this is True, the query will be retried even if cached results for
        it exist.

    cachedir : str
        This points to the directory where results will be downloaded.

    verbose : bool
        If True, will indicate progress and warn of any issues.

    timeout : float
        This sets the amount of time in seconds to wait for the service to
        respond to our initial request.

    refresh : float
        This sets the amount of time in seconds to wait before checking if the
        result file is available. If the results file isn't available after
        `refresh` seconds have elapsed, the function will wait for `refresh`
        seconds continuously, until `maxtimeout` is reached or the results file
        becomes available.

    maxtimeout : float
        The maximum amount of time in seconds to wait for a result to become
        available after submitting our query request.

    maxtries : int
        The maximum number of tries (across all mirrors tried) to make to either
        submit the request or download the results, before giving up.

    jitter : float
        This is used to control the scale of the random wait in seconds before
        starting the query. Useful in parallelized situations.

    raiseonfail : bool
        If this is True, the function will raise an Exception if something goes
        wrong, instead of returning None.

    Returns
    -------

    dict
        This returns a dict of the following form::

            {'params':dict of the input params used for the query,
             'provenance':'cache' or 'new download',
             'result':path to the file on disk with the downloaded data table}


    '''

    params = {'ra':ra,
              'dec':decl,
              'radius':radius_arcmin/60.0}
    service = 'Mast.Catalogs.Tic.Cone'

    return mast_query(service,
                      params,
                      jitter=jitter,
                      apiversion=apiversion,
                      forcefetch=forcefetch,
                      cachedir=cachedir,
                      verbose=verbose,
                      timeout=timeout,
                      refresh=refresh,
                      maxtimeout=maxtimeout,
                      maxtries=maxtries,
                      raiseonfail=raiseonfail)


def tic_xmatch(
        ra,
        decl,
        radius_arcsec=5.0,
        apiversion='v0',
        forcefetch=False,
        cachedir='~/.astrobase/mast-cache',
        verbose=True,
        timeout=90.0,
        refresh=5.0,
        maxtimeout=180.0,
        maxtries=3,
        jitter=5.0,
        raiseonfail=False
):
    '''This does a cross-match with TIC.

    Parameters
    ----------

    ra,decl : np.arrays or lists of floats
        The coordinates that will be cross-matched against the TIC.

    radius_arcsec : float
        The cross-match radius in arcseconds.

    apiversion : str
        The API version of the MAST service to use. This sets the URL that this
        function will call, using `apiversion` as key into the `MAST_URLS` dict
        above.

    forcefetch : bool
        If this is True, the query will be retried even if cached results for
        it exist.

    cachedir : str
        This points to the directory where results will be downloaded.

    verbose : bool
        If True, will indicate progress and warn of any issues.

    timeout : float
        This sets the amount of time in seconds to wait for the service to
        respond to our initial request.

    refresh : float
        This sets the amount of time in seconds to wait before checking if the
        result file is available. If the results file isn't available after
        `refresh` seconds have elapsed, the function will wait for `refresh`
        seconds continuously, until `maxtimeout` is reached or the results file
        becomes available.

    maxtimeout : float
        The maximum amount of time in seconds to wait for a result to become
        available after submitting our query request.

    maxtries : int
        The maximum number of tries (across all mirrors tried) to make to either
        submit the request or download the results, before giving up.

    jitter : float
        This is used to control the scale of the random wait in seconds before
        starting the query. Useful in parallelized situations.

    raiseonfail : bool
        If this is True, the function will raise an Exception if something goes
        wrong, instead of returning None.

    Returns
    -------

    dict
        This returns a dict of the following form::

            {'params':dict of the input params used for the query,
             'provenance':'cache' or 'new download',
             'result':path to the file on disk with the downloaded data table}

    '''

    service = 'Mast.Tic.Crossmatch'

    xmatch_input = {'fields':[{'name':'ra','type':'float'},
                              {'name':'dec','type':'float'}]}
    xmatch_input['data'] = [{'ra':x, 'dec':y} for (x,y) in zip(ra, decl)]

    params = {'raColumn':'ra',
              'decColumn':'dec',
              'radius':radius_arcsec/3600.0}

    return mast_query(service,
                      params,
                      data=xmatch_input,
                      jitter=jitter,
                      apiversion=apiversion,
                      forcefetch=forcefetch,
                      cachedir=cachedir,
                      verbose=verbose,
                      timeout=timeout,
                      refresh=refresh,
                      maxtimeout=maxtimeout,
                      maxtries=maxtries,
                      raiseonfail=raiseonfail)


def tic_objectsearch(
        objectid,
        idcol_to_use="ID",
        apiversion='v0',
        forcefetch=False,
        cachedir='~/.astrobase/mast-cache',
        verbose=True,
        timeout=90.0,
        refresh=5.0,
        maxtimeout=180.0,
        maxtries=3,
        jitter=5.0,
        raiseonfail=False
):
    '''
    This runs a TIC search for a specified TIC ID.

    Parameters
    ----------

    objectid : str
        The object ID to look up information for.

    idcol_to_use : str
        This is the name of the object ID column to use when looking up the
        provided `objectid`. This is one of {'ID', 'HIP', 'TYC', 'UCAC',
        'TWOMASS', 'ALLWISE', 'SDSS', 'GAIA', 'APASS', 'KIC'}.

    apiversion : str
        The API version of the MAST service to use. This sets the URL that this
        function will call, using `apiversion` as key into the `MAST_URLS` dict
        above.

    forcefetch : bool
        If this is True, the query will be retried even if cached results for
        it exist.

    cachedir : str
        This points to the directory where results will be downloaded.

    verbose : bool
        If True, will indicate progress and warn of any issues.

    timeout : float
        This sets the amount of time in seconds to wait for the service to
        respond to our initial request.

    refresh : float
        This sets the amount of time in seconds to wait before checking if the
        result file is available. If the results file isn't available after
        `refresh` seconds have elapsed, the function will wait for `refresh`
        seconds continuously, until `maxtimeout` is reached or the results file
        becomes available.

    maxtimeout : float
        The maximum amount of time in seconds to wait for a result to become
        available after submitting our query request.

    maxtries : int
        The maximum number of tries (across all mirrors tried) to make to either
        submit the request or download the results, before giving up.

    jitter : float
        This is used to control the scale of the random wait in seconds before
        starting the query. Useful in parallelized situations.

    raiseonfail : bool
        If this is True, the function will raise an Exception if something goes
        wrong, instead of returning None.

    Returns
    -------

    dict
        This returns a dict of the following form::

            {'params':dict of the input params used for the query,
             'provenance':'cache' or 'new download',
             'result':path to the file on disk with the downloaded data table}
    '''

    params = {
        'columns':'*',
        'filters':[
            {"paramName": idcol_to_use,
             "values":[str(objectid)]}
        ]
    }
    service = 'Mast.Catalogs.Filtered.Tic'

    return mast_query(service,
                      params,
                      jitter=jitter,
                      apiversion=apiversion,
                      forcefetch=forcefetch,
                      cachedir=cachedir,
                      verbose=verbose,
                      timeout=timeout,
                      refresh=refresh,
                      maxtimeout=maxtimeout,
                      maxtries=maxtries,
                      raiseonfail=raiseonfail)
