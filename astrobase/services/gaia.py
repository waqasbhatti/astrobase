#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''gaia - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
License: MIT. See the LICENSE file for more details.

This queries the GAIA catalog for object lists in specified areas of the
sky. The main use of this module is to generate realistic spatial distributions
of stars for variability recovery simulations in combination with colors and
luminosities from the TRILEGAL galaxy model.

If you use this module, please cite the GAIA papers as outlined at:

https://gaia.esac.esa.int/documentation//GDR1/Miscellaneous/sec_credit_and_citation_instructions.html

Much of this module is derived from the example given at:

http://gea.esac.esa.int/archive-help/commandline/index.html

For a more general and useful interface to the GAIA catalog, see the astroquery
package by A. Ginsburg, B. Sipocz, et al.:

http://astroquery.readthedocs.io/en/latest/gaia/gaia.html

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

# to read the XML returned by the TAP service
from xml.dom.minidom import parseString


###################
## FORM SETTINGS ##
###################

TAP_URL = "http://gea.esac.esa.int/tap-server/tap/async"


TAP_PARAMS = {
    'REQUEST':'doQuery',
    'LANG':'ADQL',
    'FORMAT':'json',
    'PHASE':'RUN',
    'JOBNAME':'',
    'JOBDESCRIPTION':'',
    'QUERY':''
}

RETURN_FORMATS = {
    'json':'json.gz',
    'csv':'csv.gz',
    'votable':'vot',
}


#####################
## QUERY FUNCTIONS ##
#####################


def tap_query(querystr,
              returnformat='csv',
              forcefetch=False,
              cachedir='~/.astrobase/gaia-cache',
              verbose=True,
              timeout=60.0,
              refresh=2.0,
              maxtimeout=700.0):
    '''This queries the GAIA TAP service using the ADQL querystr.

    querystr is an ADQL query. See: http://www.ivoa.net/documents/ADQL/2.0 for
    the specification and http://gea.esac.esa.int/archive-help/adql/index.html
    for GAIA-specific additions.

    returnformat is one of 'csv', 'votable', or 'json'.

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

    # get the default params
    inputparams = TAP_PARAMS.copy()

    # update them with our input params
    inputparams['QUERY'] = querystr
    if returnformat in RETURN_FORMATS:
        inputparams['FORMAT'] = returnformat
    else:
        LOGWARNING('unknown result format: %s requested, using CSV' %
                   returnformat)
        inputparams['FORMAT'] = 'csv'

    # see if the cachedir exists
    if '~' in cachedir:
        cachedir = os.path.expanduser(cachedir)
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    # generate the cachefname and look for it
    cachekey = repr(inputparams)
    cachekey = hashlib.sha256(cachekey.encode()).hexdigest()
    cachefname = os.path.join(
        cachedir,
        '%s.%s' % (cachekey, RETURN_FORMATS[returnformat])
    )
    provenance = 'cache'

    # generate a jobid here and update the input params
    jobid = 'ab-gaia-%i' % time.time()
    inputparams['JOBNAME'] = jobid
    inputparams['JOBDESCRIPTION'] = 'astrobase-gaia-tap-ADQL-query'

    # run the query if results not found in the cache
    if forcefetch or (not os.path.exists(cachefname)):

        provenance = 'new download'

        try:

            if verbose:
                LOGINFO('submitting GAIA TAP query request for input params: %s'
                        % repr(inputparams))

            waitdone = False
            timeelapsed = 0.0

            # send the query and get status
            req = requests.post(TAP_URL, data=inputparams, timeout=timeout)
            resp_status = req.status_code
            req.raise_for_status()

            # NOTE: python-requests follows the "303 See Other" redirect
            # automatically, so we get the XML status doc immediately. We don't
            # need to look up the location of it in the initial response's
            # header as in the GAIA example.
            status_url = req.url

            # parse the response XML and get the job status
            resxml = parseString(req.text)
            jobstatuselem = resxml.getElementsByTagName('uws:phase')[0]
            jobstatus = jobstatuselem.firstChild.toxml()

            # if the job completed already, jump down to retrieving results
            if jobstatus == 'COMPLETED':

                if verbose:

                    LOGINFO('GAIA query completed, '
                            'retrieving results...')

                    waitdone = True

            elif jobstatus == 'ERROR':

                if verbose:

                    LOGERROR(
                        'GAIA query failed immediately '
                        '(probably an ADQL error): %s, '
                        'status URL: %s, status contents: %s' %
                        (repr(inputparams),
                         status_url,
                         req.text)
                    )
                    return None

            # we wait for the job to complete if it's not done already
            else:

                if verbose:
                    LOGINFO(
                        'request submitted successfully, '
                        'current status is: %s. '
                        'waiting for results...' % jobstatus
                    )

                while not waitdone:

                    if timeelapsed > maxtimeout:
                        LOGERROR('GAIA TAP timed out after waiting for results,'
                                 ' request was: '
                                 '%s' % repr(inputparams))
                        return None

                    time.sleep(refresh)
                    timeelapsed = timeelapsed + refresh

                    try:

                        resreq = requests.get(status_url)
                        resreq.raise_for_status()

                        # parse the response XML and get the job status
                        resxml = parseString(resreq.text)

                        jobstatuselem = (
                            resxml.getElementsByTagName('uws:phase')[0]
                        )
                        jobstatus = jobstatuselem.firstChild.toxml()

                        if jobstatus == 'COMPLETED':

                            if verbose:

                                LOGINFO('GAIA query completed, '
                                        'retrieving results...')
                            waitdone = True

                        else:
                            if verbose:
                                LOGINFO('elapsed time: %.1f, status URL: %s '
                                        'not ready yet...'
                                        % (timeelapsed, status_url))
                            continue

                    except Exception as e:

                        LOGEXCEPTION(
                            'GAIA query failed while waiting for results: %s, '
                            'status URL: %s, status contents: %s' %
                            (repr(inputparams),
                             status_url,
                             resreq.text)
                        )
                        return None

            #
            # at this point, we should be ready to get the query results
            #
            result_url_elem = resxml.getElementsByTagName('uws:result')[0]
            result_url = result_url_elem.getAttribute('xlink:href')
            result_nrows = result_url_elem.getAttribute('rows')

            try:

                resreq = requests.get(result_url)
                resreq.raise_for_status()

                if cachefname.endswith('.gz'):

                    with gzip.open(cachefname,'wb') as outfd:
                        for chunk in resreq.iter_content(chunk_size=65536):
                            outfd.write(chunk)

                else:

                    with open(cachefname,'wb') as outfd:
                        for chunk in resreq.iter_content(chunk_size=65536):
                            outfd.write(chunk)

                if verbose:
                    LOGINFO('done. rows in result: %s' % result_nrows)
                tablefname = cachefname

            except Exception as e:

                LOGEXCEPTION(
                    'GAIA query failed while trying to download results: %s, '
                    'result URL: %s, response status: %s' %
                    (repr(inputparams),
                     result_url,
                     resreq.status_code)
                )
                return None

        except requests.exceptions.HTTPError as e:
            LOGEXCEPTION('GAIA TAP query failed, request status was: '
                         '%s' % resp_status)
            return None


        except requests.exceptions.Timeout as e:
            LOGERROR('GAIA TAP query submission timed out, '
                     'site is probably down. Request was: '
                     '%s' % repr(inputparams))
            return None

        except Exception as e:
            LOGEXCEPTION('GAIA TAP query request failed for '
                         '%s' % repr(inputparams))
            return None

    else:

        if verbose:
            LOGINFO('getting cached GAIA query result for '
                    'request: %s' %
                    (repr(inputparams)))

        tablefname = cachefname


    #
    # all done with retrieval, now return the result dict
    #

    # return a dict pointing to the result file
    # we'll parse this later
    resdict = {'params':inputparams,
               'provenance':provenance,
               'result':tablefname}

    return resdict



def objectlist_conesearch(racenter,
                          declcenter,
                          searchradiusarcsec,
                          table='gaiadr1.gaia_source',
                          columns=['source_id',
                                   'ra','dec',
                                   'phot_g_mean_mag',
                                   'l','b',
                                   'parallax', 'parallax_error'],
                          returnformat='csv',
                          forcefetch=False,
                          cachedir='~/.astrobase/gaia-cache',
                          verbose=True,
                          timeout=60.0,
                          refresh=2.0,
                          maxtimeout=700.0):
    '''This queries the GAIA TAP service for a list of objects near the coords.

    Uses a conesearch around racenter, declcenter with radius in arcsec of
    searchradiusarcsec.

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

    # this was generated using the awesome query generator at:
    # https://gea.esac.esa.int/archive/
    query = (
        "select {columns}, "
        "(DISTANCE(POINT('ICRS', "
        "{table}.ra, {table}.dec), "
        "POINT('ICRS', {ra_center}, {decl_center})))*3600.0 AS dist_arcsec "
        "from {table} where "
        "CONTAINS(POINT('ICRS',{table}.ra,"
        "{table}.dec),"
        "CIRCLE('ICRS',{ra_center},{decl_center},{search_radius}))=1 "
        "ORDER by dist_arcsec asc "
    )

    formatted_query = query.format(table=table,
                                   ra_center=racenter,
                                   decl_center=declcenter,
                                   search_radius=searchradiusarcsec/3600.0,
                                   columns=', '.join(columns))

    return tap_query(formatted_query,
                     returnformat=returnformat,
                     forcefetch=forcefetch,
                     cachedir=cachedir,
                     verbose=verbose,
                     timeout=timeout,
                     refresh=refresh,
                     maxtimeout=maxtimeout)



def objectlist_radeclbox(radeclbox,
                         table='gaiadr1.gaia_source',
                         columns=['source_id',
                                  'ra','dec',
                                  'phot_g_mean_mag',
                                  'l','b',
                                  'parallax, parallax_error'],
                         returnformat='csv',
                         forcefetch=False,
                         cachedir='~/.astrobase/gaia-cache',
                         verbose=True,
                         timeout=60.0,
                         refresh=2.0,
                         maxtimeout=700.0):

    '''
    This queries the GAIA TAP service for a list of objects in radeclbox.

    radecbox = [ra_min, ra_max, decl_min, decl_max]

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

    # this was generated using the awesome query generator at:
    # https://gea.esac.esa.int/archive/
    query = (
        "select {columns} from {table} where "
        "CONTAINS(POINT('ICRS',{table}.ra, {table}.dec),"
        "BOX('ICRS',{ra_center},{decl_center},{ra_width},{decl_height}))=1"
    )

    ra_min, ra_max, decl_min, decl_max = radeclbox
    ra_center = (ra_max + ra_min)/2.0
    decl_center = (decl_max + decl_min)/2.0
    ra_width = ra_max - ra_min
    decl_height = decl_max - decl_min

    formatted_query = query.format(table=table,
                                   columns=', '.join(columns),
                                   ra_center=ra_center,
                                   decl_center=decl_center,
                                   ra_width=ra_width,
                                   decl_height=decl_height)

    return tap_query(formatted_query,
                     returnformat=returnformat,
                     forcefetch=forcefetch,
                     cachedir=cachedir,
                     verbose=verbose,
                     timeout=timeout,
                     refresh=refresh,
                     maxtimeout=maxtimeout)
