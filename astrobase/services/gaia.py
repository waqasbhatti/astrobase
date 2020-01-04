#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gaia - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
# License: MIT. See the LICENSE file for more details.

'''
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
import time
import pickle
import random

# to do the queries
import requests
import requests.exceptions

# to read the XML returned by the TAP service
from xml.dom.minidom import parseString


###################
## FORM SETTINGS ##
###################

GAIA_URLS = {
    'gaia':{'url':"https://gea.esac.esa.int/tap-server/tap/async",
            'table':'gaiadr2.gaia_source',
            'phasekeyword':'uws:phase',
            'resultkeyword':'uws:result'},
    'heidelberg':{'url':"http://gaia.ari.uni-heidelberg.de/tap/async",
                  'table':'gaiadr2.gaia_source',
                  'phasekeyword':'phase',
                  'resultkeyword':'result'},
    'vizier':{'url':"http://tapvizier.u-strasbg.fr/TAPVizieR/tap/async",
              'table':'"I/345/gaia2"',
              'phasekeyword':'phase',
              'resultkeyword':'result'},
}

# default TAP query params, will be copied and overridden
TAP_PARAMS = {
    'REQUEST':'doQuery',
    'LANG':'ADQL',
    'FORMAT':'json',
    'PHASE':'RUN',
    'JOBNAME':'',
    'JOBDESCRIPTION':'',
    'QUERY':''
}

# valid return formats
RETURN_FORMATS = {
    'json':'json.gz',
    'csv':'csv.gz',
    'votable':'vot',
}


#####################
## QUERY FUNCTIONS ##
#####################

def tap_query(querystr,
              gaia_mirror=None,
              returnformat='csv',
              forcefetch=False,
              cachedir='~/.astrobase/gaia-cache',
              verbose=True,
              timeout=15.0,
              refresh=2.0,
              maxtimeout=300.0,
              maxtries=3,
              complete_query_later=False):
    '''This queries the GAIA TAP service using an ADQL query string.

    Parameters
    ----------

    querystr : str
        This is the ADQL query string. See:
        http://www.ivoa.net/documents/ADQL/2.0 for the specification and
        http://gea.esac.esa.int/archive-help/adql/index.html for GAIA-specific
        additions.

    gaia_mirror : {'gaia','heidelberg','vizier'} or None
        This is the key used to select a GAIA catalog mirror from the
        `GAIA_URLS` dict above. If set, the specified mirror will be used. If
        None, a random mirror chosen from that dict will be used.

    returnformat : {'csv','votable','json'}
        The returned file format to request from the GAIA catalog service.

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

    completequerylater : bool
        If set to True, a submitted query that does not return a result before
        `maxtimeout` has passed will be cancelled but its input request
        parameters and the result URL provided by the service will be saved. If
        this function is then called later with these same input request
        parameters, it will check if the query finally finished and a result is
        available. If so, will download the results instead of submitting a new
        query. If it's not done yet, will start waiting for results again. To
        force launch a new query with the same request parameters, set the
        `forcefetch` kwarg to True.

    Returns
    -------

    dict
        This returns a dict of the following form::

            {'params':dict of the input params used for the query,
             'provenance':'cache' or 'new download',
             'result':path to the file on disk with the downloaded data table}

    '''

    # get the default params
    inputparams = TAP_PARAMS.copy()

    # update them with our input params

    inputparams['QUERY'] = querystr[::]

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
    xcachekey = '-'.join([repr(inputparams[x])
                         for x in sorted(inputparams.keys())])
    cachekey = hashlib.sha256(xcachekey.encode()).hexdigest()
    cachefname = os.path.join(
        cachedir,
        '%s.%s' % (cachekey, RETURN_FORMATS[returnformat])
    )
    provenance = 'cache'

    incomplete_qpklf = os.path.join(
        cachedir,
        'incomplete-query-%s' % cachekey
    )

    ##########################################
    ## COMPLETE A QUERY THAT MAY BE RUNNING ##
    ##########################################

    # first, check if this query can be resurrected
    if (not forcefetch and
        complete_query_later and
        os.path.exists(incomplete_qpklf)):

        with open(incomplete_qpklf, 'rb') as infd:
            incomplete_qinfo = pickle.load(infd)

        LOGWARNING('complete_query_later = True, and '
                   'this query was not completed on a '
                   'previous run, will check if it is done now...')

        # get the status URL and go into a loop to see if the query completed
        waitdone = False
        timeelapsed = 0.0

        gaia_mirror = incomplete_qinfo['gaia_mirror']
        status_url = incomplete_qinfo['status_url']
        phasekeyword = incomplete_qinfo['phase_keyword']
        resultkeyword = incomplete_qinfo['result_keyword']

        while not waitdone:

            if timeelapsed > maxtimeout:

                LOGERROR('GAIA TAP query still not done '
                         'after waiting %s seconds for results.\n'
                         'status URL is: %s' %
                         (maxtimeout,
                          repr(inputparams),
                          status_url))

                return None

            try:

                resreq = requests.get(status_url,
                                      timeout=timeout)

                resreq.raise_for_status()

                # parse the response XML and get the job status
                resxml = parseString(resreq.text)

                jobstatuselem = (
                    resxml.getElementsByTagName(phasekeyword)[0]
                )
                jobstatus = jobstatuselem.firstChild.toxml()

                if jobstatus == 'COMPLETED':

                    if verbose:

                        LOGINFO('GAIA query completed, '
                                'retrieving results...')
                    waitdone = True

                # if we're not done yet, then wait some more
                elif jobstatus != 'ERROR':

                    if verbose:
                        LOGINFO('elapsed time: %.1f, '
                                'current status: %s, '
                                'status URL: %s, waiting...'
                                % (timeelapsed, jobstatus, status_url))

                    time.sleep(refresh)
                    timeelapsed = timeelapsed + refresh

                # if the JOB failed, then bail out immediately
                else:

                    LOGERROR('GAIA TAP query failed due to a server error.\n'
                             'status URL: %s\n'
                             'status contents: %s' %
                             (status_url,
                              resreq.text))

                    # since this job failed, remove the incomplete query pickle
                    # so we can try this from scratch
                    os.remove(incomplete_qpklf)

                    return None

            except requests.exceptions.Timeout:

                LOGEXCEPTION(
                    'GAIA query timed out while waiting for status '
                    'download results.\n'
                    'query: %s\n'
                    'status URL: %s' %
                    (repr(inputparams), status_url)
                )

                return None

            except Exception:

                LOGEXCEPTION(
                    'GAIA query failed while waiting for status\n'
                    'query: %s\n'
                    'status URL: %s\n'
                    'status contents: %s' %
                    (repr(inputparams),
                     status_url,
                     resreq.text)
                )

                # if the query fails completely, then either the status URL
                # doesn't exist any more or something else went wrong. we'll
                # remove the incomplete query pickle so we can try this from
                # scratch
                os.remove(incomplete_qpklf)

                return None

        #
        # at this point, we should be ready to get the query results
        #
        LOGINFO('query completed, retrieving results...')
        result_url_elem = resxml.getElementsByTagName(resultkeyword)[0]
        result_url = result_url_elem.getAttribute('xlink:href')
        result_nrows = result_url_elem.getAttribute('rows')

        try:

            resreq = requests.get(result_url, timeout=timeout)
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

            provenance = 'cache'

            # return a dict pointing to the result file
            # we'll parse this later
            resdict = {'params':inputparams,
                       'provenance':provenance,
                       'result':tablefname}

            # all went well, so we'll remove the incomplete query pickle
            os.remove(incomplete_qpklf)

            return resdict

        except requests.exceptions.Timeout:

            LOGEXCEPTION(
                'GAIA query timed out while trying to '
                'download results.\n'
                'query: %s\n'
                'result URL: %s' %
                (repr(inputparams), result_url)
            )
            return None

        except Exception:

            LOGEXCEPTION(
                'GAIA query failed because of an error '
                'while trying to download results.\n'
                'query: %s\n'
                'result URL: %s\n'
                'response status code: %s' %
                (repr(inputparams),
                 result_url,
                 resreq.status_code)
            )

            # if the result download fails, then either the result URL doesn't
            # exist any more or something else went wrong. we'll remove the
            # incomplete query pickle so we can try this from scratch
            os.remove(incomplete_qpklf)

            return None

    #####################
    ## RUN A NEW QUERY ##
    #####################

    # otherwise, we check the cache if it's done already, or run it again if not
    if forcefetch or (not os.path.exists(cachefname)):

        provenance = 'new download'

        # generate a jobid here and update the input params
        jobid = 'ab-gaia-%i' % time.time()
        inputparams['JOBNAME'] = jobid
        inputparams['JOBDESCRIPTION'] = 'astrobase-gaia-tap-ADQL-query'

        try:

            waitdone = False
            timeelapsed = 0.0

            # set the gaia mirror to use
            if gaia_mirror is not None and gaia_mirror in GAIA_URLS:

                tapurl = GAIA_URLS[gaia_mirror]['url']
                resultkeyword = GAIA_URLS[gaia_mirror]['resultkeyword']
                phasekeyword = GAIA_URLS[gaia_mirror]['phasekeyword']
                randkey = gaia_mirror

                # sub in a table name if this is left unresolved in the input
                # query
                if '{table}' in querystr:
                    inputparams['QUERY'] = (
                        querystr.format(
                            table=GAIA_URLS[gaia_mirror]['table']
                        )
                    )

            else:

                randkey = random.choice(list(GAIA_URLS.keys()))
                tapurl = GAIA_URLS[randkey]['url']
                resultkeyword = GAIA_URLS[randkey]['resultkeyword']
                phasekeyword = GAIA_URLS[randkey]['phasekeyword']

                # sub in a table name if this is left unresolved in the input
                # query
                if '{table}' in querystr:
                    inputparams['QUERY'] = (
                        querystr.format(
                            table=GAIA_URLS[randkey]['table']
                        )
                    )

                if verbose:
                    LOGINFO('using GAIA mirror TAP URL: %s' % tapurl)

            # send the query and get status
            if verbose:
                LOGINFO('submitting GAIA TAP query request for input params: %s'
                        % repr(inputparams))

            # here, we'll make sure the GAIA mirror works before doing anything
            # else
            mirrorok = False
            ntries = 1

            while not mirrorok:

                if ntries > maxtries:

                    LOGERROR('maximum number of allowed GAIA query '
                             'submission tries (%s) reached, bailing out...' %
                             maxtries)
                    return None

                try:

                    req = requests.post(tapurl,
                                        data=inputparams,
                                        timeout=timeout)
                    resp_status = req.status_code
                    req.raise_for_status()

                    mirrorok = True

                # this handles immediate 503s
                except requests.exceptions.HTTPError:

                    LOGWARNING(
                        'GAIA TAP server: %s not responding, '
                        'trying another mirror...'
                        % tapurl
                    )
                    mirrorok = False

                    # make sure not to hit current mirror again if it's down
                    remainingmirrors = list(GAIA_URLS.keys())
                    remainingmirrors.remove(randkey)

                    randkey = random.choice(remainingmirrors)
                    tapurl = GAIA_URLS[randkey]['url']
                    resultkeyword = GAIA_URLS[randkey]['resultkeyword']
                    phasekeyword = GAIA_URLS[randkey]['phasekeyword']
                    if '{table}' in querystr:
                        inputparams['QUERY'] = (
                            querystr.format(
                                table=GAIA_URLS[randkey]['table']
                            )
                        )

                # this handles initial query submission timeouts
                except requests.exceptions.Timeout:

                    LOGWARNING(
                        'GAIA TAP query submission timed out, '
                        'mirror is probably down. Trying another mirror...'
                    )
                    mirrorok = False

                    # make sure not to hit current mirror again if it's down
                    remainingmirrors = list(GAIA_URLS.keys())
                    remainingmirrors.remove(randkey)

                    randkey = random.choice(remainingmirrors)
                    tapurl = GAIA_URLS[randkey]['url']
                    resultkeyword = GAIA_URLS[randkey]['resultkeyword']
                    phasekeyword = GAIA_URLS[randkey]['phasekeyword']
                    if '{table}' in querystr:
                        inputparams['QUERY'] = (
                            querystr.format(
                                table=GAIA_URLS[randkey]['table']
                            )
                        )

                # update the number of submission tries
                ntries = ntries + 1

            # NOTE: python-requests follows the "303 See Other" redirect
            # automatically, so we get the XML status doc immediately. We don't
            # need to look up the location of it in the initial response's
            # header as in the GAIA example.
            status_url = req.url

            # parse the response XML and get the job status
            resxml = parseString(req.text)
            jobstatuselem = resxml.getElementsByTagName(phasekeyword)

            if jobstatuselem:

                jobstatuselem = jobstatuselem[0]

            else:
                LOGERROR('could not parse job phase using '
                         'keyword %s in result XML' % phasekeyword)
                LOGERROR('%s' % req.txt)

                req.close()
                return None

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

                        LOGERROR('GAIA TAP query timed out '
                                 'after waiting %s seconds for results.\n'
                                 'request was: %s\n'
                                 'status URL is: %s\n'
                                 'last status was: %s' %
                                 (maxtimeout,
                                  repr(inputparams),
                                  status_url,
                                  jobstatus))

                        # here, we'll check if we're allowed to sleep on a query
                        # for a bit and return to it later if the last status
                        # was QUEUED or EXECUTING
                        if complete_query_later and jobstatus in ('EXECUTING',
                                                                  'QUEUED'):

                            # write a pickle with the query params that we can
                            # pick up later to finish this query
                            incomplete_qpklf = os.path.join(
                                cachedir,
                                'incomplete-query-%s' % cachekey
                            )
                            with open(incomplete_qpklf, 'wb') as outfd:

                                savedict = inputparams.copy()

                                savedict['status_url'] = status_url
                                savedict['last_status'] = jobstatus
                                savedict['gaia_mirror'] = gaia_mirror
                                savedict['phase_keyword'] = phasekeyword
                                savedict['result_keyword'] = resultkeyword

                                pickle.dump(savedict,
                                            outfd,
                                            pickle.HIGHEST_PROTOCOL)

                            LOGINFO('complete_query_later = True, '
                                    'last state of query was: %s, '
                                    'will resume later if this function '
                                    'is called again with the same query' %
                                    jobstatus)

                        return None

                    time.sleep(refresh)
                    timeelapsed = timeelapsed + refresh

                    try:

                        resreq = requests.get(status_url, timeout=timeout)
                        resreq.raise_for_status()

                        # parse the response XML and get the job status
                        resxml = parseString(resreq.text)

                        jobstatuselem = (
                            resxml.getElementsByTagName(phasekeyword)[0]
                        )
                        jobstatus = jobstatuselem.firstChild.toxml()

                        if jobstatus == 'COMPLETED':

                            if verbose:

                                LOGINFO('GAIA query completed, '
                                        'retrieving results...')
                            waitdone = True

                        else:
                            if verbose:
                                LOGINFO('elapsed time: %.1f, '
                                        'current status: %s, '
                                        'status URL: %s, waiting...'
                                        % (timeelapsed, jobstatus, status_url))
                            continue

                    except requests.exceptions.Timeout:

                        LOGEXCEPTION(
                            'GAIA query timed out while waiting for results '
                            'download results.\n'
                            'query: %s\n'
                            'status URL: %s' %
                            (repr(inputparams), status_url)
                        )
                        return None

                    except Exception:

                        LOGEXCEPTION(
                            'GAIA query failed while waiting for results\n'
                            'query: %s\n'
                            'status URL: %s\n'
                            'status contents: %s' %
                            (repr(inputparams),
                             status_url,
                             resreq.text)
                        )
                        return None

            #
            # at this point, we should be ready to get the query results
            #
            result_url_elem = resxml.getElementsByTagName(resultkeyword)[0]
            result_url = result_url_elem.getAttribute('xlink:href')
            result_nrows = result_url_elem.getAttribute('rows')

            try:

                resreq = requests.get(result_url, timeout=timeout)
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

            except requests.exceptions.Timeout:

                LOGEXCEPTION(
                    'GAIA query timed out while trying to '
                    'download results.\n'
                    'query: %s\n'
                    'result URL: %s' %
                    (repr(inputparams), result_url)
                )
                return None

            except Exception:

                LOGEXCEPTION(
                    'GAIA query failed because of an error '
                    'while trying to download results.\n'
                    'query: %s\n'
                    'result URL: %s\n'
                    'response status code: %s' %
                    (repr(inputparams),
                     result_url,
                     resreq.status_code)
                )
                return None

        except requests.exceptions.HTTPError:
            LOGEXCEPTION('GAIA TAP query failed.\nrequest status was: '
                         '%s.\nquery was: %s' % (resp_status,
                                                 repr(inputparams)))
            return None

        except requests.exceptions.Timeout:
            LOGERROR('GAIA TAP query submission timed out, '
                     'site is probably down. Request was: '
                     '%s' % repr(inputparams))
            return None

        except Exception:
            LOGEXCEPTION('GAIA TAP query request failed for '
                         '%s' % repr(inputparams))

            if 'resxml' in locals():
                LOGERROR('HTTP response from service:\n%s' % req.text)

            return None

    ############################
    ## GET RESULTS FROM CACHE ##
    ############################

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
                          gaia_mirror=None,
                          columns=('source_id',
                                   'ra','dec',
                                   'phot_g_mean_mag',
                                   'l','b',
                                   'parallax', 'parallax_error',
                                   'pmra','pmra_error',
                                   'pmdec','pmdec_error'),
                          extra_filter=None,
                          returnformat='csv',
                          forcefetch=False,
                          cachedir='~/.astrobase/gaia-cache',
                          verbose=True,
                          timeout=15.0,
                          refresh=2.0,
                          maxtimeout=300.0,
                          maxtries=3,
                          complete_query_later=True):
    '''This queries the GAIA TAP service for a list of objects near the coords.

    Runs a conesearch around `(racenter, declcenter)` with radius in arcsec of
    `searchradiusarcsec`.

    Parameters
    ----------

    racenter,declcenter : float
        The center equatorial coordinates in decimal degrees.

    searchradiusarcsec : float
        The search radius of the cone-search in arcseconds.

    gaia_mirror : {'gaia','heidelberg','vizier'} or None
        This is the key used to select a GAIA catalog mirror from the
        `GAIA_URLS` dict above. If set, the specified mirror will be used. If
        None, a random mirror chosen from that dict will be used.

    columns : sequence of str
        This indicates which columns from the GAIA table to request for the
        objects found within the search radius.

    extra_filter: str or None
        If this is provided, must be a valid ADQL filter string that is used to
        further filter the cone-search results.

    returnformat : {'csv','votable','json'}
        The returned file format to request from the GAIA catalog service.

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

    completequerylater : bool
        If set to True, a submitted query that does not return a result before
        `maxtimeout` has passed will be cancelled but its input request
        parameters and the result URL provided by the service will be saved. If
        this function is then called later with these same input request
        parameters, it will check if the query finally finished and a result is
        available. If so, will download the results instead of submitting a new
        query. If it's not done yet, will start waiting for results again. To
        force launch a new query with the same request parameters, set the
        `forcefetch` kwarg to True.

    Returns
    -------

    dict
        This returns a dict of the following form::

            {'params':dict of the input params used for the query,
             'provenance':'cache' or 'new download',
             'result':path to the file on disk with the downloaded data table}

    '''

    # this was generated using the awesome query generator at:
    # https://gea.esac.esa.int/archive/

    # NOTE: here we don't resolve the table name right away. this is because
    # some of the GAIA mirrors use different table names, so we leave the table
    # name to be resolved by the lower level tap_query function. this is done by
    # the {{table}} construct.
    query = (
        "select {columns}, "
        "(DISTANCE(POINT('ICRS', "
        "{{table}}.ra, {{table}}.dec), "
        "POINT('ICRS', {ra_center:.5f}, {decl_center:.5f})))*3600.0 "
        "AS dist_arcsec "
        "from {{table}} where "
        "CONTAINS(POINT('ICRS',{{table}}.ra, {{table}}.dec),"
        "CIRCLE('ICRS',{ra_center:.5f},{decl_center:.5f},"
        "{search_radius:.6f}))=1 "
        "{extra_filter_str}"
        "ORDER by dist_arcsec asc "
    )

    if extra_filter is not None:
        extra_filter_str = ' and %s ' % extra_filter
    else:
        extra_filter_str = ''

    formatted_query = query.format(ra_center=racenter,
                                   decl_center=declcenter,
                                   search_radius=searchradiusarcsec/3600.0,
                                   extra_filter_str=extra_filter_str,
                                   columns=', '.join(columns))

    return tap_query(formatted_query,
                     gaia_mirror=gaia_mirror,
                     returnformat=returnformat,
                     forcefetch=forcefetch,
                     cachedir=cachedir,
                     verbose=verbose,
                     timeout=timeout,
                     refresh=refresh,
                     maxtimeout=maxtimeout,
                     maxtries=maxtries,
                     complete_query_later=complete_query_later)


def objectlist_radeclbox(radeclbox,
                         gaia_mirror=None,
                         columns=('source_id',
                                  'ra','dec',
                                  'phot_g_mean_mag',
                                  'l','b',
                                  'parallax, parallax_error',
                                  'pmra','pmra_error',
                                  'pmdec','pmdec_error'),
                         extra_filter=None,
                         returnformat='csv',
                         forcefetch=False,
                         cachedir='~/.astrobase/gaia-cache',
                         verbose=True,
                         timeout=15.0,
                         refresh=2.0,
                         maxtimeout=300.0,
                         maxtries=3,
                         complete_query_later=True):

    '''This queries the GAIA TAP service for a list of objects in an equatorial
    coordinate box.

    Parameters
    ----------

    radeclbox : sequence of four floats
        This defines the box to search in::

            [ra_min, ra_max, decl_min, decl_max]

    gaia_mirror : {'gaia','heidelberg','vizier'} or None
        This is the key used to select a GAIA catalog mirror from the
        `GAIA_URLS` dict above. If set, the specified mirror will be used. If
        None, a random mirror chosen from that dict will be used.

    columns : sequence of str
        This indicates which columns from the GAIA table to request for the
        objects found within the search radius.

    extra_filter: str or None
        If this is provided, must be a valid ADQL filter string that is used to
        further filter the cone-search results.

    returnformat : {'csv','votable','json'}
        The returned file format to request from the GAIA catalog service.

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

    completequerylater : bool
        If set to True, a submitted query that does not return a result before
        `maxtimeout` has passed will be cancelled but its input request
        parameters and the result URL provided by the service will be saved. If
        this function is then called later with these same input request
        parameters, it will check if the query finally finished and a result is
        available. If so, will download the results instead of submitting a new
        query. If it's not done yet, will start waiting for results again. To
        force launch a new query with the same request parameters, set the
        `forcefetch` kwarg to True.

    Returns
    -------

    dict
        This returns a dict of the following form::

            {'params':dict of the input params used for the query,
             'provenance':'cache' or 'new download',
             'result':path to the file on disk with the downloaded data table}

    '''

    # this was generated using the awesome query generator at:
    # https://gea.esac.esa.int/archive/

    # NOTE: here we don't resolve the table name right away. this is because
    # some of the GAIA mirrors use different table names, so we leave the table
    # name to be resolved by the lower level tap_query function. this is done by
    # the {{table}} construct.
    query = (
        "select {columns} from {{table}} where "
        "CONTAINS(POINT('ICRS',{{table}}.ra, {{table}}.dec),"
        "BOX('ICRS',{ra_center:.5f},{decl_center:.5f},"
        "{ra_width:.5f},{decl_height:.5f}))=1"
        "{extra_filter_str}"
    )

    ra_min, ra_max, decl_min, decl_max = radeclbox
    ra_center = (ra_max + ra_min)/2.0
    decl_center = (decl_max + decl_min)/2.0
    ra_width = ra_max - ra_min
    decl_height = decl_max - decl_min

    if extra_filter is not None:
        extra_filter_str = ' and %s ' % extra_filter
    else:
        extra_filter_str = ''

    formatted_query = query.format(columns=', '.join(columns),
                                   extra_filter_str=extra_filter_str,
                                   ra_center=ra_center,
                                   decl_center=decl_center,
                                   ra_width=ra_width,
                                   decl_height=decl_height)

    return tap_query(formatted_query,
                     gaia_mirror=gaia_mirror,
                     returnformat=returnformat,
                     forcefetch=forcefetch,
                     cachedir=cachedir,
                     verbose=verbose,
                     timeout=timeout,
                     refresh=refresh,
                     maxtimeout=maxtimeout,
                     maxtries=maxtries,
                     complete_query_later=complete_query_later)


def objectid_search(gaiaid,
                    gaia_mirror=None,
                    columns=('source_id',
                             'ra','dec',
                             'phot_g_mean_mag',
                             'phot_bp_mean_mag',
                             'phot_rp_mean_mag',
                             'l','b',
                             'parallax, parallax_error',
                             'pmra','pmra_error',
                             'pmdec','pmdec_error'),
                    returnformat='csv',
                    forcefetch=False,
                    cachedir='~/.astrobase/gaia-cache',
                    verbose=True,
                    timeout=15.0,
                    refresh=2.0,
                    maxtimeout=300.0,
                    maxtries=3,
                    complete_query_later=True):

    '''This queries the GAIA TAP service for a single GAIA source ID.

    Parameters
    ----------

    gaiaid : str
        The source ID of the object whose info will be collected.

    gaia_mirror : {'gaia','heidelberg','vizier'} or None
        This is the key used to select a GAIA catalog mirror from the
        `GAIA_URLS` dict above. If set, the specified mirror will be used. If
        None, a random mirror chosen from that dict will be used.

    columns : sequence of str
        This indicates which columns from the GAIA table to request for the
        objects found within the search radius.

    returnformat : {'csv','votable','json'}
        The returned file format to request from the GAIA catalog service.

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

    completequerylater : bool
        If set to True, a submitted query that does not return a result before
        `maxtimeout` has passed will be cancelled but its input request
        parameters and the result URL provided by the service will be saved. If
        this function is then called later with these same input request
        parameters, it will check if the query finally finished and a result is
        available. If so, will download the results instead of submitting a new
        query. If it's not done yet, will start waiting for results again. To
        force launch a new query with the same request parameters, set the
        `forcefetch` kwarg to True.

    Returns
    -------

    dict
        This returns a dict of the following form::

            {'params':dict of the input params used for the query,
             'provenance':'cache' or 'new download',
             'result':path to the file on disk with the downloaded data table}

    '''

    # NOTE: here we don't resolve the table name right away. this is because
    # some of the GAIA mirrors use different table names, so we leave the table
    # name to be resolved by the lower level tap_query function. this is done by
    # the {{table}} construct.
    query = (
        "select {columns} from {{table}} where "
        "source_id = {gaiaid}"
    )

    formatted_query = query.format(columns=', '.join(columns),
                                   gaiaid=gaiaid)

    return tap_query(formatted_query,
                     gaia_mirror=gaia_mirror,
                     returnformat=returnformat,
                     forcefetch=forcefetch,
                     cachedir=cachedir,
                     verbose=verbose,
                     timeout=timeout,
                     refresh=refresh,
                     maxtimeout=maxtimeout,
                     maxtries=maxtries,
                     complete_query_later=complete_query_later)
