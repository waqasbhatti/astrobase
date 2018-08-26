#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

'''lccs.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Aug 2018
License: MIT - see LICENSE for the full text.

This contains functions to search for objects and get light curves from a Light
Curve Collection (LCC) server (https://github.com/waqasbhatti/lcc-server) using
its HTTP API.


SERVICES SUPPORTED
------------------

This currently supports the following LCC server functions:

- conesearch   -> use cone_search(lcc_server, center_ra, center_decl, ...)

- ftsquery     -> use fulltext_search(lcc_server, searchterm, ...)

- columnsearch -> use column_search(lcc_server, filters, ...)

- xmatch       -> use xmatch_search(lcc_server, file_to_upload, ...)

The functions above will download the data products (data table CSVs, light
curve ZIP files) of the search results automatically, or in case the query takes
too long, will return within a configurable timeout. The query information is
cached to ~/.astrobase/lccs, and can be used to download data products for
long-running queries later.

The functions below support various auxiliary LCC services:

- get-dataset  -> use get_dataset(lcc_server, dataset_id)

- objectinfo   -> use object_info(lcc_server, objectid, collection, ...)

- dataset-list -> use list_recent_datasets(lcc_server, nrecent=25, ...)

- collections  -> use list_lc_collections(lcc_server)


COMMAND-LINE USAGE
------------------

TODO: make this actually happen...

You can use this module as a command line tool. If you installed astrobase from
pip or setup.py, you will have the `lccs` script available on your $PATH. If you
just downloaded this file as a standalone module, make it executable (using
chmod u+x or similar), then run `./lccs.py`. Use the --help flag to see all
available commands and options.

'''

# put this in here because lccs can be used as a standalone module
__version__ = '0.3.18'


#############
## LOGGING ##
#############

try:
    from datetime import datetime, timezone
    utc = timezone.utc
except:
    from datetime import datetime, timedelta, tzinfo

    # we'll need to instantiate a tzinfo object because py2.7's datetime
    # doesn't have the super convenient timezone object (seriously)
    # https://docs.python.org/2/library/datetime.html#datetime.tzinfo.fromutc
    ZERO = timedelta(0)

    class UTC(tzinfo):
        """UTC"""

        def utcoffset(self, dt):
            return ZERO

        def tzname(self, dt):
            return "UTC"

        def dst(self, dt):
            return ZERO

    utc = UTC()

import logging
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
import json
import sys
import time

try:
    import cPickle as pickle
except:
    import pickle



# import url methods here.  we use built-ins because we want this module to be
# usable as a single file. otherwise, we'd use something sane like Requests.

# Python 2
try:
    from urllib import urlretrieve, urlencode
    from urlparse import urlparse
    from urllib2 import urlopen, Request, HTTPError
# Python 3
except:
    from urllib.request import urlretrieve, urlopen, Request
    from urllib.error import HTTPError
    from urllib.parse import urlencode, urlparse


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
                              'lccs',
                              'apikey-%s' % lcc_server.replace(
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
            now = datetime.now(utc)

            if sys.version_info[:2] < (3,7):
                # this hideous incantation is required for lesser Pythons
                expdt = datetime.strptime(
                    expires.replace('Z',''),
                    '%Y-%m-%dT%H:%M:%S.%f'
                ).replace(tzinfo=utc)
            else:
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
                              'lccs',
                              'apikey-%s' % lcc_server.replace(
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

    if resp.code == 200:

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

    with open(APIKEYFILE,'w') as outfd:
        outfd.write('%s %s\n' % (apikey, expires))

    # chmod it to the correct value
    os.chmod(APIKEYFILE, 0o100600)

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



##############################
## QUERY HANDLING FUNCTIONS ##
##############################

def submit_get_searchquery(url, params, apikey=None):
    '''This submits a GET query to an LCC server API endpoint.

    Handles streaming of the results, and returns the final JSON stream. Also
    handles results that time out.

    url is the URL to hit for getting the results. This will probably be
    something like https://lcc.server.org/api/xyz.

    params is a dict of args to generate a query string for the API.

    apikey is not currently required for GET requests, but may be in the future,
    so it's handled here anyway. This is passed in the header of the HTTP
    request.

    '''

    # first, we need to convert any columns and collections items to broken out
    # params
    urlparams = {}

    for key in params:

        if key == 'columns':
            urlparams['columns[]'] = params[key]
        elif key == 'collections':
            urlparams['collections[]'] = params[key]
        else:
            urlparams[key] = params[key]

    # do the urlencode with doseq=True
    urlqs = urlencode(urlparams, doseq=True)
    qurl = "%s?%s" % (url, urlqs)

    # if apikey is not None, add it in as an Authorization: Bearer [apikey]
    # header
    if apikey:
        headers = {'Authorization':'Bearer: %s' % apikey}
    else:
        headers = {}

    LOGINFO('submitting search query to LCC server API URL: %s' % url)

    try:

        # hit the server
        req = Request(qurl, data=None, headers=headers)
        resp = urlopen(req)

        if resp.code == 200:

            # we'll iterate over the lines in the response
            # this works super-well for ND-JSON!
            for line in resp:

                data = json.loads(line)
                msg = data['message']
                status = data['status']

                if status != 'failed':
                    LOGINFO('status: %s, %s' % (status, msg))
                else:
                    LOGERROR('status: %s, %s' % (status, msg))

                # here, we'll decide what to do about the query

                # completed query or query sent to background...
                if status in ('ok','background'):

                    setid = data['result']['setid']
                    # save the data pickle to astrobase lccs directory
                    outpickle = os.path.join(os.path.expanduser('~'),
                                             '.astrobase',
                                             'lccs',
                                             'query-%s.pkl' % setid)
                    if not os.path.exists(os.path.dirname(outpickle)):
                        os.makedirs(os.path.dirname(outpickle))

                    with open(outpickle,'wb') as outfd:
                        pickle.dump(data, outfd, pickle.HIGHEST_PROTOCOL)
                        LOGINFO('saved query info to %s, use this to '
                                'download results later with '
                                'retrieve_dataset_files' % outpickle)

                    # we're done at this point, return
                    return status, data, data['result']['setid']

                # the query probably failed...
                elif status == 'failed':

                    # we're done at this point, return
                    return status, data, None


        # if the response was not OK, then we probably failed
        else:

            try:
                data = json.load(resp)
                msg = data['message']

                LOGERROR(msg)
                return 'failed', None, None

            except:

                LOGEXCEPTION('failed to submit query to %s' % url)
                return 'failed', None, None

    except HTTPError as e:

        LOGERROR('could not submit query to LCC API at: %s' % url)
        LOGERROR('HTTP status code was %s, reason: %s' % (e.code, e.reason))
        return 'failed', None, None



def submit_post_searchquery(url, data, apikey):
    '''This submits a POST query to an LCC server API endpoint.

    Handles streaming of the results, and returns the final JSON stream. Also
    handles results that time out.

    apikey is currently required for any POST requests.

    '''

    # first, we need to convert any columns and collections items to broken out
    # params
    postdata = {}

    for key in data:

        if key == 'columns':
            postdata['columns[]'] = data[key]
        elif key == 'collections':
            postdata['collections[]'] = data[key]
        else:
            postdata[key] = data[key]

    # do the urlencode with doseq=True
    # we also need to encode to bytes
    encoded_postdata = urlencode(postdata, doseq=True).encode()

    # if apikey is not None, add it in as an Authorization: Bearer [apikey]
    # header
    if apikey:
        headers = {'Authorization':'Bearer: %s' % apikey}
    else:
        headers = {}

    LOGINFO('submitting search query to LCC server API URL: %s' % url)

    try:

        # hit the server with a POST request
        req = Request(url, data=encoded_postdata, headers=headers)
        resp = urlopen(req)

        if resp.code == 200:

            # we'll iterate over the lines in the response
            # this works super-well for ND-JSON!
            for line in resp:

                data = json.loads(line)
                msg = data['message']
                status = data['status']

                if status != 'failed':
                    LOGINFO('status: %s, %s' % (status, msg))
                else:
                    LOGERROR('status: %s, %s' % (status, msg))

                # here, we'll decide what to do about the query

                # completed query or query sent to background...
                if status in ('ok','background'):

                    setid = data['result']['setid']
                    # save the data pickle to astrobase lccs directory
                    outpickle = os.path.join(os.path.expanduser('~'),
                                             '.astrobase',
                                             'lccs',
                                             'query-%s.pkl' % setid)
                    if not os.path.exists(os.path.dirname(outpickle)):
                        os.makedirs(os.path.dirname(outpickle))

                    with open(outpickle,'wb') as outfd:
                        pickle.dump(data, outfd, pickle.HIGHEST_PROTOCOL)
                        LOGINFO('saved query info to %s, use this to '
                                'download results later with '
                                'retrieve_dataset_files' % outpickle)

                    # we're done at this point, return
                    return status, data, data['result']['setid']

                # the query probably failed...
                elif status == 'failed':

                    # we're done at this point, return
                    return status, data, None


        # if the response was not OK, then we probably failed
        else:

            try:
                data = json.load(resp)
                msg = data['message']

                LOGERROR(msg)
                return 'failed', None, None

            except:

                LOGEXCEPTION('failed to submit query to %s' % url)
                return 'failed', None, None

    except HTTPError as e:

        LOGERROR('could not submit query to LCC API at: %s' % url)
        LOGERROR('HTTP status code was %s, reason: %s' % (e.code, e.reason))
        return 'failed', None, None



def retrieve_dataset_files(searchresult,
                           getpickle=False,
                           outdir=None):
    '''This retrieves the dataset's CSV and any LC zip files.

    Takes the output from submit_*_query functions above or a pickle file
    generated from these if the query timed out.

    If getpickle is True, will also download the dataset's pickle. Note that LCC
    server is a Python 3.6+ package and it saves its pickles in
    pickle.HIGHEST_PROTOCOL, so these pickles may be unreadable in lower
    Pythons. The dataset CSV contains the full data table and all the
    information about the dataset in its header, which is JSON parseable. You
    can also use the function get_dataset to get the dataset pickle information
    in JSON form.

    Puts the files in outdir. If it's None, they will be placed in the current
    directory.

    '''

    # this handles the direct result case from submit_*_query functions
    if isinstance(searchresult, tuple):

        info, setid = searchresult[1:]

    # handles the case where we give the function a existing query pickle
    elif isinstance(searchresult, str) and os.path.exists(searchresult):

        with open(searchresult,'rb') as infd:
            info = pickle.load(infd)
        setid = info['result']['setid']

    else:

        LOGERROR('could not understand input, '
                 'we need a searchresult from a '
                 'lccs.submit_*_query function or '
                 'the path to an existing query pickle')
        return None, None, None

    # now that we have everything, let's download some files!

    dataset_pickle = 'dataset-%s.pkl.gz' % setid
    dataset_csv = 'dataset-%s.csv' % setid
    dataset_lczip = 'lightcurves-%s.zip' % setid

    if outdir is None:
        localdir = os.getcwd()
    else:
        localdir = outdir

    server_scheme, server_netloc = urlparse(info['result']['seturl'])[:2]
    dataset_pickle_link = '%s://%s/d/%s' % (server_scheme,
                                            server_netloc,
                                            dataset_pickle)
    dataset_csv_link = '%s://%s/d/%s' % (server_scheme,
                                         server_netloc,
                                         dataset_csv)
    dataset_lczip_link = '%s://%s/p/%s' % (server_scheme,
                                           server_netloc,
                                           dataset_lczip)

    if getpickle:

        # get the dataset pickle
        LOGINFO('getting %s...' % dataset_pickle_link)
        try:

            if os.path.exists(os.path.join(localdir, dataset_pickle)):

                LOGWARNING('dataset pickle already exists, '
                           'not downloading again..')
                local_dataset_pickle = os.path.join(localdir,
                                                    dataset_pickle)

            else:

                localf, header = urlretrieve(dataset_pickle_link,
                                             os.path.join(localdir,
                                                          dataset_pickle),
                                             reporthook=on_download_chunk)
                LOGINFO('OK -> %s' % localf)
                local_dataset_pickle = localf

        except HTTPError as e:
            LOGERROR('could not download %s, '
                     'HTTP status code was: %s, reason: %s' %
                     (dataset_pickle_link, e.code, e.reason))
            local_dataset_pickle = None

    else:
        local_dataset_pickle = None


    # get the dataset CSV
    LOGINFO('getting %s...' % dataset_csv_link)
    try:

        if os.path.exists(os.path.join(localdir, dataset_csv)):

            LOGWARNING('dataset CSV already exists, not downloading again...')
            local_dataset_csv = os.path.join(localdir, dataset_csv)

        else:

            localf, header = urlretrieve(dataset_csv_link,
                                         os.path.join(localdir, dataset_csv),
                                         reporthook=on_download_chunk)
            LOGINFO('OK -> %s' % localf)
            local_dataset_csv = localf

    except HTTPError as e:
        LOGERROR('could not download %s, HTTP status code was: %s, reason: %s' %
                 (dataset_csv_link, e.code, e.reason))
        local_dataset_csv = None


    # get the dataset LC zip
    LOGINFO('getting %s...' % dataset_lczip_link)
    try:

        if os.path.exists(os.path.join(localdir, dataset_lczip)):

            LOGWARNING('dataset LC ZIP already exists, '
                       'not downloading again...')
            local_dataset_lczip = os.path.join(localdir, dataset_lczip)

        else:

            localf, header = urlretrieve(dataset_lczip_link,
                                         os.path.join(localdir, dataset_lczip),
                                         reporthook=on_download_chunk)
            LOGINFO('OK -> %s' % localf)
            local_dataset_lczip = localf

    except HTTPError as e:
        LOGERROR('could not download %s, HTTP status code was: %s, reason: %s' %
                 (dataset_lczip_link, e.code, e.reason))
        local_dataset_lczip = None


    return local_dataset_csv, local_dataset_lczip, local_dataset_pickle



###########################
## MAIN SEARCH FUNCTIONS ##
###########################

def cone_search(lcc_server,
                center_ra,
                center_decl,
                radiusarcmin=5.0,
                collections=None,
                columns=None,
                filters=None,
                download_data=True,
                outdir=None,
                maxtimeout=300.0,
                refresh=15.0,
                result_ispublic=True):

    '''This runs a cone-search query.

    Returns a tuple with the following elements:

    (search result status dict,
     search result CSV file path,
     search result LC ZIP path)

    lcc_server is the base URL of the LCC server to talk to.
    (e.g. for HAT, use: https://data.hatsurveys.org)

    center_ra, center_decl are the central coordinates of the search to
    conduct. These can be either decimal degrees of type float, or sexagesimal
    coordinates of type str:

    OK: 290.0, 45.0
    OK: 15:00:00 +45:00:00
    OK: 15 00 00.0 -45 00 00.0
    NOT OK: 290.0 +45:00:00
    NOT OK: 15:00:00 45.0

    radiusarcmin is the search radius. This is in arcminutes. The maximum radius
    you can use is 60 arcminutes.

    collections is a list of LC collections to search in. If this is None, all
    collections will be searched.

    columns is a list of columns to return in the results. Matching objects'
    object IDs, RAs, DECs, and links to light curve files will always be
    returned so there is no need to specify these columns.

    filters is an SQL-like string to use to filter on database columns in the
    LCC server's collections. To see the columns available for a search, visit
    the Collections tab in the LCC server's browser UI. The filter operators
    allowed are:

    lt -> less than
    gt -> greater than
    ge -> greater than or equal to
    le -> less than or equal to
    eq -> equal to
    ne -> not equal to
    ct -> contains text

    You may use the 'and' and 'or' operators between filter specifications to
    chain them together logically.

    Example filter strings:

    "(propermotion gt 200.0) and (sdssr lt 11.0)"
    "(dered_jmag_kmag gt 2.0) and (aep_000_stetsonj gt 10.0)"
    "(gaia_status ct 'ok') and (propermotion gt 300.0)"
    "(simbad_best_objtype ct 'RR') and (dered_sdssu_sdssg lt 0.5)"

    download_data sets if the accompanying data from the search results will be
    downloaded automatically. This includes the data table CSV, the dataset
    pickle file, and a light curve ZIP file. Note that if the search service
    indicates that your query is still in progress, this function will block
    until the light curve ZIP file becomes available. The maximum wait time in
    seconds is set by maxtimeout and the refresh interval is set by refresh.

    To avoid the wait block, set download_data to False and the function will
    write a pickle file to ~/.astrobase/lccs/query-[setid].pkl containing all
    the information necessary to retrieve these data files later when the query
    is done. To do so, call the retrieve_dataset_files with the path to this
    pickle file (it will be returned).

    outdir if not None, sets the output directory of the downloaded dataset
    files. If None, they will be downloaded to the current directory.

    result_ispublic sets if you want your dataset to be publicly visible on the
    Recent Datasets tab and /datasets page of the LCC server you're talking
    to. If False, only people who know the unique dataset URL can view and fetch
    data files from it later.

    '''

    # turn the input into a param dict

    coords = '%s %s %s' % (center_ra, center_decl, radiusarcmin)
    params = {'coords':coords}

    if collections:
        params['collections'] = collections
    if columns:
        params['columns'] = columns
    if filters:
        params['filters'] = filters

    params['result_ispublic'] = 1 if result_ispublic else 0

    # hit the server
    api_url = '%s/api/conesearch' % lcc_server

    # no API key is required for now, but we'll load one automatically if we
    # require it in the future
    searchresult = submit_get_searchquery(api_url, params, apikey=None)

    # check the status of the search
    status = searchresult[0]

    # now we'll check if we want to download the data
    if download_data:

        if status == 'ok':

            LOGINFO('query complete, downloading associated data...')
            csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                     outdir=outdir)

            if pkl:
                return searchresult[1], csv, lczip, pkl
            else:
                return searchresult[1], csv, lczip

        elif status == 'background':

            LOGINFO('query is not yet complete, '
                    'waiting up to %.1f minutes, '
                    'updates every %s seconds (hit Ctrl+C to cancel)...' %
                    (maxtimeout/60.0, refresh))

            timewaited = 0.0

            while timewaited < maxtimeout:

                try:

                    time.sleep(refresh)
                    csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                             outdir=outdir)

                    if (csv and os.path.exists(csv) and
                        lczip and os.path.exists(lczip)):

                        LOGINFO('all dataset products collected')
                        return searchresult[1], csv, lczip

                    timewaited = timewaited + refresh

                except KeyboardInterrupt:

                    LOGWARNING('abandoned wait for downloading data')
                    return searchresult[1], None, None

            LOGERROR('wait timed out.')
            return searchresult[1], None, None

        else:

            LOGERROR('could not download the data for this query result')
            return searchresult[1], None, None

    else:

        return searchresult[1], None, None



def fulltext_search(lcc_server,
                    searchterm,
                    collections=None,
                    columns=None,
                    filters=None,
                    download_data=True,
                    outdir=None,
                    maxtimeout=300.0,
                    refresh=15.0,
                    result_ispublic=True):

    '''This runs a full-text search query.

    Returns a tuple with the following elements:

    (search result status dict,
     search result CSV file path,
     search result LC ZIP path)

    lcc_server is the base URL of the LCC server to talk to.
    (e.g. for HAT, use: https://data.hatsurveys.org)

    searchterm is the term to look for in a full-text search of the LCC server's
    collections. This can be an object name, tag, description, etc., as noted in
    the LCC server's full-text search tab in its browser UI. To search for an
    exact match to a string (like an object name), you can add double quotes
    around the string, e.g. searchitem = '"exact match to me needed"'.

    collections is a list of LC collections to search in. If this is None, all
    collections will be searched.

    columns is a list of columns to return in the results. Matching objects'
    object IDs, RAs, DECs, and links to light curve files will always be
    returned so there is no need to specify these columns.

    filters is an SQL-like string to use to filter on database columns in the
    LCC server's collections. To see the columns available for a search, visit
    the Collections tab in the LCC server's browser UI. The filter operators
    allowed are:

    lt -> less than
    gt -> greater than
    ge -> greater than or equal to
    le -> less than or equal to
    eq -> equal to
    ne -> not equal to
    ct -> contains text

    You may use the 'and' and 'or' operators between filter specifications to
    chain them together logically.

    Example filter strings:

    "(propermotion gt 200.0) and (sdssr lt 11.0)"
    "(dered_jmag_kmag gt 2.0) and (aep_000_stetsonj gt 10.0)"
    "(gaia_status ct 'ok') and (propermotion gt 300.0)"
    "(simbad_best_objtype ct 'RR') and (dered_sdssu_sdssg lt 0.5)"

    download_data sets if the accompanying data from the search results will be
    downloaded automatically. This includes the data table CSV, the dataset
    pickle file, and a light curve ZIP file. Note that if the search service
    indicates that your query is still in progress, this function will block
    until the light curve ZIP file becomes available. The maximum wait time in
    seconds is set by maxtimeout and the refresh interval is set by refresh.

    To avoid the wait block, set download_data to False and the function will
    write a pickle file to ~/.astrobase/lccs/query-[setid].pkl containing all
    the information necessary to retrieve these data files later when the query
    is done. To do so, call the retrieve_dataset_files with the path to this
    pickle file (it will be returned).

    outdir if not None, sets the output directory of the downloaded dataset
    files. If None, they will be downloaded to the current directory.

    result_ispublic sets if you want your dataset to be publicly visible on the
    Recent Datasets tab and /datasets page of the LCC server you're talking
    to. If False, only people who know the unique dataset URL can view and fetch
    data files from it later.

    '''

    # turn the input into a param dict
    params = {'ftstext':searchterm}

    if collections:
        params['collections'] = collections
    if columns:
        params['columns'] = columns
    if filters:
        params['filters'] = filters

    params['result_ispublic'] = 1 if result_ispublic else 0

    # hit the server
    api_url = '%s/api/ftsquery' % lcc_server

    # no API key is required for now, but we'll load one automatically if we
    # require it in the future
    searchresult = submit_get_searchquery(api_url, params, apikey=None)

    # check the status of the search
    status = searchresult[0]

    # now we'll check if we want to download the data
    if download_data:

        if status == 'ok':

            LOGINFO('query complete, downloading associated data...')
            csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                     outdir=outdir)

            if pkl:
                return searchresult[1], csv, lczip, pkl
            else:
                return searchresult[1], csv, lczip

        elif status == 'background':

            LOGINFO('query is not yet complete, '
                    'waiting up to %.1f minutes, '
                    'updates every %s seconds (hit Ctrl+C to cancel)...' %
                    (maxtimeout/60.0, refresh))

            timewaited = 0.0

            while timewaited < maxtimeout:

                try:

                    time.sleep(refresh)
                    csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                             outdir=outdir)

                    if (csv and os.path.exists(csv) and
                        lczip and os.path.exists(lczip)):

                        LOGINFO('all dataset products collected')
                        return searchresult[1], csv, lczip

                    timewaited = timewaited + refresh

                except KeyboardInterrupt:

                    LOGWARNING('abandoned wait for downloading data')
                    return searchresult[1], None, None

            LOGERROR('wait timed out.')
            return searchresult[1], None, None

        else:

            LOGERROR('could not download the data for this query result')
            return searchresult[1], None, None

    else:

        return searchresult[1], None, None



def column_search(lcc_server,
                  filters,
                  sortcolumn='sdssr',
                  sortorder='asc',
                  collections=None,
                  columns=None,
                  download_data=True,
                  outdir=None,
                  maxtimeout=300.0,
                  refresh=15.0,
                  result_ispublic=True):

    '''This runs a column search query.

    Returns a tuple with the following elements:

    (search result status dict,
     search result CSV file path,
     search result LC ZIP path)

    lcc_server is the base URL of the LCC server to talk to.
    (e.g. for HAT, use: https://data.hatsurveys.org)

    filters is an SQL-like string to use to filter on database columns in the
    LCC server's collections. To see the columns available for a search, visit
    the Collections tab in the LCC server's browser UI. The filter operators
    allowed are:

    lt -> less than
    gt -> greater than
    ge -> greater than or equal to
    le -> less than or equal to
    eq -> equal to
    ne -> not equal to
    ct -> contains text

    You may use the 'and' and 'or' operators between filter specifications to
    chain them together logically.

    Example filter strings:

    "(propermotion gt 200.0) and (sdssr lt 11.0)"
    "(dered_jmag_kmag gt 2.0) and (aep_000_stetsonj gt 2.0)"
    "(gaia_status ct 'ok') and (propermotion gt 300.0)"
    "(simbad_best_objtype ct 'RR') and (dered_sdssu_sdssg lt 0.5)"

    sortcolumn is the database column to sort the results by. sortorder is the
    order to sort in: 'asc' -> ascending, 'desc' -> descending.

    collections is a list of LC collections to search in. If this is None, all
    collections will be searched.

    columns is a list of columns to return in the results. Matching objects'
    object IDs, RAs, DECs, and links to light curve files will always be
    returned so there is no need to specify these columns.

    download_data sets if the accompanying data from the search results will be
    downloaded automatically. This includes the data table CSV, the dataset
    pickle file, and a light curve ZIP file. Note that if the search service
    indicates that your query is still in progress, this function will block
    until the light curve ZIP file becomes available. The maximum wait time in
    seconds is set by maxtimeout and the refresh interval is set by refresh.

    To avoid the wait block, set download_data to False and the function will
    write a pickle file to ~/.astrobase/lccs/query-[setid].pkl containing all
    the information necessary to retrieve these data files later when the query
    is done. To do so, call the retrieve_dataset_files with the path to this
    pickle file (it will be returned).

    outdir if not None, sets the output directory of the downloaded dataset
    files. If None, they will be downloaded to the current directory.

    result_ispublic sets if you want your dataset to be publicly visible on the
    Recent Datasets tab and /datasets page of the LCC server you're talking
    to. If False, only people who know the unique dataset URL can view and fetch
    data files from it later.

    '''

    # turn the input into a param dict
    params = {'filters':filters,
              'sortcol':sortcolumn,
              'sortorder':sortorder}

    if collections:
        params['collections'] = collections
    if columns:
        params['columns'] = columns

    params['result_ispublic'] = 1 if result_ispublic else 0

    # hit the server
    api_url = '%s/api/columnsearch' % lcc_server

    # no API key is required for now, but we'll load one automatically if we
    # require it in the future
    searchresult = submit_get_searchquery(api_url, params, apikey=None)

    # check the status of the search
    status = searchresult[0]

    # now we'll check if we want to download the data
    if download_data:

        if status == 'ok':

            LOGINFO('query complete, downloading associated data...')
            csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                     outdir=outdir)

            if pkl:
                return searchresult[1], csv, lczip, pkl
            else:
                return searchresult[1], csv, lczip

        elif status == 'background':

            LOGINFO('query is not yet complete, '
                    'waiting up to %.1f minutes, '
                    'updates every %s seconds (hit Ctrl+C to cancel)...' %
                    (maxtimeout/60.0, refresh))

            timewaited = 0.0

            while timewaited < maxtimeout:

                try:

                    time.sleep(refresh)
                    csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                             outdir=outdir)

                    if (csv and os.path.exists(csv) and
                        lczip and os.path.exists(lczip)):

                        LOGINFO('all dataset products collected')
                        return searchresult[1], csv, lczip

                    timewaited = timewaited + refresh

                except KeyboardInterrupt:

                    LOGWARNING('abandoned wait for downloading data')
                    return searchresult[1], None, None

            LOGERROR('wait timed out.')
            return searchresult[1], None, None

        else:

            LOGERROR('could not download the data for this query result')
            return searchresult[1], None, None

    else:

        return searchresult[1], None, None



def xmatch_search(lcc_server,
                  file_to_upload,
                  xmatch_dist_arcsec=3.0,
                  collections=None,
                  columns=None,
                  filters=None,
                  download_data=True,
                  outdir=None,
                  maxtimeout=300.0,
                  refresh=15.0,
                  result_ispublic=True):

    '''This runs a column search query.

    Returns a tuple with the following elements:

    (search result status dict,
     search result CSV file path,
     search result LC ZIP path)

    lcc_server is the base URL of the LCC server to talk to.
    (e.g. for HAT, use: https://data.hatsurveys.org)

    file_to_upload is a text file containing objectid, RA, declination rows for
    the objects to cross-match against the LCC server collections. This should
    follow the format of the following example:

    # example object and coordinate list
    # objectid ra dec
    aaa 289.99698 44.99839
    bbb 293.358 -23.206
    ccc 294.197 +23.181
    ddd 19 25 27.9129 +42 47 03.693
    eee 19:25:27 -42:47:03.21
    # .
    # .
    # .
    # etc. lines starting with '#' will be ignored
    # (max 5000 objects)

    xmatch_dist_arcsec is the maximum distance in arcseconds to consider when
    cross-matching objects in the uploaded file to the LCC server's
    collections. The maximum allowed distance is 30 arcseconds. Multiple matches
    to an uploaded object are possible and will be returned in order of
    increasing distance.

    collections is a list of LC collections to search in. If this is None, all
    collections will be searched.

    columns is a list of columns to return in the results. Matching objects'
    object IDs, RAs, DECs, and links to light curve files will always be
    returned so there is no need to specify these columns.

    filters is an SQL-like string to use to filter on database columns in the
    LCC server's collections. To see the columns available for a search, visit
    the Collections tab in the LCC server's browser UI. The filter operators
    allowed are:

    lt -> less than
    gt -> greater than
    ge -> greater than or equal to
    le -> less than or equal to
    eq -> equal to
    ne -> not equal to
    ct -> contains text

    You may use the 'and' and 'or' operators between filter specifications to
    chain them together logically.

    Example filter strings:

    "(propermotion gt 200.0) and (sdssr lt 11.0)"
    "(dered_jmag_kmag gt 2.0) and (aep_000_stetsonj gt 10.0)"
    "(gaia_status ct 'ok') and (propermotion gt 300.0)"
    "(simbad_best_objtype ct 'RR') and (dered_sdssu_sdssg lt 0.5)"

    download_data sets if the accompanying data from the search results will be
    downloaded automatically. This includes the data table CSV, the dataset
    pickle file, and a light curve ZIP file. Note that if the search service
    indicates that your query is still in progress, this function will block
    until the light curve ZIP file becomes available. The maximum wait time in
    seconds is set by maxtimeout and the refresh interval is set by refresh.

    To avoid the wait block, set download_data to False and the function will
    write a pickle file to ~/.astrobase/lccs/query-[setid].pkl containing all
    the information necessary to retrieve these data files later when the query
    is done. To do so, call the retrieve_dataset_files with the path to this
    pickle file (it will be returned).

    outdir if not None, sets the output directory of the downloaded dataset
    files. If None, they will be downloaded to the current directory.

    result_ispublic sets if you want your dataset to be publicly visible on the
    Recent Datasets tab and /datasets page of the LCC server you're talking
    to. If False, only people who know the unique dataset URL can view and fetch
    data files from it later.

    '''

    with open(file_to_upload) as infd:
        xmq = infd.read()

    # check the number of lines in the input
    xmqlines = len(xmq.split('\n')[:-1])

    if xmqlines > 5000:

        LOGERROR('you have more than 5000 lines in the file to upload: %s' %
                 file_to_upload)
        return None, None, None

    # turn the input into a param dict
    params = {'xmq':xmq,
              'xmd':xmatch_dist_arcsec}

    if collections:
        params['collections'] = collections
    if columns:
        params['columns'] = columns
    if filters:
        params['filters'] = filters

    params['result_ispublic'] = 1 if result_ispublic else 0

    # hit the server
    api_url = '%s/api/xmatch' % lcc_server

    # we need an API key for xmatch

    # check if we have one already
    have_apikey, apikey, expires = check_existing_apikey(lcc_server)

    # if not, get a new one
    if not have_apikey:
        apikey, expires = get_new_apikey(lcc_server)

    # no API key is required for now, but we'll load one automatically if we
    # require it in the future
    searchresult = submit_post_searchquery(api_url, params, apikey)

    # check the status of the search
    status = searchresult[0]

    # now we'll check if we want to download the data
    if download_data:

        if status == 'ok':

            LOGINFO('query complete, downloading associated data...')
            csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                     outdir=outdir)

            if pkl:
                return searchresult[1], csv, lczip, pkl
            else:
                return searchresult[1], csv, lczip

        elif status == 'background':

            LOGINFO('query is not yet complete, '
                    'waiting up to %.1f minutes, '
                    'updates every %s seconds (hit Ctrl+C to cancel)...' %
                    (maxtimeout/60.0, refresh))

            timewaited = 0.0

            while timewaited < maxtimeout:

                try:

                    time.sleep(refresh)
                    csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                             outdir=outdir)

                    if (csv and os.path.exists(csv) and
                        lczip and os.path.exists(lczip)):

                        LOGINFO('all dataset products collected')
                        return searchresult[1], csv, lczip

                    timewaited = timewaited + refresh

                except KeyboardInterrupt:

                    LOGWARNING('abandoned wait for downloading data')
                    return searchresult[1], None, None

            LOGERROR('wait timed out.')
            return searchresult[1], None, None

        else:

            LOGERROR('could not download the data for this query result')
            return searchresult[1], None, None

    else:

        return searchresult[1], None, None



#######################################
## DATASET AND OBJECT INFO FUNCTIONS ##
#######################################

def get_dataset(lcc_server, dataset_id, strformat=False):
    '''This downloads a JSON form of the dataset from the specified lcc_server.

    This returns a dict from the parsed JSON. The interesting keys in the dict
    to look at are: 'coldesc' for the column descriptions and 'rows' for the
    actual data rows. Note that if there are more than 3000 objects in the
    dataset, the JSON will only contain the top 3000 objects. In this case, it's
    better to use retrieve_dataset_files to get the dataset CSV, which contains
    the full data table.

    lcc_server is the base URL of the LCC server to talk to.

    dataset_id is the unique setid of the dataset you want to get. In the
    results from the *_search functions above, this is the value of the
    infodict['result']['setid'] key in the first item (the infodict) in the
    returned tuple.

    strformat sets if you want the returned data rows to be formatted in their
    string representations already. This can be useful if you're piping the
    returned JSON straight into some sort of UI and you don't want to deal with
    formatting floats, etc. To do this manually when strformat is set to False,
    look at the 'coldesc' item in the returned dict, which gives the Python and
    Numpy string format specifiers for each column in the data table.

    The JSON contains metadata about the query that produced the dataset,
    information about the data table's columns, and links to download the
    dataset's products including the light curve ZIP and the dataset CSV.

    '''

    dataset_url = '%s/set/%s?json=1' % (lcc_server, dataset_id)

    if strformat:
        dataset_url = '%s?strformat=1' % dataset_url

    LOGINFO('retrieving dataset %s from %s, using URL: %s ...' % (lcc_server,
                                                                  dataset_id,
                                                                  dataset_url))

    try:

        resp = urlopen(dataset_url)
        dataset = json.loads(resp.read())

        return dataset

    except Exception as e:

        LOGEXCEPTION('could not retrieve the dataset JSON!')
        return None



def object_info(lcc_server, objectid, db_collection_id):
    '''This gets information on a single object from the LCC server.

    Returns a dict with all of the available information on an object, including
    finding charts, comments, object type and variability tags, and
    period-search results (if available).

    lcc_server is the base URL of the LCC server to talk to.

    objectid is the unique database ID of the object to retrieve info for. This
    is always returned as the `db_oid` column in LCC server search results.

    db_collection_id is the collection ID which will be searched for the
    object. This is always returned as the `collection` column in LCC server
    search results.

    NOTE: you can pass the result dict returned by this function directly into
    the astrobase.checkplot function:

    astrobase.checkplot.checkplot_pickle_to_png(result_dict, 'object-info.png')

    to generate a quick PNG overview of the object information.


    Some important items in the result dict returned from this function:

    `objectinfo`: all object magnitude, color, GAIA cross-match, and object type
                  information available for this object

    `objectcomments`: comments on the object's variability if available

    `varinfo`: variability comments, variability features, type tags, period and
               epoch information if available

    `neighbors`: information on the neighboring objects of this object in its
                 parent light curve collection

    `xmatch`: information on any cross-matches to external catalogs (e.g. KIC,
              EPIC, TIC, APOGEE, etc.)

    `finderchart`: a base-64 encoded PNG image of the object's DSS2 RED finder
                   chart. To convert this to an actual PNG, try the function
                   astrobase.checkplot._b64_to_file.

    `magseries`: a base-64 encoded PNG image of the object's light curve. To
                 convert this to an actual PNG, try the function
                 astrobase.checkplot._b64_to_file.

    `pfmethods`: a list of period-finding methods applied to the object if
                 any. If this list is present, use the keys in it to get to the
                 actual period-finding results for each method. These will
                 contain base-64 encoded PNGs of the periodogram and phased
                 light curves using the best three peaks in the periodogram, as
                 well as period and epoch information.

    '''

    url = '%s/api/object?objectid=%s&collection=%s' % (lcc_server,
                                                       objectid,
                                                       db_collection_id)

    try:

        LOGINFO(
            'getting info for %s in collection %s from %s' % (
                objectid,
                db_collection_id,
                lcc_server
            )
        )
        resp = urlopen(url)
        objectinfo = json.loads(resp.read())['result']
        return objectinfo

    except HTTPError as e:

        if e.code == 404:

            LOGERROR(
                'additional info for object %s not '
                'found in collection: %s' % (objectid,
                                             db_collection_id)
            )

        else:

            LOGERROR('could not retrieve object info, '
                     'URL used: %s, error code: %s, reason: %s' %
                     (url, e.code, e.reason))


        return None



def list_recent_datasets(lcc_server, nrecent=25):
    '''This lists recent publicly visible datasets available on the LCC server.

    Returns a list of dicts, with each dict containing info on each dataset.

    lcc_server is the base URL of the LCC server to talk to.

    nrecent indicates how many recent public datasets you want to list. This is
    always capped at 1000.

    '''

    url = '%s/api/datasets?nsets=%s' % (lcc_server, nrecent)

    try:

        LOGINFO(
            'getting list of recent publicly visible datasets from %s' % (
                lcc_server,
            )
        )
        resp = urlopen(url)
        objectinfo = json.loads(resp.read())['result']
        return objectinfo

    except HTTPError as e:

        LOGERROR('could not retrieve recent datasets list, '
                 'URL used: %s, error code: %s, reason: %s' %
                 (url, e.code, e.reason))

        return None



def list_lc_collections(lcc_server):
    '''This lists all light curve collections made available on the LCC server.

    Returns a dict containing lists of info items per collection. This includes
    collection_ids, lists of columns, lists of indexed columns, lists of
    full-text indexed columns, detailed column descriptions, number of objects
    in each collection, collection sky coverage, etc.

    '''

    url = '%s/api/collections' % lcc_server

    try:

        LOGINFO(
            'getting list of recent publicly visible datasets from %s' % (
                lcc_server,
            )
        )
        resp = urlopen(url)
        objectinfo = json.loads(resp.read())['result']['collections']
        return objectinfo

    except HTTPError as e:

        LOGERROR('could not retrieve list of collections, '
                 'URL used: %s, error code: %s, reason: %s' %
                 (url, e.code, e.reason))

        return None



###################################
## SUPPORT FOR EXECUTION AS MAIN ##
###################################

def main():

    '''
    This supports execution of the module as a script.

    TODO: finish this.

    '''

    # handle SIGPIPE sent by less, head, et al.
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    import argparse

    aparser = argparse.ArgumentParser(
        description='interact with an LCC server on the command-line'
    )

    aparser.add_argument('server',
                         action='store',
                         type='str',
                         help=("the base URL of the LCC server "
                               "you want to talk to"))

    aparser.add_argument('action',
                         choices=['conesearch',
                                  'columnsearch',
                                  'ftsearch',
                                  'xmatch',
                                  'datasets',
                                  'collections',
                                  'objectinfo',
                                  'getdata'],
                         action='store',
                         type='str',
                         help=("the action you want to perform"))
