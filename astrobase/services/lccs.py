#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# lccs.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Aug 2018
# License: MIT - see LICENSE for the full text.

'''This contains functions to search for objects and get light curves from a
Light Curve Collection server (https://github.com/waqasbhatti/lcc-server) using
its HTTP API.

The LCC-Server requires an API key to access most services. The service
functions in this module will automatically acquire an anonymous user API key on
first use (and upon API key expiry afterwards). If you sign up for an LCC-Server
user account, you can import the API key generated for that account on the user
home page. To do this, use the import_apikey function in this module.

This currently supports the following LCC-Server services::

    conesearch   : cone_search(lcc_server_url, center_ra, center_decl, ...)
    ftsquery     : fulltext_search(lcc_server_url, searchtxt, sesame=False, ...)
    columnsearch : column_search(lcc_server_url, filters, ...)
    xmatch       : xmatch_search(lcc_server_url, file_to_upload, ...

The functions above will download the data products (data table CSVs, light
curve ZIP files) of the search results automatically, or in case the query takes
too long, will return within a configurable timeout. The query information is
cached to `~/.astrobase/lccs`, and can be used to download data products for
long-running queries later.

The functions below support various auxiliary LCC services::

    get-dataset  : get_dataset(lcc_server_url, dataset_id)
    objectinfo   : object_info(lcc_server_url, objectid, collection, ...)
    dataset-list : list_recent_datasets(lcc_server_url, nrecent=25, ...)
    collections  : list_lc_collections(lcc_server_url)

'''

# put this in here because lccs can be used as a standalone module
__version__ = '0.5.0'


#############
## LOGGING ##
#############

import logging

# the basic logging styles common to all astrobase modules
log_sub = '{'
log_fmt = '[{levelname:1.1} {asctime} {module}:{lineno}] {message}'
log_date_fmt = '%y%m%d %H:%M:%S'

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


# get the correct datetime bits
try:
    from datetime import datetime, timezone
    utc = timezone.utc
except Exception:
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
except Exception:
    import pickle


# import url methods here.  we use built-ins because we want this module to be
# usable as a single file. otherwise, we'd use something sane like Requests.
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse


####################
## API KEY CONFIG ##
####################

def check_existing_apikey(lcc_server):
    '''This validates if an API key for the specified LCC-Server is available.

    API keys are stored using the following file scheme::

        ~/.astrobase/lccs/apikey-domain.of.lccserver.org

    e.g. for the HAT LCC-Server at https://data.hatsurveys.org::

        ~/.astrobase/lccs/apikey-https-data.hatsurveys.org

    Parameters
    ----------

    lcc_server : str
        The base URL of the LCC-Server for which the existence of API keys will
        be checked.

    Returns
    -------

    (apikey_ok, apikey_str, expiry) : tuple
        The returned tuple contains the status of the API key, the API key
        itself if present, and its expiry date if present.

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
        LOGWARNING('No LCC-Server API key '
                   'found in: {apikeyfile}'.format(apikeyfile=APIKEYFILE))

        return False, None, None


def get_new_apikey(lcc_server):
    '''This gets a new API key from the specified LCC-Server.

    NOTE: this only gets an anonymous API key. To get an API key tied to a user
    account (and associated privilege level), see the `import_apikey` function
    below.

    Parameters
    ----------

    lcc_server : str
        The base URL of the LCC-Server from where the API key will be fetched.

    Returns
    -------

    (apikey, expiry) : tuple
        This returns a tuple with the API key and its expiry date.

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

        LOGERROR('could not fetch the API key from LCC-Server at: %s' %
                 lcc_server)
        LOGERROR('the HTTP status code was: %s' % resp.status_code)
        return None

    #
    # now that we have an API key dict, get the API key out of it and write it
    # to the APIKEYFILE
    #
    apikey = respdict['result']['apikey']
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


def import_apikey(lcc_server, apikey_json):
    '''This imports an API key from text and writes it to the cache dir.

    Use this with the JSON file downloaded from API key download link on your
    LCC-Server user home page. The API key will thus be tied to the privileges
    of that user account and can then access objects, datasets, and collections
    marked as private for the user only or shared with that user.

    Parameters
    ----------

    lcc_server : str
        The base URL of the LCC-Server to get the API key for.

    apikey_text_json : str
        The JSON string from the API key text box on the user's LCC-Server home
        page at `lcc_server/users/home`.

    Returns
    -------

    (apikey, expiry) : tuple
        This returns a tuple with the API key and its expiry date.

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

    # get the JSON
    with open(apikey_json,'r') as infd:
        respdict = json.load(infd)

    #
    # now that we have an API key dict, get the API key out of it and write it
    # to the APIKEYFILE
    #
    apikey = respdict['apikey']
    expires = respdict['expires']

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


##############################
## QUERY HANDLING FUNCTIONS ##
##############################

def submit_post_searchquery(url, data, apikey):
    '''This submits a POST query to an LCC-Server search API endpoint.

    Handles streaming of the results, and returns the final JSON stream. Also
    handles results that time out.

    Parameters
    ----------

    url : str
        The URL of the search API endpoint to hit. This is something like
        `https://data.hatsurveys.org/api/conesearch`

    data : dict
        A dict of the search query parameters to pass to the search service.

    apikey : str
        The API key to use to access the search service. API keys are required
        for all POST request made to an LCC-Server's API endpoints.

    Returns
    -------

    (status_flag, data_dict, dataset_id) : tuple
        This returns a tuple containing the status of the request: ('complete',
        'failed', 'background', etc.), a dict parsed from the JSON result of the
        request, and a dataset ID, which can be used to reconstruct the URL on
        the LCC-Server where the results can be browsed.

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

    LOGINFO('submitting search query to LCC-Server API URL: %s' % url)

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

            except Exception:

                LOGEXCEPTION('failed to submit query to %s' % url)
                return 'failed', None, None

    except HTTPError as e:

        LOGERROR('could not submit query to LCC API at: %s' % url)
        LOGERROR('HTTP status code was %s, reason: %s' % (e.code, e.reason))
        return 'failed', None, None


def retrieve_dataset_files(searchresult,
                           getpickle=False,
                           outdir=None,
                           apikey=None):
    '''This retrieves a search result dataset's CSV and any LC zip files.

    Takes the output from the `submit_post_searchquery` function above or a
    pickle file generated from that function's output if the query timed out.

    Parameters
    ----------

    searchresult : str or tuple
        If provided as a str, points to the pickle file created using the output
        from the `submit_post_searchquery` function. If provided as a tuple,
        this is the result tuple from the `submit_post_searchquery` function.

    getpickle : False
        If this is True, will also download the dataset's pickle. Note that
        LCC-Server is a Python 3.6+ package (while lccs.py still works with
        Python 2.7) and it saves its pickles in pickle.HIGHEST_PROTOCOL for
        efficiency, so these pickles may be unreadable in lower Pythons. As an
        alternative, the dataset CSV contains the full data table and all the
        information about the dataset in its header, which is JSON
        parseable. You can also use the function `get_dataset` below to get the
        dataset pickle information in JSON form.

    outdir : None or str
        If this is a str, points to the output directory where the results will
        be placed. If it's None, they will be placed in the current directory.

    apikey : str or None
        If this is a str, uses the given API key to authenticate the download
        request. This is useful when you have a private dataset you want to get
        products for.

    Returns
    -------

    (local_dataset_csv, local_dataset_lczip, local_dataset_pickle) : tuple
        This returns a tuple containing paths to the dataset CSV, LC zipfile,
        and the dataset pickle if getpickle was set to True (None otherwise).

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
                 'we need a searchresult from the '
                 'lccs.submit_post_searchquery function or '
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

                # if apikey is not None, add it in as an Authorization: Bearer
                # [apikey] header
                if apikey:
                    headers = {'Authorization':'Bearer: %s' % apikey}
                else:
                    headers = {}

                req = Request(
                    dataset_pickle_link,
                    data=None,
                    headers=headers
                )
                resp = urlopen(req)

                # save the file
                LOGINFO('saving %s' % dataset_pickle)
                localf = os.path.join(localdir, dataset_pickle)
                with open(localf, 'wb') as outfd:
                    with resp:
                        data = resp.read()
                        outfd.write(data)

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

            # if apikey is not None, add it in as an Authorization: Bearer
            # [apikey] header
            if apikey:
                headers = {'Authorization':'Bearer: %s' % apikey}
            else:
                headers = {}

            req = Request(
                dataset_csv_link,
                data=None,
                headers=headers
            )
            resp = urlopen(req)

            # save the file
            LOGINFO('saving %s' % dataset_csv)
            localf = os.path.join(localdir, dataset_csv)
            with open(localf, 'wb') as outfd:
                with resp:
                    data = resp.read()
                    outfd.write(data)

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

            # if apikey is not None, add it in as an Authorization: Bearer
            # [apikey] header
            if apikey:
                headers = {'Authorization':'Bearer: %s' % apikey}
            else:
                headers = {}

            req = Request(
                dataset_lczip_link,
                data=None,
                headers=headers
            )
            resp = urlopen(req)

            # save the file
            LOGINFO('saving %s' % dataset_lczip)
            localf = os.path.join(localdir, dataset_lczip)
            with open(localf, 'wb') as outfd:
                with resp:
                    data = resp.read()
                    outfd.write(data)

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
                result_visibility='unlisted',
                email_when_done=False,
                collections=None,
                columns=None,
                filters=None,
                sortspec=None,
                samplespec=None,
                limitspec=None,
                download_data=True,
                outdir=None,
                maxtimeout=300.0,
                refresh=15.0):

    '''This runs a cone-search query.

    Parameters
    ----------

    lcc_server : str
        This is the base URL of the LCC-Server to talk to.  (e.g. for HAT, use:
        https://data.hatsurveys.org)

    center_ra,center_decl : float
        These are the central coordinates of the search to conduct. These can be
        either decimal degrees of type float, or sexagesimal coordinates of type
        str:

        - OK: 290.0, 45.0
        - OK: 15:00:00 +45:00:00
        - OK: 15 00 00.0 -45 00 00.0
        - NOT OK: 290.0 +45:00:00
        - NOT OK: 15:00:00 45.0

    radiusarcmin : float
        This is the search radius to use for the cone-search. This is in
        arcminutes. The maximum radius you can use is 60 arcminutes = 1 degree.

    result_visibility : {'private', 'unlisted', 'public'}
        This sets the visibility of the dataset produced from the search
        result::

               'private' -> the dataset and its products are not visible or
                            accessible by any user other than the one that
                            created the dataset.

               'unlisted' -> the dataset and its products are not visible in the
                             list of public datasets, but can be accessed if the
                             dataset URL is known

               'public' -> the dataset and its products are visible in the list
                           of public datasets and can be accessed by anyone.

    email_when_done : bool
        If True, the LCC-Server will email you when the search is complete. This
        will also set `download_data` to False. Using this requires an
        LCC-Server account and an API key tied to that account.

    collections : list of str or None
        This is a list of LC collections to search in. If this is None, all
        collections will be searched.

    columns : list of str or None
        This is a list of columns to return in the results. Matching objects'
        object IDs, RAs, DECs, and links to light curve files will always be
        returned so there is no need to specify these columns. If None, only
        these columns will be returned: 'objectid', 'ra', 'decl', 'lcfname'

    filters : str or None
        This is an SQL-like string to use to filter on database columns in the
        LCC-Server's collections. To see the columns available for a search,
        visit the Collections tab in the LCC-Server's browser UI. The filter
        operators allowed are::

            lt      -> less than
            gt      -> greater than
            ge      -> greater than or equal to
            le      -> less than or equal to
            eq      -> equal to
            ne      -> not equal to
            ct      -> contains text
            isnull  -> column value is null
            notnull -> column value is not null

        You may use the `and` and `or` operators between filter specifications
        to chain them together logically.

        Example filter strings::

            "(propermotion gt 200.0) and (sdssr lt 11.0)"
            "(dered_jmag_kmag gt 2.0) and (aep_000_stetsonj gt 10.0)"
            "(gaia_status ct 'ok') and (propermotion gt 300.0)"
            "(simbad_best_objtype ct 'RR') and (dered_sdssu_sdssg lt 0.5)"

    sortspec : tuple of two strs or None
        If not None, this should be a tuple of two items::

            ('column to sort by', 'asc|desc')

        This sets the column to sort the results by. For cone_search, the
        default column and sort order are 'dist_arcsec' and 'asc', meaning the
        distance from the search center in ascending order.

    samplespec : int or None
        If this is an int, will indicate how many rows from the initial search
        result will be uniformly random sampled and returned.

    limitspec : int or None
        If this is an int, will indicate how many rows from the initial search
        result to return in total.

        `sortspec`, `samplespec`, and `limitspec` are applied in this order:

            sample -> sort -> limit

    download_data : bool
        This sets if the accompanying data from the search results will be
        downloaded automatically. This includes the data table CSV, the dataset
        pickle file, and a light curve ZIP file. Note that if the search service
        indicates that your query is still in progress, this function will block
        until the light curve ZIP file becomes available. The maximum wait time
        in seconds is set by maxtimeout and the refresh interval is set by
        refresh.

        To avoid the wait block, set download_data to False and the function
        will write a pickle file to `~/.astrobase/lccs/query-[setid].pkl`
        containing all the information necessary to retrieve these data files
        later when the query is done. To do so, call the
        `retrieve_dataset_files` with the path to this pickle file (it will be
        returned).

    outdir : str or None
        If this is provided, sets the output directory of the downloaded dataset
        files. If None, they will be downloaded to the current directory.

    maxtimeout : float
        The maximum time in seconds to wait for the LCC-Server to respond with a
        result before timing out. You can use the `retrieve_dataset_files`
        function to get results later as needed.

    refresh : float
        The time to wait in seconds before pinging the LCC-Server to see if a
        search query has completed and dataset result files can be downloaded.

    Returns
    -------

    tuple
        Returns a tuple with the following elements::

            (search result status dict,
             search result CSV file path,
             search result LC ZIP path)

    '''

    # turn the input into a param dict

    coords = '%.5f %.5f %.1f' % (center_ra, center_decl, radiusarcmin)
    params = {
        'coords':coords
    }

    if collections:
        params['collections'] = collections
    if columns:
        params['columns'] = columns
    if filters:
        params['filters'] = filters
    if sortspec:
        params['sortspec'] = json.dumps([sortspec])
    if samplespec:
        params['samplespec'] = int(samplespec)
    if limitspec:
        params['limitspec'] = int(limitspec)

    params['visibility'] = result_visibility
    params['emailwhendone'] = email_when_done

    # we won't wait for the LC ZIP to complete if email_when_done = True
    if email_when_done:
        download_data = False

    # check if we have an API key already
    have_apikey, apikey, expires = check_existing_apikey(lcc_server)

    # if not, get a new one
    if not have_apikey:
        apikey, expires = get_new_apikey(lcc_server)

    # hit the server
    api_url = '%s/api/conesearch' % lcc_server

    searchresult = submit_post_searchquery(api_url, params, apikey)

    # check the status of the search
    status = searchresult[0]

    # now we'll check if we want to download the data
    if download_data:

        if status == 'ok':

            LOGINFO('query complete, downloading associated data...')
            csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                     outdir=outdir,
                                                     apikey=apikey)

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
                                                             outdir=outdir,
                                                             apikey=apikey)

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
                    sesame_lookup=False,
                    result_visibility='unlisted',
                    email_when_done=False,
                    collections=None,
                    columns=None,
                    filters=None,
                    sortspec=None,
                    samplespec=None,
                    limitspec=None,
                    download_data=True,
                    outdir=None,
                    maxtimeout=300.0,
                    refresh=15.0):

    '''This runs a full-text search query.

    Parameters
    ----------

    lcc_server : str
        This is the base URL of the LCC-Server to talk to.  (e.g. for HAT, use:
        https://data.hatsurveys.org)

    searchterm : str
        This is the term to look for in a full-text search of the LCC-Server's
        collections. This can be an object name, tag, description, etc., as
        noted in the LCC-Server's full-text search tab in its browser UI. To
        search for an exact match to a string (like an object name), you can add
        double quotes around the string, e.g. searchitem = '"exact match to me
        needed"'.

    sesame_lookup : bool
        If True, means the LCC-Server will assume the provided search term is a
        single object's name, look up its coordinates using the CDS SIMBAD
        SESAME name resolution service, and then search the LCC-Server for any
        matching objects. The object name can be either a star name known to
        SIMBAD, or it can be an extended source name (e.g. an open cluster or
        nebula). In the first case, a search radius of 5 arcseconds will be
        used. In the second case, a search radius of 1 degree will be used to
        find all nearby database objects associated with an extended source
        name.

    result_visibility : {'private', 'unlisted', 'public'}
        This sets the visibility of the dataset produced from the search
        result::

               'private' -> the dataset and its products are not visible or
                            accessible by any user other than the one that
                            created the dataset.

               'unlisted' -> the dataset and its products are not visible in the
                             list of public datasets, but can be accessed if the
                             dataset URL is known

               'public' -> the dataset and its products are visible in the list
                           of public datasets and can be accessed by anyone.

    email_when_done : bool
        If True, the LCC-Server will email you when the search is complete. This
        will also set `download_data` to False. Using this requires an
        LCC-Server account and an API key tied to that account.

    collections : list of str or None
        This is a list of LC collections to search in. If this is None, all
        collections will be searched.

    columns : list of str or None
        This is a list of columns to return in the results. Matching objects'
        object IDs, RAs, DECs, and links to light curve files will always be
        returned so there is no need to specify these columns. If None, only
        these columns will be returned: 'objectid', 'ra', 'decl', 'lcfname'

    filters : str or None
        This is an SQL-like string to use to filter on database columns in the
        LCC-Server's collections. To see the columns available for a search,
        visit the Collections tab in the LCC-Server's browser UI. The filter
        operators allowed are::

            lt      -> less than
            gt      -> greater than
            ge      -> greater than or equal to
            le      -> less than or equal to
            eq      -> equal to
            ne      -> not equal to
            ct      -> contains text
            isnull  -> column value is null
            notnull -> column value is not null

        You may use the `and` and `or` operators between filter specifications
        to chain them together logically.

        Example filter strings::

            "(propermotion gt 200.0) and (sdssr lt 11.0)"
            "(dered_jmag_kmag gt 2.0) and (aep_000_stetsonj gt 10.0)"
            "(gaia_status ct 'ok') and (propermotion gt 300.0)"
            "(simbad_best_objtype ct 'RR') and (dered_sdssu_sdssg lt 0.5)"

    sortspec : tuple of two strs or None
        If not None, this should be a tuple of two items::

            ('column to sort by', 'asc|desc')

        This sets the column to sort the results by. For cone_search, the
        default column and sort order are 'dist_arcsec' and 'asc', meaning the
        distance from the search center in ascending order.

    samplespec : int or None
        If this is an int, will indicate how many rows from the initial search
        result will be uniformly random sampled and returned.

    limitspec : int or None
        If this is an int, will indicate how many rows from the initial search
        result to return in total.

        `sortspec`, `samplespec`, and `limitspec` are applied in this order:

            sample -> sort -> limit

    download_data : bool
        This sets if the accompanying data from the search results will be
        downloaded automatically. This includes the data table CSV, the dataset
        pickle file, and a light curve ZIP file. Note that if the search service
        indicates that your query is still in progress, this function will block
        until the light curve ZIP file becomes available. The maximum wait time
        in seconds is set by maxtimeout and the refresh interval is set by
        refresh.

        To avoid the wait block, set download_data to False and the function
        will write a pickle file to `~/.astrobase/lccs/query-[setid].pkl`
        containing all the information necessary to retrieve these data files
        later when the query is done. To do so, call the
        `retrieve_dataset_files` with the path to this pickle file (it will be
        returned).

    outdir : str or None
        If this is provided, sets the output directory of the downloaded dataset
        files. If None, they will be downloaded to the current directory.

    maxtimeout : float
        The maximum time in seconds to wait for the LCC-Server to respond with a
        result before timing out. You can use the `retrieve_dataset_files`
        function to get results later as needed.

    refresh : float
        The time to wait in seconds before pinging the LCC-Server to see if a
        search query has completed and dataset result files can be downloaded.

    Returns
    -------

    tuple
        Returns a tuple with the following elements::

            (search result status dict,
             search result CSV file path,
             search result LC ZIP path)

    '''

    # turn the input into a param dict
    params = {'ftstext':searchterm}

    if collections:
        params['collections'] = collections
    if columns:
        params['columns'] = columns
    if filters:
        params['filters'] = filters
    if sortspec:
        params['sortspec'] = json.dumps([sortspec])
    if samplespec:
        params['samplespec'] = int(samplespec)
    if limitspec:
        params['limitspec'] = int(limitspec)

    params['visibility'] = result_visibility
    params['emailwhendone'] = email_when_done
    params['sesame'] = sesame_lookup

    # we won't wait for the LC ZIP to complete if email_when_done = True
    if email_when_done:
        download_data = False

    # check if we have an API key already
    have_apikey, apikey, expires = check_existing_apikey(lcc_server)

    # if not, get a new one
    if not have_apikey:
        apikey, expires = get_new_apikey(lcc_server)

    # hit the server
    api_url = '%s/api/ftsquery' % lcc_server

    searchresult = submit_post_searchquery(api_url, params, apikey)

    # check the status of the search
    status = searchresult[0]

    # now we'll check if we want to download the data
    if download_data:

        if status == 'ok':

            LOGINFO('query complete, downloading associated data...')
            csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                     outdir=outdir,
                                                     apikey=apikey)

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
                                                             outdir=outdir,
                                                             apikey=apikey)

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
                  result_visibility='unlisted',
                  email_when_done=False,
                  collections=None,
                  columns=None,
                  sortspec=('sdssr','asc'),
                  samplespec=None,
                  limitspec=None,
                  download_data=True,
                  outdir=None,
                  maxtimeout=300.0,
                  refresh=15.0):

    '''This runs a column search query.

    Parameters
    ----------

    lcc_server : str
        This is the base URL of the LCC-Server to talk to.  (e.g. for HAT, use:
        https://data.hatsurveys.org)

    filters : str or None
        This is an SQL-like string to use to filter on database columns in the
        LCC-Server's collections. To see the columns available for a search,
        visit the Collections tab in the LCC-Server's browser UI. The filter
        operators allowed are::

            lt      -> less than
            gt      -> greater than
            ge      -> greater than or equal to
            le      -> less than or equal to
            eq      -> equal to
            ne      -> not equal to
            ct      -> contains text
            isnull  -> column value is null
            notnull -> column value is not null

        You may use the `and` and `or` operators between filter specifications
        to chain them together logically.

        Example filter strings::

            "(propermotion gt 200.0) and (sdssr lt 11.0)"
            "(dered_jmag_kmag gt 2.0) and (aep_000_stetsonj gt 10.0)"
            "(gaia_status ct 'ok') and (propermotion gt 300.0)"
            "(simbad_best_objtype ct 'RR') and (dered_sdssu_sdssg lt 0.5)"

    result_visibility : {'private', 'unlisted', 'public'}
        This sets the visibility of the dataset produced from the search
        result::

               'private' -> the dataset and its products are not visible or
                            accessible by any user other than the one that
                            created the dataset.

               'unlisted' -> the dataset and its products are not visible in the
                             list of public datasets, but can be accessed if the
                             dataset URL is known

               'public' -> the dataset and its products are visible in the list
                           of public datasets and can be accessed by anyone.

    email_when_done : bool
        If True, the LCC-Server will email you when the search is complete. This
        will also set `download_data` to False. Using this requires an
        LCC-Server account and an API key tied to that account.

    collections : list of str or None
        This is a list of LC collections to search in. If this is None, all
        collections will be searched.

    columns : list of str or None
        This is a list of columns to return in the results. Matching objects'
        object IDs, RAs, DECs, and links to light curve files will always be
        returned so there is no need to specify these columns. If None, only
        these columns will be returned: 'objectid', 'ra', 'decl', 'lcfname'

    sortspec : tuple of two strs or None
        If not None, this should be a tuple of two items::

            ('column to sort by', 'asc|desc')

        This sets the column to sort the results by. For cone_search, the
        default column and sort order are 'dist_arcsec' and 'asc', meaning the
        distance from the search center in ascending order.

    samplespec : int or None
        If this is an int, will indicate how many rows from the initial search
        result will be uniformly random sampled and returned.

    limitspec : int or None
        If this is an int, will indicate how many rows from the initial search
        result to return in total.

        `sortspec`, `samplespec`, and `limitspec` are applied in this order:

            sample -> sort -> limit

    download_data : bool
        This sets if the accompanying data from the search results will be
        downloaded automatically. This includes the data table CSV, the dataset
        pickle file, and a light curve ZIP file. Note that if the search service
        indicates that your query is still in progress, this function will block
        until the light curve ZIP file becomes available. The maximum wait time
        in seconds is set by maxtimeout and the refresh interval is set by
        refresh.

        To avoid the wait block, set download_data to False and the function
        will write a pickle file to `~/.astrobase/lccs/query-[setid].pkl`
        containing all the information necessary to retrieve these data files
        later when the query is done. To do so, call the
        `retrieve_dataset_files` with the path to this pickle file (it will be
        returned).

    outdir : str or None
        If this is provided, sets the output directory of the downloaded dataset
        files. If None, they will be downloaded to the current directory.

    maxtimeout : float
        The maximum time in seconds to wait for the LCC-Server to respond with a
        result before timing out. You can use the `retrieve_dataset_files`
        function to get results later as needed.

    refresh : float
        The time to wait in seconds before pinging the LCC-Server to see if a
        search query has completed and dataset result files can be downloaded.

    Returns
    -------

    tuple
        Returns a tuple with the following elements::

            (search result status dict,
             search result CSV file path,
             search result LC ZIP path)

    '''

    # turn the input into a param dict
    params = {
        'filters':filters
    }

    if collections:
        params['collections'] = collections
    if columns:
        params['columns'] = columns
    if sortspec:
        params['sortspec'] = json.dumps([sortspec])
    if samplespec:
        params['samplespec'] = int(samplespec)
    if limitspec:
        params['limitspec'] = int(limitspec)

    params['visibility'] = result_visibility
    params['emailwhendone'] = email_when_done

    # we won't wait for the LC ZIP to complete if email_when_done = True
    if email_when_done:
        download_data = False

    # check if we have an API key already
    have_apikey, apikey, expires = check_existing_apikey(lcc_server)

    # if not, get a new one
    if not have_apikey:
        apikey, expires = get_new_apikey(lcc_server)

    # hit the server
    api_url = '%s/api/columnsearch' % lcc_server

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
                                                     outdir=outdir,
                                                     apikey=apikey)

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
                                                             outdir=outdir,
                                                             apikey=apikey)

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
                  result_visibility='unlisted',
                  email_when_done=False,
                  collections=None,
                  columns=None,
                  filters=None,
                  sortspec=None,
                  limitspec=None,
                  samplespec=None,
                  download_data=True,
                  outdir=None,
                  maxtimeout=300.0,
                  refresh=15.0):

    '''This runs a cross-match search query.

    Parameters
    ----------

    lcc_server : str
        This is the base URL of the LCC-Server to talk to.  (e.g. for HAT, use:
        https://data.hatsurveys.org)

    file_to_upload : str
        This is the path to a text file containing objectid, RA, declination
        rows for the objects to cross-match against the LCC-Server
        collections. This should follow the format of the following example::

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

    xmatch_dist_arcsec : float
        This is the maximum distance in arcseconds to consider when
        cross-matching objects in the uploaded file to the LCC-Server's
        collections. The maximum allowed distance is 30 arcseconds. Multiple
        matches to an uploaded object are possible and will be returned in order
        of increasing distance grouped by input `objectid`.

    result_visibility : {'private', 'unlisted', 'public'}
        This sets the visibility of the dataset produced from the search
        result::

               'private' -> the dataset and its products are not visible or
                            accessible by any user other than the one that
                            created the dataset.

               'unlisted' -> the dataset and its products are not visible in the
                             list of public datasets, but can be accessed if the
                             dataset URL is known

               'public' -> the dataset and its products are visible in the list
                           of public datasets and can be accessed by anyone.

    email_when_done : bool
        If True, the LCC-Server will email you when the search is complete. This
        will also set `download_data` to False. Using this requires an
        LCC-Server account and an API key tied to that account.

    collections : list of str or None
        This is a list of LC collections to search in. If this is None, all
        collections will be searched.

    columns : list of str or None
        This is a list of columns to return in the results. Matching objects'
        object IDs, RAs, DECs, and links to light curve files will always be
        returned so there is no need to specify these columns. If None, only
        these columns will be returned: 'objectid', 'ra', 'decl', 'lcfname'

    filters : str or None
        This is an SQL-like string to use to filter on database columns in the
        LCC-Server's collections. To see the columns available for a search,
        visit the Collections tab in the LCC-Server's browser UI. The filter
        operators allowed are::

            lt      -> less than
            gt      -> greater than
            ge      -> greater than or equal to
            le      -> less than or equal to
            eq      -> equal to
            ne      -> not equal to
            ct      -> contains text
            isnull  -> column value is null
            notnull -> column value is not null

        You may use the `and` and `or` operators between filter specifications
        to chain them together logically.

        Example filter strings::

            "(propermotion gt 200.0) and (sdssr lt 11.0)"
            "(dered_jmag_kmag gt 2.0) and (aep_000_stetsonj gt 10.0)"
            "(gaia_status ct 'ok') and (propermotion gt 300.0)"
            "(simbad_best_objtype ct 'RR') and (dered_sdssu_sdssg lt 0.5)"

    sortspec : tuple of two strs or None
        If not None, this should be a tuple of two items::

            ('column to sort by', 'asc|desc')

        This sets the column to sort the results by. For cone_search, the
        default column and sort order are 'dist_arcsec' and 'asc', meaning the
        distance from the search center in ascending order.

    samplespec : int or None
        If this is an int, will indicate how many rows from the initial search
        result will be uniformly random sampled and returned.

    limitspec : int or None
        If this is an int, will indicate how many rows from the initial search
        result to return in total.

        `sortspec`, `samplespec`, and `limitspec` are applied in this order:

            sample -> sort -> limit

    download_data : bool
        This sets if the accompanying data from the search results will be
        downloaded automatically. This includes the data table CSV, the dataset
        pickle file, and a light curve ZIP file. Note that if the search service
        indicates that your query is still in progress, this function will block
        until the light curve ZIP file becomes available. The maximum wait time
        in seconds is set by maxtimeout and the refresh interval is set by
        refresh.

        To avoid the wait block, set download_data to False and the function
        will write a pickle file to `~/.astrobase/lccs/query-[setid].pkl`
        containing all the information necessary to retrieve these data files
        later when the query is done. To do so, call the
        `retrieve_dataset_files` with the path to this pickle file (it will be
        returned).

    outdir : str or None
        If this is provided, sets the output directory of the downloaded dataset
        files. If None, they will be downloaded to the current directory.

    maxtimeout : float
        The maximum time in seconds to wait for the LCC-Server to respond with a
        result before timing out. You can use the `retrieve_dataset_files`
        function to get results later as needed.

    refresh : float
        The time to wait in seconds before pinging the LCC-Server to see if a
        search query has completed and dataset result files can be downloaded.

    Returns
    -------

    tuple
        Returns a tuple with the following elements::

            (search result status dict,
             search result CSV file path,
             search result LC ZIP path)

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

    if sortspec:
        params['sortspec'] = json.dumps([sortspec])
    if samplespec:
        params['samplespec'] = int(samplespec)
    if limitspec:
        params['limitspec'] = int(limitspec)

    params['visibility'] = result_visibility
    params['emailwhendone'] = email_when_done

    # we won't wait for the LC ZIP to complete if email_when_done = True
    if email_when_done:
        download_data = False

    # check if we have an API key already
    have_apikey, apikey, expires = check_existing_apikey(lcc_server)

    # if not, get a new one
    if not have_apikey:
        apikey, expires = get_new_apikey(lcc_server)

    # hit the server
    api_url = '%s/api/xmatch' % lcc_server

    searchresult = submit_post_searchquery(api_url, params, apikey)

    # check the status of the search
    status = searchresult[0]

    # now we'll check if we want to download the data
    if download_data:

        if status == 'ok':

            LOGINFO('query complete, downloading associated data...')
            csv, lczip, pkl = retrieve_dataset_files(searchresult,
                                                     outdir=outdir,
                                                     apikey=apikey)

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
                                                             outdir=outdir,
                                                             apikey=apikey)

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

def get_dataset(lcc_server,
                dataset_id,
                strformat=False,
                page=1):
    '''This downloads a JSON form of a dataset from the specified lcc_server.

    If the dataset contains more than 1000 rows, it will be paginated, so you
    must use the `page` kwarg to get the page you want. The dataset JSON will
    contain the keys 'npages', 'currpage', and 'rows_per_page' to help with
    this. The 'rows' key contains the actual data rows as a list of tuples.

    The JSON contains metadata about the query that produced the dataset,
    information about the data table's columns, and links to download the
    dataset's products including the light curve ZIP and the dataset CSV.

    Parameters
    ----------

    lcc_server : str
        This is the base URL of the LCC-Server to talk to.

    dataset_id : str
        This is the unique setid of the dataset you want to get. In the results
        from the `*_search` functions above, this is the value of the
        `infodict['result']['setid']` key in the first item (the infodict) in
        the returned tuple.

    strformat : bool
        This sets if you want the returned data rows to be formatted in their
        string representations already. This can be useful if you're piping the
        returned JSON straight into some sort of UI and you don't want to deal
        with formatting floats, etc. To do this manually when strformat is set
        to False, look at the `coldesc` item in the returned dict, which gives
        the Python and Numpy string format specifiers for each column in the
        data table.

    page : int
        This sets which page of the dataset should be retrieved.

    Returns
    -------

    dict
        This returns the dataset JSON loaded into a dict.

    '''

    urlparams = {'strformat':1 if strformat else 0,
                 'page':page,
                 'json':1}
    urlqs = urlencode(urlparams)

    dataset_url = '%s/set/%s?%s' % (lcc_server, dataset_id, urlqs)

    LOGINFO('retrieving dataset %s from %s, using URL: %s ...' % (lcc_server,
                                                                  dataset_id,
                                                                  dataset_url))

    try:

        # check if we have an API key already
        have_apikey, apikey, expires = check_existing_apikey(lcc_server)

        # if not, get a new one
        if not have_apikey:
            apikey, expires = get_new_apikey(lcc_server)

        # if apikey is not None, add it in as an Authorization: Bearer [apikey]
        # header
        if apikey:
            headers = {'Authorization':'Bearer: %s' % apikey}
        else:
            headers = {}

        # hit the server
        req = Request(dataset_url, data=None, headers=headers)
        resp = urlopen(req)
        dataset = json.loads(resp.read())
        return dataset

    except Exception:

        LOGEXCEPTION('could not retrieve the dataset JSON!')
        return None


def object_info(lcc_server, objectid, db_collection_id):
    '''This gets information on a single object from the LCC-Server.

    Returns a dict with all of the available information on an object, including
    finding charts, comments, object type and variability tags, and
    period-search results (if available).

    If you have an LCC-Server API key present in `~/.astrobase/lccs/` that is
    associated with an LCC-Server user account, objects that are visible to this
    user will be returned, even if they are not visible to the public. Use this
    to look up objects that have been marked as 'private' or 'shared'.

    NOTE: you can pass the result dict returned by this function directly into
    the `astrobase.checkplot.checkplot_pickle_to_png` function, e.g.::

        astrobase.checkplot.checkplot_pickle_to_png(result_dict,
                                                    'object-%s-info.png' %
                                                    result_dict['objectid'])

    to generate a quick PNG overview of the object information.

    Parameters
    ----------

    lcc_server : str
        This is the base URL of the LCC-Server to talk to.

    objectid : str
        This is the unique database ID of the object to retrieve info for. This
        is always returned as the `db_oid` column in LCC-Server search results.

    db_collection_id : str
        This is the collection ID which will be searched for the object. This is
        always returned as the `collection` column in LCC-Server search results.

    Returns
    -------

    dict
        A dict containing the object info is returned. Some important items in
        the result dict:

        - `objectinfo`: all object magnitude, color, GAIA cross-match, and
          object type information available for this object

        - `objectcomments`: comments on the object's variability if available

        - `varinfo`: variability comments, variability features, type tags,
          period and epoch information if available

        - `neighbors`: information on the neighboring objects of this object in
          its parent light curve collection

        - `xmatch`: information on any cross-matches to external catalogs
          (e.g. KIC, EPIC, TIC, APOGEE, etc.)

        - `finderchart`: a base-64 encoded PNG image of the object's DSS2 RED
          finder chart. To convert this to an actual PNG, try the function:
          `astrobase.checkplot.pkl_io._b64_to_file`.

        - `magseries`: a base-64 encoded PNG image of the object's light
          curve. To convert this to an actual PNG, try the function:
          `astrobase.checkplot.pkl_io._b64_to_file`.

        - `pfmethods`: a list of period-finding methods applied to the object if
          any. If this list is present, use the keys in it to get to the actual
          period-finding results for each method. These will contain base-64
          encoded PNGs of the periodogram and phased light curves using the best
          three peaks in the periodogram, as well as period and epoch
          information.

    '''

    urlparams = {
        'objectid':objectid,
        'collection':db_collection_id
    }

    urlqs = urlencode(urlparams)
    url = '%s/api/object?%s' % (lcc_server, urlqs)

    try:

        LOGINFO(
            'getting info for %s in collection %s from %s' % (
                objectid,
                db_collection_id,
                lcc_server
            )
        )

        # check if we have an API key already
        have_apikey, apikey, expires = check_existing_apikey(lcc_server)

        # if not, get a new one
        if not have_apikey:
            apikey, expires = get_new_apikey(lcc_server)

        # if apikey is not None, add it in as an Authorization: Bearer [apikey]
        # header
        if apikey:
            headers = {'Authorization':'Bearer: %s' % apikey}
        else:
            headers = {}

        # hit the server
        req = Request(url, data=None, headers=headers)
        resp = urlopen(req)
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
    '''This lists recent publicly visible datasets available on the LCC-Server.

    If you have an LCC-Server API key present in `~/.astrobase/lccs/` that is
    associated with an LCC-Server user account, datasets that belong to this
    user will be returned as well, even if they are not visible to the public.

    Parameters
    ----------

    lcc_server : str
        This is the base URL of the LCC-Server to talk to.

    nrecent : int
        This indicates how many recent public datasets you want to list. This is
        always capped at 1000.

    Returns
    -------

    list of dicts
        Returns a list of dicts, with each dict containing info on each dataset.

    '''

    urlparams = {'nsets':nrecent}
    urlqs = urlencode(urlparams)

    url = '%s/api/datasets?%s' % (lcc_server, urlqs)

    try:

        LOGINFO(
            'getting list of recent publicly '
            'visible and owned datasets from %s' % (
                lcc_server,
            )
        )

        # check if we have an API key already
        have_apikey, apikey, expires = check_existing_apikey(lcc_server)

        # if not, get a new one
        if not have_apikey:
            apikey, expires = get_new_apikey(lcc_server)

        # if apikey is not None, add it in as an Authorization: Bearer [apikey]
        # header
        if apikey:
            headers = {'Authorization':'Bearer: %s' % apikey}
        else:
            headers = {}

        # hit the server
        req = Request(url, data=None, headers=headers)
        resp = urlopen(req)
        recent_datasets = json.loads(resp.read())['result']
        return recent_datasets

    except HTTPError as e:

        LOGERROR('could not retrieve recent datasets list, '
                 'URL used: %s, error code: %s, reason: %s' %
                 (url, e.code, e.reason))

        return None


def list_lc_collections(lcc_server):
    '''This lists all light curve collections made available on the LCC-Server.

    If you have an LCC-Server API key present in `~/.astrobase/lccs/` that is
    associated with an LCC-Server user account, light curve collections visible
    to this user will be returned as well, even if they are not visible to the
    public.

    Parameters
    ----------

    lcc_server : str
        The base URL of the LCC-Server to talk to.

    Returns
    -------

    dict
        Returns a dict containing lists of info items per collection. This
        includes collection_ids, lists of columns, lists of indexed columns,
        lists of full-text indexed columns, detailed column descriptions, number
        of objects in each collection, collection sky coverage, etc.

    '''

    url = '%s/api/collections' % lcc_server

    try:

        LOGINFO(
            'getting list of recent publicly visible '
            'and owned LC collections from %s' % (
                lcc_server,
            )
        )

        # check if we have an API key already
        have_apikey, apikey, expires = check_existing_apikey(lcc_server)

        # if not, get a new one
        if not have_apikey:
            apikey, expires = get_new_apikey(lcc_server)

        # if apikey is not None, add it in as an Authorization: Bearer [apikey]
        # header
        if apikey:
            headers = {'Authorization':'Bearer: %s' % apikey}
        else:
            headers = {}

        # hit the server
        req = Request(url, data=None, headers=headers)
        resp = urlopen(req)
        lcc_list = json.loads(resp.read())['result']['collections']
        return lcc_list

    except HTTPError as e:

        LOGERROR('could not retrieve list of collections, '
                 'URL used: %s, error code: %s, reason: %s' %
                 (url, e.code, e.reason))

        return None
