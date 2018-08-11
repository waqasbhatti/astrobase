#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

'''lccs.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Aug 2018
License: MIT - see LICENSE for the full text.

This contains functions to search for objects and get light curves from a Light
Curve Collection (LCC) server (https://github.com/waqasbhatti/lcc-server) using
its HTTP API.

This currently supports the following LCC server services:

- conesearch      -> use function cone_search
- ftsquery        -> use function fulltext_search
- columnsearch    -> use function column_search
- xmatch          -> use function xmatch_search
- objectinfo      -> use function get_object_info
- datasets        -> use function get_recent_datasets
- collections     -> use function get_collections


TODO: make this actually happen...

You can use this module as a command line tool. If you installed astrobase from
pip or setup.py, you will have the `lccs` script available on your $PATH. If you
just downloaded this file as a standalone module, make it executable (using
chmod u+x or similar), then run `./lccs.py`. Use the --help flag to see all
available commands and options.

'''

# put this in here because lccs can be used as a standalone module
__version__ = '0.3.16'


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



def retrieve_dataset_files(searchresult,
                           getpickle=False,
                           outdir=None):
    '''This retrieves the dataset's CSV and any LC zip files.

    Takes the output from submit_*_query functions above or a pickle file
    generated from these if the query timed out.

    If getpickle is True, will also download the dataset's pickle. Note that LCC
    server is a Python 3.6+ package and it saves its pickles in
    pickle.HIGHEST_PROTOCOL, so these pickles may be unreadable in lower
    Pythons.

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
            localf, header = urlretrieve(dataset_pickle_link,
                                         os.path.join(localdir, dataset_pickle),
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



######################
## SEARCH FUNCTIONS ##
######################

def cone_search(lcc_server,
                center_ra,
                center_decl,
                radiusarcmin=5.0,
                collections=None,
                columns=None,
                filters=None,
                download_data=True,
                outdir=None,
                result_ispublic=True):
    '''This runs a cone-search query.

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

    filters is a filter string to use to filtering the objects that match the
    initial search parameters.

    download_data sets if the accompanying data from the search results will be
    downloaded automatically. This includes the data table CSV, the dataset
    pickle file, and a light curve ZIP file. Note that if the search service
    indicates that your query is still in progress, this function will block
    until the light curve ZIP file becomes available. To avoid this, set
    download_data to False and the function will write a pickle file to
    ~/.astrobase/lccs/query-[setid].pkl containing all the information necessary
    to retrieve these data files later when the query is done. To do so, call
    the retrieve_dataset_files with the path to this pickle file (it will be
    returned).

    outdir if not None, sets the output directory of the downloaded dataset
    files. If None, they will be downloaded to the current directory.

    result_ispublic sets if you want your dataset to be publicly visible on the
    Recent Datasets and /datasets page of the LCC server you're talking to. If
    False, only people who know the unique dataset URL can view and fetch data
    files from it later.

    '''



#######################################
## DATASET AND OBJECT INFO FUNCTIONS ##
#######################################



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
