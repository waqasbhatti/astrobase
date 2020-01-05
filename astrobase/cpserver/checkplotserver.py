#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# checkplotserver.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

'''`checkplotserver` is a Tornado web-server for visualizing the information
stored in checkplot pickles, editing them, and exporting information to a
variable star classification pipeline.

This is the main module used to launch the server.

'''

####################
## SYSTEM IMPORTS ##
####################

import os
import os.path
import signal
import logging
import json
import time
import sys
import socket
import stat

# this handles async updates of the checkplot pickles so the UI remains
# responsive
from concurrent.futures import ProcessPoolExecutor


# setup signal trapping on SIGINT
def _recv_sigint(signum, stack):
    '''
    handler function to receive and process a SIGINT

    '''
    raise KeyboardInterrupt


#####################
## TORNADO IMPORTS ##
#####################

# significant speedup if uvloop is available
try:
    import asyncio
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except Exception:
    pass

import tornado.ioloop
import tornado.httpserver
import tornado.web
import tornado.options
from tornado.options import define, options


###########################
## DEFINING URL HANDLERS ##
###########################

from . import checkplotserver_handlers as basehandlers
from . import checkplotserver_cphandlers as cphandlers
from . import checkplotserver_toolhandlers as toolhandlers
from . import checkplotserver_standalone as standalone


###############################
### APPLICATION SETUP BELOW ###
###############################

modpath = os.path.abspath(os.path.dirname(__file__))

# define our commandline options
define('port',
       default=5225,
       help='Run on the given port.',
       type=int)
define('serve',
       default='127.0.0.1',
       help='Bind to given address and serve content.',
       type=str)
define('assetpath',
       default=os.path.abspath(os.path.join(modpath,'cps-assets')),
       help=('Sets the asset (server images, css, js, DB) path for '
             'checkplotserver.'),
       type=str)
define('checkplotlist',
       default=None,
       help=('The path to the checkplot-filelist.json file '
             'listing checkplots to load and serve. If this is not provided, '
             'checkplotserver will look for a '
             'checkplot-pickle-flist.json in the directory '
             'that it was started in'),
       type=str)
define('debugmode',
       default=0,
       help='start up in debug mode if set to 1.',
       type=int)
define('maxprocs',
       default=2,
       help=('Number of background processes to use '
             'for saving/loading checkplot files and '
             'running light curves tools'),
       type=int)
define('readonly',
       default=False,
       help=("Run the server in readonly mode. This is useful for a "
             "public-facing instance of checkplotserver where you just "
             "want to allow collaborators to "
             "review objects but not edit them."),
       type=bool)
define('baseurl',
       default='/',
       help=("Set the base URL of the checkplotserver. "
             "This is useful when you're running checkplotserver "
             "on a remote machine and are reverse-proxying more than one "
             "instances of it so you can access them "
             "using HTTP from outside on different base URLs "
             "like /cpserver1/, /cpserver2/, etc. "
             "If this is set, all URLs will take the form [baseurl]/..., "
             "instead of /..."),
       type=str)

#
# special stand-alone mode
#
# this is used for checkplotserver is serving checkplots to another service via
# HTTP. two options are required below:
#
# --standalone=1
# --sharedsecret=/path/to/shared/secret/file
#
# the shared secret file contains a key that is required for any access via the
# standalone method. we do this because the standalone mode can open any file
# anywhere (being used for opening a checkplot pickle, serializing it to JSON,
# and sending it back to another process)
define('standalone',
       default=0,
       help=("This starts the server in standalone mode."),
       type=int)
define('sharedsecret',
       default='',
       help=("a file containing a cryptographically "
             "secure string that is used to authenticate "
             "requests that come into the special standalone mode."),
       type=str)


############
### MAIN ###
############

def main():
    '''
    This launches the server. The current script args are shown below::

      Usage: checkplotserver [OPTIONS]

      Options:

        --help                           show this help information
        --assetpath                      Sets the asset (server images, css, js, DB)
                                         path for checkplotserver.
                                         (default <astrobase install dir>
                                          /astrobase/cpserver/cps-assets)
        --baseurl                        Set the base URL of the checkplotserver.
                                         This is useful when you're running
                                         checkplotserver on a remote machine and are
                                         reverse-proxying more than one instances of
                                         it so you can access them using HTTP from
                                         outside on different base URLs like
                                         /cpserver1/, /cpserver2/, etc. If this is
                                         set, all URLs will take the form
                                         [baseurl]/..., instead of /... (default /)
        --checkplotlist                  The path to the checkplot-filelist.json file
                                         listing checkplots to load and serve. If
                                         this is not provided, checkplotserver will
                                         look for a checkplot-pickle-flist.json in
                                         the directory that it was started in
        --debugmode                      start up in debug mode if set to 1. (default
                                         0)
        --maxprocs                       Number of background processes to use for
                                         saving/loading checkplot files and running
                                         light curves tools (default 2)
        --port                           Run on the given port. (default 5225)
        --readonly                       Run the server in readonly mode. This is
                                         useful for a public-facing instance of
                                         checkplotserver where you just want to allow
                                         collaborators to review objects but not edit
                                         them. (default False)
        --serve                          Bind to given address and serve content.
                                         (default 127.0.0.1)
        --sharedsecret                   a file containing a cryptographically secure
                                         string that is used to authenticate requests
                                         that come into the special standalone mode.
        --standalone                     This starts the server in standalone mode.
                                         (default 0)

    '''
    # parse the command line
    tornado.options.parse_command_line()

    DEBUG = True if options.debugmode == 1 else False

    # get a logger
    LOGGER = logging.getLogger('checkplotserver')
    if DEBUG:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    ###################
    ## SET UP CONFIG ##
    ###################

    MAXPROCS = options.maxprocs
    ASSETPATH = options.assetpath
    BASEURL = options.baseurl

    ###################################
    ## PERSISTENT CHECKPLOT EXECUTOR ##
    ###################################

    EXECUTOR = ProcessPoolExecutor(MAXPROCS)

    #######################################
    ## CHECK IF WE'RE IN STANDALONE MODE ##
    #######################################

    if options.standalone:

        if ( (not options.sharedsecret) or
             (options.sharedsecret and
              not os.path.exists(options.sharedsecret)) ):

            LOGGER.error('Could not find a shared secret file to use in \n'
                         'standalone mode. Generate one using: \n\n'
                         'python3 -c "import secrets; '
                         'print(secrets.token_urlsafe(32))" '
                         '> secret-key-file.txt\n\nSet user-only ro '
                         'permissions on the generated file (chmod 400)')
            sys.exit(1)

        elif options.sharedsecret and os.path.exists(options.sharedsecret):

            # check if this file is readable/writeable by user only
            fileperm = oct(os.stat(options.sharedsecret)[stat.ST_MODE])

            if fileperm == '0100400' or fileperm == '0o100400':

                with open(options.sharedsecret,'r') as infd:

                    SHAREDSECRET = infd.read().strip('\n')

                    # this is the URLSpec for the standalone Handler
                    standalonespec = (
                        r'/standalone',
                        standalone.StandaloneHandler,
                        {'executor':EXECUTOR,
                         'secret':SHAREDSECRET}
                    )
            else:
                LOGGER.error('permissions on the shared secret file '
                             'should be 0100400')
                sys.exit(1)

        else:
            LOGGER.error('could not find the specified '
                         'shared secret file: %s' %
                         options.sharedsecret)
            sys.exit(1)

        # only one handler in standalone mode
        HANDLERS = [standalonespec]

    # if we're not in standalone mode, proceed normally
    else:

        if not BASEURL.endswith('/'):
            BASEURL = BASEURL + '/'

        READONLY = options.readonly
        if READONLY:
            LOGGER.warning('checkplotserver running in readonly mode.')

        # this is the directory checkplotserver.py was executed from. used to
        # figure out checkplot locations
        CURRENTDIR = os.getcwd()

        # if a checkplotlist is provided, then load it.  NOTE: all paths in this
        # file are relative to the path of the checkplotlist file itself.
        cplistfile = options.checkplotlist

        # if the provided cplistfile is OK
        if cplistfile and os.path.exists(cplistfile):

            with open(cplistfile,'r') as infd:
                CHECKPLOTLIST = json.load(infd)
            LOGGER.info('using provided checkplot list file: %s' % cplistfile)

        # if a cplist is provided, but doesn't exist
        elif cplistfile and not os.path.exists(cplistfile):
            helpmsg = (
                "Couldn't find the file %s\n"
                "NOTE: To make a checkplot list file, "
                "try running the following command:\n"
                "python %s pkl "
                "/path/to/folder/where/the/checkplot.pkl.gz/files/are" %
                (cplistfile, os.path.join(modpath,'checkplotlist.py'))
            )
            LOGGER.error(helpmsg)
            sys.exit(1)

        # finally, if no cplistfile is provided at all, search for a
        # checkplot-filelist.json in the current directory
        else:
            LOGGER.warning('No checkplot list file provided!\n'
                           '(use --checkplotlist=... for this, '
                           'or use --help to see all options)\n'
                           'looking for checkplot-filelist.json in the '
                           'current directory %s ...' % CURRENTDIR)

            # this is for single checkplot lists
            if os.path.exists(
                    os.path.join(CURRENTDIR,'checkplot-filelist.json')
            ):

                cplistfile = os.path.join(CURRENTDIR,'checkplot-filelist.json')
                with open(cplistfile,'r') as infd:
                    CHECKPLOTLIST = json.load(infd)
                LOGGER.info('using checkplot list file: %s' % cplistfile)

            # this is for chunked checkplot lists
            elif os.path.exists(os.path.join(CURRENTDIR,
                                             'checkplot-filelist-00.json')):

                cplistfile = os.path.join(CURRENTDIR,
                                          'checkplot-filelist-00.json')
                with open(cplistfile,'r') as infd:
                    CHECKPLOTLIST = json.load(infd)
                LOGGER.info('using checkplot list file: %s' % cplistfile)

            # if we can't find a checkplot list, bail out
            else:

                helpmsg = (
                    "No checkplot file list JSON found, "
                    "can't continue without one.\n"
                    "Did you make a checkplot list file? "
                    "To make one, try running the following command:\n"
                    "checkplotlist pkl "
                    "/path/to/folder/where/the/checkplot.pkl.gz/files/are"
                )
                LOGGER.error(helpmsg)
                sys.exit(1)

        ##################################
        ## URL HANDLERS FOR NORMAL MODE ##
        ##################################

        HANDLERS = [
            # index page
            (r'{baseurl}'.format(baseurl=BASEURL),
             basehandlers.IndexHandler,
             {'currentdir':CURRENTDIR,
              'assetpath':ASSETPATH,
              'cplist':CHECKPLOTLIST,
              'cplistfile':cplistfile,
              'executor':EXECUTOR,
              'readonly':READONLY,
              'baseurl':BASEURL}),
            # loads and interacts with checkplot pickles
            (r'{baseurl}cp/?(.*)'.format(baseurl=BASEURL),
             cphandlers.CheckplotHandler,
             {'currentdir':CURRENTDIR,
              'assetpath':ASSETPATH,
              'cplist':CHECKPLOTLIST,
              'cplistfile':cplistfile,
              'executor':EXECUTOR,
              'readonly':READONLY}),
            # loads and interacts with the current checkplot list JSON file
            (r'{baseurl}list'.format(baseurl=BASEURL),
             cphandlers.CheckplotListHandler,
             {'currentdir':CURRENTDIR,
              'assetpath':ASSETPATH,
              'cplist':CHECKPLOTLIST,
              'cplistfile':cplistfile,
              'executor':EXECUTOR,
              'readonly':READONLY}),
            # light curve variability and period-finding tool endpoints
            (r'{baseurl}tools/?(.*)'.format(baseurl=BASEURL),
             toolhandlers.LCToolHandler,
             {'currentdir':CURRENTDIR,
              'assetpath':ASSETPATH,
              'cplist':CHECKPLOTLIST,
              'cplistfile':cplistfile,
              'executor':EXECUTOR,
              'readonly':READONLY}),
            # download any file in the current base directory, mostly used for
            # downloading checkplot pickles and updated checkplot list JSONs
            (r'{baseurl}download/(.*)'.format(baseurl=BASEURL),
             tornado.web.StaticFileHandler, {'path': CURRENTDIR})
        ]

    #######################
    ## APPLICATION SETUP ##
    #######################

    app = tornado.web.Application(
        handlers=HANDLERS,
        static_path=ASSETPATH,
        template_path=ASSETPATH,
        static_url_prefix='{baseurl}static/'.format(baseurl=BASEURL),
        compress_response=True,
        debug=DEBUG,
    )

    # start up the HTTP server and our application. xheaders = True turns on
    # X-Forwarded-For support so we can see the remote IP in the logs
    http_server = tornado.httpserver.HTTPServer(app, xheaders=True)

    ######################
    ## start the server ##
    ######################

    # make sure the port we're going to listen on is ok
    # inspired by how Jupyter notebook does this
    portok = False
    serverport = options.port
    maxtrys = 5
    thistry = 0
    while not portok and thistry < maxtrys:
        try:
            http_server.listen(serverport, options.serve)
            portok = True
        except socket.error:
            LOGGER.warning('%s:%s is already in use, trying port %s' %
                           (options.serve, serverport, serverport + 1))
            serverport = serverport + 1

    if not portok:
        LOGGER.error('could not find a free port after 5 tries, giving up')
        sys.exit(1)

    LOGGER.info('started checkplotserver. listening on http://%s:%s%s' %
                (options.serve, serverport, BASEURL))

    # register the signal callbacks
    signal.signal(signal.SIGINT,_recv_sigint)
    signal.signal(signal.SIGTERM,_recv_sigint)

    # start the IOLoop and begin serving requests
    try:

        tornado.ioloop.IOLoop.instance().start()

    except KeyboardInterrupt:

        LOGGER.info('received Ctrl-C: shutting down...')
        tornado.ioloop.IOLoop.instance().stop()
        # close down the processpool

    EXECUTOR.shutdown()
    time.sleep(3)


# run the server
if __name__ == '__main__':
    main()
