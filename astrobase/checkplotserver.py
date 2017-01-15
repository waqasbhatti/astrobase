#!/usr/bin/env python

'''checkplotserver.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This is the Tornado web-server for serving checkplots.

'''

####################
## SYSTEM IMPORTS ##
####################

import os
import os.path
import gzip
try:
    import cPickle as pickle
except:
    import pickle
import base64
import hashlib
import signal
import logging
from datetime import time
import json
import sys

# setup signal trapping on SIGINT
def recv_sigint(signum, stack):
    '''
    handler function to receive and process a SIGINT

    '''
    raise KeyboardInterrupt

# register the signal callback
signal.signal(signal.SIGINT,recv_sigint)
signal.signal(signal.SIGTERM,recv_sigint)


#####################
## TORNADO IMPORTS ##
#####################

import tornado.ioloop
import tornado.httpserver
import tornado.web
import tornado.options
from tornado.options import define, options


###########################
## DEFINING URL HANDLERS ##
###########################

import astrobase.checkplotserver_handlers as cphandlers


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
       default=os.path.join(modpath,'data','cps-assets'),
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

############
### MAIN ###
############

def main():
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

    ASSETPATH = options.assetpath

    # this is the directory checkplotserver.py was executed from. used to figure
    # out checkplot locations
    CURRENTDIR = os.getcwd()

    # if a checkplotlist is provided, then load it.
    # NOTE: all paths in this file are relative to the path of the checkplotlist
    # file itself.
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

        if os.path.exists(os.path.join(CURRENTDIR,'checkplot-filelist.json')):

            cplistfile = os.path.join(CURRENTDIR,'checkplot-filelist.json')
            with open(cplistfile,'r') as infd:
                CHECKPLOTLIST = json.load(infd)
            LOGGER.info('found! using checkplot list file: %s' % cplistfile)

        else:

            helpmsg = (
                "No checkplot-filelist.json found, "
                "can't continue without one.\n"
                "Did you make a checkplot list file? "
                "To make one, try running the following command:\n"
                "python %s pkl "
                "/path/to/folder/where/the/checkplot.pkl.gz/files/are" %
                (os.path.join(modpath,'checkplotlist.py'))
            )
            LOGGER.error(helpmsg)
            sys.exit(1)

    ##################
    ## URL HANDLERS ##
    ##################

    HANDLERS = [
        # index page
        (r'/',
         cphandlers.IndexHandler,
         {'currentdir':CURRENTDIR,
          'assetpath':ASSETPATH,
          'cplist':CHECKPLOTLIST,
          'cplistfile':cplistfile}),
        (r'/cp/?(.*)',
         cphandlers.CheckplotHandler,
         {'currentdir':CURRENTDIR,
          'assetpath':ASSETPATH,
          'cplist':CHECKPLOTLIST,
          'cplistfile':cplistfile}),
        (r'/op',
         cphandlers.OperationsHandler,
         {'currentdir':CURRENTDIR,
          'assetpath':ASSETPATH,
          'cplist':CHECKPLOTLIST,
          'cplistfile':cplistfile}),
    ]

    #######################
    ## APPLICATION SETUP ##
    #######################

    app = tornado.web.Application(
        handlers=HANDLERS,
        static_path=ASSETPATH,
        template_path=ASSETPATH,
        static_url_prefix='/static/',
        debug=DEBUG,
    )

    # start up the HTTP server and our application. xheaders = True turns on
    # X-Forwarded-For support so we can see the remote IP in the logs
    http_server = tornado.httpserver.HTTPServer(app, xheaders=True)
    http_server.listen(options.port, options.serve)

    LOGGER.info('starting event loop. listening on http://localhost:%s' %
                options.port)

    # start the IOLoop and begin serving requests
    try:
        tornado.ioloop.IOLoop.instance().start()

    except KeyboardInterrupt:
        LOGGER.info('received Ctrl-C: shutting down...')


# run the server
if __name__ == '__main__':
    main()
