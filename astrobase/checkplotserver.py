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

try:
    import simplejson as json
except:
    import json

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


###################
## LOCAL IMPORTS ##
###################

from .checkplot import checkplot_pickle_to_dict, checkplot_pickle_update, \
    checkplot_pickle_to_png, _make_phased_magseries_plot


###########################
## DEFINING URL HANDLERS ##
###########################

class CheckplotHandler(tornado.web.RequestHandler):
    '''
    This handles everything to do with loading and serving checkplots.

    '''

    def initialize(self, currentdir):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        LOGGER.info('working in directory %s' % self.currentdir)

        # search for a checkplot-filelist.json file in this directory
        # load it, and then make everything ready for plotting, etc.


    def get(self):
        '''
        This handles GET requests to the index page.

        '''

        self.render('index.html',websock_url=self.websockurl)

    def post(self):
        '''
        This handles GET requests to the index page.

        '''

        self.render('index.html',websock_url=self.websockurl)



###############################
### APPLICATION SETUP BELOW ###
###############################

modpath = os.path.abspath(os.path.dirname(__file__))

# define our commandline options
define('port',
       default=5225,
       help='run on the given port.',
       type=int)
define('serve',
       default='127.0.0.1',
       help='bind to given address and serve content.',
       type=str)
define('assetpath',
       default=os.path.join(modpath,'data'),
       help='sets the asset (server images, css, js) path for checkplotserver.',
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

    ##################
    ## URL HANDLERS ##
    ##################

    HANDLERS = [
        # index page
        (r'/',
         CheckplotHandler,
         {'currentdir':CURRENTDIR}),
    ]

    #######################
    ## APPLICATION SETUP ##
    #######################

    app = tornado.web.Application(
        handlers=HANDLERS,
        cookie_secret=SESSIONSECRET,
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
