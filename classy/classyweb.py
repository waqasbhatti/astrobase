#!/usr/bin/env python

'''classyweb.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Nov 2016

This is the Tornado web-server for classy-web. It uses URL handlers defined
in classyhandlers.py.

'''

import os
import os.path
import ConfigParser
import sqlite3

import signal
import logging

from datetime import time
from pytz import utc

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

####################################
## LOCAL IMPORTS FOR URL HANDLERS ##
####################################

import classyhandlers


###############################
### APPLICATION SETUP BELOW ###
###############################

modpath = os.path.abspath(os.path.dirname(__file__))
CONF_FILE = os.path.join(modpath,'classy.conf')

# define our commandline options
define('port',
       default=5005,
       help='run on the given port.',
       type=int)
define('serve',
       default='127.0.0.1',
       help='bind to given address and serve content.',
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
    LOGGER = logging.getLogger('classyweb')
    if DEBUG:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)


    ###################
    ## SET UP CONFIG ##
    ###################

    # read the conf files
    CONF = ConfigParser.ConfigParser()
    CONF.read(CONF_FILE)

    # get the web config vars

    # the session secret file
    # FIXME: this should check if the permissions are 0600
    SECRETF = os.path.join(modpath,CONF.get('web','secret'))

    # get the session secret
    SESSIONSECRET = open(SECRETF).read().strip('\n')

    # get the path used for static files such as css, js, and images
    STATICPATH = os.path.abspath(
        os.path.join(modpath, CONF.get('web','static'))
    )

    # this is the path to the Tornado template files
    TEMPLATEPATH = os.path.join(STATICPATH,'templates')

    # this is the directory classyweb.py was executed from. used to figure
    # out light curve and LSP locations
    CURRENTDIR = os.getcwd()

    ##################
    ## URL HANDLERS ##
    ##################

    HANDLERS = [
        (r'/',classyhandlers.IndexHandler, {'currentdir':CURRENTDIR}),
        (r'/websock',classyhandlers.WebsockHandler, {'currentdir':CURRENTDIR}),
        (r'/about',classyhandlers.AboutHandler, {'currentdir':CURRENTDIR}),
        (r'/about/',classyhandlers.AboutHandler, {'currentdir':CURRENTDIR}),
    ]

    #######################
    ## APPLICATION SETUP ##
    #######################

    app = tornado.web.Application(
        handlers=HANDLERS,
        cookie_secret=SESSIONSECRET,
        static_path=STATICPATH,
        template_path=TEMPLATEPATH,
        static_url_prefix='/static/',
        xsrf_cookies=True,
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
