#!/usr/bin/env python

'''coffeehandlers.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jul 2014

This contains the URL handlers for the astroph-coffee web-server.

'''

import os.path
import logging
import base64
import re

LOGGER = logging.getLogger(__name__)

import tornado.web, tornado.websocket
from tornado.escape import xhtml_escape, xhtml_unescape, url_unescape



##################
## URL HANDLERS ##
##################

class IndexHandler(tornado.web.RequestHandler):
    '''
    This handles the index page.

    '''

    def initialize(self, currentdir):
        '''
        initial set up and stuff goes here

        '''
        self.currentdir = currentdir

        LOGGER.debug('initialized IndexHandler')
        LOGGER.info('working in directory %s' % self.currentdir)



    def get(self):
        '''
        This handles GET requests to the index page.

        '''

        self.render('index.html')



class AboutHandler(tornado.web.RequestHandler):
    '''
    This handles the index page.

    '''

    def initialize(self, currentdir):
        '''
        initial set up and stuff goes here

        '''
        self.currentdir = currentdir

        LOGGER.debug('initialized AboutHandler')
        LOGGER.info('working in directory %s' % self.currentdir)



    def get(self):
        '''
        This handles GET requests to the index page.

        '''

        self.render('about.html')


######################
## WEBSOCK HANDLERS ##
######################

class WebsockHandler (tornado.websocket.WebSocketHandler):
    '''This handles the main websocket that does all the work.

    we'll use self.write_message to return messages to the frontend from the
    backend.

    '''

    def initialize(self):
        '''
        This does initialization stuff using ZMQ.

        '''
        self.currentdir = currentdir

        LOGGER.debug('initialized WebsockHandler')
        LOGGER.info('working in directory %s' % self.currentdir)


    def open(self):
        '''
        This handles the initial load of the websocket from the client side.

        '''


    def on_message(self):
        '''
        This handles each incoming websocket message.

        '''

    def on_close(self):
        '''
        This handles a closed websocket from the client side.

        '''
