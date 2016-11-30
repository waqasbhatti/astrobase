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

import zmq



##################
## URL HANDLERS ##
##################

class IndexHandler(tornado.web.RequestHandler):
    '''
    This handles the index page.

    '''

    def initialize(self, currentdir, websockurl):
        '''
        initial set up and stuff goes here

        '''
        self.currentdir = currentdir
        self.websockurl = websockurl

        LOGGER.debug('initialized IndexHandler')
        LOGGER.info('working in directory %s' % self.currentdir)



    def get(self):
        '''
        This handles GET requests to the index page.

        '''

        self.render('index.html',websock_url=self.websockurl)



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

    def initialize(self, currentdir, context):
        '''
        This does initialization stuff using ZMQ.

        '''
        self.currentdir = currentdir
        self.ctx = context

        LOGGER.debug('initialized WebsockHandler')
        LOGGER.info('working in directory %s' % self.currentdir)


    def _sock_setup(self, handlerfunc):
        '''
        This sets up the ZMQ sockets for this URL handler.

        '''

        # we need to setup a new insock
        self.insock = self.ctx.socket(zmq.SUB)
        self.insock.connect('tcp://127.0.0.1:33001')

        # subscribe to a topic
        self.subtopic = subtopic
        self.insock.setsockopt(zmq.SUBSCRIBE, self.subtopic)

        # set up the input SUB socket's input stream and add our
        # callback function
        self.insockstream = zmq.eventloop.zmqstream.ZMQStream(self.insock)
        self.insockstream.on_recv(handlerfunc)
        self.sock_is_closed = False



    def _sock_shutdown(self):
        '''
        This cleans up the ZMQ sockets.

        '''

        # unsubscribe the socket from the message topic and shut it down
        if self.subtopic:
            self.insock.setsockopt(zmq.UNSUBSCRIBE, self.subtopic)
        self.insockstream.stop_on_recv()
        self.insockstream.close()
        self.insock.close(linger=0)
        self.sock_is_closed = True


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
