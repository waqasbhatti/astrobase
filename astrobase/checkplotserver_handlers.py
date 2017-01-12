#!/usr/bin/env python

'''checkplotserver_handlers.py - Waqas Bhatti (wbhatti@astro.princeton.edu) -
                                 Jan 2017

These are Tornado handlers for serving checkplots and operating on them.

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
import logging
from datetime import time

try:
    import simplejson as json
except:
    import json

# get a logger
LOGGER = logging.getLogger(__name__)

#####################
## TORNADO IMPORTS ##
#####################

import tornado.ioloop
import tornado.httpserver
import tornado.web
from tornado.escape import xhtml_escape, xhtml_unescape, url_unescape

###################
## LOCAL IMPORTS ##
###################

from .checkplot import checkplot_pickle_update, checkplot_pickle_to_png, \
    _read_checkplot_picklefile, _base64_to_file

# FIXME: import these for updating plots due to user input
# from .checkplot import _pkl_finder_objectinfo, _pkl_periodogram, \
#     _pkl_magseries_plot, _pkl_phased_magseries_plot,
# from .periodbase import pgen_lsp, aov_periodfind, \
#     stellingwerf_pdm, bls_parallel_pfind


#######################
## UTILITY FUNCTIONS ##
#######################




#####################
## HANDLER CLASSES ##
#####################


class IndexHandler(tornado.web.RequestHandler):
    '''This handles the index page.

    This page shows the current project, saved projects, and allows people to
    load, save, and delete these projects. The project database is a json file
    stored in $MODULEPATH/data. If a checkplotlist is provided, then we jump
    straight into the current project view.

    '''

    def initialize(self, currentdir, assetpath, allprojects,
                   cplist, cplistfile):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.allprojects = allprojects
        self.currentproject = cplist
        self.cplistfile = cplistfile

        LOGGER.info('working in directory %s' % self.currentdir)
        LOGGER.info('working on checkplot list file %s' % self.cplistfile)



    def get(self):
        '''
        This handles GET requests to the index page.

        '''

        # generate the project's list of checkplots
        project_checkplots = sorted(self.currentproject['checkplots'])

        self.render('cpindex.html',
                    project_checkplots=project_checkplots)





class CheckplotHandler(tornado.web.RequestHandler):
    '''This handles loading and saving checkplots.

    This includes GET requests to get to and load a specific checkplot pickle
    file and POST requests to save the checkplot changes back to the file.

    '''

    def initialize(self, currentdir, assetpath, allprojects,
                   cplist, cplistfile):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.allprojects = allprojects
        self.currentproject = cplist
        self.cplistfile = cplistfile

        LOGGER.info('working in directory %s' % self.currentdir)
        LOGGER.info('working on checkplot list file %s' % self.cplistfile)



    def get(self, checkplotfname):
        '''This handles GET requests.

        This is an AJAX endpoint; returns JSON that gets converted by the
        frontend into things to render.

        '''



    def post(self):
        '''
        This handles POST requests.

        Also an AJAX endpoint.

        '''


class OperationsHandler(tornado.web.RequestHandler):
    '''This handles operations for checkplot stuff.

    This includes GET requests to get the components (finder, objectinfo,
    varinfo, magseries plots, for each lspinfo: periodogram + best phased
    magseries plots).

    Also includes POST requests to redo any of these components (e.g. redo a
    phased mag series plot using twice or half the current period).

    '''

    def initialize(self, currentdir, assetpath, allprojects,
                   cplist, cplistfile):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.allprojects = allprojects
        self.currentproject = cplist
        self.cplistfile = cplistfile

        LOGGER.info('working in directory %s' % self.currentdir)
        LOGGER.info('working on checkplot list file %s' % self.cplistfile)



    def get(self):
        '''
        This handles GET requests.

        '''



    def post(self):
        '''
        This handles POST requests.

        '''
