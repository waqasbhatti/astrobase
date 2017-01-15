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

    def initialize(self, currentdir, assetpath, cplist, cplistfile, currentcp):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.currentcp = currentcp



    def get(self):
        '''
        This handles GET requests to the index page.

        '''

        # generate the project's list of checkplots
        project_checkplots = sorted(self.currentproject['checkplots'])
        project_checkplotbasenames = [os.path.basename(x)
                                      for x in project_checkplots]
        project_checkplotindices = range(len(project_checkplots))

        self.render('cpindex.html',
                    project_checkplots=project_checkplots,
                    project_checkplotbasenames=project_checkplotbasenames,
                    project_checkplotindices=project_checkplotindices)



class CheckplotHandler(tornado.web.RequestHandler):
    '''This handles loading and saving checkplots.

    This includes GET requests to get to and load a specific checkplot pickle
    file and POST requests to save the checkplot changes back to the file.

    '''

    def initialize(self, currentdir, assetpath, cplist, cplistfile, currentcp):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.currentcp = currentcp

        LOGGER.info('working on checkplot list file %s' % self.cplistfile)



    def get(self, checkplotfname):
        '''This handles GET requests.

        This is an AJAX endpoint; returns JSON that gets converted by the
        frontend into things to render.

        NOTE: this saves the loaded checkplot into the persistent server-wide
        dict. if we run into any errors, we should re-initialize the persistent
        stored dict

        '''

        if checkplotfname:

            # do the usual safing
            self.checkplotfname = xhtml_escape(
                base64.b64decode(checkplotfname)
            )

            # see if this plot is in the current project
            if self.checkplotfname in self.currentproject['checkplots']:

                # make sure this file exists
                cpfpath = os.path.join(
                    os.path.abspath(os.path.dirname(self.cplistfile)),
                    self.checkplotfname
                )

                LOGGER.info('loading %s...' % cpfpath)

                if not os.path.exists(cpfpath):

                    msg = "couldn't find checkplot %s" % cpfpath
                    LOGGER.error(msg)
                    resultdict = {'status':'error',
                                  'message':msg,
                                  'result':None}
                    self.currentcp = dict()

                    self.write(resultdict)


                # load it if it does exist
                self.currentcp = _read_checkplot_picklefile(cpfpath)

                # break out the initial info
                objectid = self.currentcp['objectid']
                objectinfo = self.currentcp['objectinfo']
                varinfo = self.currentcp['varinfo']

                if 'comments' in self.currentcp:
                    objectcomments = self.currentcp['comments']
                else:
                    objectcomments = None

                # these are base64 which can be provided directly to JS to
                # generate images (neat!)
                finderchart = self.currentcp['finderchart']
                magseries = self.currentcp['magseries']['plot']

                if isinstance(finderchart,bytes):
                    finderchart = finderchart.decode()
                if isinstance(magseries,bytes):
                    magseries = magseries.decode()

                cpstatus = self.currentcp['status']

                resultdict = {
                    'status':'ok',
                    'message':'found checkplot %s' % self.checkplotfname,
                    'result':{'objectid':objectid,
                              'objectinfo':objectinfo,
                              'objectcomments':objectcomments,
                              'varinfo':varinfo,
                              'finderchart':finderchart,
                              'magseries':magseries,
                              'cpstatus':cpstatus}
                }

                # now get the other stuff
                for key in ('pdm','aov','bls','gls','sls'):

                    if key in self.currentcp:

                        periodogram = self.currentcp[key]['periodogram']
                        if isinstance(periodogram,bytes):
                            periodogram = periodogram.decode()

                        phasedlc0plot = self.currentcp[key][0]['plot']
                        if isinstance(phasedlc0plot,bytes):
                            phasedlc0plot = phasedlc0plot.decode()

                        phasedlc1plot = self.currentcp[key][1]['plot']
                        if isinstance(phasedlc1plot,bytes):
                            phasedlc1plot = phasedlc1plot.decode()

                        phasedlc2plot = self.currentcp[key][2]['plot']
                        if isinstance(phasedlc2plot,bytes):
                            phasedlc2plot = phasedlc2plot.decode()

                        resultdict['result'][key] = {
                            'nbestperiods':self.currentcp[key]['nbestperiods'],
                            'periodogram':periodogram,
                            'bestperiod':self.currentcp[key]['bestperiod'],
                            'phasedlc0':{
                                'plot':phasedlc0plot,
                                'period':float(self.currentcp[key][0]['period']),
                                'epoch':float(self.currentcp[key][0]['epoch'])
                            },
                            'phasedlc1':{
                                'plot':phasedlc1plot,
                                'period':float(self.currentcp[key][1]['period']),
                                'epoch':float(self.currentcp[key][1]['epoch'])
                            },
                            'phasedlc2':{
                                'plot':phasedlc2plot,
                                'period':float(self.currentcp[key][2]['period']),
                                'epoch':float(self.currentcp[key][2]['epoch'])
                            },
                        }

                # return this via JSON
                self.write(resultdict)

            else:

                LOGGER.error('could not find %s' % self.checkplotfname)

                resultdict = {'status':'error',
                              'message':"This checkplot doesn't exist.",
                              'result':None}
                self.currentcp = dict()
                self.write(resultdict)


        else:

            resultdict = {'status':'error',
                          'message':'No checkplot provided to load.',
                          'result':None}

            self.write(resultdict)
            self.currentcp = dict()



    def post(self):
        '''This handles POST requests.

        Also an AJAX endpoint. Updates the persistent checkplot dict using the
        changes from the UI, and then saves it back to disk. This could
        definitely be faster by just loading the checkplot into a server-wide
        shared dict or something.

        FIXME: should this be async and run a processpoolexecutor or something?

        FIXME: should this zero out the persistent dict?

        '''


class OperationsHandler(tornado.web.RequestHandler):
    '''This handles operations for checkplot stuff.

    This includes GET requests to get the components (finder, objectinfo,
    varinfo, magseries plots, for each lspinfo: periodogram + best phased
    magseries plots).

    Also includes POST requests to redo any of these components (e.g. redo a
    phased mag series plot using twice or half the current period).

    '''

    def initialize(self, currentdir, assetpath, cplist, cplistfile, currentcp):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.currentcp = currentcp

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
