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
import time

# get a logger
LOGGER = logging.getLogger(__name__)

#####################
## TORNADO IMPORTS ##
#####################

import tornado.ioloop
import tornado.httpserver
import tornado.web
from tornado.escape import xhtml_escape, xhtml_unescape, url_unescape
from tornado import gen

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

    This page shows the current project.

    FUTURE: this should show a list of all projects the server knows about and
    then allow loading them, etc.

    '''

    def initialize(self, currentdir, assetpath, cplist,
                   cplistfile, executor):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor



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
                    project_checkplotindices=project_checkplotindices,
                    project_checkplotfile=self.cplistfile)



class CheckplotHandler(tornado.web.RequestHandler):
    '''This handles loading and saving checkplots.

    This includes GET requests to get to and load a specific checkplot pickle
    file and POST requests to save the checkplot changes back to the file.

    '''

    def initialize(self, currentdir, assetpath, cplist, cplistfile, executor):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor


    @gen.coroutine
    def get(self, checkplotfname):
        '''This handles GET requests.

        This is an AJAX endpoint; returns JSON that gets converted by the
        frontend into things to render.

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

                    self.write(resultdict)
                    self.finish()

                # this is the async call to the executor
                cpdict = yield self.executor.submit(
                    _read_checkplot_picklefile, cpfpath
                )

                #####################################
                ## continue after we're good to go ##
                #####################################

                # break out the initial info
                objectid = cpdict['objectid']
                objectinfo = cpdict['objectinfo']
                varinfo = cpdict['varinfo']

                if 'comments' in cpdict:
                    objectcomments = cpdict['comments']
                else:
                    objectcomments = None

                # these are base64 which can be provided directly to JS to
                # generate images (neat!)
                finderchart = cpdict['finderchart']
                magseries = cpdict['magseries']['plot']

                if isinstance(finderchart,bytes):
                    finderchart = finderchart.decode()
                if isinstance(magseries,bytes):
                    magseries = magseries.decode()

                cpstatus = cpdict['status']

                resultdict = {
                    'status':'ok',
                    'message':'found checkplot %s' % self.checkplotfname,
                    'result':{
                        'objectid':objectid,
                        'objectinfo':objectinfo,
                        'objectcomments':objectcomments,
                        'varinfo':varinfo,
                        'finderchart':finderchart,
                        'magseries':magseries,
                        # fallback in case objectinfo doesn't have ndet
                        'magseries_ndet':cpdict['magseries']['times'].size,
                        'cpstatus':cpstatus
                    }
                }

                # now get the other stuff
                for key in ('pdm','aov','bls','gls','sls'):

                    if key in cpdict:

                        periodogram = cpdict[key]['periodogram']
                        if isinstance(periodogram,bytes):
                            periodogram = periodogram.decode()

                        phasedlc0plot = cpdict[key][0]['plot']
                        if isinstance(phasedlc0plot,bytes):
                            phasedlc0plot = phasedlc0plot.decode()

                        phasedlc1plot = cpdict[key][1]['plot']
                        if isinstance(phasedlc1plot,bytes):
                            phasedlc1plot = phasedlc1plot.decode()

                        phasedlc2plot = cpdict[key][2]['plot']
                        if isinstance(phasedlc2plot,bytes):
                            phasedlc2plot = phasedlc2plot.decode()

                        resultdict['result'][key] = {
                            'nbestperiods':cpdict[key]['nbestperiods'],
                            'periodogram':periodogram,
                            'bestperiod':cpdict[key]['bestperiod'],
                            'phasedlc0':{
                                'plot':phasedlc0plot,
                                'period':float(cpdict[key][0]['period']),
                                'epoch':float(cpdict[key][0]['epoch'])
                            },
                            'phasedlc1':{
                                'plot':phasedlc1plot,
                                'period':float(cpdict[key][1]['period']),
                                'epoch':float(cpdict[key][1]['epoch'])
                            },
                            'phasedlc2':{
                                'plot':phasedlc2plot,
                                'period':float(cpdict[key][2]['period']),
                                'epoch':float(cpdict[key][2]['epoch'])
                            },
                        }

                # return this via JSON
                self.write(resultdict)
                self.finish()

            else:

                LOGGER.error('could not find %s' % self.checkplotfname)

                resultdict = {'status':'error',
                              'message':"This checkplot doesn't exist.",
                              'result':None}
                self.write(resultdict)
                self.finish()


        else:

            resultdict = {'status':'error',
                          'message':'No checkplot provided to load.',
                          'result':None}

            self.write(resultdict)


    @gen.coroutine
    def post(self, cpfile):
        '''This handles POST requests.

        Also an AJAX endpoint. Updates the persistent checkplot dict using the
        changes from the UI, and then saves it back to disk. This could
        definitely be faster by just loading the checkplot into a server-wide
        shared dict or something.

        FIXME: this will be async and run a processpoolexecutor:
        - receive post request
        - fork to background save process
        - write save-started message to frontend with filename that's busy
        - frontend marks this file as busy and carries on with the next op
        - once the save returns, this writes a save-complete message
        - then closes the request

        '''

        # now try to update the contents
        try:

            self.cpfile = base64.b64decode(cpfile).decode()
            cpcontents = self.get_argument('cpcontents', default=None)

            if not self.cpfile or not cpcontents:

                msg = "did not receive a checkplot update payload"
                resultdict = {'status':'error',
                              'message':msg,
                              'result':None}
                self.set_status(400)
                self.write(resultdict)
                self.finish()

            cpcontents = json.loads(cpcontents)

            # the only keys in cpdict that can updated from the UI are from
            # varinfo, objectinfo (objecttags) and comments
            updated = {'varinfo': cpcontents['varinfo'],
                       'objectinfo':cpcontents['objectinfo'],
                       'comments':cpcontents['comments']}

            # we need reform the self.cpfile so it points to the full path
            cpfpath = os.path.join(
                os.path.abspath(os.path.dirname(self.cplistfile)),
                self.cpfile
            )

            LOGGER.info('loading %s...' % cpfpath)

            if not os.path.exists(cpfpath):

                msg = "couldn't find checkplot %s" % cpfpath
                LOGGER.error(msg)
                resultdict = {'status':'error',
                              'message':msg,
                              'result':None}

                self.write(resultdict)
                self.finish()

            # dispatch the task
            updated = yield self.executor.submit(checkplot_pickle_update,
                                                 cpfpath, updated)

            # continue processing after this is done
            if updated:

                LOGGER.info('updated checkplot %s successfully' % updated)
                resultdict = {'status':'success',
                              'message':'checkplot update successful',
                              'result':{'checkplot':updated,
                                        'unixtime':time.time(),
                                        'changes':cpcontents}}
                self.write(resultdict)
                self.finish()

            else:
                LOGGER.error('could not handle checkplot update for %s: %s' %
                             (self.cpfile, cpcontents))
                msg = "checkplot update failed because of a backend error"
                resultdict = {'status':'error',
                              'message':msg,
                              'result':None}
                self.set_status(500)
                self.write(resultdict)
                self.finish()

        # if something goes wrong, inform the user
        except Exception as e:

            LOGGER.exception('could not handle checkplot update for %s: %s' %
                             (self.cpfile, cpcontents))
            msg = "checkplot update failed because of an exception"
            resultdict = {'status':'error',
                          'message':msg,
                          'result':None}
            self.set_status(500)
            self.write(resultdict)
            self.finish()



class CheckplotListHandler(tornado.web.RequestHandler):
    '''This handles loading and saving the checkplot-filelist.json file.

    GET requests just return the current contents of the checkplot-filelist.json
    file. POST requests will put in changes that the user made from the
    frontend.

    '''

    def initialize(self, currentdir, assetpath, cplist, cplistfile, executor):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor



    def get(self):
        '''
        This handles GET requests. Used with AJAX from frontend.

        '''

        # add the reviewed key to the current dict if it doesn't exist
        # this will hold all the reviewed objects for the frontend
        if not 'reviewed' in self.currentproject:
            self.currentproject['reviewed'] = {}

        # just returns the current project as JSON
        self.write(self.currentproject)



    def post(self):
        '''
        This handles POST requests. Saves the changes made by the user.


        '''

        objectid = self.get_argument('objectid', None)
        changes = self.get_argument('changes',None)

        if not objectid or not changes:
            msg = ("could not parse changes to the checkplot filelist "
                   "from the frontend")
            LOGGER.error(msg)
            resultdict = {'status':'error',
                          'message':msg,
                          'result':None}

            self.write(resultdict)

        objectid = xhtml_escape(objectid)
        changes = json.loads(changes)

        # update the dictionary
        if 'reviewed' not in self.currentproject:
            self.currentproject['reviewed'] = {}

        self.currentproject['reviewed'][objectid] = changes

        # update the JSON file
        with open(self.cplistfile,'w') as outfd:
            json.dump(self.currentproject, outfd)

        # return status
        msg = ("wrote all changes to the checkplot filelist "
               "from the frontend for object: %s" % objectid)
        LOGGER.info(msg)
        resultdict = {'status':'success',
                      'message':msg,
                      'result':{'objectid':objectid,
                                'changes':changes}}

        self.write(resultdict)
