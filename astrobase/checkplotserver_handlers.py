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
import time

######################################
## CUSTOM JSON ENCODER FOR FRONTEND ##
######################################

# we need this to send objects with the following types to the frontend:
# - bytes
# - ndarray
import json

class FrontendEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode()
        else:
            return json.JSONEncoder.default(self, obj)

# this replaces the default encoder and makes it so Tornado will do the right
# thing when it converts dicts to JSON when a
# tornado.web.RequestHandler.write(dict) is called.
json._default_encoder = FrontendEncoder()

#############
## LOGGING ##
#############

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

from numpy import ndarray

from . import checkplot
checkplot.set_logger_parent(__name__)

from .checkplot import checkplot_pickle_update, checkplot_pickle_to_png, \
    _read_checkplot_picklefile, _base64_to_file, _write_checkplot_picklefile

# import these for updating plots due to user input
from .checkplot import _pkl_finder_objectinfo, _pkl_periodogram, \
    _pkl_magseries_plot, _pkl_phased_magseries_plot

from .varbase import features
features.set_logger_parent(__name__)

from .varbase import lcfit
lcfit.set_logger_parent(__name__)

from .varbase import signals
signals.set_logger_parent(__name__)

from .periodbase import zgls
zgls.set_logger_parent(__name__)
from .periodbase import saov
saov.set_logger_parent(__name__)
from .periodbase import spdm
spdm.set_logger_parent(__name__)
from .periodbase import kbls
kbls.set_logger_parent(__name__)


#######################
## UTILITY FUNCTIONS ##
#######################

# this is the function map for arguments
CPTOOLMAP = {
    'psearch-gls':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes','autofreq','stepsize'),
        'kwargtypes':(float, float, bool, bool, float),
        'kwargdefs':(None, None, False, True, 1.0e-4),
        'func':zgls.pgen_lsp,
        'resloc':['gls'],
    },
    'psearch-bls':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes','autofreq','stepsize'),
        'kwargtypes':(float, float, bool, bool, float),
        'kwargdefs':(0.1, 100.0, False, True, 1.0e-4),
        'func':kbls.bls_parallel_pfind,
        'resloc':['bls'],
    },
    'psearch-pdm':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes','autofreq','stepsize'),
        'kwargtypes':(float, float, bool, bool, float),
        'kwargdefs':(None, None, False, True, 1.0e-4),
        'func':spdm.stellingwerf_pdm,
        'resloc':['pdm'],
    },
    'psearch-aov':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes','autofreq','stepsize'),
        'kwargtypes':(float, float, bool, bool, float),
        'kwargdefs':(None, None, False, True, 1.0e-4),
        'func':saov.aov_periodfind,
        'resloc':['aov'],
    },
    'var-varfeatures':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray,ndarray,ndarray),
        'kwargs':(),
        'kwargtypes':(),
        'kwargdefs':(),
        'func':features.all_nonperiodic_features,
        'resloc':['varinfo','features'],
    },
    'var-prewhiten':{
        'args':('times','mags','errs','whiteperiod'),
        'argtypes':(ndarray, ndarray, ndarray, float),
        'kwargs':('magsarefluxes','fourierorder'),
        'kwargtypes':(bool,float),
        'kwargdefs':(False,3),
        'func':signals.whiten_magseries,
        'resloc':['signals','whiten'],
    },
    'var-masksig':{
        'args':('times','mags','errs','signalperiod','signalepoch'),
        'argtypes':(ndarray, ndarray, ndarray, float, float),
        'kwargs':('magsarefluxes','maskphases','maskphaselength'),
        'kwargtypes':(bool, list, float),
        'kwargdefs':(False, [0.0,0.5,1.0], 0.1),
        'func':signals.mask_signal,
        'resloc':['signals','mask'],
    },
    'phasedlc-newplot':{
        'args':(None,'lspmethod','periodind',
                'times','mags','errs','varperiod','varepoch',
                'phasewrap','phasesort','phasebin','minbinelem',
                'plotxlim'),
        'argtypes':(None, str, int,
                    ndarray, ndarray, ndarray, float, float,
                    bool, bool, float, int, list),
        'kwargs':('xliminsetmode','magsarefluxes'),
        'kwargtypes':(bool, bool),
        'kwargdefs':(False, False),
        'func':_pkl_phased_magseries_plot,
        'resloc':[],
    },
    'lcfit-fourier':{
        'args':('times','mags','errs','period'),
        'argtypes':(ndarray, ndarray, ndarray, float),
        'kwargs':('fourierorder','magsarefluxes'),
        'kwargtypes':(int, bool),
        'kwargdefs':(6, False),
        'func':lcfit.fourier_fit_magseries,
        'resloc':['fitinfo','fourier'],
    },
    'lcfit-spline':{
        'args':('times','mags','errs','period'),
        'argtypes':(ndarray, ndarray, ndarray, float),
        'kwargs':('knotfraction','magsarefluxes'),
        'kwargtypes':(int, bool),
        'kwargdefs':(0.01, False),
        'func':lcfit.spline_fit_magseries,
        'resloc':['fitinfo','spline'],
    },
    'lcfit-legendre':{
        'args':('times','mags','errs','period'),
        'argtypes':(ndarray, ndarray, ndarray, float),
        'kwargs':('legendredeg','magsarefluxes'),
        'kwargtypes':(int, bool),
        'kwargdefs':(10, False),
        'func':lcfit.legendre_fit_magseries,
        'resloc':['fitinfo','legendre'],
    },
    'lcfit-savgol':{
        'args':('times','mags','errs','period'),
        'argtypes':(ndarray, ndarray, ndarray, float),
        'kwargs':('windowlength','magsarefluxes'),
        'kwargtypes':(int, bool),
        'kwargdefs':(None, False),
        'func':lcfit.savgol_fit_magseries,
        'resloc':['fitinfo','savgol'],
    },
}




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
                   cplistfile, executor, readonly):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor
        self.readonly = readonly



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

    def initialize(self, currentdir, assetpath, cplist,
                   cplistfile, executor, readonly):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor
        self.readonly = readonly


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
                    raise tornado.web.Finish()

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
                cpstatus = cpdict['status']

                # FIXME: add in other stuff required by the frontend
                # - signals


                # FIXME: the frontend should load these other things as well
                # into the various elems on the period-search-tools and
                # variability-tools tabs

                # this is the initial dict
                resultdict = {
                    'status':'ok',
                    'message':'found checkplot %s' % self.checkplotfname,
                    'readonly':self.readonly,
                    'result':{
                        'objectid':objectid,
                        'objectinfo':objectinfo,
                        'objectcomments':objectcomments,
                        'varinfo':varinfo,
                        'finderchart':finderchart,
                        'magseries':magseries,
                        # fallback in case objectinfo doesn't have ndet
                        'magseries_ndet':cpdict['magseries']['times'].size,
                        'cpstatus':cpstatus,
                    }
                }

                # now get the other stuff
                for key in ('pdm','aov','bls','gls','sls'):

                    # we return only the first three phased LCs per periodogram
                    if key in cpdict:

                        # get the periodogram for this method
                        periodogram = cpdict[key]['periodogram']

                        # get the phased LC with best period
                        phasedlc0plot = cpdict[key][0]['plot']

                        # get the associated fitinfo for this period if it
                        # exists
                        if ('lcfit' in cpdict[key][0] and
                            isinstance(cpdict[key][0]['lcfit'], dict)):
                            phasedlc0fit = {
                                'method':(
                                    cpdict[key][0]['lcfit']['fittype']
                                    ),
                                'redchisq':(
                                    cpdict[key][0]['lcfit']['fitredchisq']
                                    ),
                                'chisq':(
                                    cpdict[key][0]['lcfit']['fitchisq']
                                    ),
                                'params':(
                                    cpdict[key][0][
                                        'lcfit'
                                    ]['fitinfo']['finalparams'] if
                                    'finalparams' in
                                    cpdict[key][0]['lcfit']['fitinfo'] else None
                                    )
                                }
                        else:
                            phasedlc0fit = None


                        # get the phased LC with 2nd best period
                        phasedlc1plot = cpdict[key][1]['plot']

                        # get the associated fitinfo for this period if it
                        # exists
                        if ('lcfit' in cpdict[key][1] and
                            isinstance(cpdict[key][1]['lcfit'], dict)):
                            phasedlc1fit = {
                                'method':(
                                    cpdict[key][1]['lcfit']['fittype']
                                    ),
                                'redchisq':(
                                    cpdict[key][1]['lcfit']['fitredchisq']
                                    ),
                                'chisq':(
                                    cpdict[key][1]['lcfit']['fitchisq']
                                    ),
                                'params':(
                                    cpdict[key][1][
                                        'lcfit'
                                    ]['fitinfo']['finalparams'] if
                                    'finalparams' in
                                    cpdict[key][1]['lcfit']['fitinfo'] else None
                                    )
                                }
                        else:
                            phasedlc1fit = None


                        # get the phased LC with 3rd best period
                        phasedlc2plot = cpdict[key][2]['plot']

                        # get the associated fitinfo for this period if it
                        # exists
                        if ('lcfit' in cpdict[key][2] and
                            isinstance(cpdict[key][2]['lcfit'], dict)):
                            phasedlc2fit = {
                                'method':(
                                    cpdict[key][2]['lcfit']['fittype']
                                    ),
                                'redchisq':(
                                    cpdict[key][2]['lcfit']['fitredchisq']
                                    ),
                                'chisq':(
                                    cpdict[key][2]['lcfit']['fitchisq']
                                    ),
                                'params':(
                                    cpdict[key][2][
                                        'lcfit'
                                    ]['fitinfo']['finalparams'] if
                                    'finalparams' in
                                    cpdict[key][2]['lcfit']['fitinfo'] else None
                                    )
                                }
                        else:
                            phasedlc2fit = None

                        resultdict['result'][key] = {
                            'nbestperiods':cpdict[key]['nbestperiods'],
                            'periodogram':periodogram,
                            'bestperiod':cpdict[key]['bestperiod'],
                            'phasedlc0':{
                                'plot':phasedlc0plot,
                                'period':float(cpdict[key][0]['period']),
                                'epoch':float(cpdict[key][0]['epoch']),
                                'lcfit':phasedlc0fit,
                            },
                            'phasedlc1':{
                                'plot':phasedlc1plot,
                                'period':float(cpdict[key][1]['period']),
                                'epoch':float(cpdict[key][1]['epoch']),
                                'lcfit':phasedlc1fit,
                            },
                            'phasedlc2':{
                                'plot':phasedlc2plot,
                                'period':float(cpdict[key][2]['period']),
                                'epoch':float(cpdict[key][2]['epoch']),
                                'lcfit':phasedlc2fit,
                            },
                        }

                # return this via JSON
                self.write(resultdict)
                self.finish()

            else:

                LOGGER.error('could not find %s' % self.checkplotfname)

                resultdict = {'status':'error',
                              'message':"This checkplot doesn't exist.",
                              'readonly':self.readonly,
                              'result':None}
                self.write(resultdict)
                self.finish()


        else:

            resultdict = {'status':'error',
                          'message':'No checkplot provided to load.',
                          'readonly':self.readonly,
                          'result':None}

            self.write(resultdict)


    @gen.coroutine
    def post(self, cpfile):
        '''This handles POST requests.

        Also an AJAX endpoint. Updates the persistent checkplot dict using the
        changes from the UI, and then saves it back to disk. This could
        definitely be faster by just loading the checkplot into a server-wide
        shared dict or something.

        '''

        # if self.readonly is set, then don't accept any changes
        # return immediately with a 400
        if self.readonly:

            msg = "checkplotserver is in readonly mode. no updates allowed."
            resultdict = {'status':'error',
                          'message':msg,
                          'readonly':self.readonly,
                          'result':None}
            self.set_status(400)
            self.write(resultdict)
            raise tornado.web.Finish()

        # now try to update the contents
        try:

            self.cpfile = base64.b64decode(cpfile).decode()
            cpcontents = self.get_argument('cpcontents', default=None)
            savetopng = self.get_argument('savetopng', default=None)

            if not self.cpfile or not cpcontents:

                msg = "did not receive a checkplot update payload"
                resultdict = {'status':'error',
                              'message':msg,
                              'readonly':self.readonly,
                              'result':None}
                self.set_status(400)
                self.write(resultdict)
                raise tornado.web.Finish()

            cpcontents = json.loads(cpcontents)

            # the only keys in cpdict that can updated from the UI are from
            # varinfo, objectinfo (objecttags) and comments
            updated = {'varinfo': cpcontents['varinfo'],
                       'objectinfo':cpcontents['objectinfo'],
                       'comments':cpcontents['comments']}

            # we need to reform the self.cpfile so it points to the full path
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
                              'readonly':self.readonly,
                              'result':None}

                self.write(resultdict)
                raise tornado.web.Finish()

            # dispatch the task
            updated = yield self.executor.submit(checkplot_pickle_update,
                                                 cpfpath, updated)

            # continue processing after this is done
            if updated:

                LOGGER.info('updated checkplot %s successfully' % updated)

                resultdict = {'status':'success',
                              'message':'checkplot update successful',
                              'readonly':self.readonly,
                              'result':{'checkplot':updated,
                                        'unixtime':time.time(),
                                        'changes':cpcontents,
                                        'cpfpng': None}}

                # handle a savetopng trigger
                if savetopng:

                    cpfpng = os.path.abspath(cpfpath.replace('.pkl','.png'))

                    pngdone = yield self.executor.submit(
                        checkplot_pickle_to_png,
                        cpfpath, cpfpng
                    )

                    if os.path.exists(cpfpng):
                        resultdict['result']['cpfpng'] = cpfpng
                    else:
                        resultdict['result']['cpfpng'] = 'png making failed'


                self.write(resultdict)
                self.finish()

            else:
                LOGGER.error('could not handle checkplot update for %s: %s' %
                             (self.cpfile, cpcontents))
                msg = "checkplot update failed because of a backend error"
                resultdict = {'status':'error',
                              'message':msg,
                              'readonly':self.readonly,
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
                          'readonly':self.readonly,
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

    def initialize(self, currentdir, assetpath, cplist,
                   cplistfile, executor, readonly):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor
        self.readonly = readonly



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

        # if self.readonly is set, then don't accept any changes
        # return immediately with a 400
        if self.readonly:

            msg = "checkplotserver is in readonly mode. no updates allowed."
            resultdict = {'status':'error',
                          'message':msg,
                          'readonly':self.readonly,
                          'result':None}
            self.set_status(400)
            self.write(resultdict)
            raise tornado.web.Finish()


        objectid = self.get_argument('objectid', None)
        changes = self.get_argument('changes',None)

        # if either of the above is invalid, return nothing
        if not objectid or not changes:

            msg = ("could not parse changes to the checkplot filelist "
                   "from the frontend")
            LOGGER.error(msg)
            resultdict = {'status':'error',
                          'message':msg,
                          'readonly':self.readonly,
                          'result':None}

            self.write(resultdict)
            raise tornado.web.Finish()


        # otherwise, update the checkplot list JSON
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
                      'readonly':self.readonly,
                      'result':{'objectid':objectid,
                                'changes':changes}}

        self.write(resultdict)
        self.finish()



class LCToolHandler(tornado.web.RequestHandler):
    '''This handles dispatching light curve analysis tasks.

    GET requests run the light curve tools specified in the URI with arguments
    as specified in the args to the URI.

    POST requests write the results to the JSON file. The frontend JS object is
    automatically updated by the frontend code.

    '''

    def initialize(self, currentdir, assetpath, cplist,
                   cplistfile, executor, readonly):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor
        self.readonly = readonly


    @gen.coroutine
    def get(self, cpfile):
        '''This handles a GET request.

        The URI structure is:

        /tools/<cpfile>?[args]

        where args are:

        ?lctool=<lctool>&argkey1=argval1&argkey2=argval2&...

        &forcereload=1 <- if this is present, then reload values from original
        checkplot.

        lctool is one of the functions below

        PERIODSEARCH FUNCTIONS
        ----------------------

        psearch-gls: run Lomb-Scargle with given params
        psearch-bls: run BLS with given params
        psearch-pdm: run phase dispersion minimization with given params
        psearch-aov: run analysis-of-variance with given params

        arguments:

        startp=XX
        endp=XX
        magsarefluxes=True|False
        autofreq=True|False
        stepsize=XX


        VARIABILITY FUNCTIONS
        ---------------------

        var-varfeatures: gets the variability from the checkplot or recalculates
                         if it's not present

        var-prewhiten: pre-whitens the light curve with a sinusoidal signal

        var-masksig: masks a given phase location with given width from the
                     light curve


        LIGHT CURVE FUNCTIONS
        ---------------------

        phasedlc-newplot: make phased LC with new provided period/epoch
        lcfit-fourier: fit a Fourier function to the phased LC
        lcfit-spline: fit a spline function to the phased LC
        lcfit-legendre: fit a Legendre polynomial to the phased LC
        lcfit-savgol: fit a Savitsky-Golay polynomial to the phased LC


        FIXME: figure out how to cache the results of these functions
        temporarily and save them back to the checkplot after we click on save
        in the frontend.

        look for a checkplot-blah-blah.pkl-cps-processing file in the same
        place as the usual pickle file. if this exists and is newer than the pkl
        file, load it instead.

        OR

        have a checkplotdict['cpservertemp'] item.

        '''

        if cpfile:

            self.cpfile = xhtml_escape(base64.b64decode(cpfile))

            # see if this plot is in the current project
            if self.cpfile in self.currentproject['checkplots']:

                # make sure this file exists
                cpfpath = os.path.join(
                    os.path.abspath(os.path.dirname(self.cplistfile)),
                    self.cpfile
                )

                # if we can't find the pickle, quit immediately
                if not os.path.exists(cpfpath):

                    msg = "couldn't find checkplot %s" % cpfpath
                    LOGGER.error(msg)
                    resultdict = {'status':'error',
                                  'message':msg,
                                  'readonly':self.readonly,
                                  'result':None}

                    self.write(resultdict)
                    raise tornado.web.Finish()

                ###########################
                # now parse the arguments #
                ###########################

                # check if we have to force-reload
                forcereload = self.get_argument('forcereload',False)
                if forcereload and xhtml_escape(forcereload):
                    forcereload = True if forcereload == 'true' else False

                # get the light curve tool to use
                lctool = self.get_argument('lctool', None)

                # preemptive dict to fill out
                resultdict = {'status':None,
                              'message':None,
                              'readonly':self.readonly,
                              'result':None}

                # check if the lctool arg is provided
                if lctool:

                    lctool = xhtml_escape(lctool)
                    lctoolargs = []
                    lctoolkwargs = {}

                    # check if this lctool is OK and has all the required args
                    if lctool in CPTOOLMAP:

                        try:

                            # all args should have been parsed
                            # successfully. parse the kwargs now
                            for xkwarg, xkwargtype, xkwargdef in zip(
                                    CPTOOLMAP[lctool]['kwargs'],
                                    CPTOOLMAP[lctool]['kwargtypes'],
                                    CPTOOLMAP[lctool]['kwargdefs']
                            ):

                                # get the kwarg
                                wbkwarg = self.get_argument(xkwarg, None)

                                if wbkwarg:
                                    wbkwarg = url_unescape(
                                        xhtml_escape(wbkwarg)
                                    )
                                LOGGER.info('xkwarg = %s, wbkwarg = %s' %
                                            (xkwarg, repr(wbkwarg)))

                                # if it's None, sub with the default
                                if wbkwarg is None:

                                    wbkwarg = xkwargdef

                                # otherwise, cast it to the required type
                                else:

                                    # special handling for lists of floats
                                    if isinstance(xkwargtype, list):
                                        wbkwarg = json.loads(wbkwarg)
                                        wbkwarg = [float(x) for
                                                   x in wbkwarg]
                                    # usual casting for other types
                                    else:
                                        wbkwarg = xkwargtype(wbkwarg)

                                # update the lctools kwarg dict
                                lctoolkwargs.update({xkwarg:wbkwarg})

                        except Exception as e:

                            LOGGER.exception('lctool %s, kwarg %s '
                                             'will not work' %
                                             (lctool, xkwarg))
                            resultdict['status'] = 'error'
                            resultdict['message'] = (
                                'lctool %s, kwarg %s '
                                'will not work' %
                                (lctool, xkwarg)
                            )

                            self.write(resultdict)
                            raise tornado.web.Finish()

                    # if the tool is not in the CPTOOLSMAP
                    else:
                        LOGGER.error('lctool %s, does not exist' % lctool)
                        resultdict['status'] = 'error'
                        resultdict['message'] = (
                        'lctool %s does not exist' % lctool
                        )
                        self.set_status(400)
                        self.write(resultdict)
                        raise tornado.web.Finish()

                # if no lctool arg is provided
                else:

                    LOGGER.error('lctool argument not provided')
                    resultdict['status'] = 'error'
                    resultdict['message'] = (
                    'lctool argument not provided'
                    )
                    self.set_status(400)
                    self.write(resultdict)
                    raise tornado.web.Finish()


                ##############################################
                ## NOW WE'RE READY TO ACTUALLY DO SOMETHING ##
                ##############################################

                LOGGER.info('loading %s...' % cpfpath)

                # this loads the actual checkplot pickle
                cpdict = yield self.executor.submit(
                    _read_checkplot_picklefile, cpfpath
                )

                # we check for the existence of a cpfpath + '-cpserver-temp'
                # file first. this is where we store stuff before we write it
                # back to the actual checkplot.
                tempfpath = cpfpath + '-cpserver-temp'

                # load the temp checkplot if it exists
                if os.path.exists(tempfpath):

                    tempcpdict = yield self.executor.submit(
                    _read_checkplot_picklefile, tempfpath
                    )

                # if it doesn't exist, read the times, mags, errs from the
                # actual checkplot in prep for working on it
                else:

                    tempcpdict = {
                        'objectid':cpdict['objectid'],
                        'magseries':{
                            'times':cpdict['magseries']['times'],
                            'mags':cpdict['magseries']['mags'],
                            'errs':cpdict['magseries']['errs'],
                        }
                    }


                # if we're not forcing a rerun from the original checkplot dict
                if not forcereload:

                    cptimes, cpmags, cperrs = (
                        tempcpdict['magseries']['times'],
                        tempcpdict['magseries']['mags'],
                        tempcpdict['magseries']['errs'],
                    )
                    LOGGER.info('forcereload = False')

                # otherwise, reload the original times, mags, errs
                else:

                    cptimes, cpmags, cperrs = (cpdict['magseries']['times'],
                                               cpdict['magseries']['mags'],
                                               cpdict['magseries']['errs'])
                    LOGGER.info('forcereload = True')


                # collect the args
                for xarg, xargtype in zip(CPTOOLMAP[lctool]['args'],
                                          CPTOOLMAP[lctool]['argtypes']):

                    # handle special args
                    if xarg is None:
                        lctoolargs.append(None)
                    elif xarg == 'times':
                        lctoolargs.append(cptimes)
                    elif xarg == 'mags':
                        lctoolargs.append(cpmags)
                    elif xarg == 'errs':
                        lctoolargs.append(cperrs)

                    # handle other args
                    else:

                        try:

                            # get the arg
                            wbarg = url_unescape(
                                xhtml_escape(
                                    self.get_argument(xarg, None)
                                )
                            )

                            # cast the arg to the required type

                            # special handling for lists
                            if isinstance(xargtype, list):
                                wbarg = json.loads(wbarg)
                                wbarg = [float(x) for x in wbarg]
                            # usual casting for other types
                            else:
                                wbarg = xargtype(wbarg)

                            lctoolargs.append(wbarg)

                        except Exception as e:

                            LOGGER.exception('lctool %s, arg %s '
                                             'will not work' %
                                             (lctool, xarg))
                            resultdict['status'] = 'error'
                            resultdict['message'] = (
                                'lctool %s, arg %s '
                                'will not work' %
                                (lctool, xarg)
                            )
                            self.write(resultdict)
                            raise tornado.web.Finish()


                LOGGER.info(lctool)
                LOGGER.info(lctoolargs)
                LOGGER.info(lctoolkwargs)

                ############################
                ## handle the lctools now ##
                ############################

                # make sure the results aren't there already.
                # if they are and force-reload is not True,
                # just return them instead.
                resloc = CPTOOLMAP[lctool]['resloc']

                # if lctool is a periodogram method
                if lctool in ('psearch-gls',
                              'psearch-bls',
                              'psearch-pdm',
                              'psearch-aov'):

                    lspmethod = resloc[0]

                    # if we can return the results from a previous run
                    if (lspmethod in tempcpdict and
                        isinstance(tempcpdict[lspmethod], dict) and
                        (not forcereload)):

                        # for a periodogram method, we need the
                        # following items
                        bestperiod = (
                            tempcpdict[lspmethod]['bestperiod']
                        )
                        nbestperiods = (
                            tempcpdict[lspmethod]['nbestperiods']
                        )
                        nbestlspvals = (
                            tempcpdict[lspmethod]['nbestlspvals']
                        )
                        periodogram = (
                            tempcpdict[lspmethod]['periodogram']
                        )

                        # get the first phased LC plot and its period
                        # and epoch
                        phasedlc0plot = (
                            tempcpdict[lspmethod][0]['plot']
                        )
                        phasedlc0period = float(
                            tempcpdict[lspmethod][0]['period']
                        )
                        phasedlc0epoch = float(
                            tempcpdict[lspmethod][0]['epoch']
                        )

                        LOGGER.warning(
                            'returning previously unsaved '
                            'results for lctool %s from %s' %
                            (lctool, tempfpath)
                        )

                        #
                        # assemble the returndict
                        #

                        resultdict['status'] = 'warning'
                        resultdict['message'] = (
                            'previous '
                            'unsaved results from %s' %
                            lctool
                        )
                        resultdict['result'] = {
                            lspmethod:{
                                'nbestperiods':nbestperiods,
                                'periodogram':periodogram,
                                'bestperiod':bestperiod,
                                'nbestpeaks':nbestlspvals,
                                'phasedlc0':{
                                    'plot':phasedlc0plot,
                                    'period':phasedlc0period,
                                    'epoch':phasedlc0epoch,
                                }
                            }
                        }

                        self.write(resultdict)
                        self.finish()

                    # otherwise, we have to rerun the periodogram method
                    else:

                        # run the period finder
                        lctoolfunction = CPTOOLMAP[lctool]['func']
                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs,
                        )

                        # get what we need out of funcresults when it
                        # returns
                        nbestperiods = funcresults['nbestperiods']
                        nbestlspvals = funcresults['nbestlspvals']
                        bestperiod = funcresults['bestperiod']

                        # generate the periodogram png
                        pgramres = yield self.executor.submit(
                            _pkl_periodogram,
                            funcresults,
                        )

                        # generate the phased LC for the best period only. we
                        # show this in the frontend along with the
                        # periodogram. the user decides which other peaks they
                        # want a phased LC for, and we save them to the
                        # tempcpdict as required. for now, we'll save only the
                        # best phased LC back to tempcpdict.
                        phasedlcargs = (None,
                                        lspmethod,
                                        0,
                                        cptimes,
                                        cpmags,
                                        cperrs,
                                        bestperiod,
                                        'min',
                                        True,
                                        True,
                                        0.002,
                                        7,
                                        [-0.8,0.8])

                        # here, we set a bestperiodhighlight to distinguish this
                        # plot from the ones existing in the checkplot already
                        phasedlckwargs = {
                            'xliminsetmode':False,
                            'magsarefluxes':lctoolkwargs['magsarefluxes'],
                            'bestperiodhighlight':'#e1f5fe',
                        }

                        # dispatch the plot function
                        phasedlc = yield self.executor.submit(
                            _pkl_phased_magseries_plot,
                            *phasedlcargs,
                            **phasedlckwargs
                        )

                        # save these to the cpservertemp key
                        # save the pickle only if readonly is not true
                        if not self.readonly:

                            tempcpdict[lspmethod] = {
                                'periods':funcresults['periods'],
                                'lspvals':funcresults['lspvals'],
                                'bestperiod':funcresults['bestperiod'],
                                'nbestperiods':funcresults['nbestperiods'],
                                'nbestlspvals':funcresults['nbestlspvals'],
                                'periodogram':pgramres[lspmethod]['periodogram'],
                                0:phasedlc
                            }


                            savekwargs = {
                                'outfile':tempfpath,
                                'protocol':pickle.HIGHEST_PROTOCOL
                            }
                            savedcpf = yield self.executor.submit(
                                _write_checkplot_picklefile,
                                cpdict,
                                **savekwargs
                            )

                            LOGGER.info(
                                'saved temp results from '
                                '%s to checkplot: %s' %
                                (lctool, savedcpf)
                            )

                        else:

                            LOGGER.warning(
                                'not saving temp results to checkplot '
                                ' because readonly = True'
                            )

                        #
                        # assemble the return dict
                        #

                        # the periodogram
                        periodogram = pgramres[lspmethod]['periodogram']

                        # the best period phasedlc plot, period, and
                        # epoch
                        phasedlc0plot = phasedlc['plot']
                        phasedlc0period = float(phasedlc['period'])
                        phasedlc0epoch = float(phasedlc['epoch'])

                        resultdict['status'] = 'success'
                        resultdict['message'] = (
                            'new results for %s' %
                            lctool
                        )
                        resultdict['result'] = {
                            lspmethod:{
                                'nbestperiods':nbestperiods,
                                'periodogram':periodogram,
                                'bestperiod':bestperiod,
                                'nbestpeaks':nbestlspvals,
                                'phasedlc0':{
                                    'plot':phasedlc0plot,
                                    'period':phasedlc0period,
                                    'epoch':phasedlc0epoch,
                                }
                            }
                        }

                        # return to frontend
                        self.write(resultdict)
                        self.finish()


                # if the lctool is a call to the phased LC plot itself
                elif lctool == 'phasedlc-newplot':

                    lspmethod = lctoolargs[1]

                    # if we can return the results from a previous run
                    if ('cpservertemp' in cpdict and
                        lspmethod in cpdict['cpservertemp'] and
                        isinstance(cpdict['cpservertemp'][lspmethod],
                                   dict) and
                        (not forcereload)):

                        # TODO:
                        stuff()

                        # otherwise, we need to dispatch the function
                    else:

                        # run the phased LC function
                        lctoolfunction = CPTOOLMAP[lctool]['func']
                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs,
                        )

                # if the lctool is var-varfeatures
                elif lctool == 'var-varfeatures':

                    key1, key2 = resloc
                    # TODO: stuff


                # if the lctool is var-prewhiten or var-masksig
                elif lctool in ('var-prewhiten','var-masksig'):

                    key1, key2 = resloc
                    # TODO: stuff


                # if the lctool is a lcfit method
                elif lctool in ('lcfit-fourier',
                                'lcfit-spline',
                                'lcfit-legendre',
                                'lcfit-savgol'):

                    key1, key2 = resloc
                    # TODO: stuff

                # otherwise, this is an unrecognized lctool
                else:

                    LOGGER.error('lctool %s, does not exist' % lctool)
                    resultdict['status'] = 'error'
                    resultdict['message'] = (
                        'lctool %s does not exist' % lctool
                    )
                    self.set_status(400)
                    self.write(resultdict)
                    raise tornado.web.Finish()

            # if the cpfile doesn't exist
            else:

                LOGGER.error('could not find %s' % self.cpfile)

                resultdict = {'status':'error',
                              'message':"This checkplot doesn't exist.",
                              'readonly':self.readonly,
                              'result':None}
                self.set_status(404)
                self.write(resultdict)
                raise tornado.web.Finish()

        # if no checkplot was provided to load
        else:

            resultdict = {'status':'error',
                          'message':'No checkplot provided to load.',
                          'readonly':self.readonly,
                          'result':None}
            self.set_status(400)
            self.write(resultdict)
            raise tornado.web.Finish()


    def post(self, cpfile):
        '''This handles a POST request.

        This will save the results of the previous tool run to the checkplot
        file and the JSON filelist.

        This is only called when the user explicitly clicks on the 'permanently
        update checkplot with results' button. If the server is in readonly
        mode, this has no effect.

        '''
