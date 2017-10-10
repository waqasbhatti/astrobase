#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

try:
    from cStringIO import StringIO as strio
except:
    from io import BytesIO as strio

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
    # this is a special tool to remove all unsaved lctool results from the
    # current checkplot pickle
    'lctool-reset':{
        'args':(),
        'argtypes':(),
        'kwargs':(),
        'kwargtypes':(),
        'kwargdefs':(),
        'func':None,
        'resloc':[],
    },
    # this is a special tool to get all unsaved lctool results from the
    # current checkplot pickle
    'lctool-results':{
        'args':(),
        'argtypes':(),
        'kwargs':(),
        'kwargtypes':(),
        'kwargdefs':(),
        'func':None,
        'resloc':[],
    },
    ## PERIOD SEARCH METHODS ##
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
    ## PLOTTING A NEW PHASED LC ##
    'phasedlc-newplot':{
        'args':(None,'lspmethod','periodind',
                'times','mags','errs','varperiod','varepoch'),
        'argtypes':(None, str, int, ndarray, ndarray, ndarray, float, float),
        'kwargs':('xliminsetmode','magsarefluxes',
                  'phasewrap','phasesort',
                  'phasebin','plotxlim[]'),
        'kwargtypes':(bool, bool, bool, bool, float, list),
        'kwargdefs':(False, False, True, True, 0.002, [-0.8,0.8]),
        'func':_pkl_phased_magseries_plot,
        'resloc':[],
    },
    ## VARIABILITY TOOLS ##
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
        'args':('times','mags','errs','whitenperiod', 'whitenparams[]'),
        'argtypes':(ndarray, ndarray, ndarray, float, list),
        'kwargs':('magsarefluxes',),
        'kwargtypes':(bool,),
        'kwargdefs':(False,),
        'func':signals.prewhiten_magseries,
        'resloc':['signals','prewhiten'],
    },
    'var-masksig':{
        'args':('times','mags','errs','signalperiod','signalepoch'),
        'argtypes':(ndarray, ndarray, ndarray, float, float),
        'kwargs':('magsarefluxes','maskphases[]','maskphaselength'),
        'kwargtypes':(bool, list, float),
        'kwargdefs':(False, [0.0,0.5,1.0], 0.1),
        'func':signals.mask_signal,
        'resloc':['signals','mask'],
    },
    ## FITTING FUNCTIONS TO LIGHT CURVES ##
    # this is a special call to just subtract an already fit function from the
    # current light curve
    'lcfit-subtract':{
        'args':('fitmethod', 'periodind'),
        'argtypes':(str, int),
        'kwargs':(),
        'kwargtypes':(),
        'kwargdefs':(),
        'func':None,
        'resloc':[],
    },
    'lcfit-fourier':{
        'args':('times','mags','errs','period'),
        'argtypes':(ndarray, ndarray, ndarray, float),
        'kwargs':('fourierorder','magsarefluxes', 'fourierparams[]'),
        'kwargtypes':(int, bool, list),
        'kwargdefs':(6, False, []),
        'func':lcfit.fourier_fit_magseries,
        'resloc':['fitinfo','fourier'],
    },
    'lcfit-spline':{
        'args':('times','mags','errs','period'),
        'argtypes':(ndarray, ndarray, ndarray, float),
        'kwargs':('maxknots','knotfraction','magsarefluxes'),
        'kwargtypes':(int, float, bool),
        'kwargdefs':(30, 0.01, False),
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
        '''This handles GET requests to the index page.

        TODO: fix this so it loads a modified version of the usual index.html
        template for readonly mode. This should replace all of the text boxes
        with readonly versions.

        TODO: maybe also provide the correct baseurl from the checkplotserver
        options dict, so the frontend JS can just read that off immediately.

        '''

        # generate the project's list of checkplots
        project_checkplots = self.currentproject['checkplots']
        project_checkplotbasenames = [os.path.basename(x)
                                      for x in project_checkplots]
        project_checkplotindices = range(len(project_checkplots))

        # get the sortkey and order
        project_cpsortkey = self.currentproject['sortkey']
        if self.currentproject['sortorder'] == 'asc':
            project_cpsortorder = 'ascending'
        elif self.currentproject['sortorder'] == 'desc':
            project_cpsortorder = 'descending'

        # get the filterkey and condition
        project_cpfilterstatements = self.currentproject['filterstatements']

        self.render('cpindex.html',
                    project_checkplots=project_checkplots,
                    project_cpsortorder=project_cpsortorder,
                    project_cpsortkey=project_cpsortkey,
                    project_cpfilterstatements=project_cpfilterstatements,
                    project_checkplotbasenames=project_checkplotbasenames,
                    project_checkplotindices=project_checkplotindices,
                    project_checkplotfile=self.cplistfile,
                    readonly=self.readonly)



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

        &forcereload=true <- if this is present, then reload values from
        original checkplot.

        &objectid=<objectid>

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

                # get the objectid
                cpobjectid = self.get_argument('objectid',None)

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
                                if xkwargtype is list:
                                    wbkwarg = self.get_arguments(xkwarg)
                                    if len(wbkwarg) > 0:
                                        wbkwarg = [url_unescape(xhtml_escape(x))
                                                   for x in wbkwarg]
                                    else:
                                        wbkwarg = None

                                else:
                                    wbkwarg = self.get_argument(xkwarg, None)
                                    if wbkwarg is not None:
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
                                    if xkwargtype is list:
                                        wbkwarg = [float(x) for x in wbkwarg]

                                    # special handling for booleans
                                    elif xkwargtype is bool:

                                        if wbkwarg == 'false':
                                            wbkwarg = False
                                        elif wbkwarg == 'true':
                                            wbkwarg = True
                                        else:
                                            wbkwarg = xkwargdef

                                    # usual casting for other types
                                    else:

                                        wbkwarg = xkwargtype(wbkwarg)

                                # update the lctools kwarg dict

                                # make sure to remove any [] from the kwargs
                                # this was needed to parse the input query
                                # string correctly
                                if xkwarg.endswith('[]'):
                                    xkwarg = xkwarg.rstrip('[]')

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
                            resultdict['result'] = {'objectid':cpobjectid}

                            self.write(resultdict)
                            raise tornado.web.Finish()

                    # if the tool is not in the CPTOOLSMAP
                    else:
                        LOGGER.error('lctool %s, does not exist' % lctool)
                        resultdict['status'] = 'error'
                        resultdict['message'] = (
                        'lctool %s does not exist' % lctool
                        )
                        resultdict['result'] = {'objectid':cpobjectid}

                        self.write(resultdict)
                        raise tornado.web.Finish()

                # if no lctool arg is provided
                else:

                    LOGGER.error('lctool argument not provided')
                    resultdict['status'] = 'error'
                    resultdict['message'] = (
                    'lctool argument not provided'
                    )
                    resultdict['result'] = {'objectid':cpobjectid}

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

                            if xargtype is list:

                                wbarg = self.get_arguments(xarg)

                            else:

                                wbarg = url_unescape(
                                    xhtml_escape(
                                        self.get_argument(xarg, None)
                                    )
                                )

                            # cast the arg to the required type

                            # special handling for lists
                            if xargtype is list:
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
                            resultdict['result'] = {'objectid':cpobjectid}

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

                # TODO: figure out a way to make the dispatched tasks
                # cancellable. This can probably be done by having a global
                # TOOLQUEUE object that gets imported on initialize(). In this
                # object, we could put in key:vals like so:
                #
                # TOOLQUEUE['lctool-<toolname>-cpfpath'] = (
                #    yield self.executor.submit(blah, *blah_args, **blah_kwargs)
                # )
                #
                # then we probably need some sort of frontend AJAX call that
                # enqueues things and can then cancel stuff from the queue. see
                # stuff we need to figure out:
                # - if the above scheme actually yields so we remain async
                # - if the Future object supports cancellation
                # - if the Future object that isn't resolved actually works


                # get the objectid. we'll send this along with every
                # result. this should handle the case of the current objectid
                # not being the same as the objectid being looked at by the
                # user. in effect, this will allow the user to launch a
                # long-running process and come back to it later since the
                # frontend will load the older results when they are complete.
                objectid = cpdict['objectid']

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
                            'objectid':objectid,
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
                        # returns. we get the first three peaks/periods
                        nbestperiods = funcresults['nbestperiods'][:3]
                        nbestlspvals = funcresults['nbestlspvals'][:3]
                        bestperiod = funcresults['bestperiod']

                        # generate the periodogram png
                        pgramres = yield self.executor.submit(
                            _pkl_periodogram,
                            funcresults,
                        )

                        # generate the phased LCs. we show these in the frontend
                        # along with the periodogram.
                        phasedlcargs0 = (None,
                                        lspmethod,
                                        -1,
                                        cptimes,
                                        cpmags,
                                        cperrs,
                                        nbestperiods[0],
                                        'min')
                        phasedlcargs1 = (None,
                                        lspmethod,
                                        -1,
                                        cptimes,
                                        cpmags,
                                        cperrs,
                                        nbestperiods[1],
                                        'min')
                        phasedlcargs2 = (None,
                                        lspmethod,
                                        -1,
                                        cptimes,
                                        cpmags,
                                        cperrs,
                                        nbestperiods[2],
                                        'min')

                        # here, we set a bestperiodhighlight to distinguish this
                        # plot from the ones existing in the checkplot already
                        phasedlckwargs = {
                            'xliminsetmode':False,
                            'magsarefluxes':lctoolkwargs['magsarefluxes'],
                            'bestperiodhighlight':'#defa75',
                        }

                        # dispatch the plot functions
                        phasedlc0 = yield self.executor.submit(
                            _pkl_phased_magseries_plot,
                            *phasedlcargs0,
                            **phasedlckwargs
                        )

                        phasedlc1 = yield self.executor.submit(
                            _pkl_phased_magseries_plot,
                            *phasedlcargs1,
                            **phasedlckwargs
                        )

                        phasedlc2 = yield self.executor.submit(
                            _pkl_phased_magseries_plot,
                            *phasedlcargs2,
                            **phasedlckwargs
                        )


                        # save these to the tempcpdict
                        # save the pickle only if readonly is not true
                        if not self.readonly:

                            tempcpdict[lspmethod] = {
                                'periods':funcresults['periods'],
                                'lspvals':funcresults['lspvals'],
                                'bestperiod':funcresults['bestperiod'],
                                'nbestperiods':funcresults['nbestperiods'],
                                'nbestlspvals':funcresults['nbestlspvals'],
                                'periodogram':pgramres[lspmethod]['periodogram'],
                                0:phasedlc0,
                                1:phasedlc1,
                                2:phasedlc2,
                            }


                            savekwargs = {
                                'outfile':tempfpath,
                                'protocol':pickle.HIGHEST_PROTOCOL
                            }
                            savedcpf = yield self.executor.submit(
                                _write_checkplot_picklefile,
                                tempcpdict,
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

                        # phasedlc plot, period, and epoch for best 3 peaks
                        phasedlc0plot = phasedlc0['plot']
                        phasedlc0period = float(phasedlc0['period'])
                        phasedlc0epoch = float(phasedlc0['epoch'])

                        phasedlc1plot = phasedlc1['plot']
                        phasedlc1period = float(phasedlc1['period'])
                        phasedlc1epoch = float(phasedlc1['epoch'])

                        phasedlc2plot = phasedlc2['plot']
                        phasedlc2period = float(phasedlc2['period'])
                        phasedlc2epoch = float(phasedlc2['epoch'])

                        resultdict['status'] = 'success'
                        resultdict['message'] = (
                            'new results for %s' %
                            lctool
                        )
                        resultdict['result'] = {
                            'objectid':objectid,
                            lspmethod:{
                                'nbestperiods':nbestperiods,
                                'periodogram':periodogram,
                                'bestperiod':bestperiod,
                                'nbestpeaks':nbestlspvals,
                                'phasedlc0':{
                                    'plot':phasedlc0plot,
                                    'period':phasedlc0period,
                                    'epoch':phasedlc0epoch,
                                },
                                'phasedlc1':{
                                    'plot':phasedlc1plot,
                                    'period':phasedlc1period,
                                    'epoch':phasedlc1epoch,
                                },
                                'phasedlc2':{
                                    'plot':phasedlc2plot,
                                    'period':phasedlc2period,
                                    'epoch':phasedlc2epoch,
                                }
                            }
                        }

                        # return to frontend
                        self.write(resultdict)
                        self.finish()


                # if the lctool is a call to the phased LC plot itself
                # this requires lots of parameters
                # these should all be present in the frontend
                elif lctool == 'phasedlc-newplot':

                    lspmethod = lctoolargs[1]
                    periodind = lctoolargs[2]

                    # if we can return the results from a previous run
                    if (not forcereload and lspmethod in tempcpdict and
                        isinstance(tempcpdict[lspmethod], dict) and
                        periodind in tempcpdict[lspmethod] and
                        isinstance(tempcpdict[lspmethod][periodind], dict)):

                        # we get phased LC at periodind from a previous run
                        phasedlc = tempcpdict[lspmethod][periodind]

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
                        retkey = 'phasedlc%s' % periodind
                        resultdict['result'] = {
                            'objectid':objectid,
                            lspmethod:{
                                retkey:phasedlc
                            }
                        }

                        self.write(resultdict)
                        self.finish()

                    # otherwise, we need to dispatch the function
                    else:

                        # add the highlight to distinguish this plot from usual
                        # checkplot plots
                        # full disclosure: http://c0ffee.surge.sh/
                        lctoolkwargs['bestperiodhighlight'] = '#defa75'

                        # set the input periodind to -1 to make sure we still
                        # have the highlight on the plot. we use the correct
                        # periodind when returning
                        lctoolargs[2] = -1

                        # run the phased LC function
                        lctoolfunction = CPTOOLMAP[lctool]['func']
                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs,
                        )

                        # save these to the tempcpdict
                        # save the pickle only if readonly is not true
                        if not self.readonly:

                            if (lspmethod in tempcpdict and
                                isinstance(tempcpdict[lspmethod], dict)):

                                if periodind in tempcpdict[lspmethod]:

                                    tempcpdict[lspmethod][periodind] = (
                                        funcresults
                                    )

                                else:

                                    tempcpdict[lspmethod].update(
                                        {periodind: funcresults}
                                    )

                            else:

                                tempcpdict[lspmethod] = {periodind: funcresults}


                            savekwargs = {
                                'outfile':tempfpath,
                                'protocol':pickle.HIGHEST_PROTOCOL
                            }
                            savedcpf = yield self.executor.submit(
                                _write_checkplot_picklefile,
                                tempcpdict,
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
                        resultdict['status'] = 'success'
                        resultdict['message'] = (
                            'new results for %s' %
                            lctool
                        )
                        retkey = 'phasedlc%s' % periodind
                        resultdict['result'] = {
                            'objectid':objectid,
                            lspmethod:{
                                retkey:funcresults
                            }
                        }

                        self.write(resultdict)
                        self.finish()


                # if the lctool is var-varfeatures
                elif lctool == 'var-varfeatures':

                    # see if we can return results from a previous iteration of
                    # this tool
                    if (not forcereload and
                        'varinfo' in tempcpdict and
                        isinstance(tempcpdict['varinfo'], dict) and
                        'varfeatures' in tempcpdict['varinfo'] and
                        isinstance(tempcpdict['varinfo']['varfeatures'], dict)):

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
                            'objectid':objectid,
                            'varinfo': {
                                'varfeatures': (
                                    tempcpdict['varinfo']['varfeatures']
                                )
                            }
                        }

                        self.write(resultdict)
                        self.finish()

                    # otherwise, we need to dispatch the function
                    else:

                        lctoolfunction = CPTOOLMAP[lctool]['func']
                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs,
                        )

                        # save these to the tempcpdict
                        # save the pickle only if readonly is not true
                        if not self.readonly:

                            if ('varinfo' in tempcpdict and
                                isinstance(tempcpdict['varinfo'], dict)):

                                if 'varfeatures' in tempcpdict['varinfo']:

                                    tempcpdict['varinfo']['varfeatures'] = (
                                        funcresults
                                    )

                                else:

                                    tempcpdict['varinfo'].update(
                                        {'varfeatures': funcresults}
                                    )

                            else:

                                tempcpdict['varinfo'] = {'varfeatures':
                                                         funcresults}


                            savekwargs = {
                                'outfile':tempfpath,
                                'protocol':pickle.HIGHEST_PROTOCOL
                            }
                            savedcpf = yield self.executor.submit(
                                _write_checkplot_picklefile,
                                tempcpdict,
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
                        resultdict['status'] = 'success'
                        resultdict['message'] = (
                            'new results for %s' %
                            lctool
                        )
                        resultdict['result'] = {
                            'objectid':objectid,
                            'varinfo':{
                                'varfeatures':funcresults
                            }
                        }

                        self.write(resultdict)
                        self.finish()


                # if the lctool is var-prewhiten or var-masksig
                elif lctool in ('var-prewhiten','var-masksig'):

                    key1, key2 = resloc

                    # see if we can return results from a previous iteration of
                    # this tool
                    if (not forcereload and
                        key1 in tempcpdict and
                        isinstance(tempcpdict[key1], dict) and
                        key2 in tempcpdict[key1] and
                        isinstance(tempcpdict[key1][key2], dict)):

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
                            'objectid':objectid,
                            key1: {
                                key2: (
                                    tempcpdict[key1][key2]
                                )
                            }
                        }

                        self.write(resultdict)
                        self.finish()

                    # otherwise, we need to dispatch the function
                    else:

                        lctoolfunction = CPTOOLMAP[lctool]['func']

                        # send in a stringio object for the fitplot kwarg
                        lctoolkwargs['plotfit'] = strio()
                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs,
                        )

                        # we turn the returned fitplotfile fd into a base64
                        # encoded string after reading it
                        fitfd = funcresults['fitplotfile']
                        fitfd.seek(0)
                        fitbin = fitfd.read()
                        fitb64 = base64.b64encode(fitbin)
                        fitfd.close()
                        funcresults['fitplotfile'] = fitb64

                        # save these to the tempcpdict
                        # save the pickle only if readonly is not true
                        if not self.readonly:

                            if (key1 in tempcpdict and
                                isinstance(tempcpdict[key1], dict)):

                                if key2 in tempcpdict[key1]:

                                    tempcpdict[key1][key2] = (
                                        funcresults
                                    )

                                else:

                                    tempcpdict[key1].update(
                                        {key2: funcresults}
                                    )

                            else:

                                tempcpdict[key1] = {key2: funcresults}


                            savekwargs = {
                                'outfile':tempfpath,
                                'protocol':pickle.HIGHEST_PROTOCOL
                            }
                            savedcpf = yield self.executor.submit(
                                _write_checkplot_picklefile,
                                tempcpdict,
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
                        # for this operation, we'll return:
                        # - fitplotfile
                        fitreturndict = {'fitplotfile':fitb64}

                        resultdict['status'] = 'success'
                        resultdict['message'] = (
                            'new results for %s' %
                            lctool
                        )
                        resultdict['result'] = {
                            'objectid':objectid,
                            key1:{
                                key2:fitreturndict
                            }
                        }

                        self.write(resultdict)
                        self.finish()


                # if the lctool is a lcfit method
                elif lctool in ('lcfit-fourier',
                                'lcfit-spline',
                                'lcfit-legendre',
                                'lcfit-savgol'):

                    key1, key2 = resloc

                    # see if we can return results from a previous iteration of
                    # this tool
                    if (not forcereload and
                        key1 in tempcpdict and
                        isinstance(tempcpdict[key1], dict) and
                        key2 in tempcpdict[key1] and
                        isinstance(tempcpdict[key1][key2], dict)):

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

                        # these are the full results
                        phasedfitlc = tempcpdict[key1][key2]

                        # we only want a few things from them
                        fitresults = {
                            'method':phasedfitlc['lcfit']['fittype'],
                            'chisq':phasedfitlc['lcfit']['fitchisq'],
                            'redchisq':phasedfitlc['lcfit']['fitredchisq'],
                            'period':phasedfitlc['period'],
                            'epoch':phasedfitlc['epoch'],
                            'plot':phasedfitlc['plot'],
                        }

                        # add fitparams if there are any
                        if ('finalparams' in phasedfitlc['lcfit']['fitinfo'] and
                            phasedfitlc['lcfit']['fitinfo']['finalparams']
                            is not None):
                            fitresults['fitparams'] = (
                                phasedfitlc['lcfit']['fitinfo']['finalparams']
                            )

                        # this is the final result object
                        resultdict['result'] = {
                            'objectid':objectid,
                            key1: {
                                key2: (
                                    fitresults
                                )
                            }
                        }

                        self.write(resultdict)
                        self.finish()

                    # otherwise, we need to dispatch the function
                    else:

                        lctoolfunction = CPTOOLMAP[lctool]['func']

                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs,
                        )

                        # now that we have the fit results, generate a fitplot.
                        # these args are for the special fitplot mode of
                        # _pkl_phased_magseries_plot
                        phasedlcargs = (None,
                                        'lcfit',
                                        -1,
                                        cptimes,
                                        cpmags,
                                        cperrs,
                                        lctoolargs[3], # this is the fit period
                                        'min')

                        # here, we set a bestperiodhighlight to distinguish this
                        # plot from the ones existing in the checkplot already
                        # also add the overplotfit information
                        phasedlckwargs = {
                            'xliminsetmode':False,
                            'magsarefluxes':lctoolkwargs['magsarefluxes'],
                            'bestperiodhighlight':'#defa75',
                            'overplotfit':funcresults
                        }

                        # dispatch the plot function
                        phasedlc = yield self.executor.submit(
                            _pkl_phased_magseries_plot,
                            *phasedlcargs,
                            **phasedlckwargs
                        )

                        # save these to the tempcpdict
                        # save the pickle only if readonly is not true
                        if not self.readonly:

                            if (key1 in tempcpdict and
                                isinstance(tempcpdict[key1], dict)):

                                if key2 in tempcpdict[key1]:

                                    tempcpdict[key1][key2] = (
                                        phasedlc
                                    )

                                else:

                                    tempcpdict[key1].update(
                                        {key2: phasedlc}
                                    )

                            else:

                                tempcpdict[key1] = {key2: phasedlc}


                            savekwargs = {
                                'outfile':tempfpath,
                                'protocol':pickle.HIGHEST_PROTOCOL
                            }
                            savedcpf = yield self.executor.submit(
                                _write_checkplot_picklefile,
                                tempcpdict,
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
                        fitresults = {
                            'method':phasedlc['lcfit']['fittype'],
                            'chisq':phasedlc['lcfit']['fitchisq'],
                            'redchisq':phasedlc['lcfit']['fitredchisq'],
                            'period':phasedlc['period'],
                            'epoch':phasedlc['epoch'],
                            'plot':phasedlc['plot'],
                        }

                        # add fitparams if there are any
                        if ('finalparams' in funcresults['fitinfo'] and
                            funcresults['fitinfo']['finalparams'] is not None):
                            fitresults['fitparams'] = (
                                funcresults['fitinfo']['finalparams']
                            )


                        resultdict['status'] = 'success'
                        resultdict['message'] = (
                            'new results for %s' %
                            lctool
                        )
                        resultdict['result'] = {
                            'objectid':objectid,
                            key1:{
                                key2:fitresults
                            }
                        }

                        self.write(resultdict)
                        self.finish()


                # if this is the special lcfit subtract tool
                elif lctool == 'lcfit-subtract':

                    fitmethod, periodind = lctoolargs

                    # find the fit requested

                    # subtract it from the cptimes, cpmags, cperrs

                    # if not readonly, write back to cptimes, cpmags, cperrs

                    # make a new phasedlc plot for the current periodind using
                    # these new cptimes, cpmags, cperrs

                    # return this plot


                # if this is the special full reset tool
                elif lctool == 'lctool-reset':

                    if os.path.exists(tempfpath):
                        os.remove(tempfpath)
                        LOGWARNING('reset all LC tool results '
                                   'for %s by removing %s' %
                                   (tempfpath, cpfpath))
                        resultdict['status'] = 'success'
                    else:
                        resultdict['status'] = 'error'
                        LOGWARNING('tried to reset LC tool results for %s, '
                                   'but temp checkplot result pickle %s '
                                   'does not exist' %
                                   (tempfpath, cpfpath))

                    resultdict['message'] = (
                        'all unsynced results for this object have been purged'
                    )
                    resultdict['result'] = {'objectid':cpobjectid}

                    self.write(resultdict)
                    self.finish()


                # if this is the special load results tool
                elif lctool == 'lctool-results':

                    target = self.get_argument('resultsfor',None)

                    if target is not None:

                        target = xhtml_escape(target)

                        # get rid of invalid targets
                        if (target not in CPTOOL or
                            target == 'lctool-reset' or
                            target == 'lctool-results' or
                            target == 'phasedlc-newplot' or
                            target == 'lcfit-subtract'):

                            LOGGER.error("can't get results for %s" % target)
                            resultdict['status'] = 'error'
                            resultdict['message'] = (
                                "can't get results for %s" % target
                            )
                            resultdict['result'] = {'objectid':cpobjectid}

                            self.write(resultdict)
                            raise tornado.web.Finish()

                        # if we're good to go, get the target location
                        targetloc = CPTOOLS[target]['resloc']

                        # first, search the cptempdict for this target
                        # if found, return it

                        # second, search the actual cpdict for this target
                        # if found, return it

                    # otherwise, we're being asked for everything
                    # return the whole
                    else:

                        pass


                # otherwise, this is an unrecognized lctool
                else:

                    LOGGER.error('lctool %s, does not exist' % lctool)
                    resultdict['status'] = 'error'
                    resultdict['message'] = (
                        'lctool %s does not exist' % lctool
                    )
                    resultdict['result'] = {'objectid':cpobjectid}

                    self.write(resultdict)
                    raise tornado.web.Finish()

            # if the cpfile doesn't exist
            else:

                LOGGER.error('could not find %s' % self.cpfile)

                resultdict = {'status':'error',
                              'message':"This checkplot doesn't exist.",
                              'readonly':self.readonly,
                              'result':None}

                self.write(resultdict)
                raise tornado.web.Finish()

        # if no checkplot was provided to load
        else:

            resultdict = {'status':'error',
                          'message':'No checkplot provided to load.',
                          'readonly':self.readonly,
                          'result':None}

            self.write(resultdict)
            raise tornado.web.Finish()



    def post(self, cpfile):
        '''This handles a POST request.

        This will save the results of the previous tool run to the checkplot
        file and the JSON filelist.

        This is only called when the user explicitly clicks on the 'permanently
        update checkplot with results' button. If the server is in readonly
        mode, this has no effect.

        This will copy everything from the '.pkl-cpserver-temp' file to the
        actual checkplot pickle and then remove that file.

        '''
