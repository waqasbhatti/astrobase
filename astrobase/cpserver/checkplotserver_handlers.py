#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# checkplotserver_handlers.py - Waqas Bhatti (wbhatti@astro.princeton.edu) -
#                               Jan 2017

'''
These are Tornado handlers for serving checkplots and operating on them.

'''

####################
## SYSTEM IMPORTS ##
####################

import os
import os.path
import logging

import numpy as np
from numpy import ndarray

######################################
## CUSTOM JSON ENCODER FOR FRONTEND ##
######################################

# we need this to send objects with the following types to the frontend:
# - bytes
# - ndarray
import json


class FrontendEncoder(json.JSONEncoder):
    '''This overrides Python's default JSONEncoder so we can serialize custom
    objects.

    '''

    def default(self, obj):
        '''Overrides the default serializer for `JSONEncoder`.

        This can serialize the following objects in addition to what
        `JSONEncoder` can already do.

        - `np.array`
        - `bytes`
        - `complex`
        - `np.float64` and other `np.dtype` objects

        Parameters
        ----------

        obj : object
            A Python object to serialize to JSON.

        Returns
        -------

        str
            A JSON encoded representation of the input object.

        '''

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode()
        elif isinstance(obj, complex):
            return (obj.real, obj.imag)
        elif (isinstance(obj, (float, np.float64, np.float_)) and
              not np.isfinite(obj)):
            return None
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
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

###################
## LOCAL IMPORTS ##
###################

from ..varclass import varfeatures
from .. import lcfit
from ..varbase import signals
from ..checkplot.pkl_utils import _pkl_phased_magseries_plot

from ..periodbase import zgls
from ..periodbase import saov
from ..periodbase import smav
from ..periodbase import spdm
from ..periodbase import kbls
from ..periodbase import macf


############
## CONFIG ##
############

PFMETHODS = ['gls','pdm','acf','aov','mav','bls','win']


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
        'kwargs':('startp','endp','magsarefluxes',
                  'autofreq','stepsize','nbestpeaks',
                  'sigclip[]', 'lctimefilters',
                  'lcmagfilters','periodepsilon'),
        'kwargtypes':(float, float, bool,
                      bool, float, int,
                      list, str,
                      str, float),
        'kwargdefs':(None, None, False,
                     True, 1.0e-4, 10,
                     None, None,
                     None, 0.1),
        'func':zgls.pgen_lsp,
        'resloc':['gls'],
    },
    'psearch-bls':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes',
                  'autofreq','stepsize','nbestpeaks',
                  'sigclip[]','lctimefilters','lcmagfilters',
                  'periodepsilon','mintransitduration','maxtransitduration'),
        'kwargtypes':(float, float, bool,
                      bool, float, int,
                      list, str, str,
                      float, float, float),
        'kwargdefs':(0.1, 100.0, False,
                     True, 1.0e-4, 10,
                     None, None, None,
                     0.1, 0.01, 0.08),
        'func':kbls.bls_parallel_pfind,
        'resloc':['bls'],
    },
    'psearch-pdm':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes',
                  'autofreq','stepsize','nbestpeaks',
                  'sigclip[]','lctimefilters','lcmagfilters',
                  'periodepsilon','phasebinsize','mindetperbin'),
        'kwargtypes':(float, float, bool,
                      bool, float, int,
                      list, str, str,
                      float, float, int),
        'kwargdefs':(None, None, False,
                     True, 1.0e-4, 10,
                     None, None, None,
                     0.1, 0.05, 9),
        'func':spdm.stellingwerf_pdm,
        'resloc':['pdm'],
    },
    'psearch-aov':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes',
                  'autofreq','stepsize','nbestpeaks',
                  'sigclip[]','lctimefilters','lcmagfilters',
                  'periodepsilon','phasebinsize','mindetperbin'),
        'kwargtypes':(float, float, bool,
                      bool, float, int,
                      list, str, str,
                      float, float, int),
        'kwargdefs':(None, None, False,
                     True, 1.0e-4, 10,
                     None, None, None,
                     0.1, 0.05, 9),
        'func':saov.aov_periodfind,
        'resloc':['aov'],
    },
    'psearch-mav':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes',
                  'autofreq','stepsize','nbestpeaks',
                  'sigclip[]','lctimefilters','lcmagfilters',
                  'periodepsilon','nharmonics'),
        'kwargtypes':(float, float, bool,
                      bool, float, int,
                      list, str, str,
                      float, int),
        'kwargdefs':(None, None, False,
                     True, 1.0e-4, 10,
                     None, None, None,
                     0.1, 6),
        'func':smav.aovhm_periodfind,
        'resloc':['mav'],
    },
    'psearch-acf':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes',
                  'autofreq','stepsize','smoothacf',
                  'sigclip[]','lctimefilters', 'lcmagfilters',
                  'periodepsilon', 'fillgaps'),
        'kwargtypes':(float, float, bool,
                      bool, float, int,
                      list, str, str,
                      float, float),
        'kwargdefs':(None, None, False,
                     True, 1.0e-4, 721,
                     None, None, None,
                     0.1, 0.0),
        'func':macf.macf_period_find,
        'resloc':['acf'],
    },
    'psearch-win':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray, ndarray, ndarray),
        'kwargs':('startp','endp','magsarefluxes',
                  'autofreq','stepsize','nbestpeaks',
                  'sigclip[]','lctimefilters','lcmagfilters',
                  'periodepsilon'),
        'kwargtypes':(float, float, bool,
                      bool, float, int,
                      list, str, str,
                      float),
        'kwargdefs':(None, None, False,
                     True, 1.0e-4, 10,
                     None, None, None,
                     0.1),
        'func':zgls.specwindow_lsp,
        'resloc':['win'],
    },
    ## PLOTTING A NEW PHASED LC ##
    'phasedlc-newplot':{
        'args':(None,'lspmethod','periodind',
                'times','mags','errs','varperiod','varepoch'),
        'argtypes':(None, str, int, ndarray, ndarray, ndarray, float, float),
        'kwargs':('xliminsetmode','magsarefluxes',
                  'phasewrap','phasesort',
                  'phasebin','plotxlim[]',
                  'sigclip[]','lctimefilters','lcmagfilters'),
        'kwargtypes':(bool, bool, bool, bool, float, list, list, str, str),
        'kwargdefs':(False, False, True, True, 0.002, [-0.8,0.8],
                     None, None, None),
        'func':_pkl_phased_magseries_plot,
        'resloc':[],
    },
    # FIXME: add sigclip, lctimefilters, and lcmagfilters for all of these
    ## VARIABILITY TOOLS ##
    'var-varfeatures':{
        'args':('times','mags','errs'),
        'argtypes':(ndarray,ndarray,ndarray),
        'kwargs':(),
        'kwargtypes':(),
        'kwargdefs':(),
        'func':varfeatures.all_nonperiodic_features,
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
    # FIXME: add sigclip, lctimefilters, and lcmagfilters for all of these
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
                   cplistfile, executor, readonly, baseurl):
        '''
        handles initial setup.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor
        self.readonly = readonly
        self.baseurl = baseurl

    def get(self):
        '''This handles GET requests to the index page.

        TODO: provide the correct baseurl from the checkplotserver options dict,
        so the frontend JS can just read that off immediately.

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
                    readonly=self.readonly,
                    baseurl=self.baseurl)
