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
import base64
import logging
import time as utime
from io import BytesIO as StrIO

import numpy as np
from numpy import ndarray

import json
from .checkplotserver_handlers import FrontendEncoder

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
from tornado.escape import xhtml_escape, url_unescape
from tornado import gen

###################
## LOCAL IMPORTS ##
###################

from ..checkplot.pkl_io import (
    _read_checkplot_picklefile,
)
from ..checkplot.pkl_png import checkplot_pickle_to_png
from ..checkplot.pkl import checkplot_pickle_update

from .checkplotserver_handlers import PFMETHODS


class CheckplotHandler(tornado.web.RequestHandler):
    '''This handles loading and saving checkplots.

    This includes GET requests to get to and load a specific checkplot pickle
    file and POST requests to save the checkplot changes back to the file.

    '''

    def initialize(self, currentdir, assetpath, cplist,
                   cplistfile, executor, readonly):
        '''
        This handles initial setup of this `RequestHandler`.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor
        self.readonly = readonly

    @gen.coroutine
    def get(self, checkplotfname):
        '''This handles GET requests to serve a specific checkplot pickle.

        This is an AJAX endpoint; returns JSON that gets converted by the
        frontend into things to render.

        '''

        if checkplotfname:

            # do the usual safing
            self.checkplotfname = xhtml_escape(
                base64.b64decode(url_unescape(checkplotfname))
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

                LOGGER.info('loaded %s' % cpfpath)

                # break out the initial info
                objectid = cpdict['objectid']
                objectinfo = cpdict['objectinfo']
                varinfo = cpdict['varinfo']

                if 'pfmethods' in cpdict:
                    pfmethods = cpdict['pfmethods']
                else:
                    pfmethods = []
                    for pfm in PFMETHODS:
                        if pfm in cpdict:
                            pfmethods.append(pfm)

                # handle neighbors for this object
                neighbors = []

                if ('neighbors' in cpdict and
                    cpdict['neighbors'] is not None and
                    len(cpdict['neighbors'])) > 0:

                    nbrlist = cpdict['neighbors']

                    # get each neighbor, its info, and its phased LCs
                    for nbr in nbrlist:

                        if 'magdiffs' in nbr:
                            nbrmagdiffs = nbr['magdiffs']
                        else:
                            nbrmagdiffs = None

                        if 'colordiffs' in nbr:
                            nbrcolordiffs = nbr['colordiffs']
                        else:
                            nbrcolordiffs = None

                        thisnbrdict = {
                            'objectid':nbr['objectid'],
                            'objectinfo':{
                                'ra':nbr['ra'],
                                'decl':nbr['decl'],
                                'xpix':nbr['xpix'],
                                'ypix':nbr['ypix'],
                                'distarcsec':nbr['dist'],
                                'magdiffs':nbrmagdiffs,
                                'colordiffs':nbrcolordiffs
                            }
                        }

                        try:

                            nbr_magseries = nbr['magseries']['plot']
                            thisnbrdict['magseries'] = nbr_magseries

                        except Exception:

                            LOGGER.error(
                                "could not load magseries plot for "
                                "neighbor %s for object %s"
                                % (nbr['objectid'],
                                   cpdict['objectid'])
                            )

                        try:

                            for pfm in pfmethods:
                                if pfm in nbr:
                                    thisnbrdict[pfm] = {
                                        'plot':nbr[pfm][0]['plot'],
                                        'period':nbr[pfm][0]['period'],
                                        'epoch':nbr[pfm][0]['epoch']
                                    }

                        except Exception:

                            LOGGER.error(
                                "could not load phased LC plots for "
                                "neighbor %s for object %s"
                                % (nbr['objectid'],
                                   cpdict['objectid'])
                            )

                        neighbors.append(thisnbrdict)

                # load object comments
                if 'comments' in cpdict:
                    objectcomments = cpdict['comments']
                else:
                    objectcomments = None

                # load the xmatch results, if any
                if 'xmatch' in cpdict:

                    objectxmatch = cpdict['xmatch']

                    # get rid of those pesky nans
                    for xmcat in objectxmatch:
                        if isinstance(objectxmatch[xmcat]['info'], dict):
                            xminfo = objectxmatch[xmcat]['info']
                            for xmek in xminfo:
                                if (isinstance(xminfo[xmek], float) and
                                    (not np.isfinite(xminfo[xmek]))):
                                    xminfo[xmek] = None

                else:
                    objectxmatch = None

                # load the colormagdiagram object
                if 'colormagdiagram' in cpdict:
                    colormagdiagram = cpdict['colormagdiagram']
                else:
                    colormagdiagram = None

                # these are base64 which can be provided directly to JS to
                # generate images (neat!)

                if 'finderchart' in cpdict:
                    finderchart = cpdict['finderchart']
                else:
                    finderchart = None

                if ('magseries' in cpdict and
                    isinstance(cpdict['magseries'], dict) and
                    'plot' in cpdict['magseries']):
                    magseries = cpdict['magseries']['plot']
                    time0 = cpdict['magseries']['times'].min()
                    magseries_ndet = cpdict['magseries']['times'].size
                else:
                    magseries = None
                    time0 = 0.0
                    magseries_ndet = 0
                    LOGGER.warning(
                        "no 'magseries' key present in this "
                        "checkplot, some plots may be broken..."
                    )

                if 'status' in cpdict:
                    cpstatus = cpdict['status']
                else:
                    cpstatus = 'unknown, possibly incomplete checkplot'

                # load the uifilters if present
                if 'uifilters' in cpdict:
                    uifilters = cpdict['uifilters']
                else:
                    uifilters = {'psearch_magfilters':None,
                                 'psearch_sigclip':None,
                                 'psearch_timefilters':None}

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
                        'time0':'%.3f' % time0,
                        'objectid':objectid,
                        'objectinfo':objectinfo,
                        'colormagdiagram':colormagdiagram,
                        'objectcomments':objectcomments,
                        'varinfo':varinfo,
                        'uifilters':uifilters,
                        'neighbors':neighbors,
                        'xmatch':objectxmatch,
                        'finderchart':finderchart,
                        'magseries':magseries,
                        # fallback in case objectinfo doesn't have ndet
                        'magseries_ndet':magseries_ndet,
                        'cpstatus':cpstatus,
                        'pfmethods':pfmethods
                    }
                }

                # make sure to replace nans with Nones. frontend JS absolutely
                # hates NaNs and for some reason, the JSON encoder defined at
                # the top of this file doesn't deal with them even though it
                # should
                for key in resultdict['result']['objectinfo']:

                    if (isinstance(resultdict['result']['objectinfo'][key],
                                   (float, np.float64, np.float_)) and
                        (not np.isfinite(resultdict['result'][
                            'objectinfo'
                        ][key]))):
                        resultdict['result']['objectinfo'][key] = None

                    elif (isinstance(resultdict['result']['objectinfo'][key],
                                     ndarray)):

                        thisval = resultdict['result']['objectinfo'][key]
                        thisval = thisval.tolist()
                        for i, v in enumerate(thisval):
                            if (isinstance(v,(float, np.float64, np.float_)) and
                                (not(np.isfinite(v)))):
                                thisval[i] = None
                        resultdict['result']['objectinfo'][key] = thisval

                # remove nans from varinfo itself
                for key in resultdict['result']['varinfo']:

                    if (isinstance(
                            resultdict['result']['varinfo'][key],
                            (float, np.float64, np.float_)) and
                        (not np.isfinite(
                            resultdict['result']['varinfo'][key]
                        ))):
                        resultdict['result']['varinfo'][key] = None

                    elif (isinstance(
                            resultdict['result']['varinfo'][key],
                            ndarray)):

                        thisval = (
                            resultdict['result']['varinfo'][key]
                        )
                        thisval = thisval.tolist()
                        for i, v in enumerate(thisval):
                            if (isinstance(v,(float, np.float64, np.float_)) and
                                (not(np.isfinite(v)))):
                                thisval[i] = None
                        resultdict['result']['varinfo'][key] = (
                            thisval
                        )

                # remove nans from varinfo['features']
                if ('features' in resultdict['result']['varinfo'] and
                    isinstance(resultdict['result']['varinfo']['features'],
                               dict)):

                    for key in resultdict['result']['varinfo']['features']:

                        if (isinstance(
                                resultdict[
                                    'result'
                                ]['varinfo']['features'][key],
                                (float, np.float64, np.float_)) and
                            (not np.isfinite(
                                resultdict[
                                    'result'
                                ]['varinfo']['features'][key]))):
                            resultdict[
                                'result'
                            ]['varinfo']['features'][key] = None

                        elif (isinstance(
                                resultdict[
                                    'result'
                                ]['varinfo']['features'][key],
                                ndarray)):

                            thisval = (
                                resultdict['result']['varinfo']['features'][key]
                            )
                            thisval = thisval.tolist()
                            for i, v in enumerate(thisval):
                                if (isinstance(v,(float,
                                                  np.float64,
                                                  np.float_)) and
                                    (not(np.isfinite(v)))):
                                    thisval[i] = None
                            resultdict['result']['varinfo']['features'][key] = (
                                thisval
                            )

                # now get the periodograms and phased LCs
                for key in pfmethods:

                    # get the periodogram for this method
                    periodogram = cpdict[key]['periodogram']

                    # get the phased LC with best period
                    if 0 in cpdict[key] and isinstance(cpdict[key][0], dict):
                        phasedlc0plot = cpdict[key][0]['plot']
                        phasedlc0period = float(cpdict[key][0]['period'])
                        phasedlc0epoch = float(cpdict[key][0]['epoch'])
                    else:
                        phasedlc0plot = None
                        phasedlc0period = None
                        phasedlc0epoch = None

                    # get the associated fitinfo for this period if it
                    # exists
                    if (0 in cpdict[key] and
                        isinstance(cpdict[key][0], dict) and
                        'lcfit' in cpdict[key][0] and
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
                    if 1 in cpdict[key] and isinstance(cpdict[key][1], dict):
                        phasedlc1plot = cpdict[key][1]['plot']
                        phasedlc1period = float(cpdict[key][1]['period'])
                        phasedlc1epoch = float(cpdict[key][1]['epoch'])
                    else:
                        phasedlc1plot = None
                        phasedlc1period = None
                        phasedlc1epoch = None

                    # get the associated fitinfo for this period if it
                    # exists
                    if (1 in cpdict[key] and
                        isinstance(cpdict[key][1], dict) and
                        'lcfit' in cpdict[key][1] and
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
                    if 2 in cpdict[key] and isinstance(cpdict[key][2], dict):
                        phasedlc2plot = cpdict[key][2]['plot']
                        phasedlc2period = float(cpdict[key][2]['period'])
                        phasedlc2epoch = float(cpdict[key][2]['epoch'])
                    else:
                        phasedlc2plot = None
                        phasedlc2period = None
                        phasedlc2epoch = None

                    # get the associated fitinfo for this period if it
                    # exists
                    if (2 in cpdict[key] and
                        isinstance(cpdict[key][2], dict) and
                        'lcfit' in cpdict[key][2] and
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
                            'period':phasedlc0period,
                            'epoch':phasedlc0epoch,
                            'lcfit':phasedlc0fit,
                        },
                        'phasedlc1':{
                            'plot':phasedlc1plot,
                            'period':phasedlc1period,
                            'epoch':phasedlc1epoch,
                            'lcfit':phasedlc1fit,
                        },
                        'phasedlc2':{
                            'plot':phasedlc2plot,
                            'period':phasedlc2period,
                            'epoch':phasedlc2epoch,
                            'lcfit':phasedlc2fit,
                        },
                    }

                #
                # end of processing per pfmethod
                #

                # return the checkplot via JSON
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

            self.cpfile = base64.b64decode(url_unescape(cpfile)).decode()
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
            # varinfo, objectinfo (objecttags), uifilters, and comments
            updated = {'varinfo': cpcontents['varinfo'],
                       'objectinfo':cpcontents['objectinfo'],
                       'comments':cpcontents['comments'],
                       'uifilters':cpcontents['uifilters']}

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
                                        'unixtime':utime.time(),
                                        'changes':cpcontents,
                                        'cpfpng': None}}

                # handle a savetopng trigger
                if savetopng:

                    cpfpng = os.path.abspath(cpfpath.replace('.pkl','.png'))
                    cpfpng = StrIO()
                    pngdone = yield self.executor.submit(
                        checkplot_pickle_to_png,
                        cpfpath, cpfpng
                    )

                    if pngdone is not None:

                        # we'll send back the PNG, which can then be loaded by
                        # the frontend and reformed into a download
                        pngdone.seek(0)
                        pngbin = pngdone.read()
                        pngb64 = base64.b64encode(pngbin)
                        pngdone.close()
                        del pngbin
                        resultdict['result']['cpfpng'] = pngb64

                    else:
                        resultdict['result']['cpfpng'] = ''

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
        except Exception:

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
        This handles initial setup of the `RequestHandler`.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor
        self.readonly = readonly

    def get(self):
        '''
        This handles GET requests for the current checkplot-list.json file.

        Used with AJAX from frontend.

        '''

        # add the reviewed key to the current dict if it doesn't exist
        # this will hold all the reviewed objects for the frontend
        if 'reviewed' not in self.currentproject:
            self.currentproject['reviewed'] = {}

        # just returns the current project as JSON
        self.write(self.currentproject)

    def post(self):
        '''This handles POST requests.

        Saves the changes made by the user on the frontend back to the current
        checkplot-list.json file.

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
