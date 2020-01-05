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
from .checkplotserver_handlers import PFMETHODS


###########################################################
## STANDALONE CHECKPLOT PICKLE -> JSON over HTTP HANDLER ##
###########################################################

def _time_independent_equals(a, b):
    '''
    This compares two values in constant time.

    Taken from tornado:

    https://github.com/tornadoweb/tornado/blob/
    d4eb8eb4eb5cc9a6677e9116ef84ded8efba8859/tornado/web.py#L3060

    '''
    if len(a) != len(b):
        return False
    result = 0
    if isinstance(a[0], int):  # python3 byte strings
        for x, y in zip(a, b):
            result |= x ^ y
    else:  # python2
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)
    return result == 0


class StandaloneHandler(tornado.web.RequestHandler):
    '''This handles loading checkplots into JSON and sending that back.

    This is a special handler used when `checkplotserver` is in 'stand-alone'
    mode, i.e. only serving up checkplot pickles anywhere on disk as JSON when
    requested.

    '''

    def initialize(self, executor, secret):
        '''
        This handles initial setup of the `RequestHandler`.

        '''

        self.executor = executor
        self.secret = secret

    @gen.coroutine
    def get(self):
        '''This handles GET requests.

        Returns the requested checkplot pickle's information as JSON.

        Requires a pre-shared secret `key` argument for the operation to
        complete successfully. This is obtained from a command-line argument.

        '''

        provided_key = self.get_argument('key',default=None)

        if not provided_key:

            LOGGER.error('standalone URL hit but no secret key provided')
            retdict = {'status':'error',
                       'message':('standalone URL hit but '
                                  'no secret key provided'),
                       'result':None,
                       'readonly':True}
            self.set_status(401)
            self.write(retdict)
            raise tornado.web.Finish()

        else:

            provided_key = xhtml_escape(provided_key)

            if not _time_independent_equals(provided_key,
                                            self.secret):

                LOGGER.error('secret key provided does not match known key')
                retdict = {'status':'error',
                           'message':('standalone URL hit but '
                                      'no secret key provided'),
                           'result':None,
                           'readonly':True}
                self.set_status(401)
                self.write(retdict)
                raise tornado.web.Finish()

        #
        # actually start work here
        #
        LOGGER.info('key auth OK')
        checkplotfname = self.get_argument('cp', default=None)

        if checkplotfname:

            try:
                # do the usual safing
                cpfpath = xhtml_escape(
                    base64.b64decode(url_unescape(checkplotfname))
                )

            except Exception:
                msg = 'could not decode the incoming payload'
                LOGGER.error(msg)
                resultdict = {'status':'error',
                              'message':msg,
                              'result':None,
                              'readonly':True}
                self.set_status(400)
                self.write(resultdict)
                raise tornado.web.Finish()

            LOGGER.info('loading %s...' % cpfpath)

            if not os.path.exists(cpfpath):

                msg = "couldn't find checkplot %s" % cpfpath
                LOGGER.error(msg)
                resultdict = {'status':'error',
                              'message':msg,
                              'result':None,
                              'readonly':True}

                self.set_status(404)
                self.write(resultdict)
                raise tornado.web.Finish()

            #
            # load the checkplot
            #

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

            # this is the initial dict
            resultdict = {
                'status':'ok',
                'message':'found checkplot %s' % os.path.basename(cpfpath),
                'readonly':True,
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
            self.set_header('Content-Type','application/json; charset=UTF-8')
            self.write(resultdict)
            self.finish()

        else:

            LOGGER.error('no checkplot file requested')

            resultdict = {'status':'error',
                          'message':"This checkplot doesn't exist.",
                          'readonly':True,
                          'result':None}

            self.status(400)
            self.write(resultdict)
            self.finish()
