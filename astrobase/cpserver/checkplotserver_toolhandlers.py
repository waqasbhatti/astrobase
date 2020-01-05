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
from io import BytesIO as StrIO
import numpy as np
import pickle
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

from .. import lcmath

from ..checkplot.pkl_io import (
    _read_checkplot_picklefile,
    _write_checkplot_picklefile
)
from ..checkplot.pkl_utils import _pkl_periodogram, _pkl_phased_magseries_plot
from .. import lcfit

from .checkplotserver_handlers import CPTOOLMAP


#############################
## CHECKPLOT TOOL HANDLERS ##
#############################

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
        This handles initial setup of the `RequestHandler`.

        '''

        self.currentdir = currentdir
        self.assetpath = assetpath
        self.currentproject = cplist
        self.cplistfile = cplistfile
        self.executor = executor
        self.readonly = readonly

    @gen.coroutine
    def get(self, cpfile):
        '''This handles a GET request to run a specified LC tool.

        Parameters
        ----------

        cpfile : str
            This is the checkplot file to run the tool on.

        Returns
        -------

        str
            Returns a JSON response.

        Notes
        -----

        The URI structure is::

            /tools/<cpfile>?[args]

        where args are::

            ?lctool=<lctool>&argkey1=argval1&argkey2=argval2&...

            &forcereload=true <- if this is present, then reload values from
            original checkplot.

            &objectid=<objectid>

        `lctool` is one of the strings below

        Period search functions::

            psearch-gls: run Lomb-Scargle with given params
            psearch-bls: run BLS with given params
            psearch-pdm: run phase dispersion minimization with given params
            psearch-aov: run analysis-of-variance with given params
            psearch-mav: run analysis-of-variance (multi-harm) with given params
            psearch-acf: run ACF period search with given params
            psearch-win: run spectral window function search with given params

        Arguments recognized by all period-search functions are::

            startp=XX
            endp=XX
            magsarefluxes=True|False
            autofreq=True|False
            stepsize=XX

        Variability characterization functions::

            var-varfeatures: gets the variability features from the checkplot or
                             recalculates if they're not present

            var-prewhiten: pre-whitens the light curve with a sinusoidal signal

            var-masksig: masks a given phase location with given width from the
                         light curve

        Light curve manipulation functions ::

            phasedlc-newplot: make phased LC with new provided period/epoch
            lcfit-fourier: fit a Fourier function to the phased LC
            lcfit-spline: fit a spline function to the phased LC
            lcfit-legendre: fit a Legendre polynomial to the phased LC
            lcfit-savgol: fit a Savitsky-Golay polynomial to the phased LC

        FIXME: figure out how to cache the results of these functions
        temporarily and save them back to the checkplot after we click on save
        in the frontend.

        TODO: look for a checkplot-blah-blah.pkl-cps-processing file in the same
        place as the usual pickle file. if this exists and is newer than the pkl
        file, load it instead. Or have a checkplotdict['cpservertemp'] item.

        '''

        if cpfile:

            self.cpfile = (
                xhtml_escape(base64.b64decode(url_unescape(cpfile)))
            )

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

                        except Exception:

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
                            # special handling for epochs that can be optional
                            elif xargtype is float and xarg == 'varepoch':
                                try:
                                    wbarg = xargtype(wbarg)
                                except Exception:
                                    wbarg = None
                            # usual casting for other types
                            else:
                                wbarg = xargtype(wbarg)

                            lctoolargs.append(wbarg)

                        except Exception:

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
                              'psearch-aov',
                              'psearch-mav',
                              'psearch-acf',
                              'psearch-win'):

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

                        # see if sigclip is set. if so, then do the sigclip on
                        # the times, mags, errs
                        if lctoolkwargs['sigclip'] is not None:

                            wtimes, wmags, werrs = lcmath.sigclip_magseries(
                                lctoolargs[0],
                                lctoolargs[1],
                                lctoolargs[2],
                                sigclip=lctoolkwargs['sigclip'],
                                magsarefluxes=lctoolkwargs['magsarefluxes']
                            )

                            lctoolargs[0] = wtimes
                            lctoolargs[1] = wmags
                            lctoolargs[2] = werrs

                        #
                        # process the LC filters now
                        #

                        # see if the lctimefilters are set
                        if lctoolkwargs['lctimefilters']:

                            wtimes, wmags, werrs = (lctoolargs[0],
                                                    lctoolargs[1],
                                                    lctoolargs[2])
                            filtermasks = [
                                np.full_like(wtimes, False, dtype=np.bool_)
                            ]

                            # parse the time filter strings
                            filterstr = lctoolkwargs['lctimefilters']

                            filters = filterstr.split(',')
                            filters = [
                                x.strip().lstrip('(').rstrip(')').strip()
                                for x in filters
                            ]

                            for filt in filters:

                                try:

                                    thisfilt = filt.split(':')
                                    if len(thisfilt) == 2:

                                        filt_lo = float(thisfilt[0])
                                        filt_hi = float(thisfilt[1])

                                        filtermasks.append(
                                            ((wtimes -
                                              cptimes.min()) < filt_hi) &
                                            ((wtimes -
                                              cptimes.min()) > filt_lo)
                                        )

                                    elif (len(thisfilt) == 3 and
                                          thisfilt[0].strip() == 'not'):

                                        filt_lo = float(thisfilt[1])
                                        filt_hi = float(thisfilt[2])

                                        filtermasks.append(np.logical_not(
                                            (((wtimes -
                                               cptimes.min()) < filt_hi) &
                                             ((wtimes -
                                               cptimes.min()) > filt_lo))
                                        ))

                                    else:
                                        continue

                                except Exception:
                                    continue

                            # finally, apply the filters if applicable
                            if len(filtermasks) > 0:

                                # apply the filters using an OR
                                filterind = np.column_stack(filtermasks)
                                filterind = np.any(filterind, axis=1)

                                lctoolargs[0] = wtimes[filterind]
                                lctoolargs[1] = wmags[filterind]
                                lctoolargs[2] = werrs[filterind]

                        # see if the lcmagfilters are set
                        if lctoolkwargs['lcmagfilters']:

                            wtimes, wmags, werrs = (lctoolargs[0],
                                                    lctoolargs[1],
                                                    lctoolargs[2])
                            filtermasks = [
                                np.full_like(wtimes, False, dtype=np.bool_)
                            ]

                            # parse the time filter strings
                            filterstr = lctoolkwargs['lcmagfilters']

                            filters = filterstr.split(',')
                            filters = [
                                x.strip().strip()
                                for x in filters
                            ]

                            for filt in filters:

                                try:

                                    thisfilt = filt.split(':')
                                    if len(thisfilt) == 2:

                                        filt_lo = float(thisfilt[0])
                                        filt_hi = float(thisfilt[1])

                                        filtermasks.append(
                                            (wmags < filt_hi) &
                                            (wmags > filt_lo)
                                        )

                                    elif (len(thisfilt) == 3 and
                                          thisfilt[0].strip() == 'not'):

                                        filt_lo = float(thisfilt[1])
                                        filt_hi = float(thisfilt[2])

                                        filtermasks.append(np.logical_not(
                                            ((wmags < filt_hi) &
                                             (wmags > filt_lo))
                                        ))

                                    else:
                                        continue

                                except Exception:
                                    continue

                            # finally, apply the filters if applicable
                            if len(filtermasks) > 0:

                                # apply the filters using an OR
                                filterind = np.column_stack(filtermasks)
                                filterind = np.any(filterind, axis=1)

                                lctoolargs[0] = wtimes[filterind]
                                lctoolargs[1] = wmags[filterind]
                                lctoolargs[2] = werrs[filterind]

                        # at the end of processing, remove from lctookwargs
                        # since the pfmethod doesn't know about this
                        del lctoolkwargs['lctimefilters']
                        del lctoolkwargs['lcmagfilters']

                        #
                        # now run the period finder and get results
                        #

                        lctoolfunction = CPTOOLMAP[lctool]['func']

                        # run the period finder
                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs
                        )

                        # get what we need out of funcresults when it
                        # returns.
                        nbestperiods = funcresults['nbestperiods']
                        nbestlspvals = funcresults['nbestlspvals']
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
                                         lctoolargs[0],
                                         lctoolargs[1],
                                         lctoolargs[2],
                                         nbestperiods[0],
                                         'min')

                        if len(nbestperiods) > 1:
                            phasedlcargs1 = (None,
                                             lspmethod,
                                             -1,
                                             lctoolargs[0],
                                             lctoolargs[1],
                                             lctoolargs[2],
                                             nbestperiods[1],
                                             'min')
                        else:
                            phasedlcargs1 = None

                        if len(nbestperiods) > 2:
                            phasedlcargs2 = (None,
                                             lspmethod,
                                             -1,
                                             lctoolargs[0],
                                             lctoolargs[1],
                                             lctoolargs[2],
                                             nbestperiods[2],
                                             'min')
                        else:
                            phasedlcargs2 = None

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

                        if phasedlcargs1 is not None:
                            phasedlc1 = yield self.executor.submit(
                                _pkl_phased_magseries_plot,
                                *phasedlcargs1,
                                **phasedlckwargs
                            )
                        else:
                            phasedlc1 = None

                        if phasedlcargs2 is not None:
                            phasedlc2 = yield self.executor.submit(
                                _pkl_phased_magseries_plot,
                                *phasedlcargs2,
                                **phasedlckwargs
                            )
                        else:
                            phasedlc2 = None

                        # save these to the tempcpdict
                        # save the pickle only if readonly is not true
                        if not self.readonly:

                            tempcpdict[lspmethod] = {
                                'periods':funcresults['periods'],
                                'lspvals':funcresults['lspvals'],
                                'bestperiod':funcresults['bestperiod'],
                                'nbestperiods':funcresults['nbestperiods'],
                                'nbestlspvals':funcresults['nbestlspvals'],
                                'periodogram':(
                                    pgramres[lspmethod]['periodogram']
                                ),
                                0:phasedlc0,
                            }

                            if phasedlc1 is not None:
                                tempcpdict[lspmethod][1] = phasedlc1

                            if phasedlc2 is not None:
                                tempcpdict[lspmethod][2] = phasedlc2

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

                        if phasedlc1 is not None:

                            phasedlc1plot = phasedlc1['plot']
                            phasedlc1period = float(phasedlc1['period'])
                            phasedlc1epoch = float(phasedlc1['epoch'])

                        if phasedlc2 is not None:

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
                                'nbestpeaks':nbestlspvals,
                                'periodogram':periodogram,
                                'bestperiod':bestperiod,
                                'phasedlc0':{
                                    'plot':phasedlc0plot,
                                    'period':phasedlc0period,
                                    'epoch':phasedlc0epoch,
                                },
                            }
                        }

                        if phasedlc1 is not None:
                            resultdict['result'][lspmethod]['phasedlc1'] = {
                                'plot':phasedlc1plot,
                                'period':phasedlc1period,
                                'epoch':phasedlc1epoch,
                            }

                        if phasedlc2 is not None:
                            resultdict['result'][lspmethod]['phasedlc2'] = {
                                'plot':phasedlc2plot,
                                'period':phasedlc2period,
                                'epoch':phasedlc2epoch,
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

                        # see if sigclip is set. if so, then do the sigclip on
                        # the times, mags, errs
                        if lctoolkwargs['sigclip'] is not None:
                            stimes, smags, serrs = lcmath.sigclip_magseries(
                                lctoolargs[3],
                                lctoolargs[4],
                                lctoolargs[5],
                                sigclip=lctoolkwargs['sigclip'],
                                magsarefluxes=lctoolkwargs['magsarefluxes']
                            )
                        else:
                            stimes, smags, serrs = (lctoolargs[3],
                                                    lctoolargs[4],
                                                    lctoolargs[5])

                        #
                        # process the LC filters now
                        #

                        # see if the lctimefilters are set
                        if lctoolkwargs['lctimefilters']:

                            wtimes, wmags, werrs = stimes, smags, serrs

                            filtermasks = [
                                np.full_like(wtimes, False, dtype=np.bool_)
                            ]

                            # parse the time filter strings
                            filterstr = lctoolkwargs['lctimefilters']

                            filters = filterstr.split(',')
                            filters = [
                                x.strip().lstrip('(').rstrip(')').strip()
                                for x in filters
                            ]

                            for filt in filters:

                                try:

                                    thisfilt = filt.split(':')
                                    if len(thisfilt) == 2:

                                        filt_lo = float(thisfilt[0])
                                        filt_hi = float(thisfilt[1])

                                        filtermasks.append(
                                            ((wtimes -
                                              cptimes.min()) < filt_hi) &
                                            ((wtimes -
                                              cptimes.min()) > filt_lo)
                                        )

                                    elif (len(thisfilt) == 3 and
                                          thisfilt[0].strip() == 'not'):

                                        filt_lo = float(thisfilt[1])
                                        filt_hi = float(thisfilt[2])

                                        filtermasks.append(np.logical_not(
                                            (((wtimes -
                                               cptimes.min()) < filt_hi) &
                                             ((wtimes -
                                               cptimes.min()) > filt_lo))
                                        ))

                                    else:
                                        continue

                                except Exception:
                                    continue

                            # finally, apply the filters if applicable
                            if len(filtermasks) > 0:

                                # apply the filters using an OR
                                filterind = np.column_stack(filtermasks)
                                filterind = np.any(filterind, axis=1)

                                stimes = wtimes[filterind]
                                smags = wmags[filterind]
                                serrs = werrs[filterind]

                        # see if the lcmagfilters are set
                        if lctoolkwargs['lcmagfilters']:

                            wtimes, wmags, werrs = stimes, smags, serrs
                            filtermasks = [
                                np.full_like(wtimes, False, dtype=np.bool_)
                            ]

                            # parse the time filter strings
                            filterstr = lctoolkwargs['lcmagfilters']

                            filters = filterstr.split(',')
                            filters = [
                                x.strip().strip()
                                for x in filters
                            ]

                            for filt in filters:

                                try:

                                    thisfilt = filt.split(':')
                                    if len(thisfilt) == 2:

                                        filt_lo = float(thisfilt[0])
                                        filt_hi = float(thisfilt[1])

                                        filtermasks.append(
                                            (wmags < filt_hi) &
                                            (wmags > filt_lo)
                                        )

                                    elif (len(thisfilt) == 3 and
                                          thisfilt[0].strip() == 'not'):

                                        filt_lo = float(thisfilt[1])
                                        filt_hi = float(thisfilt[2])

                                        filtermasks.append(np.logical_not(
                                            ((wmags < filt_hi) &
                                             (wmags > filt_lo))
                                        ))

                                    else:
                                        continue

                                except Exception:
                                    continue

                            # finally, apply the filters if applicable
                            if len(filtermasks) > 0:

                                # apply the filters using an OR
                                filterind = np.column_stack(filtermasks)
                                filterind = np.any(filterind, axis=1)

                                stimes = wtimes[filterind]
                                smags = wmags[filterind]
                                serrs = werrs[filterind]

                        # at the end of processing, remove from lctookwargs
                        # since the pfmethod doesn't know about this
                        del lctoolkwargs['lctimefilters']
                        del lctoolkwargs['lcmagfilters']

                        # if the varepoch is set to None, try to get the
                        # minimum-light epoch using a spline fit
                        if lctoolargs[-1] is None:
                            LOGGER.warning(
                                'automatically getting min epoch '
                                'for phased LC plot'
                            )
                            try:
                                spfit = lcfit.spline_fit_magseries(
                                    stimes,         # times
                                    smags,          # mags
                                    serrs,          # errs
                                    lctoolargs[6],  # period
                                    magsarefluxes=lctoolkwargs['magsarefluxes'],
                                    sigclip=None,
                                    verbose=True
                                )

                                # set the epoch correctly now for the plot
                                lctoolargs[-1] = spfit['fitinfo']['fitepoch']

                                if len(spfit['fitinfo']['fitepoch']) != 1:
                                    lctoolargs[-1] = (
                                        spfit['fitinfo']['fitepoch'][0]
                                    )

                            # if the spline fit fails, use the minimum of times
                            # as epoch as usual
                            except Exception:

                                LOGGER.exception(
                                    'spline fit failed, '
                                    'using min(times) as epoch'
                                )

                                lctoolargs[-1] = np.min(stimes)

                        # now run the phased LC function with provided args,
                        # kwargs

                        # final times, mags, errs
                        lctoolargs[3] = stimes
                        lctoolargs[4] = smags
                        lctoolargs[5] = serrs

                        # the sigclip kwarg isn't used here since we did this
                        # already earlier
                        del lctoolkwargs['sigclip']

                        lctoolfunction = CPTOOLMAP[lctool]['func']

                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs
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
                            **lctoolkwargs
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
                        lctoolkwargs['plotfit'] = StrIO()

                        funcresults = yield self.executor.submit(
                            lctoolfunction,
                            *lctoolargs,
                            **lctoolkwargs
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
                            **lctoolkwargs
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
                                        lctoolargs[3],  # this is the fit period
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
                        LOGGER.warning('reset all LC tool results '
                                       'for %s by removing %s' %
                                       (tempfpath, cpfpath))
                        resultdict['status'] = 'success'
                    else:
                        resultdict['status'] = 'error'
                        LOGGER.warning('tried to reset LC tool results for %s, '
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
                        if (target not in CPTOOLMAP or
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
                        targetloc = CPTOOLMAP[target]['resloc']

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

        TODO: implement this.

        This will save the results of the previous tool run to the checkplot
        file and the JSON filelist.

        This is only called when the user explicitly clicks on the 'permanently
        update checkplot with results' button. If the server is in readonly
        mode, this has no effect.

        This will copy everything from the '.pkl-cpserver-temp' file to the
        actual checkplot pickle and then remove that file.

        '''
