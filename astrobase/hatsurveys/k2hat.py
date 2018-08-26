#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
k2hat.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 07/15
License: MIT. See the LICENCE file for license text.

This contains functions for reading K2 CSV light-curves produced by the HAT
Project into a Python dictionary. Requires numpy.

The only external function here is:

read_csv_lightcurve(lcfile)

EXAMPLE:

Reading the best aperture LC for EPIC201183188 = UCAC4-428-055298 (see
http://k2.hatsurveys.org to search for this object and download the light
curve):

>>> import k2hat
>>> lcdict = k2hat.read_csv_lightcurve('UCAC4-428-055298-75d3f4357b314ff5ac458e917e6dfeb964877b60affe9193d4f65088-k2lc.csv.gz')

The Python dict lcdict contains the metadata and all columns.

>>> lcdict.keys()
['decl', 'objectid', 'bjdoffset', 'qualflag', 'fovchannel', 'BGV',
'aperpixradius', 'IM04', 'TF17', 'EP01', 'CF01', 'ra', 'fovmodule', 'columns',
'k2campaign', 'EQ01', 'fovccd', 'FRN', 'IE04', 'kepid', 'YCC', 'XCC', 'BJD',
'napertures', 'ucac4id', 'IQ04', 'kepmag', 'ndet','kernelspec']

The columns for the light curve are stored in the columns key of the dict. To
get a list of the columns:

>>> lcdict['columns']
['BJD', 'BGV', 'FRN', 'XCC', 'YCC', 'IM04', 'IE04', 'IQ04', 'EP01', 'EQ01',
'TF17', 'CF01']

To get columns:

>>> bjd, epdmags = lcdict['BJD'], lcdict['EP01']
>>> bjd
array([ 2456808.1787283,  2456808.1991608,  2456808.2195932, ...,
        2456890.2535691,  2456890.274001 ,  2456890.2944328])
>>> epdmags
array([ 16.03474,  16.02773,  16.01826, ...,  15.76997,  15.76577,
        15.76263])

'''

# put this in here because k2hat can be used as a standalone module
__version__ = '0.3.18'

#############
## LOGGING ##
#############

import logging
from datetime import datetime
from traceback import format_exc

# setup a logger
LOGGER = None
LOGMOD = __name__
DEBUG = False

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.%s' % (parent_name, LOGMOD))

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('[%s - DBUG] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('[%s - INFO] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('[%s - ERR!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('[%s - WRN!] %s' % (
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            message)
        )

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '[%s - EXC!] %s\nexception was: %s' % (
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                message, format_exc()
            )
        )


#############
## IMPORTS ##
#############

import os.path
import gzip
import numpy as np


########################
## COLUMN DEFINITIONS ##
########################

# LC column definitions
# the first elem is the column description, the second is the format to use when
# writing a CSV LC column, the third is the type to use when parsing a CSV LC
# column
COLUMNDEFS = {
    'bjd':['time in Baryocentric Julian Date','%.7f',float],
    'bgv':['Background value (ADU)','%.5f',float],
    'bge':['Background value (ADU)','%.5f',float],
    'frn':['cadence number of observation','%i',int],
    'xcc':['x coordinate on module', '%.3f',float],
    'ycc':['y coordinate on module', '%.3f',float],
    'arc':['arc length parameter', '%.3f', float],
    # aperture 00
    'im00':['K2 instrumental magnitude (aperture 00)','%.5f',float],
    'ie00':['K2 instrumental mag. error (aperture 00)','%.5f',float],
    'iq00':['K2 instrumental mag. quality flag (aperture 00)','%s',str],
    'ep00':['detrended magnitude (aperture 00)','%.5f',float],
    'eq00':['detrended mag. quality flag (aperture 00)','%i',int],
    'tf00':['TFA magnitude (aperture 00)','%.5f',float],
    'cf00':['Cosine filtered magnitude (aperture 00)','%.5f',float],
    # aperture 01
    'im01':['K2 instrumental magnitude (aperture 01)','%.5f',float],
    'ie01':['K2 instrumental mag. error (aperture 01)','%.5f',float],
    'iq01':['K2 instrumental mag. quality flag (aperture 01)','%s',str],
    'ep01':['detrended magnitude (aperture 01)','%.5f',float],
    'eq01':['detrended mag. quality flag (aperture 01)','%i',int],
    'tf01':['TFA magnitude (aperture 01)','%.5f',float],
    'cf01':['Cosine filtered magnitude (aperture 01)','%.5f',float],
    # aperture 02
    'im02':['K2 instrumental magnitude (aperture 02)','%.5f',float],
    'ie02':['K2 instrumental mag. error (aperture 02)','%.5f',float],
    'iq02':['K2 instrumental mag. quality flag (aperture 02)','%s',str],
    'ep02':['detrended magnitude (aperture 02)','%.5f',float],
    'eq02':['detrended mag. quality flag (aperture 02)','%i',int],
    'tf02':['TFA magnitude (aperture 02)','%.5f',float],
    'cf02':['Cosine filtered magnitude (aperture 02)','%.5f',float],
    # aperture 03
    'im03':['K2 instrumental magnitude (aperture 03)','%.5f',float],
    'ie03':['K2 instrumental mag. error (aperture 03)','%.5f',float],
    'iq03':['K2 instrumental mag. quality flag (aperture 03)','%s',str],
    'ep03':['detrended magnitude (aperture 03)','%.5f',float],
    'eq03':['detrended mag. quality flag (aperture 03)','%i',int],
    'tf03':['TFA magnitude (aperture 03)','%.5f',float],
    'cf03':['Cosine filtered magnitude (aperture 03)','%.5f',float],
    # aperture 04
    'im04':['K2 instrumental magnitude (aperture 04)','%.5f',float],
    'ie04':['K2 instrumental mag. error (aperture 04)','%.5f',float],
    'iq04':['K2 instrumental mag. quality flag (aperture 04)','%s',str],
    'ep04':['detrended magnitude (aperture 04)','%.5f',float],
    'eq04':['detrended mag. quality flag (aperture 04)','%i',int],
    'tf04':['TFA magnitude (aperture 04)','%.5f',float],
    'cf04':['Cosine filtered magnitude (aperture 04)','%.5f',float],
    # aperture 05
    'im05':['K2 instrumental magnitude (aperture 05)','%.5f',float],
    'ie05':['K2 instrumental mag. error (aperture 05)','%.5f',float],
    'iq05':['K2 instrumental mag. quality flag (aperture 05)','%s',str],
    'ep05':['detrended magnitude (aperture 05)','%.5f',float],
    'eq05':['detrended mag. quality flag (aperture 05)','%i',int],
    'tf05':['TFA magnitude (aperture 05)','%.5f',float],
    'cf05':['Cosine filtered magnitude (aperture 05)','%.5f',float],
    # aperture 06
    'im06':['K2 instrumental magnitude (aperture 06)','%.5f',float],
    'ie06':['K2 instrumental mag. error (aperture 06)','%.5f',float],
    'iq06':['K2 instrumental mag. quality flag (aperture 06)','%s',str],
    'ep06':['detrended magnitude (aperture 06)','%.5f',float],
    'eq06':['detrended mag. quality flag (aperture 06)','%i',int],
    'tf06':['TFA magnitude (aperture 06)','%.5f',float],
    'cf06':['Cosine filtered magnitude (aperture 06)','%.5f',float],
    # aperture 07
    'im07':['K2 instrumental magnitude (aperture 07)','%.5f',float],
    'ie07':['K2 instrumental mag. error (aperture 07)','%.5f',float],
    'iq07':['K2 instrumental mag. quality flag (aperture 07)','%s',str],
    'ep07':['detrended magnitude (aperture 07)','%.5f',float],
    'eq07':['detrended mag. quality flag (aperture 07)','%i',int],
    'tf07':['TFA magnitude (aperture 07)','%.5f',float],
    'cf07':['Cosine filtered magnitude (aperture 07)','%.5f',float],
    # aperture 08
    'im08':['K2 instrumental magnitude (aperture 08)','%.5f',float],
    'ie08':['K2 instrumental mag. error (aperture 08)','%.5f',float],
    'iq08':['K2 instrumental mag. quality flag (aperture 08)','%s',str],
    'ep08':['detrended magnitude (aperture 08)','%.5f',float],
    'eq08':['detrended mag. quality flag (aperture 08)','%i',int],
    'tf08':['TFA magnitude (aperture 08)','%.5f',float],
    'cf08':['Cosine filtered magnitude (aperture 08)','%.5f',float],
    # aperture 09
    'im09':['K2 instrumental magnitude (aperture 09)','%.5f',float],
    'ie09':['K2 instrumental mag. error (aperture 09)','%.5f',float],
    'iq09':['K2 instrumental mag. quality flag (aperture 09)','%s',str],
    'ep09':['detrended magnitude (aperture 09)','%.5f',float],
    'eq09':['detrended mag. quality flag (aperture 09)','%i',int],
    'tf09':['TFA magnitude (aperture 09)','%.5f',float],
    'cf09':['Cosine filtered magnitude (aperture 09)','%.5f',float],
    # aperture 10
    'im10':['K2 instrumental magnitude (aperture 10)','%.5f',float],
    'ie10':['K2 instrumental mag. error (aperture 10)','%.5f',float],
    'iq10':['K2 instrumental mag. quality flag (aperture 10)','%s',str],
    'ep10':['detrended magnitude (aperture 10)','%.5f',float],
    'eq10':['detrended mag. quality flag (aperture 10)','%i',int],
    'tf10':['TFA magnitude (aperture 10)','%.5f',float],
    'cf10':['Cosine filtered magnitude (aperture 10)','%.5f',float],
    # aperture 11
    'im11':['K2 instrumental magnitude (aperture 11)','%.5f',float],
    'ie11':['K2 instrumental mag. error (aperture 11)','%.5f',float],
    'iq11':['K2 instrumental mag. quality flag (aperture 11)','%s',str],
    'ep11':['detrended magnitude (aperture 11)','%.5f',float],
    'eq11':['detrended mag. quality flag (aperture 11)','%i',int],
    'tf11':['TFA magnitude (aperture 11)','%.5f',float],
    'cf11':['Cosine filtered magnitude (aperture 11)','%.5f',float],
    # aperture 12
    'im12':['K2 instrumental magnitude (aperture 12)','%.5f',float],
    'ie12':['K2 instrumental mag. error (aperture 12)','%.5f',float],
    'iq12':['K2 instrumental mag. quality flag (aperture 12)','%s',str],
    'ep12':['detrended magnitude (aperture 12)','%.5f',float],
    'eq12':['detrended mag. quality flag (aperture 12)','%i',int],
    'tf12':['TFA magnitude (aperture 12)','%.5f',float],
    'cf12':['Cosine filtered magnitude (aperture 12)','%.5f',float],
    # aperture 13
    'im13':['K2 instrumental magnitude (aperture 13)','%.5f',float],
    'ie13':['K2 instrumental mag. error (aperture 13)','%.5f',float],
    'iq13':['K2 instrumental mag. quality flag (aperture 13)','%s',str],
    'ep13':['detrended magnitude (aperture 13)','%.5f',float],
    'eq13':['detrended mag. quality flag (aperture 13)','%i',int],
    'tf13':['TFA magnitude (aperture 13)','%.5f',float],
    'cf13':['Cosine filtered magnitude (aperture 13)','%.5f',float],
    # aperture 14
    'im14':['K2 instrumental magnitude (aperture 14)','%.5f',float],
    'ie14':['K2 instrumental mag. error (aperture 14)','%.5f',float],
    'iq14':['K2 instrumental mag. quality flag (aperture 14)','%s',str],
    'ep14':['detrended magnitude (aperture 14)','%.5f',float],
    'eq14':['detrended mag. quality flag (aperture 14)','%i',int],
    'tf14':['TFA magnitude (aperture 14)','%.5f',float],
    'cf14':['Cosine filtered magnitude (aperture 14)','%.5f',float],
    # aperture 15
    'im15':['K2 instrumental magnitude (aperture 15)','%.5f',float],
    'ie15':['K2 instrumental mag. error (aperture 15)','%.5f',float],
    'iq15':['K2 instrumental mag. quality flag (aperture 15)','%s',str],
    'ep15':['detrended magnitude (aperture 15)','%.5f',float],
    'eq15':['detrended mag. quality flag (aperture 15)','%i',int],
    'tf15':['TFA magnitude (aperture 15)','%.5f',float],
    'cf15':['Cosine filtered magnitude (aperture 15)','%.5f',float],
    # aperture 16
    'im16':['K2 instrumental magnitude (aperture 16)','%.5f',float],
    'ie16':['K2 instrumental mag. error (aperture 16)','%.5f',float],
    'iq16':['K2 instrumental mag. quality flag (aperture 16)','%s',str],
    'ep16':['detrended magnitude (aperture 16)','%.5f',float],
    'eq16':['detrended mag. quality flag (aperture 16)','%i',int],
    'tf16':['TFA magnitude (aperture 16)','%.5f',float],
    'cf16':['Cosine filtered magnitude (aperture 16)','%.5f',float],
    # aperture 17
    'im17':['K2 instrumental magnitude (aperture 17)','%.5f',float],
    'ie17':['K2 instrumental mag. error (aperture 17)','%.5f',float],
    'iq17':['K2 instrumental mag. quality flag (aperture 17)','%s',str],
    'ep17':['detrended magnitude (aperture 17)','%.5f',float],
    'eq17':['detrended mag. quality flag (aperture 17)','%i',int],
    'tf17':['TFA magnitude (aperture 17)','%.5f',float],
    'cf17':['Cosine filtered magnitude (aperture 17)','%.5f',float],
    # aperture 18
    'im18':['K2 instrumental magnitude (aperture 18)','%.5f',float],
    'ie18':['K2 instrumental mag. error (aperture 18)','%.5f',float],
    'iq18':['K2 instrumental mag. quality flag (aperture 18)','%s',str],
    'ep18':['detrended magnitude (aperture 18)','%.5f',float],
    'eq18':['detrended mag. quality flag (aperture 18)','%i',int],
    'tf18':['TFA magnitude (aperture 18)','%.5f',float],
    'cf18':['Cosine filtered magnitude (aperture 18)','%.5f',float],
    # aperture 19
    'im19':['K2 instrumental magnitude (aperture 19)','%.5f',float],
    'ie19':['K2 instrumental mag. error (aperture 19)','%.5f',float],
    'iq19':['K2 instrumental mag. quality flag (aperture 19)','%s',str],
    'ep19':['detrended magnitude (aperture 19)','%.5f',float],
    'eq19':['detrended mag. quality flag (aperture 19)','%i',int],
    'tf19':['TFA magnitude (aperture 19)','%.5f',float],
    'cf19':['Cosine filtered magnitude (aperture 19)','%.5f',float],
    # aperture 20
    'im20':['K2 instrumental magnitude (aperture 20)','%.5f',float],
    'ie20':['K2 instrumental mag. error (aperture 20)','%.5f',float],
    'iq20':['K2 instrumental mag. quality flag (aperture 20)','%s',str],
    'ep20':['detrended magnitude (aperture 20)','%.5f',float],
    'eq20':['detrended mag. quality flag (aperture 20)','%i',int],
    'tf20':['TFA magnitude (aperture 20)','%.5f',float],
    'cf20':['Cosine filtered magnitude (aperture 20)','%.5f',float],
    # aperture 20
    'im21':['K2 instrumental magnitude (aperture 21)','%.5f',float],
    'ie21':['K2 instrumental mag. error (aperture 21)','%.5f',float],
    'iq21':['K2 instrumental mag. quality flag (aperture 21)','%s',str],
    'ep21':['detrended magnitude (aperture 21)','%.5f',float],
    'eq21':['detrended mag. quality flag (aperture 21)','%i',int],
    'tf21':['TFA magnitude (aperture 21)','%.5f',float],
    'cf21':['Cosine filtered magnitude (aperture 21)','%.5f',float],
    # aperture 21
    'im22':['K2 instrumental magnitude (aperture 22)','%.5f',float],
    'ie22':['K2 instrumental mag. error (aperture 22)','%.5f',float],
    'iq22':['K2 instrumental mag. quality flag (aperture 22)','%s',str],
    'ep22':['detrended magnitude (aperture 22)','%.5f',float],
    'eq22':['detrended mag. quality flag (aperture 22)','%i',int],
    'tf22':['TFA magnitude (aperture 22)','%.5f',float],
    'cf22':['Cosine filtered magnitude (aperture 22)','%.5f',float],
    # aperture 22
    'im23':['K2 instrumental magnitude (aperture 23)','%.5f',float],
    'ie23':['K2 instrumental mag. error (aperture 23)','%.5f',float],
    'iq23':['K2 instrumental mag. quality flag (aperture 23)','%s',str],
    'ep23':['detrended magnitude (aperture 23)','%.5f',float],
    'eq23':['detrended mag. quality flag (aperture 23)','%i',int],
    'tf23':['TFA magnitude (aperture 23)','%.5f',float],
    'cf23':['Cosine filtered magnitude (aperture 23)','%.5f',float],
    # aperture 23
    'im24':['K2 instrumental magnitude (aperture 24)','%.5f',float],
    'ie24':['K2 instrumental mag. error (aperture 24)','%.5f',float],
    'iq24':['K2 instrumental mag. quality flag (aperture 24)','%s',str],
    'ep24':['detrended magnitude (aperture 24)','%.5f',float],
    'eq24':['detrended mag. quality flag (aperture 24)','%i',int],
    'tf24':['TFA magnitude (aperture 24)','%.5f',float],
    'cf24':['Cosine filtered magnitude (aperture 24)','%.5f',float],
    # aperture 24
    'im25':['K2 instrumental magnitude (aperture 25)','%.5f',float],
    'ie25':['K2 instrumental mag. error (aperture 25)','%.5f',float],
    'iq25':['K2 instrumental mag. quality flag (aperture 25)','%s',str],
    'ep25':['detrended magnitude (aperture 25)','%.5f',float],
    'eq25':['detrended mag. quality flag (aperture 25)','%i',int],
    'tf25':['TFA magnitude (aperture 25)','%.5f',float],
    'cf25':['Cosine filtered magnitude (aperture 25)','%.5f',float],
    # aperture 25
    'im26':['K2 instrumental magnitude (aperture 26)','%.5f',float],
    'ie26':['K2 instrumental mag. error (aperture 26)','%.5f',float],
    'iq26':['K2 instrumental mag. quality flag (aperture 26)','%s',str],
    'ep26':['detrended magnitude (aperture 26)','%.5f',float],
    'eq26':['detrended mag. quality flag (aperture 26)','%i',int],
    'tf26':['TFA magnitude (aperture 26)','%.5f',float],
    'cf26':['Cosine filtered magnitude (aperture 26)','%.5f',float],
    # aperture 26
    'im27':['K2 instrumental magnitude (aperture 27)','%.5f',float],
    'ie27':['K2 instrumental mag. error (aperture 27)','%.5f',float],
    'iq27':['K2 instrumental mag. quality flag (aperture 27)','%s',str],
    'ep27':['detrended magnitude (aperture 27)','%.5f',float],
    'eq27':['detrended mag. quality flag (aperture 27)','%i',int],
    'tf27':['TFA magnitude (aperture 27)','%.5f',float],
    'cf27':['Cosine filtered magnitude (aperture 27)','%.5f',float],
    # aperture 27
    'im28':['K2 instrumental magnitude (aperture 28)','%.5f',float],
    'ie28':['K2 instrumental mag. error (aperture 28)','%.5f',float],
    'iq28':['K2 instrumental mag. quality flag (aperture 28)','%s',str],
    'ep28':['detrended magnitude (aperture 28)','%.5f',float],
    'eq28':['detrended mag. quality flag (aperture 28)','%i',int],
    'tf28':['TFA magnitude (aperture 28)','%.5f',float],
    'cf28':['Cosine filtered magnitude (aperture 28)','%.5f',float],
    # aperture 28
    'im29':['K2 instrumental magnitude (aperture 29)','%.5f',float],
    'ie29':['K2 instrumental mag. error (aperture 29)','%.5f',float],
    'iq29':['K2 instrumental mag. quality flag (aperture 29)','%s',str],
    'ep29':['detrended magnitude (aperture 29)','%.5f',float],
    'eq29':['detrended mag. quality flag (aperture 29)','%i',int],
    'tf29':['TFA magnitude (aperture 29)','%.5f',float],
    'cf29':['Cosine filtered magnitude (aperture 29)','%.5f',float],
    # aperture 29
    'im30':['K2 instrumental magnitude (aperture 30)','%.5f',float],
    'ie30':['K2 instrumental mag. error (aperture 30)','%.5f',float],
    'iq30':['K2 instrumental mag. quality flag (aperture 30)','%s',str],
    'ep30':['detrended magnitude (aperture 30)','%.5f',float],
    'eq30':['detrended mag. quality flag (aperture 30)','%i',int],
    'tf30':['TFA magnitude (aperture 30)','%.5f',float],
    'cf30':['Cosine filtered magnitude (aperture 30)','%.5f',float],
    # aperture 30
    'im31':['K2 instrumental magnitude (aperture 31)','%.5f',float],
    'ie31':['K2 instrumental mag. error (aperture 31)','%.5f',float],
    'iq31':['K2 instrumental mag. quality flag (aperture 31)','%s',str],
    'ep31':['detrended magnitude (aperture 31)','%.5f',float],
    'eq31':['detrended mag. quality flag (aperture 31)','%i',int],
    'tf31':['TFA magnitude (aperture 31)','%.5f',float],
    'cf31':['Cosine filtered magnitude (aperture 31)','%.5f',float],
    # aperture 31
    'im32':['K2 instrumental magnitude (aperture 32)','%.5f',float],
    'ie32':['K2 instrumental mag. error (aperture 32)','%.5f',float],
    'iq32':['K2 instrumental mag. quality flag (aperture 32)','%s',str],
    'ep32':['detrended magnitude (aperture 32)','%.5f',float],
    'eq32':['detrended mag. quality flag (aperture 32)','%i',int],
    'tf32':['TFA magnitude (aperture 32)','%.5f',float],
    'cf32':['Cosine filtered magnitude (aperture 32)','%.5f',float],
    # aperture 33
    'im33':['K2 instrumental magnitude (aperture 33)','%.5f',float],
    'ie33':['K2 instrumental mag. error (aperture 33)','%.5f',float],
    'iq33':['K2 instrumental mag. quality flag (aperture 33)','%s',str],
    'ep33':['detrended magnitude (aperture 33)','%.5f',float],
    'eq33':['detrended mag. quality flag (aperture 33)','%i',int],
    'tf33':['TFA magnitude (aperture 33)','%.5f',float],
    'cf33':['Cosine filtered magnitude (aperture 33)','%.5f',float],
    # aperture 34
    'im34':['K2 instrumental magnitude (aperture 34)','%.5f',float],
    'ie34':['K2 instrumental mag. error (aperture 34)','%.5f',float],
    'iq34':['K2 instrumental mag. quality flag (aperture 34)','%s',str],
    'ep34':['detrended magnitude (aperture 34)','%.5f',float],
    'eq34':['detrended mag. quality flag (aperture 34)','%i',int],
    'tf34':['TFA magnitude (aperture 34)','%.5f',float],
    'cf34':['Cosine filtered magnitude (aperture 34)','%.5f',float],
    # aperture 35
    'im35':['K2 instrumental magnitude (aperture 35)','%.5f',float],
    'ie35':['K2 instrumental mag. error (aperture 35)','%.5f',float],
    'iq35':['K2 instrumental mag. quality flag (aperture 35)','%s',str],
    'ep35':['detrended magnitude (aperture 35)','%.5f',float],
    'eq35':['detrended mag. quality flag (aperture 35)','%i',int],
    'tf35':['TFA magnitude (aperture 35)','%.5f',float],
    'cf35':['Cosine filtered magnitude (aperture 35)','%.5f',float],
}


##################################
## FUNCTIONS TO READ K2 HAT LCS ##
##################################

def parse_csv_header(header):
    '''This parses a CSV header from a K2 CSV LC.

    Returns a dict that can be used to update an existing lcdict with the
    relevant metadata info needed to form a full LC.

    '''

    # first, break into lines
    headerlines = header.split('\n')
    headerlines = [x.lstrip('# ') for x in headerlines]

    # next, find the indices of the '# COLUMNS' line and '# LIGHTCURVE' line
    metadatastart = headerlines.index('METADATA')
    columnstart = headerlines.index('COLUMNS')
    lcstart = headerlines.index('LIGHTCURVE')

    # get the lines for the metadata and columndefs
    metadata = headerlines[metadatastart+1:columnstart-1]
    columndefs = headerlines[columnstart+1:lcstart-1]

    # parse the metadata
    metainfo = [x.split(',') for x in metadata][:-1]
    aperpixradius = metadata[-1]

    objectid, kepid, ucac4id, kepmag = metainfo[0]
    objectid, kepid, ucac4id, kepmag = (objectid.split(' = ')[-1],
                                        kepid.split(' = ')[-1],
                                        ucac4id.split(' = ')[-1],
                                        kepmag.split(' = ')[-1])
    kepmag = float(kepmag) if kepmag else None

    ra, decl, ndet, k2campaign = metainfo[1]
    ra, decl, ndet, k2campaign = (ra.split(' = ')[-1],
                                  decl.split(' = ')[-1],
                                  int(ndet.split(' = ')[-1]),
                                  int(k2campaign.split(' = ')[-1]))

    fovccd, fovchannel, fovmodule = metainfo[2]
    fovccd, fovchannel, fovmodule = (int(fovccd.split(' = ')[-1]),
                                     int(fovchannel.split(' = ')[-1]),
                                     int(fovmodule.split(' = ')[-1]))

    try:
        qualflag, bjdoffset, napertures = metainfo[3]
        qualflag, bjdoffset, napertures = (int(qualflag.split(' = ')[-1]),
                                           float(bjdoffset.split(' = ')[-1]),
                                           int(napertures.split(' = ')[-1]))
        kernelspec = None
    except Exception as e:
        qualflag, bjdoffset, napertures, kernelspec = metainfo[3]
        qualflag, bjdoffset, napertures, kernelspec = (
            int(qualflag.split(' = ')[-1]),
            float(bjdoffset.split(' = ')[-1]),
            int(napertures.split(' = ')[-1]),
            str(kernelspec.split(' = ')[-1])
        )

    aperpixradius = aperpixradius.split(' = ')[-1].split(',')
    aperpixradius = [float(x) for x in aperpixradius]

    # parse the columndefs
    columns = [x.split(' - ')[1] for x in columndefs]

    metadict = {'objectid':objectid,
                'objectinfo':{
                    'objectid':objectid,
                    'kepid':kepid,
                    'ucac4id':ucac4id,
                    'kepmag':kepmag,
                    'ra':ra,
                    'decl':decl,
                    'ndet':ndet,
                    'k2campaign':k2campaign,
                    'fovccd':fovccd,
                    'fovchannel':fovchannel,
                    'fovmodule':fovmodule,
                    'qualflag':qualflag,
                    'bjdoffset':bjdoffset,
                    'napertures':napertures,
                    'kernelspec':kernelspec,
                    'aperpixradius':aperpixradius,
                },
                'columns':columns}

    return metadict



def read_csv_lightcurve(lcfile):
    '''
    This reads in a K2 lightcurve in CSV format. Transparently reads gzipped
    files.

    '''

    # read in the file first
    if '.gz' in os.path.basename(lcfile):
        LOGINFO('reading gzipped K2 LC: %s' % lcfile)
        infd = gzip.open(lcfile,'rb')
    else:
        LOGINFO('reading K2 LC: %s' % lcfile)
        infd = open(lcfile,'rb')

    lctext = infd.read().decode()
    infd.close()

    # figure out the header and get the LC columns
    lcstart = lctext.index('# LIGHTCURVE\n')
    lcheader = lctext[:lcstart+12]
    lccolumns = lctext[lcstart+13:].split('\n')
    lccolumns = [x.split(',') for x in lccolumns if len(x) > 0]

    # initialize the lcdict and parse the CSV header
    lcdict = parse_csv_header(lcheader)

    # tranpose the LC rows into columns
    lccolumns = list(zip(*lccolumns))

    # write the columns to the dict
    for colind, col in enumerate(lcdict['columns']):

        # this picks out the caster to use when reading each column using the
        # definitions in the lcutils.COLUMNDEFS dictionary
        lcdict[col] = np.array([COLUMNDEFS[col][2](x)
                                for x in lccolumns[colind]])

    return lcdict
