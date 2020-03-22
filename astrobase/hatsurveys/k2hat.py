#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# k2hat.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - 07/15
# License: MIT. See the LICENCE file for license text.

'''
This contains functions for reading K2 CSV light-curves produced by the HAT
Project into a Python dictionary. Requires numpy.

The only external function here is::

    read_csv_lightcurve(lcfile)

Example:

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
__version__ = '0.5.0'

#############
## LOGGING ##
#############

# the basic logging styles common to all astrobase modules
log_sub = '{'
log_fmt = '[{levelname:1.1} {asctime} {module}:{lineno}] {message}'
log_date_fmt = '%y%m%d %H:%M:%S'

import logging

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


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
    'BJD':['time in Baryocentric Julian Date','%.7f',float],
    'BGV':['Background value (ADU)','%.5f',float],
    'BGE':['Background value (ADU)','%.5f',float],
    'FRN':['cadence number of observation','%i',int],
    'XCC':['x coordinate on module', '%.3f',float],
    'YCC':['y coordinate on module', '%.3f',float],
    'ARC':['arc length parameter', '%.3f', float],
    # APERture 00
    'IM00':['K2 instrumental magnitude (aperture 00)','%.5f',float],
    'IE00':['K2 instrumental mag. error (aperture 00)','%.5f',float],
    'IQ00':['K2 instrumental mag. quality flag (aperture 00)','%s',str],
    'EP00':['detrended magnitude (aperture 00)','%.5f',float],
    'EQ00':['detrended mag. quality flag (aperture 00)','%i',int],
    'TF00':['TFA magnitude (aperture 00)','%.5f',float],
    'CF00':['Cosine filtered magnitude (aperture 00)','%.5f',float],
    # APERture 01
    'IM01':['K2 instrumental magnitude (aperture 01)','%.5f',float],
    'IE01':['K2 instrumental mag. error (aperture 01)','%.5f',float],
    'IQ01':['K2 instrumental mag. quality flag (aperture 01)','%s',str],
    'EP01':['detrended magnitude (aperture 01)','%.5f',float],
    'EQ01':['detrended mag. quality flag (aperture 01)','%i',int],
    'TF01':['TFA magnitude (aperture 01)','%.5f',float],
    'CF01':['Cosine filtered magnitude (aperture 01)','%.5f',float],
    # APERture 02
    'IM02':['K2 instrumental magnitude (aperture 02)','%.5f',float],
    'IE02':['K2 instrumental mag. error (aperture 02)','%.5f',float],
    'IQ02':['K2 instrumental mag. quality flag (aperture 02)','%s',str],
    'EP02':['detrended magnitude (aperture 02)','%.5f',float],
    'EQ02':['detrended mag. quality flag (aperture 02)','%i',int],
    'TF02':['TFA magnitude (aperture 02)','%.5f',float],
    'CF02':['Cosine filtered magnitude (aperture 02)','%.5f',float],
    # APERture 03
    'IM03':['K2 instrumental magnitude (aperture 03)','%.5f',float],
    'IE03':['K2 instrumental mag. error (aperture 03)','%.5f',float],
    'IQ03':['K2 instrumental mag. quality flag (aperture 03)','%s',str],
    'EP03':['detrended magnitude (aperture 03)','%.5f',float],
    'EQ03':['detrended mag. quality flag (aperture 03)','%i',int],
    'TF03':['TFA magnitude (aperture 03)','%.5f',float],
    'CF03':['Cosine filtered magnitude (aperture 03)','%.5f',float],
    # APERture 04
    'IM04':['K2 instrumental magnitude (aperture 04)','%.5f',float],
    'IE04':['K2 instrumental mag. error (aperture 04)','%.5f',float],
    'IQ04':['K2 instrumental mag. quality flag (aperture 04)','%s',str],
    'EP04':['detrended magnitude (aperture 04)','%.5f',float],
    'EQ04':['detrended mag. quality flag (aperture 04)','%i',int],
    'TF04':['TFA magnitude (aperture 04)','%.5f',float],
    'CF04':['Cosine filtered magnitude (aperture 04)','%.5f',float],
    # APERture 05
    'IM05':['K2 instrumental magnitude (aperture 05)','%.5f',float],
    'IE05':['K2 instrumental mag. error (aperture 05)','%.5f',float],
    'IQ05':['K2 instrumental mag. quality flag (aperture 05)','%s',str],
    'EP05':['detrended magnitude (aperture 05)','%.5f',float],
    'EQ05':['detrended mag. quality flag (aperture 05)','%i',int],
    'TF05':['TFA magnitude (aperture 05)','%.5f',float],
    'CF05':['Cosine filtered magnitude (aperture 05)','%.5f',float],
    # APERture 06
    'IM06':['K2 instrumental magnitude (aperture 06)','%.5f',float],
    'IE06':['K2 instrumental mag. error (aperture 06)','%.5f',float],
    'IQ06':['K2 instrumental mag. quality flag (aperture 06)','%s',str],
    'EP06':['detrended magnitude (aperture 06)','%.5f',float],
    'EQ06':['detrended mag. quality flag (aperture 06)','%i',int],
    'TF06':['TFA magnitude (aperture 06)','%.5f',float],
    'CF06':['Cosine filtered magnitude (aperture 06)','%.5f',float],
    # APERture 07
    'IM07':['K2 instrumental magnitude (aperture 07)','%.5f',float],
    'IE07':['K2 instrumental mag. error (aperture 07)','%.5f',float],
    'IQ07':['K2 instrumental mag. quality flag (aperture 07)','%s',str],
    'EP07':['detrended magnitude (aperture 07)','%.5f',float],
    'EQ07':['detrended mag. quality flag (aperture 07)','%i',int],
    'TF07':['TFA magnitude (aperture 07)','%.5f',float],
    'CF07':['Cosine filtered magnitude (aperture 07)','%.5f',float],
    # APERture 08
    'IM08':['K2 instrumental magnitude (aperture 08)','%.5f',float],
    'IE08':['K2 instrumental mag. error (aperture 08)','%.5f',float],
    'IQ08':['K2 instrumental mag. quality flag (aperture 08)','%s',str],
    'EP08':['detrended magnitude (aperture 08)','%.5f',float],
    'EQ08':['detrended mag. quality flag (aperture 08)','%i',int],
    'TF08':['TFA magnitude (aperture 08)','%.5f',float],
    'CF08':['Cosine filtered magnitude (aperture 08)','%.5f',float],
    # APERture 09
    'IM09':['K2 instrumental magnitude (aperture 09)','%.5f',float],
    'IE09':['K2 instrumental mag. error (aperture 09)','%.5f',float],
    'IQ09':['K2 instrumental mag. quality flag (aperture 09)','%s',str],
    'EP09':['detrended magnitude (aperture 09)','%.5f',float],
    'EQ09':['detrended mag. quality flag (aperture 09)','%i',int],
    'TF09':['TFA magnitude (aperture 09)','%.5f',float],
    'CF09':['Cosine filtered magnitude (aperture 09)','%.5f',float],
    # APERture 10
    'IM10':['K2 instrumental magnitude (aperture 10)','%.5f',float],
    'IE10':['K2 instrumental mag. error (aperture 10)','%.5f',float],
    'IQ10':['K2 instrumental mag. quality flag (aperture 10)','%s',str],
    'EP10':['detrended magnitude (aperture 10)','%.5f',float],
    'EQ10':['detrended mag. quality flag (aperture 10)','%i',int],
    'TF10':['TFA magnitude (aperture 10)','%.5f',float],
    'CF10':['Cosine filtered magnitude (aperture 10)','%.5f',float],
    # APERture 11
    'IM11':['K2 instrumental magnitude (aperture 11)','%.5f',float],
    'IE11':['K2 instrumental mag. error (aperture 11)','%.5f',float],
    'IQ11':['K2 instrumental mag. quality flag (aperture 11)','%s',str],
    'EP11':['detrended magnitude (aperture 11)','%.5f',float],
    'EQ11':['detrended mag. quality flag (aperture 11)','%i',int],
    'TF11':['TFA magnitude (aperture 11)','%.5f',float],
    'CF11':['Cosine filtered magnitude (aperture 11)','%.5f',float],
    # APERture 12
    'IM12':['K2 instrumental magnitude (aperture 12)','%.5f',float],
    'IE12':['K2 instrumental mag. error (aperture 12)','%.5f',float],
    'IQ12':['K2 instrumental mag. quality flag (aperture 12)','%s',str],
    'EP12':['detrended magnitude (aperture 12)','%.5f',float],
    'EQ12':['detrended mag. quality flag (aperture 12)','%i',int],
    'TF12':['TFA magnitude (aperture 12)','%.5f',float],
    'CF12':['Cosine filtered magnitude (aperture 12)','%.5f',float],
    # APERture 13
    'IM13':['K2 instrumental magnitude (aperture 13)','%.5f',float],
    'IE13':['K2 instrumental mag. error (aperture 13)','%.5f',float],
    'IQ13':['K2 instrumental mag. quality flag (aperture 13)','%s',str],
    'EP13':['detrended magnitude (aperture 13)','%.5f',float],
    'EQ13':['detrended mag. quality flag (aperture 13)','%i',int],
    'TF13':['TFA magnitude (aperture 13)','%.5f',float],
    'CF13':['Cosine filtered magnitude (aperture 13)','%.5f',float],
    # APERture 14
    'IM14':['K2 instrumental magnitude (aperture 14)','%.5f',float],
    'IE14':['K2 instrumental mag. error (aperture 14)','%.5f',float],
    'IQ14':['K2 instrumental mag. quality flag (aperture 14)','%s',str],
    'EP14':['detrended magnitude (aperture 14)','%.5f',float],
    'EQ14':['detrended mag. quality flag (aperture 14)','%i',int],
    'TF14':['TFA magnitude (aperture 14)','%.5f',float],
    'CF14':['Cosine filtered magnitude (aperture 14)','%.5f',float],
    # APERture 15
    'IM15':['K2 instrumental magnitude (aperture 15)','%.5f',float],
    'IE15':['K2 instrumental mag. error (aperture 15)','%.5f',float],
    'IQ15':['K2 instrumental mag. quality flag (aperture 15)','%s',str],
    'EP15':['detrended magnitude (aperture 15)','%.5f',float],
    'EQ15':['detrended mag. quality flag (aperture 15)','%i',int],
    'TF15':['TFA magnitude (aperture 15)','%.5f',float],
    'CF15':['Cosine filtered magnitude (aperture 15)','%.5f',float],
    # APERture 16
    'IM16':['K2 instrumental magnitude (aperture 16)','%.5f',float],
    'IE16':['K2 instrumental mag. error (aperture 16)','%.5f',float],
    'IQ16':['K2 instrumental mag. quality flag (aperture 16)','%s',str],
    'EP16':['detrended magnitude (aperture 16)','%.5f',float],
    'EQ16':['detrended mag. quality flag (aperture 16)','%i',int],
    'TF16':['TFA magnitude (aperture 16)','%.5f',float],
    'CF16':['Cosine filtered magnitude (aperture 16)','%.5f',float],
    # APERture 17
    'IM17':['K2 instrumental magnitude (aperture 17)','%.5f',float],
    'IE17':['K2 instrumental mag. error (aperture 17)','%.5f',float],
    'IQ17':['K2 instrumental mag. quality flag (aperture 17)','%s',str],
    'EP17':['detrended magnitude (aperture 17)','%.5f',float],
    'EQ17':['detrended mag. quality flag (aperture 17)','%i',int],
    'TF17':['TFA magnitude (aperture 17)','%.5f',float],
    'CF17':['Cosine filtered magnitude (aperture 17)','%.5f',float],
    # APERture 18
    'IM18':['K2 instrumental magnitude (aperture 18)','%.5f',float],
    'IE18':['K2 instrumental mag. error (aperture 18)','%.5f',float],
    'IQ18':['K2 instrumental mag. quality flag (aperture 18)','%s',str],
    'EP18':['detrended magnitude (aperture 18)','%.5f',float],
    'EQ18':['detrended mag. quality flag (aperture 18)','%i',int],
    'TF18':['TFA magnitude (aperture 18)','%.5f',float],
    'CF18':['Cosine filtered magnitude (aperture 18)','%.5f',float],
    # APERture 19
    'IM19':['K2 instrumental magnitude (aperture 19)','%.5f',float],
    'IE19':['K2 instrumental mag. error (aperture 19)','%.5f',float],
    'IQ19':['K2 instrumental mag. quality flag (aperture 19)','%s',str],
    'EP19':['detrended magnitude (aperture 19)','%.5f',float],
    'EQ19':['detrended mag. quality flag (aperture 19)','%i',int],
    'TF19':['TFA magnitude (aperture 19)','%.5f',float],
    'CF19':['Cosine filtered magnitude (aperture 19)','%.5f',float],
    # APERture 20
    'IM20':['K2 instrumental magnitude (aperture 20)','%.5f',float],
    'IE20':['K2 instrumental mag. error (aperture 20)','%.5f',float],
    'IQ20':['K2 instrumental mag. quality flag (aperture 20)','%s',str],
    'EP20':['detrended magnitude (aperture 20)','%.5f',float],
    'EQ20':['detrended mag. quality flag (aperture 20)','%i',int],
    'TF20':['TFA magnitude (aperture 20)','%.5f',float],
    'CF20':['Cosine filtered magnitude (aperture 20)','%.5f',float],
    # APERture 20
    'IM21':['K2 instrumental magnitude (aperture 21)','%.5f',float],
    'IE21':['K2 instrumental mag. error (aperture 21)','%.5f',float],
    'IQ21':['K2 instrumental mag. quality flag (aperture 21)','%s',str],
    'EP21':['detrended magnitude (aperture 21)','%.5f',float],
    'EQ21':['detrended mag. quality flag (aperture 21)','%i',int],
    'TF21':['TFA magnitude (aperture 21)','%.5f',float],
    'CF21':['Cosine filtered magnitude (aperture 21)','%.5f',float],
    # APERture 21
    'IM22':['K2 instrumental magnitude (aperture 22)','%.5f',float],
    'IE22':['K2 instrumental mag. error (aperture 22)','%.5f',float],
    'IQ22':['K2 instrumental mag. quality flag (aperture 22)','%s',str],
    'EP22':['detrended magnitude (aperture 22)','%.5f',float],
    'EQ22':['detrended mag. quality flag (aperture 22)','%i',int],
    'TF22':['TFA magnitude (aperture 22)','%.5f',float],
    'CF22':['Cosine filtered magnitude (aperture 22)','%.5f',float],
    # APERture 22
    'IM23':['K2 instrumental magnitude (aperture 23)','%.5f',float],
    'IE23':['K2 instrumental mag. error (aperture 23)','%.5f',float],
    'IQ23':['K2 instrumental mag. quality flag (aperture 23)','%s',str],
    'EP23':['detrended magnitude (aperture 23)','%.5f',float],
    'EQ23':['detrended mag. quality flag (aperture 23)','%i',int],
    'TF23':['TFA magnitude (aperture 23)','%.5f',float],
    'CF23':['Cosine filtered magnitude (aperture 23)','%.5f',float],
    # APERture 23
    'IM24':['K2 instrumental magnitude (aperture 24)','%.5f',float],
    'IE24':['K2 instrumental mag. error (aperture 24)','%.5f',float],
    'IQ24':['K2 instrumental mag. quality flag (aperture 24)','%s',str],
    'EP24':['detrended magnitude (aperture 24)','%.5f',float],
    'EQ24':['detrended mag. quality flag (aperture 24)','%i',int],
    'TF24':['TFA magnitude (aperture 24)','%.5f',float],
    'CF24':['Cosine filtered magnitude (aperture 24)','%.5f',float],
    # APERture 24
    'IM25':['K2 instrumental magnitude (aperture 25)','%.5f',float],
    'IE25':['K2 instrumental mag. error (aperture 25)','%.5f',float],
    'IQ25':['K2 instrumental mag. quality flag (aperture 25)','%s',str],
    'EP25':['detrended magnitude (aperture 25)','%.5f',float],
    'EQ25':['detrended mag. quality flag (aperture 25)','%i',int],
    'TF25':['TFA magnitude (aperture 25)','%.5f',float],
    'CF25':['Cosine filtered magnitude (aperture 25)','%.5f',float],
    # APERture 25
    'IM26':['K2 instrumental magnitude (aperture 26)','%.5f',float],
    'IE26':['K2 instrumental mag. error (aperture 26)','%.5f',float],
    'IQ26':['K2 instrumental mag. quality flag (aperture 26)','%s',str],
    'EP26':['detrended magnitude (aperture 26)','%.5f',float],
    'EQ26':['detrended mag. quality flag (aperture 26)','%i',int],
    'TF26':['TFA magnitude (aperture 26)','%.5f',float],
    'CF26':['Cosine filtered magnitude (aperture 26)','%.5f',float],
    # APERture 26
    'IM27':['K2 instrumental magnitude (aperture 27)','%.5f',float],
    'IE27':['K2 instrumental mag. error (aperture 27)','%.5f',float],
    'IQ27':['K2 instrumental mag. quality flag (aperture 27)','%s',str],
    'EP27':['detrended magnitude (aperture 27)','%.5f',float],
    'EQ27':['detrended mag. quality flag (aperture 27)','%i',int],
    'TF27':['TFA magnitude (aperture 27)','%.5f',float],
    'CF27':['Cosine filtered magnitude (aperture 27)','%.5f',float],
    # APERture 27
    'IM28':['K2 instrumental magnitude (aperture 28)','%.5f',float],
    'IE28':['K2 instrumental mag. error (aperture 28)','%.5f',float],
    'IQ28':['K2 instrumental mag. quality flag (aperture 28)','%s',str],
    'EP28':['detrended magnitude (aperture 28)','%.5f',float],
    'EQ28':['detrended mag. quality flag (aperture 28)','%i',int],
    'TF28':['TFA magnitude (aperture 28)','%.5f',float],
    'CF28':['Cosine filtered magnitude (aperture 28)','%.5f',float],
    # APERture 28
    'IM29':['K2 instrumental magnitude (aperture 29)','%.5f',float],
    'IE29':['K2 instrumental mag. error (aperture 29)','%.5f',float],
    'IQ29':['K2 instrumental mag. quality flag (aperture 29)','%s',str],
    'EP29':['detrended magnitude (aperture 29)','%.5f',float],
    'EQ29':['detrended mag. quality flag (aperture 29)','%i',int],
    'TF29':['TFA magnitude (aperture 29)','%.5f',float],
    'CF29':['Cosine filtered magnitude (aperture 29)','%.5f',float],
    # APERture 29
    'IM30':['K2 instrumental magnitude (aperture 30)','%.5f',float],
    'IE30':['K2 instrumental mag. error (aperture 30)','%.5f',float],
    'IQ30':['K2 instrumental mag. quality flag (aperture 30)','%s',str],
    'EP30':['detrended magnitude (aperture 30)','%.5f',float],
    'EQ30':['detrended mag. quality flag (aperture 30)','%i',int],
    'TF30':['TFA magnitude (aperture 30)','%.5f',float],
    'CF30':['Cosine filtered magnitude (aperture 30)','%.5f',float],
    # APERture 30
    'IM31':['K2 instrumental magnitude (aperture 31)','%.5f',float],
    'IE31':['K2 instrumental mag. error (aperture 31)','%.5f',float],
    'IQ31':['K2 instrumental mag. quality flag (aperture 31)','%s',str],
    'EP31':['detrended magnitude (aperture 31)','%.5f',float],
    'EQ31':['detrended mag. quality flag (aperture 31)','%i',int],
    'TF31':['TFA magnitude (aperture 31)','%.5f',float],
    'CF31':['Cosine filtered magnitude (aperture 31)','%.5f',float],
    # APERture 31
    'IM32':['K2 instrumental magnitude (aperture 32)','%.5f',float],
    'IE32':['K2 instrumental mag. error (aperture 32)','%.5f',float],
    'IQ32':['K2 instrumental mag. quality flag (aperture 32)','%s',str],
    'EP32':['detrended magnitude (aperture 32)','%.5f',float],
    'EQ32':['detrended mag. quality flag (aperture 32)','%i',int],
    'TF32':['TFA magnitude (aperture 32)','%.5f',float],
    'CF32':['Cosine filtered magnitude (aperture 32)','%.5f',float],
    # APERture 33
    'IM33':['K2 instrumental magnitude (aperture 33)','%.5f',float],
    'IE33':['K2 instrumental mag. error (aperture 33)','%.5f',float],
    'IQ33':['K2 instrumental mag. quality flag (aperture 33)','%s',str],
    'EP33':['detrended magnitude (aperture 33)','%.5f',float],
    'EQ33':['detrended mag. quality flag (aperture 33)','%i',int],
    'TF33':['TFA magnitude (aperture 33)','%.5f',float],
    'CF33':['Cosine filtered magnitude (aperture 33)','%.5f',float],
    # APERture 34
    'IM34':['K2 instrumental magnitude (aperture 34)','%.5f',float],
    'IE34':['K2 instrumental mag. error (aperture 34)','%.5f',float],
    'IQ34':['K2 instrumental mag. quality flag (aperture 34)','%s',str],
    'EP34':['detrended magnitude (aperture 34)','%.5f',float],
    'EQ34':['detrended mag. quality flag (aperture 34)','%i',int],
    'TF34':['TFA magnitude (aperture 34)','%.5f',float],
    'CF34':['Cosine filtered magnitude (aperture 34)','%.5f',float],
    # APERture 35
    'IM35':['K2 instrumental magnitude (aperture 35)','%.5f',float],
    'IE35':['K2 instrumental mag. error (aperture 35)','%.5f',float],
    'IQ35':['K2 instrumental mag. quality flag (aperture 35)','%s',str],
    'EP35':['detrended magnitude (aperture 35)','%.5f',float],
    'EQ35':['detrended mag. quality flag (aperture 35)','%i',int],
    'TF35':['TFA magnitude (aperture 35)','%.5f',float],
    'CF35':['Cosine filtered magnitude (aperture 35)','%.5f',float],
}


##################################
## FUNCTIONS TO READ K2 HAT LCS ##
##################################

def _parse_csv_header(header):
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
    except Exception:
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

    Parameters
    ----------

    lcfile : str
        The light curve file to read.

    Returns
    -------

    dict
        Returns an lcdict.

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
    lcdict = _parse_csv_header(lcheader)

    # tranpose the LC rows into columns
    lccolumns = list(zip(*lccolumns))

    # write the columns to the dict
    for colind, col in enumerate(lcdict['columns']):

        # this picks out the caster to use when reading each column using the
        # definitions in the lcutils.COLUMNDEFS dictionary
        lcdict[col.lower()] = np.array([COLUMNDEFS[col][2](x)
                                        for x in lccolumns[colind]])

    lcdict['columns'] = [x.lower() for x in lcdict['columns']]

    return lcdict
