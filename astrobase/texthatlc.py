#!/usr/bin/env python
'''texthatlc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2016
License: MIT - see LICENSE for the full text.

This contains functions to read the original text HAT light curves. Implemented
by Luke Bouma (bouma.luke@gmail.com).

Use the read_original_textlc function to read a HAT .tfalc or .epdlc light curve
file.

'''
####################
## SYSTEM IMPORTS ##
####################

import logging
from datetime import datetime
from traceback import format_exc

import numpy as np
from numpy import nan

from astropy.io import ascii as astascii

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.texthatlc' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


def read_original_textlc(lcpath):
    '''
    Read .epdlc, and .tfalc light curves and return a corresponding labelled
    dict (if LC from <2012) or astropy table (if >=2012). Each has different
    keys that can be accessed via .keys()

    Input:
    lcpath: path (string) to light curve data, which is a textfile with HAT
    LC data.

    Example:
    dat = read_original_textlc('HAT-115-0003266.epdlc')
    '''

    LOGINFO('reading original HAT text LC: {:s}'.format(lcpath))

    N_lines_to_parse_comments = 50
    with open(lcpath, 'rb') as file:
        head = [next(file) for ind in range(N_lines_to_parse_comments)]

    N_comment_lines = len([l for l in head if l.decode('UTF-8')[0]=='#'])

    # if there are too many comment lines, fail out
    if N_comment_lines < N_lines_to_parse_comments:
        LOGERROR(
            'LC file {fpath} has too many comment lines'.format(fpath=lcpath)
        )
        return None

    first_data_line = list(
        filter(None, head[N_comment_lines].decode('UTF-8').split())
    )
    N_cols = len(first_data_line)

    # There are different column formats depending on when HAT pipeline was run
    # also different formats for different types of LCs:
    # pre-2012: .epdlc -> 17 columns
    # pre-2012: .tfalc -> 20 columns
    # post-2012: .epdlc or .tfalc -> 32 columns

    if N_cols == 17:
        colformat = 'pre2012-epdlc'
    elif N_cols == 20:
        colformat = 'pre2012-tfalc'
    elif N_cols == 32:
        colformat = 'post2012-hatlc'
    else:
        LOGERROR("can't handle this column format yet, "
                 "file: {fpath}, ncols: {ncols}".format(fpath=lcpath,
                                                        ncols=N_cols))
        return None


    # deal with pre-2012 column format
    if colformat == 'pre2012-epdlc':

        col_names = ['framekey','rjd',
                     'aim_000','aie_000','aiq_000',
                     'aim_001','aie_001','aiq_001',
                     'aim_002','aie_002','aiq_002',
                     'arm_000','arm_001','arm_002',
                     'aep_000','aep_001','aep_002']
        col_dtypes = ['U8',float,
                      float,float,'U1',
                      float,float,'U1',
                      float,float,'U1',
                      float,float,float,
                      float,float,float]
        dtype_pairs = [el for el in zip(col_names, col_dtypes)]
        data = np.genfromtxt(lcpath, names=col_names, dtype=col_dtypes,
            skip_header=N_comment_lines, delimiter=None)
        out = {}
        for ix in range(len(data.dtype.names)):
            out[data.dtype.names[ix]] = data[data.dtype.names[ix]]

    elif colformat == 'pre2012-tfalc':

        col_names = ['framekey','rjd',
                     'aim_000','aie_000','aiq_000',
                     'aim_001','aie_001','aiq_001',
                     'aim_002','aie_002','aiq_002',
                     'arm_000','arm_001','arm_002',
                     'aep_000','aep_001','aep_002',
                     'atf_000','atf_001','atf_002']
        col_dtypes = ['U8',float,
                      float,float,'U1',
                      float,float,'U1',
                      float,float,'U1',
                      float,float,float,
                      float,float,float,
                      float,float,float]
        dtype_pairs = [el for el in zip(col_names, col_dtypes)]
        data = np.genfromtxt(lcpath, names=col_names, dtype=col_dtypes,
            skip_header=N_comment_lines, delimiter=None)
        out = {}
        for ix in range(len(data.dtype.names)):
            out[data.dtype.names[ix]] = data[data.dtype.names[ix]]

    elif colformat == 'post2012-hatlc':

        col_names = ['hatid', 'framekey', 'fld', 'bjd',
                     'aim_000', 'aie_000', 'aiq_000',
                     'aim_001', 'aie_001', 'aiq_001',
                     'aim_002', 'aie_002', 'aiq_002',
                     'arm_000', 'arm_001', 'arm_002',
                     'aep_000', 'aep_001', 'aep_002',
                     'atf_000', 'atf_001', 'atf_002',
                     'xcc', 'ycc', 'bgv', 'bge',
                     'fsv', 'fdv', 'fkv',
                     'iha', 'izd', 'rjd']

        out = astascii.read(lcpath, names=col_names, comment='#')

    return out
