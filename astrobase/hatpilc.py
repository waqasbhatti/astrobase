'''hatpilc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - May 2017
License: MIT. See LICENSE for full text.

This is mostly for internal use. Contains functions to read text light curves
produced by the HATPI prototype system's image-subtraction photometry pipeline.

'''

####################
## SYSTEM IMPORTS ##
####################

import logging
from datetime import datetime
from traceback import format_exc
import os
import os.path
import gzip
import re

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np



#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.hatpilc' % parent_name)

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



###################
## USEFUL CONFIG ##
###################

HATIDREGEX = re.compile(r'HAT-\d{3}-\d{7}')

COLDEFS = [('rjd',float),
           ('stf',str),
           ('hat',str),
           ('xcc',float),
           ('ycc',float),
           ('xic',float),
           ('yic',float),
           ('fsv',float),
           ('fdv',float),
           ('fkv',float),
           ('bgv',float),
           ('bge',float),
           ('ifl1',float),
           ('ife1',float),
           ('irm1',float),
           ('ire1',float),
           ('irq1',str),
           ('ifl2',float),
           ('ife2',float),
           ('irm2',float),
           ('ire2',float),
           ('irq2',str),
           ('ifl3',float),
           ('ife3',float),
           ('irm3',float),
           ('ire3',float),
           ('irq3',str),
           ('iep1',float),
           ('iep2',float),
           ('iep3',float)]



##################################
## READING AND WRITING TEXT LCS ##
##################################

def read_hatpi_txtlc(lcfile):
    '''
    This reads in a textlc that is complete up to the TFA stage.

    '''

    if 'TF1' in lcfile:
        thiscoldefs = COLDEFS + [('itf1',float)]
    elif 'TF2' in lcfile:
        thiscoldefs = COLDEFS + [('itf2',float)]
    elif 'TF3' in lcfile:
        thiscoldefs = COLDEFS + [('itf3',float)]

    LOGINFO('reading %s' % lcfile)

    with gzip.open(lcfile,'r') as infd:

        lclines = infd.read().decode().split('\n')
        lclines = [x.split() for x in lclines if ('#' not in x and len(x) > 0)]
        ndet = len(lclines)

        if ndet > 0:

            lccols = list(zip(*lclines))
            lcdict = {x[0]:y for (x,y) in zip(thiscoldefs, lccols)}

            # convert to ndarray
            for col in thiscoldefs:
                lcdict[col[0]] = np.array([col[1](x) for x in lcdict[col[0]]])

        else:

            lcdict = {}
            LOGWARNING('no detections in %s' % lcfile)
            # convert to empty ndarrays
            for col in thiscoldefs:
                lcdict[col[0]] = np.array([])

        # add the object's name to the lcdict
        hatid = HATIDREGEX.findall(lcfile)
        lcdict['objectid'] = hatid[0] if hatid else 'unknown object'

        # add the columns to the lcdict
        lcdict['columns'] = [x[0] for x in thiscoldefs]

        # add some basic info
        lcdict['objectinfo'] = {
            'ndet':ndet,
            'hatid':hatid[0] if hatid else 'unknown object'
        }

    return lcdict



def lcdict_to_pickle(lcdict, outfile=None):
    '''This just writes the lcdict to a pickle.

    If outfile is None, then will try to get the name from the
    lcdict['objectid'] and write to <objectid>-hptxtlc.pkl. If that fails, will
    write to a file named hptxtlc.pkl'.

    '''

    if not outfile and lcdict['objectid']:
        outfile = '%s-hptxtlc.pkl' % lcdict['objectid']
    elif not outfile and not lcdict['objectid']:
        outfile = 'hptxtlc.pkl'

    with open(outfile,'wb') as outfd:
        pickle.dump(lcdict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.exists(outfile):
        LOGINFO('lcdict for object: %s -> %s OK' % (lcdict['objectid'],
                                                    outfile))
        return outfile
    else:
        LOGERROR('could not make a pickle for this lcdict!')
        return None



################################
## CONCATENATING LIGHT CURVES ##
################################

def concatenate_textlcs(lclist):
    '''This concatenates a list of light curves.

    Does not care about overlaps or duplicates. The light curves must all be
    from the same aperture.

    The intended use is to concatenate light curves across CCDs or instrument
    changes for a single object. These can then be normalized later using
    standard astrobase tools to search for variablity and/or periodicity.

    '''

    # read the first light curve
    lcdict = read_hatpi_textlc(lclist[0])

    # now read the rest
    for lcf in lclist[1:]:

        thislcd = read_hatpi_textlc(lcf)
        LOGINFO('adding %s to %s' % (lcf, lclist[0]))

        if thislcd['columns'] != lcdict['columns']:
            LOGERROR('file %s does not have the '
                     'same columns as first file %s, skipping...'
                     % (lcf, lclist[0]))
            continue

        else:

            for col in lcdict['columns']:
                lcdict[col] = np.concatenate((lcdict[col], thislcf[col]))

    # now we're all done concatenatin'
    return lcdict



def concatenate_textlcs_for_hatid(lcbasedir, objectid,
                                  aperture='TF1'):
    '''
    This concatenates all text LCs for an objectid with the given aperture.

    lcbasedir is the directory to start searching in.

    objectid is the object to search for.

    aperture is the aperture postfix to use: (TF1 = aperture 1,
                                              TF2 = aperture 2,
                                              TF3 = aperture 3)

    '''

    # use os.walk to go through the directories
