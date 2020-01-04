#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pkl_io.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
# License: MIT.

'''
This contains utility functions that support the checkplot.pkl input/output
functionality.

'''

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

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

import os
import os.path
import gzip
import base64

import pickle
from io import BytesIO as StrIO

from tornado.escape import squeeze


#######################
## BASE64 OPERATIONS ##
#######################

def _base64_to_file(b64str, outfpath, writetostrio=False):
    '''This converts the base64 encoded string to a file.

    Parameters
    ----------

    b64str : str
        A base64 encoded strin that is the output of `base64.b64encode`.

    outfpath : str
        The path to where the file will be written. This should include an
        appropriate extension for the file (e.g. a base64 encoded string that
        represents a PNG should have its `outfpath` end in a '.png') so the OS
        can open these files correctly.

    writetostrio : bool
        If this is True, will return a StringIO object with the binary stream
        decoded from the base64-encoded input string `b64str`. This can be
        useful to embed these into other files without having to write them to
        disk.

    Returns
    -------

    str or StringIO object
        If `writetostrio` is False, will return the output file's path as a
        str. If it is True, will return a StringIO object directly. If writing
        the file fails in either case, will return None.

    '''

    try:

        filebytes = base64.b64decode(b64str)

        # if we're writing back to a stringio object
        if writetostrio:

            outobj = StrIO(filebytes)
            return outobj

        # otherwise, we're writing to an actual file
        else:

            with open(outfpath,'wb') as outfd:
                outfd.write(filebytes)

            if os.path.exists(outfpath):
                return outfpath
            else:
                LOGERROR('could not write output file: %s' % outfpath)
                return None

    except Exception:

        LOGEXCEPTION('failed while trying to convert '
                     'b64 string to file %s' % outfpath)
        return None


########################
## READ/WRITE PICKLES ##
########################

def _read_checkplot_picklefile(checkplotpickle):
    '''This reads a checkplot gzipped pickle file back into a dict.

    NOTE: the try-except is for Python 2 pickles that have numpy arrays in
    them. Apparently, these aren't compatible with Python 3. See here:

    http://stackoverflow.com/q/11305790

    The workaround is noted in this answer:

    http://stackoverflow.com/a/41366785

    Parameters
    ----------

    checkplotpickle : str
        The path to a checkplot pickle file. This can be a gzipped file (in
        which case the file extension should end in '.gz')

    Returns
    -------

    dict
        This returns a checkplotdict.

    '''

    if checkplotpickle.endswith('.gz'):

        try:
            with gzip.open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd)

        except UnicodeDecodeError:

            with gzip.open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd, encoding='latin1')

    else:

        try:
            with open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd)

        except UnicodeDecodeError:

            with open(checkplotpickle,'rb') as infd:
                cpdict = pickle.load(infd, encoding='latin1')

    return cpdict


def _write_checkplot_picklefile(checkplotdict,
                                outfile=None,
                                protocol=None,
                                outgzip=False):

    '''This writes the checkplotdict to a (gzipped) pickle file.

    Parameters
    ----------

    checkplotdict : dict
        This the checkplotdict to write to the pickle file.

    outfile : None or str
        The path to the output pickle file to write. If `outfile` is None,
        writes a (gzipped) pickle file of the form:

        checkplot-{objectid}.pkl(.gz)

        to the current directory.

    protocol : int
        This sets the pickle file protocol to use when writing the pickle:

        If None, will choose a protocol using the following rules:

        - 4 -> default in Python >= 3.4 - fast but incompatible with Python 2
        - 3 -> default in Python 3.0-3.3 - mildly fast
        - 2 -> default in Python 2 - very slow, but compatible with Python 2/3

        The default protocol kwarg is None, this will make an automatic choice
        for pickle protocol that's best suited for the version of Python in
        use. Note that this will make pickles generated by Py3 incompatible with
        Py2.

    outgzip : bool
        If this is True, will gzip the output file. Note that if the `outfile`
        str ends in a gzip, this will be automatically turned on.

    Returns
    -------

    str
        The absolute path to the written checkplot pickle file. None if writing
        fails.

    '''

    # for Python >= 3.4; use v4 by default
    if not protocol:
        protocol = 4

    if outgzip:

        if not outfile:

            outfile = (
                'checkplot-{objectid}.pkl.gz'.format(
                    objectid=squeeze(checkplotdict['objectid']).replace(' ','-')
                )
            )

        with gzip.open(outfile,'wb') as outfd:
            pickle.dump(checkplotdict,outfd,protocol=protocol)

    else:

        if not outfile:

            outfile = (
                'checkplot-{objectid}.pkl'.format(
                    objectid=squeeze(checkplotdict['objectid']).replace(' ','-')
                )
            )

        # make sure to do the right thing if '.gz' is in the filename but
        # outgzip was False
        if outfile.endswith('.gz'):

            LOGWARNING('output filename ends with .gz but kwarg outgzip=False. '
                       'will use gzip to compress the output pickle')
            with gzip.open(outfile,'wb') as outfd:
                pickle.dump(checkplotdict,outfd,protocol=protocol)

        else:
            with open(outfile,'wb') as outfd:
                pickle.dump(checkplotdict,outfd,protocol=protocol)

    return os.path.abspath(outfile)
