#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# imageutils.py
# Waqas Bhatti and Luke Bouma - Aug 2019
# (wbhatti@astro.princeton.edu and luke@astro.princeton.edu)

'''Utilities for operating on FITS images.

For the time being, this sub-package only contains functions to shorten patterns
used when reading headers, comments, and data from FITS files.

- :py:func:`astrobase.imageutils.get_header_keyword`: get value of a header
  keyword.

- :py:func:`astrobase.imageutils.get_data_keyword`: get value of a data
  entry.

- :py:func:`astrobase.imageutils.get_header_keyword_list`: get values of a list
  of header keywords.

- :py:func:`astrobase.imageutils.get_header_comment_list`: get values of a list
  of header comments.

- :py:func:`astrobase.imageutils.get_data_keyword_list`: get values of a list
  of data entries. (E.g., ['FLUX','TMID_BJD'] for a light curve file).

'''

import astropy.io.fits as pyfits


def get_header_keyword(fits_file,
                       keyword,
                       ext=0):
    '''This gets a single header keyword out of the FITS header.

    Parameters
    ----------
    fits_file : str
        Path to fits file to open.

    keyword : str
        Keyword to access in the header.

    ext : int
        FITS extension number.

    Returns
    -------
    val : int, float, or str
        The value from the header key requested.

    '''

    hdulist = pyfits.open(fits_file)

    if keyword in hdulist[ext].header:
        val = hdulist[ext].header[keyword]
    else:
        val = None

    hdulist.close()
    return val


def get_data_keyword(fits_file,
                     keyword,
                     ext=1):
    '''This gets a single data array out of a FITS binary table.

    Parameters
    ----------
    fits_file : str
        Path to fits file to open.

    keyword : str
        Keyword to access in the header.

    ext : int
        FITS extension number.

    Returns
    -------
    val : ndarray
        The value in the FITS data table header.

    '''

    hdulist = pyfits.open(fits_file)

    if keyword in hdulist[ext].data.names:
        val = hdulist[ext].data[keyword]
    else:
        val = None

    hdulist.close()
    return val


def get_header_keyword_list(fits_file,
                            keyword_list,
                            ext=0):
    '''This gets a list of FITS header keywords out of a FITS header.

    Parameters
    ----------
    fits_file : str
        Path to fits file to open.

    keyword_list : str
        List of keywords to return.

    ext : int
        FITS extension number.

    Returns
    -------
    out_dict : dict
        Dict with keys from ``keyword_list`` and values whatever was in the
        header.

    '''

    hdulist = pyfits.open(fits_file)

    out_dict = {}

    for keyword in keyword_list:

        if keyword in hdulist[ext].header:
            out_dict[keyword] = hdulist[ext].header[keyword]
        else:
            out_dict[keyword] = None

    hdulist.close()
    return out_dict


def get_header_comment_list(fits_file,
                            keyword_list,
                            ext=0):
    ''' This gets a list of comment keys from the FITS header.

    Parameters
    ----------
    fits_file : str
        Path to fits file to open.

    keyword_list : str
        List of keywords to return.

    ext : int
        FITS extension number.

    Returns
    -------
    out_dict : dict
        Dict with keys from ``keyword_list`` and values whatever was in the
        header comments.

    '''

    hdulist = pyfits.open(fits_file)

    out_dict = {}

    for keyword in keyword_list:

        if keyword in hdulist[ext].header:
            out_dict[keyword] = hdulist[ext].header.comments[keyword]
        else:
            out_dict[keyword] = None

    hdulist.close()
    return out_dict


def get_data_keyword_list(fits_file,
                          keyword_list,
                          ext=1):
    '''This gets FITS binary table data arrays from a list of header keys.

    Parameters
    ----------
    fits_file : str
        Path to fits file to open.

    keyword_list : str
        List of keywords to return.

    ext : int
        FITS extension number.

    Returns
    -------
    out_dict : dict
        Dict with keys from ``keyword_list`` and values whatever is in the FITS
        data extension.
    '''

    hdulist = pyfits.open(fits_file)

    out_dict = {}

    for keyword in keyword_list:

        if keyword in hdulist[ext].data.names:
            out_dict[keyword] = hdulist[ext].data[keyword]
        else:
            out_dict[keyword] = None

    hdulist.close()
    return out_dict
