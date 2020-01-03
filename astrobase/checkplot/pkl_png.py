#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pkl_png.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2019
# License: MIT.

'''
This contains utility functions that support the checkplot pickle to PNG export
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

from io import BytesIO as StrIO

import numpy as np
from numpy import nan as npnan

# import from Pillow to generate pngs from checkplot dicts
from PIL import Image, ImageDraw, ImageFont


###################
## LOCAL IMPORTS ##
###################

from ..plotbase import METHODSHORTLABELS
from .pkl_io import _read_checkplot_picklefile, _base64_to_file


###################
## MAIN FUNCTION ##
###################

def checkplot_pickle_to_png(
        checkplotin,
        outfile,
        extrarows=None
):
    '''This reads the checkplot pickle or dict provided, and writes out a PNG.

    The output PNG contains most of the information in the input checkplot
    pickle/dict, and can be used to quickly glance through the highlights
    instead of having to review the checkplot with the `checkplotserver`
    webapp. This is useful for exporting read-only views of finalized checkplots
    from the `checkplotserver` as well, to share them with other people.

    The PNG has 4 x N tiles::

        [    finder    ] [  objectinfo  ] [ varinfo/comments ] [ unphased LC  ]
        [ periodogram1 ] [ phased LC P1 ] [   phased LC P2   ] [ phased LC P3 ]
        [ periodogram2 ] [ phased LC P1 ] [   phased LC P2   ] [ phased LC P3 ]
                                         .
                                         .
        [ periodogramN ] [ phased LC P1 ] [ phased LC P2 ] [ phased LC P3 ]

    for N independent period-finding methods producing:

    - periodogram1,2,3...N: the periodograms from each method

    - phased LC P1,P2,P3: the phased lightcurves using the best 3 peaks in each
      periodogram

    Parameters
    ----------

    checkplotin : dict or str
        This is either a checkplotdict produced by
        :py:func:`astrobase.checkplot.pkl.checkplot_dict` or a checkplot pickle
        file produced by :py:func:`astrobase.checkplot.pkl.checkplot_pickle`.

    outfile : str
        The filename of the output PNG file to create.

    extrarows : list of tuples
        This is a list of 4-element tuples containing paths to PNG files that
        will be added to the end of the rows generated from the checkplotin
        pickle/dict. Each tuple represents a row in the final output PNG
        file. If there are less than 4 elements per tuple, the missing elements
        will be filled in with white-space. If there are more than 4 elements
        per tuple, only the first four will be used.

        The purpose of this kwarg is to incorporate periodograms and phased LC
        plots (in the form of PNGs) generated from an external period-finding
        function or program (like VARTOOLS) to allow for comparison with
        astrobase results.

        NOTE: the PNG files specified in `extrarows` here will be added to those
        already present in the input checkplotdict['externalplots'] if that is
        None because you passed in a similar list of external plots to the
        :py:func:`astrobase.checkplot.pkl.checkplot_pickle` function earlier. In
        this case, `extrarows` can be used to add even more external plots if
        desired.

        Each external plot PNG will be resized to 750 x 480 pixels to fit into
        an output image cell.

        By convention, each 4-element tuple should contain:

        - a periodiogram PNG
        - phased LC PNG with 1st best peak period from periodogram
        - phased LC PNG with 2nd best peak period from periodogram
        - phased LC PNG with 3rd best peak period from periodogram

        Example of extrarows::

            [('/path/to/external/bls-periodogram.png',
              '/path/to/external/bls-phasedlc-plot-bestpeak.png',
              '/path/to/external/bls-phasedlc-plot-peak2.png',
              '/path/to/external/bls-phasedlc-plot-peak3.png'),
             ('/path/to/external/pdm-periodogram.png',
              '/path/to/external/pdm-phasedlc-plot-bestpeak.png',
              '/path/to/external/pdm-phasedlc-plot-peak2.png',
              '/path/to/external/pdm-phasedlc-plot-peak3.png'),
            ...]

    Returns
    -------

    str
        The absolute path to the generated checkplot PNG.

    '''

    if (isinstance(checkplotin, str) and os.path.exists(checkplotin)):
        cpd = _read_checkplot_picklefile(checkplotin)
    elif isinstance(checkplotin, dict):
        cpd = checkplotin
    else:
        LOGERROR('checkplotin: %s of type %s is not a '
                 'valid checkplot filename (or does not exist), or a dict' %
                 (os.path.abspath(checkplotin), type(checkplotin)))
        return None

    # figure out the dimensions of the output png
    # each cell is 750 x 480 pixels
    # a row is made of four cells
    # - the first row is for object info
    # - the rest are for periodograms and phased LCs, one row per method
    # if there are more than three phased LC plots per method, we'll only plot 3
    if 'pfmethods' in cpd:
        cplspmethods = cpd['pfmethods']
    else:
        cplspmethods = []
        for pfm in METHODSHORTLABELS:
            if pfm in cpd:
                cplspmethods.append(pfm)

    cprows = len(cplspmethods)

    # add in any extra rows from neighbors
    if 'neighbors' in cpd and cpd['neighbors'] and len(cpd['neighbors']) > 0:
        nbrrows = len(cpd['neighbors'])
    else:
        nbrrows = 0

    # add in any extra rows from keyword arguments
    if extrarows and len(extrarows) > 0:
        erows = len(extrarows)
    else:
        erows = 0

    # add in any extra rows from the checkplot dict
    if ('externalplots' in cpd and
        cpd['externalplots'] and
        len(cpd['externalplots']) > 0):
        cpderows = len(cpd['externalplots'])
    else:
        cpderows = 0

    totalwidth = 3000
    totalheight = 480 + (cprows + erows + nbrrows + cpderows)*480

    # this is the output PNG
    outimg = Image.new('RGBA',(totalwidth, totalheight),(255,255,255,255))

    # now fill in the rows of the output png. we'll use Pillow to build up the
    # output image from the already stored plots and stuff in the checkplot
    # dict.

    ###############################
    # row 1, cell 1: finder chart #
    ###############################

    if cpd['finderchart']:
        finder = Image.open(
            _base64_to_file(cpd['finderchart'], None, writetostrio=True)
        )
        bigfinder = finder.resize((450,450), Image.ANTIALIAS)
        outimg.paste(bigfinder,(150,20))

    #####################################
    # row 1, cell 2: object information #
    #####################################

    # find the font we need from the package data
    fontpath = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     '..',
                     'cpserver',
                     'cps-assets',
                     'DejaVuSans.ttf')
    )
    # load the font
    if os.path.exists(fontpath):
        cpfontnormal = ImageFont.truetype(fontpath, 20)
        cpfontlarge = ImageFont.truetype(fontpath, 28)
    else:
        LOGWARNING('could not find bundled '
                   'DejaVu Sans font in the astrobase package '
                   'data, using ugly defaults...')
        cpfontnormal = ImageFont.load_default()
        cpfontlarge = ImageFont.load_default()

    # the image draw object
    objinfodraw = ImageDraw.Draw(outimg)

    # write out the object information

    # objectid
    objinfodraw.text(
        (625, 25),
        cpd['objectid'] if cpd['objectid'] else 'no objectid',
        font=cpfontlarge,
        fill=(0,0,255,255)
    )
    # twomass id
    if 'twomassid' in cpd['objectinfo']:
        objinfodraw.text(
            (625, 60),
            ('2MASS J%s' % cpd['objectinfo']['twomassid']
             if cpd['objectinfo']['twomassid']
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    # ndet
    if 'ndet' in cpd['objectinfo']:
        objinfodraw.text(
            (625, 85),
            ('LC points: %s' % cpd['objectinfo']['ndet']
             if cpd['objectinfo']['ndet'] is not None
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (625, 85),
            ('LC points: %s' % cpd['magseries']['times'].size),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    # coords and PM
    objinfodraw.text(
        (625, 125),
        ('Coords and PM'),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )
    if 'ra' in cpd['objectinfo'] and 'decl' in cpd['objectinfo']:
        objinfodraw.text(
            (900, 125),
            (('RA, Dec: %.3f, %.3f' %
              (cpd['objectinfo']['ra'], cpd['objectinfo']['decl']))
             if (cpd['objectinfo']['ra'] is not None and
                 cpd['objectinfo']['decl'] is not None)
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (900, 125),
            'RA, Dec: nan, nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    if 'propermotion' in cpd['objectinfo']:
        objinfodraw.text(
            (900, 150),
            (('Total PM: %.5f mas/yr' % cpd['objectinfo']['propermotion'])
             if (cpd['objectinfo']['propermotion'] is not None)
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (900, 150),
            'Total PM: nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    if 'rpmj' in cpd['objectinfo']:
        objinfodraw.text(
            (900, 175),
            (('Reduced PM [Jmag]: %.3f' % cpd['objectinfo']['rpmj'])
             if (cpd['objectinfo']['rpmj'] is not None)
             else ''),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
    else:
        objinfodraw.text(
            (900, 175),
            'Reduced PM [Jmag]: nan',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # here, we have to deal with two generations of objectinfo dicts

    # first, deal with the new generation of objectinfo dicts
    if 'available_dereddened_bands' in cpd['objectinfo']:

        #
        # first, we deal with the bands and mags
        #
        # magnitudes
        objinfodraw.text(
            (625, 200),
            'Magnitudes',
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        # process the various bands
        # if dereddened mags aren't available, use the observed mags
        if len(cpd['objectinfo']['available_bands']) > 0:

            # we'll get all the available mags
            for bandind, band, label in zip(
                    range(len(cpd['objectinfo']['available_bands'])),
                    cpd['objectinfo']['available_bands'],
                    cpd['objectinfo']['available_band_labels']
            ):

                thisbandmag = cpd['objectinfo'][band]

                # we'll draw stuff in three rows depending on the number of
                # bands we have to use
                if bandind in (0,1,2,3,4):

                    thispos = (900+125*bandind, 200)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (label, thisbandmag),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                elif bandind in (5,6,7,8,9):

                    rowbandind = bandind - 5

                    thispos = (900+125*rowbandind, 225)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (label, thisbandmag),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                else:

                    rowbandind = bandind - 10

                    thispos = (900+125*rowbandind, 250)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (label, thisbandmag),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

        #
        # next, deal with the colors
        #
        # colors
        if ('dereddened' in cpd['objectinfo'] and
            cpd['objectinfo']['dereddened'] is True):
            deredlabel = "(dereddened)"
        else:
            deredlabel = ""

        objinfodraw.text(
            (625, 275),
            'Colors %s' % deredlabel,
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        if len(cpd['objectinfo']['available_colors']) > 0:

            # we'll get all the available mags (dereddened versions preferred)
            for colorind, color, colorlabel in zip(
                    range(len(cpd['objectinfo']['available_colors'])),
                    cpd['objectinfo']['available_colors'],
                    cpd['objectinfo']['available_color_labels']
            ):

                thiscolor = cpd['objectinfo'][color]

                # we'll draw stuff in three rows depending on the number of
                # bands we have to use
                if colorind in (0,1,2,3,4):

                    thispos = (900+150*colorind, 275)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (colorlabel, thiscolor),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                elif colorind in (5,6,7,8,9):

                    thisrowind = colorind - 5
                    thispos = (900+150*thisrowind, 300)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (colorlabel, thiscolor),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                elif colorind in (10,11,12,13,14):

                    thisrowind = colorind - 10
                    thispos = (900+150*thisrowind, 325)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (colorlabel, thiscolor),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

                else:

                    thisrowind = colorind - 15
                    thispos = (900+150*thisrowind, 350)

                    objinfodraw.text(
                        thispos,
                        '%s: %.3f' % (colorlabel, thiscolor),
                        font=cpfontnormal,
                        fill=(0,0,0,255)
                    )

    # otherwise, deal with older generation of checkplots
    else:

        objinfodraw.text(
            (625, 200),
            ('Magnitudes'),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        objinfodraw.text(
            (900, 200),
            ('gri: %.3f, %.3f, %.3f' %
             ((cpd['objectinfo']['sdssg'] if
               ('sdssg' in cpd['objectinfo'] and
                cpd['objectinfo']['sdssg'] is not None)
               else npnan),
              (cpd['objectinfo']['sdssr'] if
               ('sdssr' in cpd['objectinfo'] and
                cpd['objectinfo']['sdssr'] is not None)
               else npnan),
              (cpd['objectinfo']['sdssi'] if
               ('sdssi' in cpd['objectinfo'] and
                cpd['objectinfo']['sdssi'] is not None)
               else npnan))),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
        objinfodraw.text(
            (900, 225),
            ('JHK: %.3f, %.3f, %.3f' %
             ((cpd['objectinfo']['jmag'] if
               ('jmag' in cpd['objectinfo'] and
                cpd['objectinfo']['jmag'] is not None)
               else npnan),
              (cpd['objectinfo']['hmag'] if
               ('hmag' in cpd['objectinfo'] and
                cpd['objectinfo']['hmag'] is not None)
               else npnan),
              (cpd['objectinfo']['kmag'] if
               ('kmag' in cpd['objectinfo'] and
                cpd['objectinfo']['kmag'] is not None)
               else npnan))),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
        objinfodraw.text(
            (900, 250),
            ('BV: %.3f, %.3f' %
             ((cpd['objectinfo']['bmag'] if
               ('bmag' in cpd['objectinfo'] and
                cpd['objectinfo']['bmag'] is not None)
               else npnan),
              (cpd['objectinfo']['vmag'] if
               ('vmag' in cpd['objectinfo'] and
                cpd['objectinfo']['vmag'] is not None)
               else npnan))),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        # colors
        if ('dereddened' in cpd['objectinfo'] and
            cpd['objectinfo']['dereddened'] is True):
            deredlabel = "(dereddened)"
        else:
            deredlabel = ""

        objinfodraw.text(
            (625, 275),
            'Colors %s' % deredlabel,
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

        objinfodraw.text(
            (900, 275),
            ('B - V: %.3f, V - K: %.3f' %
             ( (cpd['objectinfo']['bvcolor'] if
                ('bvcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['bvcolor'] is not None)
                else npnan),
               (cpd['objectinfo']['vkcolor'] if
                ('vkcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['vkcolor'] is not None)
                else npnan) )),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
        objinfodraw.text(
            (900, 300),
            ('i - J: %.3f, g - K: %.3f' %
             ( (cpd['objectinfo']['ijcolor'] if
                ('ijcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['ijcolor'] is not None)
                else npnan),
               (cpd['objectinfo']['gkcolor'] if
                ('gkcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['gkcolor'] is not None)
                else npnan) )),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )
        objinfodraw.text(
            (900, 325),
            ('J - K: %.3f' %
             ( (cpd['objectinfo']['jkcolor'] if
                ('jkcolor' in cpd['objectinfo'] and
                 cpd['objectinfo']['jkcolor'] is not None)
                else npnan),) ),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    #
    # rest of the object information
    #

    # color classification
    if ('color_classes' in cpd['objectinfo'] and
        cpd['objectinfo']['color_classes']):

        objinfodraw.text(
            (625, 375),
            ('star classification by color: %s' %
             (', '.join(cpd['objectinfo']['color_classes']))),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # GAIA neighbors
    if ( ('gaia_neighbors' in cpd['objectinfo']) and
         (cpd['objectinfo']['gaia_neighbors'] is not None) and
         (np.isfinite(cpd['objectinfo']['gaia_neighbors'])) and
         ('searchradarcsec' in cpd['objectinfo']) and
         (cpd['objectinfo']['searchradarcsec']) ):

        objinfodraw.text(
            (625, 400),
            ('%s GAIA close neighbors within %.1f arcsec' %
             (cpd['objectinfo']['gaia_neighbors'],
              cpd['objectinfo']['searchradarcsec'])),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # closest GAIA neighbor
    if ( ('gaia_closest_distarcsec' in cpd['objectinfo']) and
         (cpd['objectinfo']['gaia_closest_distarcsec'] is not None) and
         (np.isfinite(cpd['objectinfo']['gaia_closest_distarcsec'])) and
         ('gaia_closest_gmagdiff' in cpd['objectinfo']) and
         (cpd['objectinfo']['gaia_closest_gmagdiff'] is not None) and
         (np.isfinite(cpd['objectinfo']['gaia_closest_gmagdiff'])) ):

        objinfodraw.text(
            (625, 425),
            ('closest GAIA neighbor is %.1f arcsec away, '
             'GAIA mag (obj-nbr): %.3f' %
             (cpd['objectinfo']['gaia_closest_distarcsec'],
              cpd['objectinfo']['gaia_closest_gmagdiff'])),
            font=cpfontnormal,
            fill=(0,0,0,255)
        )

    # object tags
    if 'objecttags' in cpd['objectinfo'] and cpd['objectinfo']['objecttags']:

        objtagsplit = cpd['objectinfo']['objecttags'].split(',')

        # write three tags per line
        nobjtaglines = int(np.ceil(len(objtagsplit)/3.0))

        for objtagline in range(nobjtaglines):
            objtagslice = ','.join(objtagsplit[objtagline*3:objtagline*3+3])
            objinfodraw.text(
                (625, 450+objtagline*25),
                objtagslice,
                font=cpfontnormal,
                fill=(135, 54, 0, 255)
            )

    ################################################
    # row 1, cell 3: variability info and comments #
    ################################################

    # objectisvar
    objisvar = cpd['varinfo']['objectisvar']

    if objisvar == '0':
        objvarflag = 'Variable star flag not set'
    elif objisvar == '1':
        objvarflag = 'Object is probably a variable star'
    elif objisvar == '2':
        objvarflag = 'Object is probably not a variable star'
    elif objisvar == '3':
        objvarflag = 'Not sure if this object is a variable star'
    elif objisvar is None:
        objvarflag = 'Variable star flag not set'
    elif objisvar is True:
        objvarflag = 'Object is probably a variable star'
    elif objisvar is False:
        objvarflag = 'Object is probably not a variable star'
    else:
        objvarflag = 'Variable star flag: %s' % objisvar

    objinfodraw.text(
        (1650, 125),
        objvarflag,
        font=cpfontnormal,
        fill=(0,0,0,255)
    )

    # period
    objinfodraw.text(
        (1650, 150),
        ('Period [days]: %.6f' %
         (cpd['varinfo']['varperiod']
          if cpd['varinfo']['varperiod'] is not None
          else np.nan)),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )

    # epoch
    objinfodraw.text(
        (1650, 175),
        ('Epoch [JD]: %.6f' %
         (cpd['varinfo']['varepoch']
          if cpd['varinfo']['varepoch'] is not None
          else np.nan)),
        font=cpfontnormal,
        fill=(0,0,0,255)
    )

    # variability tags
    if cpd['varinfo']['vartags']:

        vartagsplit = cpd['varinfo']['vartags'].split(',')

        # write three tags per line
        nvartaglines = int(np.ceil(len(vartagsplit)/3.0))

        for vartagline in range(nvartaglines):
            vartagslice = ','.join(vartagsplit[vartagline*3:vartagline*3+3])
            objinfodraw.text(
                (1650, 225+vartagline*25),
                vartagslice,
                font=cpfontnormal,
                fill=(135, 54, 0, 255)
            )

    # object comments
    if 'comments' in cpd and cpd['comments']:

        commentsplit = cpd['comments'].split(' ')

        # write 10 words per line
        ncommentlines = int(np.ceil(len(commentsplit)/10.0))

        for commentline in range(ncommentlines):
            commentslice = ' '.join(
                commentsplit[commentline*10:commentline*10+10]
            )
            objinfodraw.text(
                (1650, 325+commentline*25),
                commentslice,
                font=cpfontnormal,
                fill=(0,0,0,255)
            )

    # this handles JSON-ified checkplots returned by LCC server
    elif 'objectcomments' in cpd and cpd['objectcomments']:

        commentsplit = cpd['objectcomments'].split(' ')

        # write 10 words per line
        ncommentlines = int(np.ceil(len(commentsplit)/10.0))

        for commentline in range(ncommentlines):
            commentslice = ' '.join(
                commentsplit[commentline*10:commentline*10+10]
            )
            objinfodraw.text(
                (1650, 325+commentline*25),
                commentslice,
                font=cpfontnormal,
                fill=(0,0,0,255)
            )

    #######################################
    # row 1, cell 4: unphased light curve #
    #######################################

    if (cpd['magseries'] and
        'plot' in cpd['magseries'] and
        cpd['magseries']['plot']):
        magseries = Image.open(
            _base64_to_file(cpd['magseries']['plot'], None, writetostrio=True)
        )
        outimg.paste(magseries,(750*3,0))

    # this handles JSON-ified checkplots from LCC server
    elif ('magseries' in cpd and isinstance(cpd['magseries'],str)):

        magseries = Image.open(
            _base64_to_file(cpd['magseries'], None, writetostrio=True)
        )
        outimg.paste(magseries,(750*3,0))

    ###############################
    # the rest of the rows in cpd #
    ###############################
    for lspmethodind, lspmethod in enumerate(cplspmethods):

        ###############################
        # the periodogram comes first #
        ###############################

        if (cpd[lspmethod] and cpd[lspmethod]['periodogram']):

            pgram = Image.open(
                _base64_to_file(cpd[lspmethod]['periodogram'], None,
                                writetostrio=True)
            )
            outimg.paste(pgram,(0,480 + 480*lspmethodind))

        #############################
        # best phased LC comes next #
        #############################

        if (cpd[lspmethod] and 0 in cpd[lspmethod] and cpd[lspmethod][0]):

            plc1 = Image.open(
                _base64_to_file(cpd[lspmethod][0]['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc1,(750,480 + 480*lspmethodind))

        # this handles JSON-ified checkplots from LCC server
        elif (cpd[lspmethod] and 'phasedlc0' in cpd[lspmethod] and
              isinstance(cpd[lspmethod]['phasedlc0']['plot'], str)):

            plc1 = Image.open(
                _base64_to_file(cpd[lspmethod]['phasedlc0']['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc1,(750,480 + 480*lspmethodind))

        #################################
        # 2nd best phased LC comes next #
        #################################

        if (cpd[lspmethod] and 1 in cpd[lspmethod] and cpd[lspmethod][1]):

            plc2 = Image.open(
                _base64_to_file(cpd[lspmethod][1]['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc2,(750*2,480 + 480*lspmethodind))

        # this handles JSON-ified checkplots from LCC server
        elif (cpd[lspmethod] and 'phasedlc1' in cpd[lspmethod] and
              isinstance(cpd[lspmethod]['phasedlc1']['plot'], str)):

            plc2 = Image.open(
                _base64_to_file(cpd[lspmethod]['phasedlc1']['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc2,(750*2,480 + 480*lspmethodind))

        #################################
        # 3rd best phased LC comes next #
        #################################

        if (cpd[lspmethod] and 2 in cpd[lspmethod] and cpd[lspmethod][2]):

            plc3 = Image.open(
                _base64_to_file(cpd[lspmethod][2]['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc3,(750*3,480 + 480*lspmethodind))

        # this handles JSON-ified checkplots from LCC server
        elif (cpd[lspmethod] and 'phasedlc2' in cpd[lspmethod] and
              isinstance(cpd[lspmethod]['phasedlc2']['plot'], str)):

            plc3 = Image.open(
                _base64_to_file(cpd[lspmethod]['phasedlc2']['plot'], None,
                                writetostrio=True)
            )
            outimg.paste(plc3,(750*3,480 + 480*lspmethodind))

    ################################
    ## ALL DONE WITH BUILDING PNG ##
    ################################

    #########################
    # add in any extra rows #
    #########################

    # from the keyword arguments
    if erows > 0:

        for erowind, erow in enumerate(extrarows):

            # make sure we never go above 4 plots in a row
            for ecolind, ecol in enumerate(erow[:4]):

                eplot = Image.open(ecol)
                eplotresized = eplot.resize((750,480), Image.ANTIALIAS)
                outimg.paste(eplotresized,
                             (750*ecolind,
                              (cprows+1)*480 + 480*erowind))

    # from the checkplotdict
    if cpderows > 0:

        for cpderowind, cpderow in enumerate(cpd['externalplots']):

            # make sure we never go above 4 plots in a row
            for cpdecolind, cpdecol in enumerate(cpderow[:4]):

                cpdeplot = Image.open(cpdecol)
                cpdeplotresized = cpdeplot.resize((750,480), Image.ANTIALIAS)
                outimg.paste(cpdeplotresized,
                             (750*cpdecolind,
                              (cprows+1)*480 + (erows*480) + 480*cpderowind))

    # from neighbors:
    if nbrrows > 0:

        # we have four tiles
        # tile 1: neighbor objectid, ra, decl, distance, unphased LC
        # tile 2: phased LC for gls
        # tile 3: phased LC for pdm
        # tile 4: phased LC for any other period finding method
        #         the priority is like so: ['bls','mav','aov','win']

        for nbrind, nbr in enumerate(cpd['neighbors']):

            # figure out which period finding methods are available for this
            # neighbor. make sure to match the ones from the actual object in
            # order of priority: 'gls','pdm','bls','aov','mav','acf','win'
            nbrlspmethods = []

            for lspmethod in cpd['pfmethods']:
                if lspmethod in nbr:
                    nbrlspmethods.append(lspmethod)

            # restrict to top three in priority
            nbrlspmethods = nbrlspmethods[:3]

            try:

                # first panel: neighbor objectid, ra, decl, distance, unphased
                # LC
                nbrlc = Image.open(
                    _base64_to_file(
                        nbr['magseries']['plot'], None, writetostrio=True
                    )
                )
                outimg.paste(nbrlc,
                             (750*0,
                              (cprows+1)*480 + (erows*480) + (cpderows*480) +
                              480*nbrind))

                # overlay the objectinfo
                objinfodraw.text(
                    (98,
                     (cprows+1)*480 + (erows*480) + (cpderows*480) +
                     480*nbrind + 15),
                    ('N%s: %s' % (nbrind + 1, nbr['objectid'])),
                    font=cpfontlarge,
                    fill=(0,0,255,255)
                )
                # overlay the objectinfo
                objinfodraw.text(
                    (98,
                     (cprows+1)*480 + (erows*480) + (cpderows*480) +
                     480*nbrind + 50),
                    ('(RA, DEC) = (%.3f, %.3f), distance: %.1f arcsec' %
                     (nbr['ra'], nbr['decl'], nbr['dist'])),
                    font=cpfontnormal,
                    fill=(0,0,255,255)
                )

                # second panel: phased LC for gls
                lsp1lc = Image.open(
                    _base64_to_file(
                        nbr[nbrlspmethods[0]][0]['plot'], None,
                        writetostrio=True
                    )
                )
                outimg.paste(lsp1lc,
                             (750*1,
                              (cprows+1)*480 + (erows*480) + (cpderows*480) +
                              480*nbrind))

                # second panel: phased LC for gls
                lsp2lc = Image.open(
                    _base64_to_file(
                        nbr[nbrlspmethods[1]][0]['plot'], None,
                        writetostrio=True
                    )
                )
                outimg.paste(lsp2lc,
                             (750*2,
                              (cprows+1)*480 + (erows*480) + (cpderows*480) +
                              480*nbrind))

                # second panel: phased LC for gls
                lsp3lc = Image.open(
                    _base64_to_file(
                        nbr[nbrlspmethods[2]][0]['plot'], None,
                        writetostrio=True
                    )
                )
                outimg.paste(lsp3lc,
                             (750*3,
                              (cprows+1)*480 + (erows*480) + (cpderows*480) +
                              480*nbrind))

            except Exception:

                LOGERROR('neighbor %s does not have a magseries plot, '
                         'measurements are probably all nan' % nbr['objectid'])

                # overlay the objectinfo
                objinfodraw.text(
                    (98,
                     (cprows+1)*480 + (erows*480) + (cpderows*480) +
                     480*nbrind + 15),
                    ('N%s: %s' %
                     (nbrind + 1, nbr['objectid'])),
                    font=cpfontlarge,
                    fill=(0,0,255,255)
                )

                if 'ra' in nbr and 'decl' in nbr and 'dist' in nbr:

                    # overlay the objectinfo
                    objinfodraw.text(
                        (98,
                         (cprows+1)*480 + (erows*480) + (cpderows*480) +
                         480*nbrind + 50),
                        ('(RA, DEC) = (%.3f, %.3f), distance: %.1f arcsec' %
                         (nbr['ra'], nbr['decl'], nbr['dist'])),
                        font=cpfontnormal,
                        fill=(0,0,255,255)
                    )

                elif 'objectinfo' in nbr:

                    # overlay the objectinfo
                    objinfodraw.text(
                        (98,
                         (cprows+1)*480 + (erows*480) + (cpderows*480) +
                         480*nbrind + 50),
                        ('(RA, DEC) = (%.3f, %.3f), distance: %.1f arcsec' %
                         (nbr['objectinfo']['ra'],
                          nbr['objectinfo']['decl'],
                          nbr['objectinfo']['distarcsec'])),
                        font=cpfontnormal,
                        fill=(0,0,255,255)
                    )

    #####################
    ## WRITE FINAL PNG ##
    #####################

    is_strio = isinstance(outfile, StrIO)

    if not is_strio:

        # check if we've stupidly copied over the same filename as the input
        # pickle to expected output file
        if outfile.endswith('pkl'):
            LOGWARNING('expected output PNG filename ends with .pkl, '
                       'changed to .png')
            outfile = outfile.replace('.pkl','.png')

    outimg.save(outfile, format='PNG', optimize=True)

    if not is_strio:
        if os.path.exists(outfile):
            LOGINFO('checkplot pickle -> checkplot PNG: %s OK' % outfile)
            return outfile
        else:
            LOGERROR('failed to write checkplot PNG')
            return None

    else:
        LOGINFO('checkplot pickle -> StringIO instance OK')
        return outfile


def cp2png(checkplotin, extrarows=None):
    '''This is just a shortened form of the function above for convenience.

    This only handles pickle files as input.

    Parameters
    ----------

    checkplotin : str
        File name of a checkplot pickle file to convert to a PNG.

    extrarows : list of tuples
        This is a list of 4-element tuples containing paths to PNG files that
        will be added to the end of the rows generated from the checkplotin
        pickle/dict. Each tuple represents a row in the final output PNG
        file. If there are less than 4 elements per tuple, the missing elements
        will be filled in with white-space. If there are more than 4 elements
        per tuple, only the first four will be used.

        The purpose of this kwarg is to incorporate periodograms and phased LC
        plots (in the form of PNGs) generated from an external period-finding
        function or program (like VARTOOLS) to allow for comparison with
        astrobase results.

        NOTE: the PNG files specified in `extrarows` here will be added to those
        already present in the input `checkplotdict['externalplots']` if that is
        None because you passed in a similar list of external plots to the
        :py:func:`astrobase.checkplot.pkl.checkplot_pickle` function earlier. In
        this case, `extrarows` can be used to add even more external plots if
        desired.

        Each external plot PNG will be resized to 750 x 480 pixels to fit into
        an output image cell.

        By convention, each 4-element tuple should contain:

        - a periodiogram PNG
        - phased LC PNG with 1st best peak period from periodogram
        - phased LC PNG with 2nd best peak period from periodogram
        - phased LC PNG with 3rd best peak period from periodogram

        Example of extrarows::

            [('/path/to/external/bls-periodogram.png',
              '/path/to/external/bls-phasedlc-plot-bestpeak.png',
              '/path/to/external/bls-phasedlc-plot-peak2.png',
              '/path/to/external/bls-phasedlc-plot-peak3.png'),
             ('/path/to/external/pdm-periodogram.png',
              '/path/to/external/pdm-phasedlc-plot-bestpeak.png',
              '/path/to/external/pdm-phasedlc-plot-peak2.png',
              '/path/to/external/pdm-phasedlc-plot-peak3.png'),
            ...]

    Returns
    -------

    str
        The absolute path to the generated checkplot PNG.

    '''

    if checkplotin.endswith('.gz'):
        outfile = checkplotin.replace('.pkl.gz','.png')
    else:
        outfile = checkplotin.replace('.pkl','.png')

    return checkplot_pickle_to_png(checkplotin, outfile, extrarows=extrarows)
