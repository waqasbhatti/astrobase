#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''hplc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - May 2017
License: MIT. See LICENSE for full text.

This is mostly for internal use. Contains functions to read text light curves
produced by the HATPI prototype system's image-subtraction photometry pipeline.

'''

# put this in here because hplc can be used as a standalone module
__version__ = '0.2.8'

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
import glob
import sys
import shutil
import multiprocessing as mp

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
    globals()['LOGGER'] = logging.getLogger('%s.hplc' % parent_name)

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

# used to find HATIDs
HATIDREGEX = re.compile(r'HAT-\d{3}-\d{7}')

# used to get the station ID, frame number, subframe id, and CCD number from a
# framekey or standard HAT FITS filename
FRAMEREGEX = re.compile(r'(\d{1})\-(\d{6})(\w{0,1})_(\d{1})')

# these are the columns in the input text LCs, common to all epdlc and tfalcs
# an additional column for the TFA magnitude in the current aperture is added
# for tfalcs. there are three tfalcs for each epdlc, one for each aperture.
COLDEFS = [('rjd',float),  # The reduced Julian date
           ('frk',str),    # Framekey: {stationid}-{framenum}{framesub}_{ccdnum}
           ('hat',str),    # The HATID of the object
           ('xcc',float),  # Original x coordinate on the imagesub astromref
           ('ycc',float),  # Original y coordinate on the imagesub astromref
           ('xic',float),  # Shifted x coordinate on this frame
           ('yic',float),  # Shifted y coordinate on this frame
           ('fsv',float),  # Measured S value
           ('fdv',float),  # Measured D value
           ('fkv',float),  # Measured K value
           ('bgv',float),  # Background value
           ('bge',float),  # Background measurement error
           ('ifl1',float), # Flux measurement in ADU, aperture 1
           ('ife1',float), # Flux error in ADU, aperture 1
           ('irm1',float), # Instrumental magnitude in aperture 1
           ('ire1',float), # Instrumental magnitude error for aperture 1
           ('irq1',str),   # Instrumental magnitude quality flag for aperture 1
           ('ifl2',float), # Flux measurement in ADU, aperture 2
           ('ife2',float), # Flux error in ADU, aperture 2
           ('irm2',float), # Instrumental magnitude in aperture 2
           ('ire2',float), # Instrumental magnitude error for aperture 2
           ('irq2',str),   # Instrumental magnitude quality flag for aperture 2
           ('ifl3',float), # Flux measurement in ADU, aperture 3
           ('ife3',float), # Flux error in ADU, aperture 3
           ('irm3',float), # Instrumental magnitude in aperture 3
           ('ire3',float), # Instrumental magnitude error for aperture 3
           ('irq3',str),   # Instrumental magnitude quality flag for aperture 3
           ('iep1',float), # EPD magnitude for aperture 1
           ('iep2',float), # EPD magnitude for aperture 2
           ('iep3',float)] # EPD magnitude for aperture 3

# these are the mag columns
MAGCOLS = ['ifl1','irm1','iep1','itf1',
           'ifl2','irm2','iep2','itf2',
           'ifl3','irm3','iep3','itf3']


##################################
## READING AND WRITING TEXT LCS ##
##################################

def read_hatpi_textlc(lcfile):
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

    if lcfile.endswith('.gz'):
        infd = gzip.open(lcfile,'r')
    else:
        infd = open(lcfile,'r')


    with infd:

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

        # add some basic info similar to usual HATLCs
        lcdict['objectinfo'] = {
            'ndet':ndet,
            'hatid':hatid[0] if hatid else 'unknown object',
            'network':'HP',
        }

        # break out the {stationid}-{framenum}{framesub}_{ccdnum} framekey
        # into separate columns
        framekeyelems = FRAMEREGEX.findall('\n'.join(lcdict['frk']))

        lcdict['stf'] = np.array([(int(x[0]) if x[0].isdigit() else np.nan)
                                  for x in framekeyelems])
        lcdict['cfn'] = np.array([(int(x[1]) if x[0].isdigit() else np.nan)
                                  for x in framekeyelems])
        lcdict['cfs'] = np.array([x[2] for x in framekeyelems])
        lcdict['ccd'] = np.array([(int(x[3]) if x[0].isdigit() else np.nan)
                                  for x in framekeyelems])

        # update the column list with these columns
        lcdict['columns'].extend(['stf','cfn','cfs','ccd'])

        # add more objectinfo: 'stations', etc.
        lcdict['objectinfo']['network'] = 'HP'
        lcdict['objectinfo']['stations'] = [
            'HP%s' % x for x in np.unique(lcdict['stf']).tolist()
        ]


    return lcdict



def lcdict_to_pickle(lcdict, outfile=None):
    '''This just writes the lcdict to a pickle.

    If outfile is None, then will try to get the name from the
    lcdict['objectid'] and write to <objectid>-hptxtlc.pkl. If that fails, will
    write to a file named hptxtlc.pkl'.

    '''

    if not outfile and lcdict['objectid']:
        outfile = '%s-hplc.pkl' % lcdict['objectid']
    elif not outfile and not lcdict['objectid']:
        outfile = 'hplc.pkl'

    with open(outfile,'wb') as outfd:
        pickle.dump(lcdict, outfd, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.exists(outfile):
        LOGINFO('lcdict for object: %s -> %s OK' % (lcdict['objectid'],
                                                    outfile))
        return outfile
    else:
        LOGERROR('could not make a pickle for this lcdict!')
        return None



def read_hatpi_pklc(lcfile):
    '''
    This just reads a pickle LC. Returns an lcdict.

    '''

    try:

        if lcfile.endswith('.gz'):
            infd = gzip.open(lcfile,'rb')
        else:
            infd = open(lcfile,'rb')

        lcdict = pickle.load(infd)
        infd.close()

        return lcdict

    except UnicodeDecodeError:

        if lcfile.endswith('.gz'):
            infd = gzip.open(lcfile,'rb')
        else:
            infd = open(lcfile,'rb')

        LOGWARNING('pickle %s was probably from Python 2 '
                   'and failed to load without using "latin1" encoding. '
                   'This is probably a numpy issue: '
                   'http://stackoverflow.com/q/11305790' % lcfile)
        lcdict = pickle.load(infd, encoding='latin1')
        infd.close()

        return lcdict



################################
## CONCATENATING LIGHT CURVES ##
################################

def concatenate_textlcs(lclist,
                        sortby='rjd',
                        normalize=True):
    '''This concatenates a list of light curves.

    Does not care about overlaps or duplicates. The light curves must all be
    from the same aperture.

    The intended use is to concatenate light curves across CCDs or instrument
    changes for a single object. These can then be normalized later using
    standard astrobase tools to search for variablity and/or periodicity.

    sortby is a column to sort the final concatenated light curve by in
    ascending order.

    If normalize is True, then each light curve's magnitude columns are
    normalized to zero.

    The returned lcdict has an extra column: 'lcn' that tracks which measurement
    belongs to which input light curve. This can be used with
    lcdict['concatenated'] which relates input light curve index to input light
    curve filepath. Finally, there is an 'nconcatenated' key in the lcdict that
    contains the total number of concatenated light curves.

    '''

    # read the first light curve
    lcdict = read_hatpi_textlc(lclist[0])

    # track which LC goes where
    # initial LC
    lccounter = 0
    lcdict['concatenated'] = {lccounter: os.path.abspath(lclist[0])}
    lcdict['lcn'] = np.full_like(lcdict['rjd'], lccounter)

    # normalize if needed
    if normalize:

        for col in MAGCOLS:

            if col in lcdict:
                thismedval = np.nanmedian(lcdict[col])

                # handle fluxes
                if col in ('ifl1','ifl2','ifl3'):
                    lcdict[col] = lcdict[col] / thismedval
                # handle mags
                else:
                    lcdict[col] = lcdict[col] - thismedval

    # now read the rest
    for lcf in lclist[1:]:

        thislcd = read_hatpi_textlc(lcf)

        # if the columns don't agree, skip this LC
        if thislcd['columns'] != lcdict['columns']:
            LOGERROR('file %s does not have the '
                     'same columns as first file %s, skipping...'
                     % (lcf, lclist[0]))
            continue

        # otherwise, go ahead and start concatenatin'
        else:

            LOGINFO('adding %s (ndet: %s) to %s (ndet: %s)'
                    % (lcf,
                       thislcd['objectinfo']['ndet'],
                       lclist[0],
                       lcdict[lcdict['columns'][0]].size))

            # update LC tracking
            lccounter = lccounter + 1
            lcdict['concatenated'][lccounter] = os.path.abspath(lcf)
            lcdict['lcn'] = np.concatenate((
                lcdict['lcn'],
                np.full_like(thislcd['rjd'],lccounter)
            ))

            # concatenate the columns
            for col in lcdict['columns']:

                # handle normalization for magnitude columns
                if normalize and col in MAGCOLS:

                    thismedval = np.nanmedian(thislcd[col])

                    # handle fluxes
                    if col in ('ifl1','ifl2','ifl3'):
                        thislcd[col] = thislcd[col] / thismedval
                    # handle mags
                    else:
                        thislcd[col] = thislcd[col] - thismedval

                # concatenate the values
                lcdict[col] = np.concatenate((lcdict[col], thislcd[col]))

    #
    # now we're all done concatenatin'
    #

    # make sure to add up the ndet
    lcdict['objectinfo']['ndet'] = lcdict[lcdict['columns'][0]].size

    # update the stations
    lcdict['objectinfo']['stations'] = [
            'HP%s' % x for x in np.unique(lcdict['stf']).tolist()
        ]

    # update the total LC count
    lcdict['nconcatenated'] = lccounter + 1

    # if we're supposed to sort by a column, do so
    if sortby and sortby in [x[0] for x in COLDEFS]:

        LOGINFO('sorting concatenated light curve by %s...' % sortby)
        sortind = np.argsort(lcdict[sortby])

        # sort all the measurement columns by this index
        for col in lcdict['columns']:
            lcdict[col] = lcdict[col][sortind]

        # make sure to sort the lcn index as well
        lcdict['lcn'] = lcdict['lcn'][sortind]

    LOGINFO('done. concatenated light curve has %s detections' %
            lcdict['objectinfo']['ndet'])
    return lcdict



def concatenate_textlcs_for_objectid(lcbasedir,
                                     objectid,
                                     aperture='TF1',
                                     postfix='.gz',
                                     sortby='rjd',
                                     normalize=True,
                                     recursive=True):
    '''This concatenates all text LCs for an objectid with the given aperture.

    Does not care about overlaps or duplicates. The light curves must all be
    from the same aperture.

    The intended use is to concatenate light curves across CCDs or instrument
    changes for a single object. These can then be normalized later using
    standard astrobase tools to search for variablity and/or periodicity.


    lcbasedir is the directory to start searching in.

    objectid is the object to search for.

    aperture is the aperture postfix to use: (TF1 = aperture 1,
                                              TF2 = aperture 2,
                                              TF3 = aperture 3)

    sortby is a column to sort the final concatenated light curve by in
    ascending order.

    If normalize is True, then each light curve's magnitude columns are
    normalized to zero, and the whole light curve is then normalized to the
    global median magnitude for each magnitude column.

    If recursive is True, then the function will search recursively in lcbasedir
    for any light curves matching the specified criteria. This may take a while,
    especially on network filesystems.

    The returned lcdict has an extra column: 'lcn' that tracks which measurement
    belongs to which input light curve. This can be used with
    lcdict['concatenated'] which relates input light curve index to input light
    curve filepath. Finally, there is an 'nconcatenated' key in the lcdict that
    contains the total number of concatenated light curves.

    '''
    LOGINFO('looking for light curves for %s, aperture %s in directory: %s'
            % (objectid, aperture, lcbasedir))

    if recursive == False:

        matching = glob.glob(os.path.join(lcbasedir,
                                          '*%s*%s*%s' % (objectid,
                                                         aperture,
                                                         postfix)))
    else:
        # use recursive glob for Python 3.5+
        if sys.version_info[:2] > (3,4):

            matching = glob.glob(os.path.join(lcbasedir,
                                              '**',
                                              '*%s*%s*%s' % (objectid,
                                                             aperture,
                                                             postfix)),
                                 recursive=True)
            LOGINFO('found %s files: %s' % (len(matching), repr(matching)))

        # otherwise, use os.walk and glob
        else:

            # use os.walk to go through the directories
            walker = os.walk(lcbasedir)
            matching = []

            for root, dirs, files in walker:
                for sdir in dirs:
                    searchpath = os.path.join(root,
                                              sdir,
                                              '*%s*%s*%s' % (objectid,
                                                             aperture,
                                                             prefix))
                    foundfiles = glob.glob(searchpath)

                    if foundfiles:
                        matching.extend(foundfiles)
                        LOGINFO(
                            'found %s in dir: %s' % (repr(foundfiles),
                                                     os.path.join(root,sdir))
                        )

    # now that we have all the files, concatenate them
    # a single file will be returned as normalized
    if matching and len(matching) > 0:
        clcdict = concatenate_textlcs(matching,
                                      sortby=sortby,
                                      normalize=normalize)
        return clcdict
    else:
        LOGERROR('did not find any light curves for %s and aperture %s' %
                 (objectid, aperture))
        return None



def concat_write_pklc(lcbasedir,
                      objectid,
                      aperture='TF1',
                      postfix='.gz',
                      sortby='rjd',
                      normalize=True,
                      outdir=None,
                      recursive=True):
    '''This concatenates all text LCs for the given object and writes to a pklc.

    Basically a rollup for the concatenate_textlcs_for_objectid and
    lcdict_to_pickle functions.

    '''

    concatlcd = concatenate_textlcs_for_objectid(lcbasedir,
                                                 objectid,
                                                 aperture=aperture,
                                                 sortby=sortby,
                                                 normalize=normalize,
                                                 recursive=recursive)

    if not outdir:
        outdir = 'pklcs'

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outfpath = os.path.join(outdir, '%s-%s-pklc.pkl' % (concatlcd['objectid'],
                                                        aperture))
    pklc = lcdict_to_pickle(concatlcd, outfile=outfpath)
    return pklc



def parallel_concat_worker(task):
    '''
    This is a worker for the function below.

    task[0] = lcbasedir
    task[1] = objectid
    task[2] = {'aperture','postfix','sortby','normalize','outdir','recursive'}

    '''

    lcbasedir, objectid, kwargs = task

    try:
        return concat_write_pklc(lcbasedir, objectid, **kwargs)
    except Exception as e:
        LOGEXCEPTION('failed LC concatenation for %s in %s'
                     % (objectid, lcbasedir))
        return None



def parallel_concat_lcdir(lcbasedir,
                          objectidlist,
                          aperture='TF1',
                          postfix='.gz',
                          sortby='rjd',
                          normalize=True,
                          outdir=None,
                          recursive=True,
                          nworkers=32,
                          maxworkertasks=1000):
    '''This concatenates all text LCs for the given objectidlist.


    '''

    if not outdir:
        outdir = 'pklcs'

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    tasks = [(lcbasedir, x, {'aperture':aperture,
                             'postfix':postfix,
                             'sortby':sortby,
                             'normalize':normalize,
                             'outdir':outdir,
                             'recursive':recursive}) for x in objectidlist]

    pool = mp.Pool(nworkers, maxtasksperchild=maxworkertasks)
    results = pool.map(parallel_concat_worker, tasks)

    pool.close()
    pool.join()

    return {x:y for (x,y) in zip(objectidlist, results)}



##############################################
## MERGING APERTURES FOR HATPI LIGHT CURVES ##
##############################################

def merge_hatpi_textlc_apertures(lclist):
    '''This merges all TFA text LCs with separate apertures for a single object.

    The framekey column will be used as the join column across all light curves
    in lclist. Missing values will be filled in with nans. This function assumes
    all light curves are in the format specified in COLDEFS above and readable
    by read_hatpi_textlc above (i.e. have a single column for TFA mags for a
    specific aperture at the end).

    '''

    lcaps = {}
    framekeys = []

    for lc in lclist:

        lcd = read_hatpi_textlc(lc)

        # figure what aperture this is and put it into the lcdict. if two LCs
        # with the same aperture (i.e. TF1 and TF1) are provided, the later one
        # in the lclist will overwrite the previous one,
        for col in lcd['columns']:
            if col.startswith('itf'):
                lcaps[col] = lcd
        thisframekeys = lcd['frk'].tolist()
        framekeys.extend(thisframekeys)

    # uniqify the framekeys
    framekeys = sorted(list(set(framekeys)))

    # FIXME: finish this



#######################################
## READING BINNED HATPI LIGHT CURVES ##
#######################################

def read_hatpi_binnedlc(binnedpklf, textlcf, timebinsec):
    '''This reads a binnedlc pickle produced by the HATPI prototype pipeline.

    Converts it into a standard lcdict as produced by the read_hatpi_textlc
    function above by using the information in unbinnedtextlc for the same
    object.

    Adds a 'binned' key to the standard lcdict containing the binned mags, etc.

    '''

    LOGINFO('reading binned LC %s' % binnedpklf)

    # read the textlc
    lcdict = read_hatpi_textlc(textlcf)

    # read the binned LC

    if binnedpklf.endswith('.gz'):
        infd = gzip.open(binnedpklf,'rb')
    else:
        infd = open(binnedpklf,'rb')


    try:
        binned = pickle.load(infd)
    except:
        infd.seek(0)
        binned = pickle.load(infd, encoding='latin1')
    infd.close()

    # now that we have both, pull out the required columns from the binnedlc
    blckeys = binned.keys()

    lcdict['binned'] = {}

    for key in blckeys:

        # get EPD stuff
        if (key == 'epdlc' and
            'AP0' in binned[key] and
            'AP1' in binned[key] and
            'AP2' in binned[key]):

            # we'll have to generate errors because we don't have any in the
            # generated binned LC.

            ap0mad = np.nanmedian(np.abs(binned[key]['AP0'] -
                                         np.nanmedian(binned[key]['AP0'])))
            ap1mad = np.nanmedian(np.abs(binned[key]['AP1'] -
                                         np.nanmedian(binned[key]['AP1'])))
            ap2mad = np.nanmedian(np.abs(binned[key]['AP2'] -
                                         np.nanmedian(binned[key]['AP2'])))


            lcdict['binned']['iep1'] = {'times':binned[key]['RJD'],
                                        'mags':binned[key]['AP0'],
                                        'errs':np.full_like(binned[key]['AP0'],
                                                            ap0mad),
                                        'nbins':binned[key]['nbins'],
                                        'timebins':binned[key]['jdbins'],
                                        'timebinsec':timebinsec}
            lcdict['binned']['iep2'] = {'times':binned[key]['RJD'],
                                        'mags':binned[key]['AP1'],
                                        'errs':np.full_like(binned[key]['AP1'],
                                                            ap1mad),
                                        'nbins':binned[key]['nbins'],
                                        'timebins':binned[key]['jdbins'],
                                        'timebinsec':timebinsec}
            lcdict['binned']['iep3'] = {'times':binned[key]['RJD'],
                                        'mags':binned[key]['AP2'],
                                        'errs':np.full_like(binned[key]['AP2'],
                                                            ap2mad),
                                        'nbins':binned[key]['nbins'],
                                        'timebins':binned[key]['jdbins'],
                                        'timebinsec':timebinsec}

        # get TFA stuff for aperture 1
        if ((key == 'tfalc.TF1' or key == 'tfalc.TF1.gz') and
            'AP0' in binned[key]):

            # we'll have to generate errors because we don't have any in the
            # generated binned LC.

            ap0mad = np.nanmedian(np.abs(binned[key]['AP0'] -
                                         np.nanmedian(binned[key]['AP0'])))


            lcdict['binned']['itf1'] = {'times':binned[key]['RJD'],
                                        'mags':binned[key]['AP0'],
                                        'errs':np.full_like(binned[key]['AP0'],
                                                            ap0mad),
                                        'nbins':binned[key]['nbins'],
                                        'timebins':binned[key]['jdbins'],
                                        'timebinsec':timebinsec}

        # get TFA stuff for aperture 1
        if ((key == 'tfalc.TF2' or key == 'tfalc.TF2.gz') and
            'AP0' in binned[key]):

            # we'll have to generate errors because we don't have any in the
            # generated binned LC.

            ap0mad = np.nanmedian(np.abs(binned[key]['AP0'] -
                                         np.nanmedian(binned[key]['AP0'])))


            lcdict['binned']['itf2'] = {'times':binned[key]['RJD'],
                                        'mags':binned[key]['AP0'],
                                        'errs':np.full_like(binned[key]['AP0'],
                                                            ap0mad),
                                        'nbins':binned[key]['nbins'],
                                        'timebins':binned[key]['jdbins'],
                                        'timebinsec':timebinsec}

        # get TFA stuff for aperture 1
        if ((key == 'tfalc.TF3' or key == 'tfalc.TF3.gz') and
            'AP0' in binned[key]):

            # we'll have to generate errors because we don't have any in the
            # generated binned LC.

            ap0mad = np.nanmedian(np.abs(binned[key]['AP0'] -
                                         np.nanmedian(binned[key]['AP0'])))


            lcdict['binned']['itf3'] = {'times':binned[key]['RJD'],
                                        'mags':binned[key]['AP0'],
                                        'errs':np.full_like(binned[key]['AP0'],
                                                            ap0mad),
                                        'nbins':binned[key]['nbins'],
                                        'timebins':binned[key]['jdbins'],
                                        'timebinsec':timebinsec}

    # all done, check if we succeeded
    if lcdict['binned']:

        return lcdict

    else:

        LOGERROR('no binned measurements found in %s!' % binnedpklf)
        return None



def generate_hatpi_binnedlc_pkl(binnedpklf, textlcf, timebinsec,
                                outfile=None):
    '''
    This reads the binned LC and writes it out to a pickle.

    '''

    binlcdict = read_hatpi_binnedlc(binnedpklf, textlcf, timebinsec)

    if binlcdict:
        if outfile is None:
            outfile = os.path.join(
                os.path.dirname(binnedpklf),
                '%s-hplc.pkl' % (
                    os.path.basename(binnedpklf).rstrip('sec-lc.pkl.gz')
                )
            )

        return lcdict_to_pickle(binlcdict, outfile=outfile)
    else:
        LOGERROR('could not read binned HATPI LC: %s' % binnedpklf)
        return None



def parallel_gen_binnedlc_pkls(binnedpkldir,
                               textlcdir,
                               timebinsec,
                               binnedpklglob='*binned*sec*.pkl',
                               textlcglob='*.tfalc.TF1*'):
    '''
    This generates the binnedlc pkls for a directory of such files.

    FIXME: finish this

    '''

    binnedpkls = sorted(glob.glob(os.path.join(binnedpkldir, binnedpklglob)))

    # find all the textlcs associated with these
    textlcs = []

    for bpkl in binnedpkls:

        objectid = HATIDREGEX.findall(bpkl)
        if objectid is not None:
            objectid = objectid[0]

        searchpath = os.path.join(textlcdir, '%s-%s' % (objectid, textlcglob))
        textlcf = glob.glob(searchpath)
        if textlcf:
            textlcs.append(textlcf)
        else:
            textlcs.append(None)




#####################
## POST-PROCESSING ##
#####################

def pklc_fovcatalog_objectinfo(
        pklcdir,
        fovcatalog,
        fovcatalog_columns=[0,1,2,
                            6,7,
                            8,9,
                            10,11,
                            13,14,15,16,
                            17,18,19,
                            20,21],
        fovcatalog_colnames=['objectid','ra','decl',
                             'jmag','jmag_err',
                             'hmag','hmag_err',
                             'kmag','kmag_err',
                             'bmag','vmag','rmag','imag',
                             'sdssu','sdssg','sdssr',
                             'sdssi','sdssz'],
        fovcatalog_colformats=('U20,f8,f8,'
                               'f8,f8,'
                               'f8,f8,'
                               'f8,f8,'
                               'f8,f8,f8,f8,'
                               'f8,f8,f8,'
                               'f8,f8')
):
    '''Adds catalog info to objectinfo key of all pklcs in lcdir.

    If fovcatalog, fovcatalog_columns, fovcatalog_colnames are provided, uses
    them to find all the additional information listed in the fovcatalog_colname
    keys, and writes this info to the objectinfo key of each lcdict. This makes
    it easier for astrobase tools to work on these light curve.

    The default set up for fovcatalog is to use a text file generated by the
    HATPI pipeline before auto-calibrating a field. The format is specified as
    above in _columns,  _colnames, and _colformats.

    '''

    if fovcatalog.endswith('.gz'):
        catfd = gzip.open(fovcatalog)
    else:
        catfd = open(fovcatalog)

    # read the catalog using the colformats, etc.
    fovcat = np.genfromtxt(catfd,
                           usecols=fovcatalog_columns,
                           names=fovcatalog_colnames,
                           dtype=fovcatalog_colformats)
    catfd.close()

    pklclist = sorted(glob.glob(os.path.join(pklcdir, '*HAT*-pklc.pkl')))

    updatedpklcs, failedpklcs = [], []

    for pklc in pklclist:

        lcdict = read_hatpi_pklc(pklc)
        objectid = lcdict['objectid']

        catind = np.where(fovcat['objectid'] == objectid)

        # if we found catalog info for this object, put it into objectinfo
        if len(catind) > 0 and catind[0]:

            lcdict['objectinfo'].update(
                {x:y for x,y in zip(
                    fovcatalog_colnames,
                    [np.asscalar(fovcat[z][catind]) for z in fovcatalog_colnames]
                )
                }
            )

            # write the LC back to the pickle (tempfile for atomicity)
            with open(pklc+'-tmp','wb') as outfd:
                pickle.dump(lcdict, outfd, pickle.HIGHEST_PROTOCOL)

            # overwrite previous once we know it exists
            if os.path.exists(pklc+'-tmp'):
                shutil.move(pklc+'-tmp',pklc)

                LOGINFO('updated %s with catalog info for %s at %.3f, %.3f OK' %
                        (pklc, objectid,
                         lcdict['objectinfo']['ra'],
                         lcdict['objectinfo']['decl']))

                updatedpklcs.append(pklc)

        # otherwise, do nothing
        else:
            failedpklcs.append(pklc)

    # end of pklclist processing
    return updatedpklcs, failedpklcs
