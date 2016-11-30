#!/usr/bin/env python

'''
imageutils.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2013

This contains various utilities for:

- generating stamps for an image, converting an image to JPEGs
- getting the value of a certain keyword from the FITS header for a series of
  FITS files

Important FITS header keywords:

FOCUS (steps)
BJD
MIDTIME (HH:MM:SS.SSS - middle of exposure)
MIDDATE (YYYY-MM-DD - middle of exposure)
TIMESYS (should be UTC)
OBJECT (field name)
JD (JD of middle exposure)
HA (hour angle)
Z (zenith distance)
ABORTED (0 = exp not aborted, 1 = aborted exp)
IMAGETYP

'''

import os
import os.path
import sys
import logging
import glob

import numpy as np
np.seterr(all='ignore')

import numpy.ma as npma
import numpy.random as npr

import scipy.misc
import scipy.ndimage
import scipy

from scipy.optimize import leastsq
USE_LEASTSQ = 1

try:
    from scipy.optimize import curve_fit
    USE_LEASTSQ=0
except:
    print('cannot import curve_fit, will use leastsq')
    USE_LEASTSQ=1

try:
    import pyfits
except:
    from astropy.io import fits as pyfits

from PIL import Image
from PIL import ImageDraw

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.imageutils' % parent_name)


## FITS UTILITIES ##


def read_fits(fits_file,ext=0):
    '''
    Shortcut function to get the header and data from a fits file and a given
    extension.

    '''

    hdulist = pyfits.open(fits_file)
    img_header = hdulist[ext].header
    img_data = hdulist[ext].data
    hdulist.close()

    return img_data, img_header


def read_fits_header(fits_file, ext=0):
    '''
    Shortcut function to just read the header of the FITS file and return it.

    '''
    hdulist = pyfits.open(fits_file)
    img_header = hdulist[ext].header
    hdulist.close()

    return img_header



def trim_image(fits_img,
               fits_hdr,
               custombox=None):
    '''
    Returns a trimmed image using the TRIMSEC header of the image header.
    FIXME: check if this does the right thing.

    custombox is a string of the form [Xlo:Xhi,Ylo:Yhi] and will trim the image
    to a custom size.

    '''

    if custombox:

        trimsec = custombox

    else:

        if 'TRIMSEC' in fits_hdr:
            trimsec = fits_hdr['TRIMSEC']
        elif 'DATASEC' in fits_hdr:
            trimsec = fits_hdr['DATASEC']
        else:
            if custombox is None:
                if LOGGER:
                    LOGGER.error('no DATASEC or TRIMSEC in image header')
                else:
                    print('cannot trim image, no DATASEC or '
                          'TRIMSEC in image header')
                return

    if trimsec != '[0:0,0:0]':

        datasec = trimsec.strip('[]').split(',')

        try:
            datasec_y = [int(x) for x in datasec[0].split(':')]
            datasec_x = [int(x) for x in datasec[1].split(':')]

            trimmed_img = fits_img[datasec_x[0]-1:datasec_x[1],
                                   datasec_y[0]-1:datasec_y[1]]
        except ValueError as e:
            if LOGGER:
                LOGGER.error('datasec/trimsec not correctly set in FITS header, '
                             ' not trimming')
            else:
                print('datasec/trimsec not correctly set in FITS header, '
                      ' not trimming')
            trimmed_img = fits_img

    else:
        if LOGGER:
            LOGGER.error('datasec/trimsec not correctly set in FITS header, '
                         ' not trimming')
        else:
            print('datasec/trimsec not correctly set in FITS header, '
                  ' not trimming')
        trimmed_img = fits_img

    return trimmed_img



def make_superflat(image_glob,
                   fits_imagetype_card = 'IMAGETYP',
                   fits_flat_keyword='flat',
                   smoothlevel=11,
                   ext=None,
                   method='mean',
                   saveto=None):
    '''
    This generates a normalized superflat image for a series of flatfield
    images.

    1. finds all flat field images in image_glob
    2. takes their average
    3. normalizes by dividing out the median value (optional)
    4. smooths the flatfield image so that small scale problems still show up
       when this flat field is divided from the object frames (optional)

    '''

    image_flist = sorted(glob.glob(image_glob))

    # go through the images and find all the flats

    flat_imgs = {}
    flat_count = 0

    for fits_image in image_flist:

        compressed_ext = compressed_fits_ext(fits_image)

        if ext is None and compressed_ext:
            img, hdr = read_fits(fits_image,
                                 ext=compressed_ext[0])
        elif (ext is not None):
            img, hdr = read_fits(fits_image,ext=ext)
        else:
            img, hdr = read_fits(fits_image)

        if hdr[fits_imagetype_card] == fits_flat_keyword:
            trimmed_img = trim_image(img, hdr)
            flat_imgs[fits_image] = trimmed_img
            flat_count = flat_count + 1
            print('found flat %s' % fits_image)

    if flat_count > 1:

        all_flats = np.asarray([flat_imgs[k] for k in flat_imgs])
        del flat_imgs

        # get the median/mean of the flats depending on method
        if method == 'mean':
            median_flat = np.mean(all_flats,axis=0)
        elif method == 'median':
            median_flat = np.median(all_flats,axis=0)

        smoothed_flat = scipy.ndimage.median_filter(median_flat,
                                                    size=smoothlevel)

        if saveto:
            pyfits.writeto(saveto,smoothed_flat)
        else:
            return smoothed_flat

    else:

        return None



def compressed_fits_ext(fits_file):
    '''
    Check if a fits file is a compressed FITS file. Return the extension numbers
    of the compressed image as a list if these exist, otherwise, return None.

    '''

    hdulist = pyfits.open(fits_file)

    compressed_img_exts = []

    for i, ext in enumerate(hdulist):
        if isinstance(ext,pyfits.hdu.compressed.CompImageHDU):
            compressed_img_exts.append(i)

    hdulist.close()

    if len(compressed_img_exts) < 1:
        return None
    else:
        return compressed_img_exts


def get_header_keyword(fits_file,
                       keyword,
                       ext=0):
    '''
    Get the value of a header keyword in a fits file optionally using an
    extension.

    '''
    hdulist = pyfits.open(fits_file)

    if keyword in hdulist[ext].header:
        val = hdulist[ext].header[keyword]
    else:
        val = None

    hdulist.close()
    return val


def get_header_keyword_list(fits_file,
                            keyword_list,
                            ext=0):

    hdulist = pyfits.open(fits_file)

    out_dict = {}

    for keyword in keyword_list:

        if keyword in hdulist[ext].header:
            out_dict[keyword] = hdulist[ext].header[keyword]
        else:
            out_dict[keyword] = None

    hdulist.close()
    return out_dict


## IMAGE SCALING FUNCTIONS ##

def pixel_scale_func(x, m, c):
    return m*x + c


def pixel_scale_func_residual(params, x, y):

    f = pixel_scale_func(x, params[0], params[1])
    return y - f


def zscale_img(img_array,
               cap=255.0,
               fracsamples=0.1):
    '''
    This scales the image pixels in a manner similar to what DS9 does when
    zscale and linear are selected in the scale menu.

    Algorithm found here:

    http://iraf.net/phpBB2/viewtopic.php?t=77998&sid=b5ee7df81074f31fa7086aa1f31a74be

    Quoting the second comment from there:

    - sample the image (1000 points or so depending on size) in a grid covering
      the full frame to get a representative sample of all pixels in the image.

    - sort the sample pixels to get min/max/median values

    - iteratively fit a line to map the sample data to the number of pixels you
      want on output (e.g. 256 for 8-bit data, ximtool uses 200 or so for the
      display). Some of the sample pixels are usually rejected at each iteration
      to get a better fit.

    - from the fitted slope derive the optimal z1/z2 end values for the
      data. When mapping the image, input values outside this range maps to the
      extremes of your output range, everything in between maps linearly. The
      brightness/contrast adjustments are done by changing the offset and slope
      of this linear transformation respectively.  In display servers this
      usually means just rewriting the colormap for the image but it could also
      be used to remap all the image pixels.

      nsamples = fraction of total pixels to use for statistics
      cap = fix the max value to be within range 0-255

    '''

    img_shape = img_array.shape

    total_pixels = img_shape[0]*img_shape[1]
    nsamples = int(np.floor(fracsamples*total_pixels))

    random_index_x = npr.random_integers(0,high=img_shape[1]-1,size=nsamples)
    random_index_y = npr.random_integers(0,high=img_shape[0]-1,size=nsamples)

    # the x values
    img_sample = img_array[random_index_x, random_index_y]

    sample_med, sample_min, sample_max = (np.median(img_sample),
                                          np.nanmin(img_sample),
                                          np.nanmax(img_sample))
    sample_std = np.std(img_sample)

    trimmed_sample_ind = np.where(abs(img_sample - sample_med) < 1.0*sample_std)
    trimmed_sample = img_sample[trimmed_sample_ind]

    trimmed_sample = np.sort(trimmed_sample)

    # the y values: we're mapping our img_sample to a range between 0 and cap
    pixel_scale = np.linspace(0, cap, num=len(trimmed_sample))

    initial_slope = np.median(pixel_scale/trimmed_sample)
    initial_intercept = (np.median(pixel_scale) -
                         initial_slope*np.median(trimmed_sample))

    if USE_LEASTSQ == 1:

        params = leastsq(pixel_scale_func_residual,
                               np.array([initial_slope,
                                         initial_intercept]),
                               args=(trimmed_sample, pixel_scale))

        scale_params = params[0]

    else:
        scale_params, scale_covariance = curve_fit(pixel_scale_func,
                                                   trimmed_sample,
                                                   pixel_scale,
                                                   p0=(initial_slope,
                                                       initial_intercept))

    sample_med, sample_min, sample_max = (np.median(trimmed_sample),
                                          np.nanmin(trimmed_sample),
                                          np.nanmax(trimmed_sample))

    min_scale_param = sample_min*scale_params[0] + scale_params[1]
    max_scale_param = sample_max*scale_params[0] + scale_params[1]

    print(min_scale_param,
          max_scale_param)

    print(np.min(img_array), np.max(img_array))

    clipped_image_array = np.clip(img_array, min_scale_param, max_scale_param)

    return scale_params[0]*clipped_image_array + scale_params[1]


def clipped_linscale_img(img_array,
                         cap=255.0,
                         lomult=2.0,
                         himult=2.0):
    '''
    This clips the image between the values:

    [median(img_array) - lomult*stdev(img_array),
     median(img_array) + himult*stdev(img_array)]

    and returns a linearly scaled image using the cap given.

    '''

    img_med, img_stdev = np.median(img_array), np.std(img_array)
    clipped_linear_img = np.clip(img_array,
                                 img_med-lomult*img_stdev,
                                 img_med+himult*img_stdev)
    return cap*clipped_linear_img/(img_med+himult*img_stdev)



def logscale_img(img_array,
                 cap=255.0,
                 coeff=1000.0):
    '''
    This scales the image according to the relation:

    logscale_img = np.log(coeff*(img/max(img))+1)/np.log(coeff)

    Taken from the DS9 scaling algorithms page at:

    http://hea-www.harvard.edu/RD/ds9/ref/how.html

    According to that page:

    coeff = 1000.0 works well for optical images
    coeff = 100.0 works well for IR images

    '''

    logscaled_img = np.log(coeff*img_array/np.nanmax(img_array)+1)/np.log(coeff)
    return cap*logscaled_img


def extract_img_background(img_array,
                           custom_limits=None,
                           median_diffbelow=200.0,
                           image_min=None):
    '''
    This extracts the background of the image array provided:

    - masks the array to only values between the median and the min of flux
    - then returns the median value in 3 x 3 stamps.

    img_array = image to find the background for

    custom_limits = use this to provide custom median and min limits for the
                    background extraction

    median_diffbelow = subtract this value from the median to get the upper
                       bound for background extraction

    image_min = use this value as the lower bound for background extraction

    '''

    if not custom_limits:

        backmax = np.median(img_array)-median_diffbelow
        backmin = image_min if image_min is not None else np.nanmin(img_array)

    else:

        backmin, backmax = custom_limits

    masked = npma.masked_outside(img_array, backmin, backmax)
    backmasked = npma.median(masked)

    return backmasked


## IMAGE SECTION FUNCTIONS ##

def img_to_stamps(img,
                  stampsize=256):
    '''
    Generate stamps for an image of size imgsizex x imgsize y. Stamps not in the
    center of the image will be generated for the edges of the image. This
    generates 3 x 3 stamps for each image.

    top_left_corner = img[:xstampsize,:ystampsize]
    bottom_right_corner = img[-xstampsize:,-ystampsize:]

    top_right_corner = img[imgsizex-xstampsize:,:ystampsize]
    bottom_left_corner = img[:xstampsize,imgsizey-ystampsize:]

    top_center = img[imgsizex/2-xstampsize/2:imgsizex/2+xstampsize/2,:ystampsize]
    bottom_center = img[imgsizex/2-xstampsize/2:imgsizex/2+xstampsize/2,
                        imgsizey-ystampsize:]

    center = img[imgsizex/2-xstampsize/2:imgsizex/2+xstampsize/2,
                 imgsizey/2-ystampsize/2:imgsizey/2+ystampsize/2]

    right_center = img[imgsizex-xstampsize:,
                       imgsizey/2-ystampsize/2:imgsizey/2+ystampsize/2]
    left_center = img[:xstampsize,imgsizey/2-ystampsize/2:imgsizey/2+ystampsize/2]


    '''

    imgsizex, imgsizey = img.shape
    xstampsize, ystampsize = stampsize, stampsize

    # get the total number of possible stamps
    n_possible_xstamps = imgsizex/float(xstampsize)
    n_possible_ystamps = imgsizey/float(ystampsize)


    # if we can actually make stamps, then go ahead
    if (n_possible_xstamps >= 3) and (n_possible_ystamps >= 3):

        # FIXME: the coordinate slices should be swapped here, i.e. x,y -> y,x,
        # because the np indexing scheme is y,x instead of x,y
        return {'topleft':img[:xstampsize,:ystampsize],
                'topcenter':img[imgsizex/2-xstampsize/2:imgsizex/2+xstampsize/2,
                                :ystampsize],
                'topright':img[imgsizex-xstampsize:,:ystampsize],
                'midleft':img[:xstampsize,
                               imgsizey/2-ystampsize/2:imgsizey/2+ystampsize/2],
                'midcenter':img[imgsizex/2-xstampsize/2:imgsizex/2+xstampsize/2,
                                imgsizey/2-ystampsize/2:imgsizey/2+ystampsize/2],
                'midright':img[imgsizex-xstampsize:,
                               imgsizey/2-ystampsize/2:imgsizey/2+ystampsize/2],
                'bottomleft':img[:xstampsize,imgsizey-ystampsize:],
                'bottomcenter':img[imgsizex/2-xstampsize/2:imgsizex/2+xstampsize/2,
                                   imgsizey-ystampsize:],
                'bottomright':img[-xstampsize:,-ystampsize:]}
    else:
        if LOGGER:
            LOGGER.error('stampsize is too large for this image')
        else:
            print('error: stampsize is too large for this image')
        return None


def stamps_background(image_stamps,
                      custom_limits=None,
                      median_diffbelow=200.0,
                      image_min=None):
    '''
    This returns background values for each of the stamps in the image_stamps
    object, using the extract_img_background function above.

    '''

    return dict(
        [(
                key,extract_img_background(
                    image_stamps[key],
                    custom_limits=custom_limits,
                    median_diffbelow=median_diffbelow,
                    image_min=image_min
                    )
                )
         for key in image_stamps]
        )


def stamps_to_jpeg(image_stamps,
                   out_fname,
                   sepwidth=1,
                   scale=False,
                   scale_func=clipped_linscale_img,
                   scale_func_params={'cap':255.0,
                                      'lomult':2,
                                      'himult':2.5}):
    '''
    This turns the stamps returned from the function img_to_stamps above into
    a single 3 x 3 postage stamp image. Uses sepwidth pixels as the separator
    between each row/line of stamps.

    '''

    toprow_xsize, toprow_ysize = image_stamps['topright'].shape
    toprow_separr = np.array([[255.0]*sepwidth]*toprow_ysize)

    # note, these should be topleft, topcenter, topright, but since a[x,y] is
    # actually a[y,x] in np array coordinates, it is backwards.
    # FIXME: fix this by fixing img_to_stamps above

    # get the stamps
    if scale:

        topleft = scale_func(image_stamps['topleft'],
                             **scale_func_params)
        midleft = scale_func(image_stamps['midleft'],
                             **scale_func_params)
        bottomleft = scale_func(image_stamps['bottomleft'],
                             **scale_func_params)

        topcenter = scale_func(image_stamps['topcenter'],
                             **scale_func_params)
        midcenter = scale_func(image_stamps['midcenter'],
                             **scale_func_params)
        bottomcenter = scale_func(image_stamps['bottomcenter'],
                             **scale_func_params)

        topright = scale_func(image_stamps['topright'],
                             **scale_func_params)
        midright = scale_func(image_stamps['midright'],
                             **scale_func_params)
        bottomright = scale_func(image_stamps['bottomright'],
                             **scale_func_params)

    else:

        topleft = image_stamps['topleft']
        midleft = image_stamps['midleft']
        bottomleft = image_stamps['bottomleft']

        topcenter = image_stamps['topcenter']
        midcenter = image_stamps['midcenter']
        bottomcenter = image_stamps['bottomcenter']

        topright = image_stamps['topright']
        midright = image_stamps['midright']
        bottomright = image_stamps['bottomright']


    toprow_stamp = np.hstack((topleft,
                              toprow_separr,
                              midleft,
                              toprow_separr,
                              bottomleft))

    midrow_xsize, midrow_ysize = midright.shape
    midrow_separr = np.array([[255.0]*sepwidth]*midrow_ysize)

    # similarly, these should be midleft, midcenter, midright
    midrow_stamp = np.hstack((topcenter,
                              midrow_separr,
                              midcenter,
                              midrow_separr,
                              bottomcenter))

    bottomrow_xsize, bottomrow_ysize = bottomright.shape
    bottomrow_ysize = bottomright.shape[1]
    bottomrow_separr = np.array([[255.0]*sepwidth]*bottomrow_ysize)

    # similarly, these should be bottomleft, bottomcenter, bottomright
    bottomrow_stamp = np.hstack((topright,
                                 bottomrow_separr,
                                 midright,
                                 bottomrow_separr,
                                 bottomright))

    full_stamp = np.vstack((toprow_stamp,
                            np.array([255.0]*(toprow_xsize*3 + sepwidth*2)),
                            midrow_stamp,
                            np.array([255.0]*(midrow_xsize*3 + sepwidth*2)),
                            bottomrow_stamp))

    scipy.misc.imsave(out_fname,full_stamp)
    return full_stamp


def fits_to_stamps_jpeg(fits_image,
                        out_fname=None,
                        ext=None,
                        stampsize=256,
                        sepwidth=1,
                        scale_func=clipped_linscale_img,
                        scale_func_params={'cap':255.0,
                                           'lomult':2,
                                           'himult':2.5}):
    '''
    This turns a FITS image into a 3 x 3 stamps JPEG.

    '''

    compressed_ext = compressed_fits_ext(fits_image)

    if ext is None and compressed_ext:
        img, hdr = read_fits(fits_image,
                             ext=compressed_ext[0])
    elif (ext is not None):
        img, hdr = read_fits(fits_image,ext=ext)
    else:
        img, hdr = read_fits(fits_image)

    trimmed_img = trim_image(img, hdr)
    scaled_img = scale_func(trimmed_img,**scale_func_params)
    stamps = img_to_stamps(scaled_img,stampsize=stampsize)
    if out_fname is None:
        out_fname = fits_image + '.stamp.jpeg'
    stamps_img = stamps_to_jpeg(stamps,out_fname,sepwidth=sepwidth)



def fits_to_full_jpeg(fits_image,
                      out_fname=None,
                      ext=None,
                      resize=False,
                      flip=True,
                      outsizex=800,
                      outsizey=800,
                      annotate=True,
                      scale_func=clipped_linscale_img,
                      scale_func_params={'cap':255.0,
                                         'lomult':2,
                                         'himult':2.5}):
    '''
    This converts a FITS image to a full frame JPEG.

    '''
    compressed_ext = compressed_fits_ext(fits_image)

    if ext is None and compressed_ext:
        img, hdr = read_fits(fits_image,
                             ext=compressed_ext[0])
    elif (ext is not None):
        img, hdr = read_fits(fits_image,ext=ext)
    else:
        img, hdr = read_fits(fits_image)

    #trimmed_img = trim_image(img, hdr)
    trimmed_img = img
    jpegaspect = float(img.shape[1])/float(img.shape[0])
    scaled_img = scale_func(trimmed_img,**scale_func_params)

    if resize:
        resized_img = scipy.misc.imresize(scaled_img,
                                          (int(img.shape[1]/2.2),
                                           int(img.shape[0]/2.2)))
    else:
        resized_img = scaled_img

    if not out_fname:

        out_fname = '%s-%s-%s-%s-proj%s-%s.jpg' % (
            fits_image.rstrip('.fits.fz'),
            hdr['IMAGETYP'].lower() if 'IMAGETYP' in hdr else 'typeunknown',
            hdr['EXPTIME'] if 'EXPTIME' in hdr else 'expunknown',
            (hdr['FILTERS'].replace('+','') if
             'FILTERS' in hdr else 'filtunknown'),
            hdr['PROJID'] if 'PROJID' in hdr else 'unknown',
            hdr['OBJECT'] if 'OBJECT' in hdr else 'objectunknown'
            )

    scipy.misc.imsave(out_fname,resized_img)

    # flip the saved image
    if flip:
        outimg = Image.open(out_fname)
        outimg = outimg.transpose(Image.FLIP_TOP_BOTTOM)
        outimg.save(out_fname)

    # annotate the image if told to do so
    if annotate:
        outimg = Image.open(out_fname)
        draw = ImageDraw.Draw(outimg)
        annotation = "%s: %s - %s - %s - PR%s - %s" % (
            os.path.basename(fits_image).rstrip('.fits.fz'),
            hdr['IMAGETYP'].lower() if 'IMAGETYP' in hdr else 'typeunknown',
            hdr['EXPTIME'] if 'EXPTIME' in hdr else 'expunknown',
            (hdr['FILTERS'].replace('+','') if
             'FILTERS' in hdr else 'filtunknown'),
            hdr['PROJID'] if 'PROJID' in hdr else 'unknown',
            hdr['OBJECT'] if 'OBJECT' in hdr else 'objectunknown'
        )
        draw.text((10,10),annotation,fill=255)
        del draw
        outimg.save(out_fname)

    return out_fname



def fitscoords_to_jpeg(fits_image,
                       out_fname=None,
                       ext=None,
                       resize=False,
                       flip=True,
                       coordbox=None,
                       coordcenter=None,
                       outsizex=770,
                       outsizey=770,
                       annotate=True,
                       scale_func=clipped_linscale_img,
                       scale_func_params={'cap':255.0,
                                          'lomult':2,
                                          'himult':2.5}):
    '''
    This converts a FITS image to a full frame JPEG.

    if coordbox and not coordcenter:
        coordbox = [xmin, xmax, ymin, max] of box to cut out of FITS

    elif coordcenter and not coordbox:
        coordcenter = [xcenter, ycenter, xwidth, ywidth]

    else:
        do nothing, since we can't have both at the same time


    '''
    compressed_ext = compressed_fits_ext(fits_image)

    if ext is None and compressed_ext:
        img, hdr = read_fits(fits_image,
                             ext=compressed_ext[0])
    elif (ext is not None):
        img, hdr = read_fits(fits_image,ext=ext)
    else:
        img, hdr = read_fits(fits_image)

    trimmed_img = img
    jpegaspect = float(img.shape[1])/float(img.shape[0])
    scaled_img = scale_func(trimmed_img,**scale_func_params)

    if coordbox and not coordcenter:
        # numpy is y,x
        scaled_img = scaled_img[coordbox[2]:coordbox[3],
                                coordbox[0]:coordbox[1]]

    elif coordcenter and not coordbox:
        # numpy is y,x
        xmin, xmax = (coordcenter[0] - coordcenter[2]/2.0,
                      coordcenter[0] + coordcenter[2]/2.0)
        ymin, ymax = (coordcenter[1] - coordcenter[3]/2.0,
                      coordcenter[1] + coordcenter[3]/2.0)
        scaled_img = scaled_img[ymin:ymax, xmin:xmax]

    if resize:
        resized_img = scipy.misc.imresize(scaled_img,
                                          (int(img.shape[1]/2.2),
                                           int(img.shape[0]/2.2)))
    else:
        resized_img = scaled_img

    if not out_fname:

        out_fname = '%s-%s-%s-%s-proj%s-%s.jpg' % (
            fits_image.rstrip('.fits.fz'),
            hdr['IMAGETYP'].lower() if 'IMAGETYP' in hdr else 'typeunknown',
            hdr['EXPTIME'] if 'EXPTIME' in hdr else 'expunknown',
            (hdr['FILTERS'].replace('+','') if
             'FILTERS' in hdr else 'filtunknown'),
            hdr['PROJID'] if 'PROJID' in hdr else 'unknown',
            hdr['OBJECT'] if 'OBJECT' in hdr else 'objectunknown'
            )

        if coordbox and not coordcenter:
            out_fname = '%s-X%sX%s-Y%sY%s.jpg' % (
                out_fname.rstrip('.jpg'),
                coordbox[0], coordbox[1],
                coordbox[2], coordbox[3]
            )

        elif coordcenter and not coordbox:
            out_fname = '%s-XC%sYC%s-XW%sYW%s.jpg' % (
                out_fname.rstrip('.jpg'),
                coordcenter[0], coordcenter[1],
                coordcenter[2], coordcenter[3]
            )

    scipy.misc.imsave(out_fname,resized_img)

    # flip the saved image
    if flip:
        outimg = Image.open(out_fname)
        outimg = outimg.transpose(Image.FLIP_TOP_BOTTOM)
        outimg.save(out_fname)

    # annotate the image if told to do so
    if annotate:

        outimg = Image.open(out_fname)
        draw = ImageDraw.Draw(outimg)
        annotation = "%s" % (
            os.path.basename(fits_image).rstrip('.fits.fz'),
        )
        draw.text((4,2),annotation,fill=255)

        # make a circle at the center of the frame
        lx, ly = outimg.size[0], outimg.size[1]
        bx0, bx1 = int(lx/2 - 15), int(lx/2 + 15)
        by0, by1 = int(ly/2 - 15), int(ly/2 + 15)
        draw.ellipse([bx0,by0,bx1,by1], outline=255)

        del draw
        outimg.save(out_fname)

    return out_fname



def nparr_to_full_jpeg(nparr,
                       out_fname,
                       outsizex=770,
                       outsizey=770,
                       scale=True,
                       scale_func=clipped_linscale_img,
                       scale_func_params={'cap':255.0,
                                          'lomult':2,
                                          'himult':2.5}):
    '''
    This just writes a numpy array to a JPEG.

    '''
    if scale:
        scaled_img = scale_func(nparr,**scale_func_params)
    else:
        scaled_img = nparr

    resized_img = scipy.misc.imresize(scaled_img,
                                      (outsizex,outsizey))
    if out_fname is None:
        out_fname = fits_image + '.jpeg'
    scipy.misc.imsave(out_fname,resized_img)



def check_frame_warping(frame,
                        margins=50,
                        threshold=10.0,
                        showplot=False):
    '''This checks if an image is warped (perhaps by a bad shift/convolution).

    Calculates the median of the rows and columns of the image taking into
    account the margin on either side (as specified by the margins kwarg). Then
    fits a straight line to the trend. If the chi-sq of the fit is above the
    specified threshold, returns False as the image is likely to be
    warped. Otherwise, returns True.

    '''

    hdu = pyfits.open(frame)
    image = hdu[0].data
    hdu.close()

    clippedimage = image[margins:-margins, margins:-margins]
    imagecoordnum = np.arange(len(clippedimage))

    # get the medians in the x and y directions
    medx = np.nanmedian(clippedimage,axis=1)
    medy = np.nanmedian(clippedimage,axis=0)

    # fit a 1-degree polynomial
    xfitcoeffs = np.polyfit(imagecoordnum,medx,1)
    yfitcoeffs = np.polyfit(imagecoordnum,medy,1)

    xfitpoly = np.poly1d(xfitcoeffs)
    yfitpoly = np.poly1d(yfitcoeffs)

    xfit = xfitpoly(imagecoordnum)
    yfit = yfitpoly(imagecoordnum)

    xfit_redchisq = np.sum((medx - xfit)*(medx - xfit))/(len(imagecoordnum) - 2)
    yfit_redchisq = np.sum((medy - yfit)*(medy - yfit))/(len(imagecoordnum) - 2)

    warpinfo = {'medx':medx,
                'medy':medy,
                'xfitpoly':xfitpoly,
                'yfitpoly':yfitpoly,
                'xfit':xfit,
                'yfit':yfit,
                'xfit_redchisq':xfit_redchisq,
                'yfit_redchisq':yfit_redchisq}

    if (xfit_redchisq > threshold) or (yfit_redchisq > threshold):
        return False, warpinfo
    else:
        return True, warpinfo
