#!/usr/bin/env python

'''epd.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2016

This is a simple External Parameter Decorrelation (EPD) module, intended for use
with HAT lightcurves.

'''

import numpy as np

from scipy.signal import medfilt
from scipy.linalg import lstsq

DEBUG = False

###################
## EPD FUNCTIONS ##
###################

def epd_diffmags(coeff, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag):
    '''
    This calculates the difference in mags after EPD coefficients are
    calculated.

    final EPD mags = median(magseries) + epd_diffmags()

    '''

    return -(coeff[0]*fsv**2. +
             coeff[1]*fsv +
             coeff[2]*fdv**2. +
             coeff[3]*fdv +
             coeff[4]*fkv**2. +
             coeff[5]*fkv +
             coeff[6] +
             coeff[7]*fsv*fdv +
             coeff[8]*fsv*fkv +
             coeff[9]*fdv*fkv +
             coeff[10]*np.sin(2*np.pi*xcc) +
             coeff[11]*np.cos(2*np.pi*xcc) +
             coeff[12]*np.sin(2*np.pi*ycc) +
             coeff[13]*np.cos(2*np.pi*ycc) +
             coeff[14]*np.sin(4*np.pi*xcc) +
             coeff[15]*np.cos(4*np.pi*xcc) +
             coeff[16]*np.sin(4*np.pi*ycc) +
             coeff[17]*np.cos(4*np.pi*ycc) +
             coeff[18]*bgv +
             coeff[19]*bge -
             mag)


def epd_magseries(mag, fsv, fdv, fkv, xcc, ycc, bgv, bge,
                  smooth=21, sigmaclip=3.0):
    '''
    Detrends a magnitude series given in mag using accompanying values of S in
    fsv, D in fdv, K in fkv, x coords in xcc, y coords in ycc, background in
    bgv, and background error in bge. smooth is used to set a smoothing
    parameter for the fit function.

    '''

    # find all the finite values of the magnitude
    finiteind = np.isfinite(mag)

    # calculate median and stdev
    mag_median = np.median(mag[finiteind])
    mag_stdev = np.nanstd(mag)

    # if we're supposed to sigma clip, do so
    if sigmaclip:
        excludeind = abs(mag - mag_median) < sigmaclip*mag_stdev
        finalind = finiteind & excludeind
    else:
        finalind = finiteind

    final_mag = mag[finalind]
    final_len = len(final_mag)

    if DEBUG:
        print('final epd fit mag len = %s' % final_len)

    # smooth the signal
    smoothedmag = medfilt(final_mag, smooth)

    # make the linear equation matrix
    epdmatrix = np.c_[fsv[finalind]**2.0,
                      fsv[finalind],
                      fdv[finalind]**2.0,
                      fdv[finalind],
                      fkv[finalind]**2.0,
                      fkv[finalind],
                      np.ones(final_len),
                      fsv[finalind]*fdv[finalind],
                      fsv[finalind]*fkv[finalind],
                      fdv[finalind]*fkv[finalind],
                      np.sin(2*np.pi*xcc[finalind]),
                      np.cos(2*np.pi*xcc[finalind]),
                      np.sin(2*np.pi*ycc[finalind]),
                      np.cos(2*np.pi*ycc[finalind]),
                      np.sin(4*np.pi*xcc[finalind]),
                      np.cos(4*np.pi*xcc[finalind]),
                      np.sin(4*np.pi*ycc[finalind]),
                      np.cos(4*np.pi*ycc[finalind]),
                      bgv[finalind],
                      bge[finalind]]

    # solve the equation epdmatrix * x = smoothedmag
    # return the EPD differential mags if the solution succeeds
    try:

        coeffs, residuals, rank, singulars = lstsq(epdmatrix, smoothedmag)

        if DEBUG:
            print('coeffs = %s, residuals = %s' % (coeffs, residuals))

        return epd_diffmags(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag)

    # if the solution fails, return nothing
    except Exception as e:

        print('%sZ: EPD solution did not converge! Error was: %s' %
              (datetime.utcnow().isoformat(), e))
        return None


def epd_lightcurve(rlcfile,
                   mags=[19,20,21],
                   sdk=[7,8,9],
                   xy=[3,4],
                   backgnd=[5,6],
                   smooth=21,
                   sigmaclip=3.0,
                   rlcext='rlc',
                   outfile=None,
                   minndet=20):
    '''
    Runs the EPD process on rlcfile, using columns specified to get the required
    parameters. If outfile is None, the .epdlc will be placeed in the same
    directory as rlcfile.

    '''

    # read the lightcurve in
    rlc = np.genfromtxt(rlcfile,
                        usecols=tuple(xy + backgnd + sdk + mags),
                        dtype='f8,f8,f8,f8,f8,f8,f8,f8,f8,f8',
                        names=['xcc','ycc','bgv','bge','fsv','fdv','fkv',
                               'rm1','rm2','rm3'])

    if len(rlc['xcc']) >= minndet:

        # calculate the EPD differential mags
        epddiffmag1 = epd_magseries(rlc['rm1'],rlc['fsv'],rlc['fdv'],rlc['fkv'],
                                    rlc['xcc'],rlc['ycc'],rlc['bgv'],rlc['bge'],
                                    smooth=smooth, sigmaclip=sigmaclip)
        epddiffmag2 = epd_magseries(rlc['rm2'],rlc['fsv'],rlc['fdv'],rlc['fkv'],
                                    rlc['xcc'],rlc['ycc'],rlc['bgv'],rlc['bge'],
                                    smooth=smooth, sigmaclip=sigmaclip)
        epddiffmag3 = epd_magseries(rlc['rm3'],rlc['fsv'],rlc['fdv'],rlc['fkv'],
                                    rlc['xcc'],rlc['ycc'],rlc['bgv'],rlc['bge'],
                                    smooth=smooth, sigmaclip=sigmaclip)

        # add the EPD diff mags back to the median mag to get the EPD mags
        if epddiffmag1 is not None:
            mag_median = np.median(rlc['rm1'][np.isfinite(rlc['rm1'])])
            epdmag1 = epddiffmag1 + mag_median
        else:
            epdmag1 = np.array([np.nan for x in rlc['rm1']])
            print('%sZ: no EP1 mags available for %s!' %
                  (datetime.utcnow().isoformat(), rlcfile))

        if epddiffmag2 is not None:
            mag_median = np.median(rlc['rm2'][np.isfinite(rlc['rm2'])])
            epdmag2 = epddiffmag2 + mag_median
        else:
            epdmag2 = np.array([np.nan for x in rlc['rm2']])
            print('%sZ: no EP2 mags available for %s!' %
                  (datetime.utcnow().isoformat(), rlcfile))

        if epddiffmag3 is not None:
            mag_median = np.median(rlc['rm3'][np.isfinite(rlc['rm3'])])
            epdmag3 = epddiffmag3 + mag_median
        else:
            epdmag3 = np.array([np.nan for x in rlc['rm3']])
            print('%sZ: no EP3 mags available for %s!' %
                  (datetime.utcnow().isoformat(), rlcfile))

        # now write the EPD LCs out to the outfile
        if not outfile:
            outfile = '%s.epdlc' % rlcfile.strip('.%s' % rlcext)

        inf = open(rlcfile,'rb')
        inflines = inf.readlines()
        inf.close()
        outf = open(outfile,'wb')

        for line, epd1, epd2, epd3 in zip(inflines, epdmag1, epdmag2, epdmag3):
            outline = '%s %.6f %.6f %.6f\n' % (line.rstrip('\n'), epd1, epd2, epd3)
            outf.write(outline)

        outf.close()
        return outfile

    else:
        print('not running EPD for %s, ndet = %s < min ndet = %s' %
              (rlcfile, len(rlc['xcc']), minndet))
        return None
