#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''starfeatures - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2017
License: MIT. See the LICENSE file for more details.

This calculates various features related to the color/proper-motion of stars.

All of the functions in this module require as input a dict of the following
form (an 'objectinfo' dict):

{

}



'''

def star_color_features(objectinfo):
    '''
    Stellar colors and dereddened stellar colors using 2MASS DUST API:

    http://irsa.ipac.caltech.edu/applications/DUST/docs/dustProgramInterface.html

    '''


def propermotion_features(objectinfo):
    '''
    Calculates proper motion features.

    - total proper motion from pmra, pmdecl
    - reduced J proper motion from propermotion and Jmag

    '''



def star_neighbor_features(objectinfo,
                           lclistpkl,
                           fwhm_arcsec):
    '''Gets several neighbor features:

    - distance to closest neighbor in arcsec
    - mag diff of closest neighbor
    - total number of all neighbors within 2 x fwhm_arcsec

    objectinfo is the objectinfo dict from an object light curve

    lclistpkl is a pickle produced by lcproc.makelclist that has a kdtree
    available for nearest neighbor searches.

    '''
