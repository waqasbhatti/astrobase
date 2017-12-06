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
    Stellar colors and dereddened stellar colors using NED extinction API.

    '''
