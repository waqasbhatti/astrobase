"""
test_convert_identifiers.py - Luke Bouma (luke@astro.princeton.edu) - Oct 2019
License: MIT - see the LICENSE file for details.

Test easy conversion between survey identifiers.
"""

###########
# imports #
###########
from astrobase.services.identifiers import (
    simbad_to_gaiadr2,
    gaiadr2_to_tic,
    simbad_to_tic,
    tic_to_gaiadr2
)


####################
# tests to execute #
####################

def test_simbad2gaiadrtwo():

    assert simbad_to_gaiadr2('WASP-4') == '6535499658122055552'

    #
    # Orion Nebula is an open cluster; doesn't have Gaia DR2 identifier.
    #
    assert simbad_to_gaiadr2('M 42') is None


def test_gaiadrtwo2tic():

    assert gaiadr2_to_tic('6535499658122055552') == '402026209'


def test_simbad2tic():

    assert simbad_to_tic('WASP-4') == '402026209'
    #
    # Orion Nebula is an open cluster; doesn't have Gaia DR2 identifier.
    #
    assert simbad_to_tic('M 42') is None


def test_tic2gaiadrtwo():

    assert tic_to_gaiadr2('402026209') == '6535499658122055552'
