"""
test_convert_identifiers.py - Luke Bouma (luke@astro.princeton.edu) - Oct 2019
License: MIT - see the LICENSE file for details.

Test easy conversion between survey identifiers.
"""

###########
# imports #
###########
from astrobase.services.convert_identifiers import (
    simbad2gaiadrtwo,
    gaiadrtwo2tic,
    simbad2tic
)

import unittest

##########
# config #
##########

class BadQueryTestCases(unittest.TestCase):
    """
    Class that enables requiring particular exceptions to be raised.
    """

    def bad_simbad2targetcatalog_test(self, bad_identifier, target_function):
        """
        bad_identifier (str): a valid SIMBAD identifier that doesn't have a
        Gaia DR2 identifier

        target_function (function): either of
        `astrobase.service.convert_identifiers.simbad2gaiadrtwo` or
        `astrobase.service.convert_identifiers.simbad2tic`
        """

        with self.assertRaises(NameError) as context:
            target_function(bad_identifier)

        self.assertTrue(
            'Failed to retrieve Gaia DR2 identifier' in str(context.exception)
        )


####################
# tests to execute #
####################

def test_simbad2gaiadrtwo():

    assert simbad2gaiadrtwo('WASP-4') == '6535499658122055552'

    #
    # Orion Nebula is an open cluster; doesn't have Gaia DR2 identifier.
    #
    b = BadQueryTestCases()
    b.bad_simbad2targetcatalog_test('M 42', simbad2gaiadrtwo)


def test_gaiadrtwo2tic():

    assert gaiadrtwo2tic('6535499658122055552') == '402026209'


def test_simbad2tic():

    assert simbad2tic('WASP-4') == '402026209'

    b = BadQueryTestCases()
    b.bad_simbad2targetcatalog_test('M 42', simbad2tic)
