"""
test_tesslightcurves.py - Luke Bouma (luke@astro.princeton.edu) - Nov 2019
License: MIT - see the LICENSE file for details.

Test TESS light-curve acquisition functions.
"""

###########
# imports #
###########

try:
    from astrobase.services import tesslightcurves

    test_ok = (
        tesslightcurves.lightkurve_dependency and
        tesslightcurves.eleanor_dependency and
        tesslightcurves.astroquery_dependency
    )

except Exception:
    test_ok = False

####################
# tests to execute #
####################

if test_ok:

    import os

    ###########
    # helpers #
    ###########

    def init_tempdir():

        cwd = os.getcwd()
        tempdir = os.path.join(cwd, 'temp_downloaddir')
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)

        return tempdir

    def test_get_two_minute_spoc_lightcurves():

        tempdir = init_tempdir()

        tic_id = '402026209'  # WASP-4
        lcfile = tesslightcurves.get_two_minute_spoc_lightcurves(
            tic_id,
            download_dir=tempdir
        )

        assert len(lcfile) == 1

        tic_id = '100100827'  # WASP-18
        lcfile = tesslightcurves.get_two_minute_spoc_lightcurves(
            tic_id,
            download_dir=tempdir
        )

        assert len(lcfile) == 2

    def test_get_hlsp_lightcurves():

        tic_id = '220314428'  # "V684 Mon"

        tempdir = init_tempdir()

        lcfile = tesslightcurves.get_hlsp_lightcurves(
            tic_id,
            hlsp_products=['CDIPS'],
            download_dir=tempdir,
            verbose=True
        )

        assert len(lcfile) == 1

    def test_get_eleanor_lightcurves():
        """NOTE: This test takes a while, because eleanor downloads lots of
        metadata onto your system for any new sector.

        """

        tic_id = '308538095'  # CDIPS PC

        tempdir = init_tempdir()

        lcfile = tesslightcurves.get_eleanor_lightcurves(
            tic_id,
            download_dir=tempdir
        )

        assert len(lcfile) == 5

    if __name__ == "__main__":
        test_get_eleanor_lightcurves()
        test_get_hlsp_lightcurves()
        test_get_two_minute_spoc_lightcurves()
