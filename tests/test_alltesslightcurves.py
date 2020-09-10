'''
A test for the alltesslightcurves module.
'''

#############
## IMPORTS ##
#############

from astrobase.services import alltesslightcurves

try:
    from astrobase.services import tesslightcurves

    test_ok = (
        tesslightcurves.lightkurve_dependency and
        tesslightcurves.eleanor_dependency and
        tesslightcurves.astroquery_dependency
    )

except Exception:
    test_ok = False

######################
## TESTS TO EXECUTE ##
######################

if test_ok:

    import os

    #############
    ## HELPERS ##
    #############

    def init_tempdir():

        cwd = os.getcwd()
        tempdir = os.path.join(cwd, 'temp_downloaddir')
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)

        return tempdir

    def test_get_all_tess_lightcurves():

        tic_id = '220314428'  # "V684 Mon"

        tempdir = init_tempdir()

        lcfile = alltesslightcurves.get_all_tess_lightcurves(
            tic_id=tic_id,
            pipelines=['CDIPS', 'PATHOS', 'TASOC', '2minSPOC', 'eleanor'],
            download_dir=tempdir,
        )

        assert len(lcfile) == 5

    if __name__ == "__main__":
        test_get_all_tess_lightcurves()
