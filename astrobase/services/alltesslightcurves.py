'''
A tool for aquiring TESS light curves from a variety of pipelines.
Wraps functions found in tesslightcurves.py

'''

#############
## IMPORTS ##
#############

import astrobase.services.tesslightcurves as tlc

methodDict = {
    '2minSPOC': tlc.get_two_minute_spoc_lightcurves,
    'eleanor': tlc.get_eleanor_lightcurves,
    'CDIPS': tlc.get_hlsp_lightcurves,
    'PATHOS': tlc.get_hlsp_lightcurves,
    'TASOC': tlc.get_hlsp_lightcurves
}

##########
## WORK ##
##########


def get_all_tess_lightcurves(
        tic_id,
        pipelines=('CDIPS', 'PATHOS', 'TASOC', '2minSPOC', 'eleanor'),
        download_dir=None
):
    """
    Gets all possible TESS light curves for a TIC ID.

    Parameters
    ----------

    tic_id : str
        The TIC ID of the object to get all the light curves for.

    pipelines : list or tuple
        The pipeline products to search for light curves of the given TIC
        ID. Must be one or more of the following::

            ['CDIPS', 'PATHOS', 'TASOC', '2minSPOC', 'eleanor']

    download_dir : str or None
        The directory to download the light curves to. If None, will download to
        the current directory.

    Returns
    -------

    lcfile_list : list of str
        Returns a list of light curve files that were downloaded for the object.

    """
    for pipeline in pipelines:

        get_method = methodDict[pipeline]

        if pipeline in ['2minSPOC', 'eleanor']:
            return get_method(tic_id=tic_id, download_dir=download_dir)

        elif pipeline in ['CDIPS', 'PATHOS', 'TASOC']:
            return get_method(tic_id=tic_id, hlsp_products=[pipeline],
                              download_dir=download_dir)

        else:
            raise NotImplementedError
