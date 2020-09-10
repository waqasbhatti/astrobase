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

def get_all_tess_lightcurves(tic_id, 
        pipelines=('CDIPS', 'PATHOS', 'TASOC', '2minSPOC', 'eleanor'), 
        download_dir=None):
    """
    pipeline (list): list including any of
        ['CDIPS', 'PATHOS', 'TASOC', '2minSPOC', 'eleanor']
    """
    for pipeline in pipelines:

        get_method = methodDict[pipeline]

        if pipeline in ['2minSPOC', 'eleanor']:
            get_method(tic_id=tic_id, download_dir=download_dir)

        elif pipeline in ['CDIPS', 'PATHOS', 'TASOC']:
            get_method(tic_id=tic_id, hlsp_products=[pipeline],
                       download_dir=download_dir)

        else:
            raise NotImplementedError
