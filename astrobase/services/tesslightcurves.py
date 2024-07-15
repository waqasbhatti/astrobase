#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tesslightcurves.py - Luke Bouma (bouma.luke@gmail.com) - Nov 2019
# License: MIT - see the LICENSE file for the full text.

"""
Useful tools for acquiring TESS light-curves.  This module contains a number of
non-standard dependencies, including lightkurve, eleanor, and astroquery.

Light-curve retrieval: get light-curves from all sectors for a tic_id::

    get_two_minute_spoc_lightcurves
    get_hlsp_lightcurves
    get_eleanor_lightcurves
    get_unpopular_lightcurve

Visibility queries: check if an ra/dec was observed::

    is_two_minute_spoc_lightcurve_available
    get_tess_visibility_given_ticid
    get_tess_visibility_given_ticids
"""

#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############

from glob import glob
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import json
from os.path import join

from astropy.coordinates import SkyCoord

# This module contains a number of non-standard dependencies, including
# lightkurve, astroquery, eleanor, and tess_cpm/unpopular.
#
# $ conda install -c conda-forge lightkurve
# $ conda install -c astropy astroquery
# $ pip install eleanor
#

try:
    from lightkurve.search import search_lightcurvefile
    lightkurve_dependency = True
except ImportError:
    lightkurve_dependency = False

try:
    from astroquery.mast import Tesscut
    from astroquery.mast import Observations
    astroquery_dependency = True
except ImportError:
    astroquery_dependency = False

try:
    import eleanor
    eleanor_dependency = True
except ImportError:
    eleanor_dependency = False

try:
    # Install using the setup.py from the fork at
    # git clone https://github.com/lgbouma/unpopular
    import tess_cpm
    tess_cpm_dependency = True
except ImportError:
    tess_cpm_dependency = False

deps = {
    'lightkurve': lightkurve_dependency,
    'astroquery': astroquery_dependency,
    'eleanor': eleanor_dependency,
    'tess_cpm': tess_cpm_dependency
}

for k,v in deps.items():
    if not v:
        wrn = (
            'Failed to import {:s} dependency. Trying anyway.'.
            format(k)
        )
        LOGWARNING(wrn)

from astrobase.services.mast import tic_objectsearch

#############
## HELPERS ##
#############

def _get_tesscutout(cache_dir=".", objectname=None, coordinates=None, size=5,
                    sector=None, inflate=True, force_download=False,
                    verbose=True):
    """
    Helper function that wraps Tesscut.download_cutouts with a local cache.

    Parameters
    ----------
    cache_dir : str
        The path to which the TESScut FFI will be written.
    objectname : str, optional
        The target around which to search, by name (objectname="M104")
        or TIC ID (objectname="TIC 141914082").
        One and only one of coordinates and objectname must be supplied.
    coordinates : str or `astropy.coordinates` object, optional
        The target around which to search. It may be specified as a
        string or as the appropriate `astropy.coordinates` object.
    size : int, array-like, `~astropy.units.Quantity`
        Optional, default 5 pixels.
        The size of the cutout array. If ``size`` is a scalar number or
        a scalar `~astropy.units.Quantity`, then a square cutout of ``size``
        will be created.  If ``size`` has two elements, they should be in
        ``(ny, nx)`` order.  Scalar numbers in ``size`` are assumed to be in
        units of pixels. `~astropy.units.Quantity` objects must be in pixel or
        angular units.
    sector : int
        Optional.
        The TESS sector to return the cutout from.  If not supplied, cutouts
        from all available sectors on which the coordinate appears will be returned.
    path : str
        Optional.
        The directory in which the cutouts will be saved.
        Defaults to current directory.
    inflate : bool
        Optional, default True.
        Cutout target pixel files are returned from the server in a zip file,
        by default they will be inflated and the zip will be removed.
        Set inflate to false to stop before the inflate step.

    Returns
    -------
    cutout_paths : list
        List of paths to tesscut FITS files.
    """

    from astroquery.mast.utils import parse_input_location

    coords = parse_input_location(coordinates, objectname)

    ra = f"{coords.ra.value:.6f}"

    matched = [m for m in glob(join(cache_dir, '*.fits')) if ra in m]

    if (len(matched) != 0) and (force_download == False):

        LOGINFO(f"Found the following FITS files in the {cache_dir} "
                f"directory with matching RA values.")
        LOGINFO(matched)
        LOGINFO("If you still want to download the file, set the "
                "force_download keyword to True.")
        return matched

    else:
        t_paths = Tesscut.download_cutouts(
            coordinates=coordinates, size=size, sector=sector, path=cache_dir,
            inflate=inflate, objectname=objectname
        )
        cutout_paths = list(t_paths['Local Path'])
        return cutout_paths


def _plot_cpm_lightcurve(df, figpath, min_cpm_reg=None):

    plt.close("all")
    fig, axs = plt.subplots(nrows=2, figsize=(12,8), sharex=True)

    axs[0].scatter(df.time, df.norm_flux, c="k", s=3,
                   label="Normalized Flux", zorder=2, rasterized=True,
                   linewidths=0)
    axs[0].plot(df.time, df.cpm_pred, "-", lw=2, c="C3",
                alpha=0.8, label="CPM Prediction", zorder=1)
    axs[1].scatter(df.time, df.dtr_flux, c='k', s=3, zorder=2,
                   rasterized=True, linewidths=0)
    if min_cpm_reg is not None:
        txt = f'Reg = {min_cpm_reg:.2e}'
        axs[1].text(0.03, 0.97, txt, transform=axs[1].transAxes,
                    ha='left', va='top')
    axs[0].legend()

    fig.text(-0.01, 0.5, 'Relative flux', va='center', rotation=90)
    fig.text(0.5, -0.01, "Time - 2457000 [Days]", ha='center', va='center')
    fig.tight_layout()

    fig.savefig(figpath, bbox_inches="tight", dpi=300)


##########
## WORK ##
##########

def get_unpopular_lightcurve(tic_id, download_dir=None, verbose=True,
                             overwrite=False):
    """This downloads and creates the default light curves using `unpopular`
    (Hattori et al., 2021,
    https://ui.adsabs.harvard.edu/abs/2021arXiv210615063H/abstract).

    NOTE: The default per-sectors FFI cutouts from this approach are ~60 Mb for
    TESS Years 1 & 2, and ~190 Mb for TESS Years 3 and 4.  For an object in the
    CVZ, this function will therefore download and cache ~3 Gb of TESS FFI
    cutout data from MAST.  Be careful when trying to scale up!

    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string, e.g., "123456789".

    download_dir : str
        The directory to which TESS FFI cutout and the light curves will be
        written.  If None, defaults to working directory.

    overwrite : bool
        If true, re-creates the CPM light curves.  Otherwise, pulls from cached
        CSV files.

    Returns
    -------
    csvpaths: list or None
        List of light-curve file paths.  None if none are found and downloaded.
    """

    if not astroquery_dependency:
        LOGERROR(
            "The astroquery package is required for this function to work."
        )
        return None

    if not tess_cpm_dependency:
        LOGERROR(
            "The tess_cpm package is required for this function to work."
        )
        return None

    assert isinstance(tic_id, str)

    if download_dir is None:
        download_dir = './'

    objectname = f"TIC{tic_id}"
    cutout_paths = _get_tesscutout(
        size=50, objectname=objectname, cache_dir=download_dir
    )

    #
    # Create a default light curve from each sector of data available.
    #
    csvpaths = glob(join(download_dir, f"TIC{tic_id}_*llc.csv"))

    if len(csvpaths)>0 and not overwrite:
        return csvpaths

    for cutout_path in cutout_paths:

        sector = os.path.basename(cutout_path).split("_")[0].split("-")[1]
        camera = os.path.basename(cutout_path).split("_")[0].split("-")[2]
        ccd = os.path.basename(cutout_path).split("_")[0].split("-")[3]
        starid = f"TIC{tic_id}_{sector}_{camera}_{ccd}"

        # Instantiate tess_cpm Source object, and remove values with non-zero
        # quality flags.
        s = tess_cpm.Source(cutout_path, remove_bad=True)

        # Plot the median image, trimmed at 10-90th percentile.
        figpath = join(download_dir, f"{starid}_10_90.png")
        s.plot_cutout(figpath=figpath)

        # Select the aperture: whatever pixel the target star landed on.
        s.set_aperture(rowlims=[25, 25], collims=[25, 25])
        # s.set_aperture(rowlims=[23, 26], collims=[23, 26])

        figpath = join(download_dir, f"{starid}_10_90_aperture.png")
        s.plot_cutout(rowlims=[20, 30], collims=[20, 30], show_aperture=True,
                      figpath=figpath)

        # Plot the zero-centered & median-divided flux.
        figpath = join(download_dir, f"{starid}_pixbypix_norm.png")
        s.plot_pix_by_pix(data_type="normalized_flux", figpath=figpath)

        s.add_cpm_model(
            exclusion_size=5, n=64, predictor_method="similar_brightness"
        )

        # This method allows us to see our above choices
        _ = s.models[0][0].plot_model(size_predictors=10, figpath=figpath)

        # Use cross-validation to set the regularization value.  K-fold to
        # split the light curve into `k` contiguous sections, and to predict
        # the `i^th` section using all other sections.  Smaller values ->
        # weaker regularization.
        figpath = join(download_dir, f"{starid}_regs.png")
        cpm_regs = 10.0 ** np.arange(-10, 10)
        k = 10
        min_cpm_reg, cdpps = s.calc_min_cpm_reg(cpm_regs, k, figpath=figpath)

        s.set_regs([min_cpm_reg])
        s.holdout_fit_predict(k=k)

        figpath = join(download_dir, f"{starid}_pixbypix_cpm_subtracted.png")
        s.plot_pix_by_pix(
            data_type="cpm_subtracted_flux", split=True, figpath=figpath
        )

        aperture_normalized_flux = s.get_aperture_lc(
            data_type="normalized_flux"
        )
        aperture_cpm_prediction = s.get_aperture_lc(
            data_type="cpm_prediction", weighting="median"
        )
        detrended_flux = s.get_aperture_lc(data_type="cpm_subtracted_flux")

        #
        # Save the light curve as a CSV file
        #
        out_df = pd.DataFrame({
            "time": s.time,
            "norm_flux": aperture_normalized_flux,
            "cpm_pred": aperture_cpm_prediction,
            "dtr_flux": detrended_flux
        })
        csvpath = join(download_dir, f"{starid}_cpm_llc.csv")
        out_df.to_csv(csvpath, index=False)
        LOGINFO(f"Wrote {csvpath}")

        figpath = join(download_dir, f"{starid}_cpm_llc.png")
        _plot_cpm_lightcurve(out_df, figpath, min_cpm_reg=min_cpm_reg)

    csvpaths = glob(join(download_dir, f"TIC{tic_id}_*llc.csv"))
    return csvpaths


def get_two_minute_spoc_lightcurves(tic_id, download_dir=None, verbose=True):
    """This downloads 2-minute TESS SPOC light curves.

    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    Returns
    -------
    lcfiles : list or None
        List of light-curve file paths. None if none are found and downloaded.

    """

    if not lightkurve_dependency:
        LOGERROR(
            "The lightkurve package is required for this function to work."
        )
        return None

    if not isinstance(download_dir, str):
        errmsg = (
            'get_two_minute_spoc_lightcurves: failed to get valid download_dir'
        )
        LOGERROR(errmsg)
        return None

    search_str = 'TIC ' + tic_id
    if verbose:
        msg = f'Searching via lightkurve.search_lightcurvefile for {search_str}'
        LOGINFO(msg)

    res = search_lightcurvefile(search_str, cadence='short', mission='TESS')

    if len(res.table) == 0:
        errmsg = (
            f'failed to get any SC data for TIC{tic_id}. need other LC source.'
        )
        LOGERROR(errmsg)
        return None
    else:
        msg = (
            f'Got {len(res.table)} sectors of SC data for TIC{tic_id}.'
        )
        if verbose:
            LOGINFO(msg)

    res.download_all(download_dir=download_dir)
    lcfiles = glob(join(download_dir, 'mastDownload', 'TESS',
                        '*{}*'.format(tic_id), '*{}*.fits'.format(tic_id) ))

    return lcfiles


def get_hlsp_lightcurves(tic_id,
                         hlsp_products=('CDIPS', 'TASOC', 'PATHOS'),
                         download_dir=None,
                         verbose=True):
    """This downloads TESS HLSP light curves for a given TIC ID.

    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    hlsp_products : sequence of str
        List of desired HLSP products to search. For instance, ["CDIPS"].

    download_dir : str
        Path of directory to which light-curve will be downloaded.

    Returns
    -------
    lcfiles : list or None
        List of light-curve file paths. None if none are found and downloaded.

    """

    assert isinstance(hlsp_products, (tuple, list))

    if not astroquery_dependency:
        LOGERROR(
            "The astroquery package is required for this function to work."
        )
        return None

    lcfiles = []

    for hlsp in hlsp_products:

        obs_table = Observations.query_criteria(
            target_name=tic_id, provenance_name=hlsp
        )

        if verbose:
            LOGINFO(f'Found {len(obs_table)} {hlsp} light-curves.')

        if len(obs_table) == 0:
            if verbose:
                LOGINFO("Did not find light-curves. Escaping.")
            return None

        # Get list of available products for this Observation.
        cdips_products = Observations.get_product_list(obs_table)

        # Download the products for this Observation.
        manifest = Observations.download_products(cdips_products,
                                                  download_dir=download_dir)
        if verbose:
            LOGINFO("Done")

        if len(manifest) >= 1:
            lcfiles.append(list(manifest['Local Path']))

    #
    # flatten lcfiles list
    #
    if len(lcfiles) >= 1:
        return_lcfiles = [item for sublist in lcfiles for item in sublist]
    else:
        return_lcfiles = None

    return return_lcfiles


def get_eleanor_lightcurves(tic_id, download_dir=None, targetdata_kwargs=None):
    """This downloads light curves from the Eleanor project for a given TIC ID.

    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    download_dir : str
        The light curve FITS files will be downloaded here.

    targetdata_kwargs : dict
        Optional dictionary of keys and values to be passed
        ``eleanor.TargetData`` (see
        https://adina.feinste.in/eleanor/api.html). For instance, you might pass
        ``{'height':8, 'width':8, 'do_pca':True, 'do_psf':True,
        'crowded_field':False}`` to run these settings through to eleanor. The
        default options used if targetdata_kwargs is None are as follows::

            {
                height=15,
                width=15,
                save_postcard=True,
                do_pca=False,
                do_psf=False,
                bkg_size=31,
                crowded_field=True,
                cal_cadences=None,
                try_load=True,
                regressors=None
            }

    Returns
    -------
    lcfiles : list or None
        List of light-curve file paths. These are saved as CSV, rather than
        FITS, by this function.

    """

    if not eleanor_dependency:
        LOGERROR(
            "The eleanor package is required for this function to work."
        )
        return None

    stars = eleanor.multi_sectors(tic=np.int64(tic_id), sectors='all', tc=False)

    for star in stars:

        if targetdata_kwargs is None:
            d = eleanor.TargetData(star, height=15, width=15,
                                   save_postcard=True, do_pca=False,
                                   do_psf=False, bkg_size=31,
                                   aperture_mode='normal', cal_cadences=None,
                                   try_load=False)
        else:
            d = eleanor.TargetData(star, **targetdata_kwargs)

        d.save(directory=download_dir)

    lcfiles = glob(join(
        download_dir, 'hlsp_eleanor_tess_ffi_tic{}*.fits'.format(tic_id)
    ))

    return lcfiles


def is_two_minute_spoc_lightcurve_available(tic_id):
    """
    This checks if a 2-minute TESS SPOC light curve is available for the TIC ID.

    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    Returns
    -------
    result : bool
        True if a 2 minute SPOC light-curve is available, else False.

    """

    if not lightkurve_dependency:
        LOGERROR(
            "The lightkurve package is required for this function to work."
        )
        return False

    search_str = 'TIC ' + tic_id
    res = search_lightcurvefile(search_str, cadence='short', mission='TESS')

    if len(res.table) == 0:
        return False
    else:
        return True


def get_tess_visibility_given_ticid(tic_id):
    """
    This checks if a given TIC ID is visible in a TESS sector.

    Parameters
    ----------
    tic_id : str
        The TIC ID of the object as a string.

    Returns
    -------
    sector_str,full_sector_str : tuple of strings
        The first element of the tuple contains a string list of the sector
        numbers where the object is visible. The second element of the tuple
        contains a string list of the full sector names where the object is
        visible.

        For example, "[16, 17]" and "[tess-s0016-1-4, tess-s0017-2-3]". If
        empty, will return "[]" and "[]".
    """

    if not astroquery_dependency:
        LOGERROR(
            "The astroquery package is required for this function to work."
        )
        return None, None

    ticres = tic_objectsearch(tic_id)

    with open(ticres['cachefname'], 'r') as json_file:
        data = json.load(json_file)

    ra = data['data'][0]['ra']
    dec = data['data'][0]['dec']

    coord = SkyCoord(ra, dec, unit="deg")
    sector_table = Tesscut.get_sectors(coord)

    sector_str = list(sector_table['sector'])
    full_sector_str = list(sector_table['sectorName'])

    return sector_str, full_sector_str


def get_tess_visibility_given_ticids(ticids):
    """This gets TESS visibility info for an iterable container of TIC IDs.

    Parameters
    ----------

    ticids : iterable of str
        The TIC IDs to look up.

    Returns
    -------

    tuple
        Returns a two-element tuple containing lists of the sector numbers and
        the full names of the sectors containing the requested TIC IDs.

    """

    if not astroquery_dependency:
        LOGERROR(
            "The astroquery package is required for this function to work."
        )
        return None, None

    sector_strs, full_sector_strs = [], []

    for ticid in ticids:
        sector_str, full_sector_str = (
            get_tess_visibility_given_ticid(ticid)
        )
        sector_strs.append(sector_str)
        full_sector_strs.append(full_sector_str)

    return sector_strs, full_sector_strs
