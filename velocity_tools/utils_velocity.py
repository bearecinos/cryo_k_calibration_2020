import pandas as pd
import numpy as np
import logging
import geopandas as gpd
import math
import os
import pyproj
import pickle
import xarray as xr
import rasterio
import matplotlib.colors as colors
import salem
from affine import Affine
from salem import wgs84
from shapely.ops import transform as shp_trafo
from functools import partial, wraps
from collections import OrderedDict
from oggm import cfg, entity_task, workflow
from oggm.utils._workflow import ncDataset, _get_centerline_lonlat

# Module logger
log = logging.getLogger(__name__)

def normalised(vel):
    vel_min = min(vel)
    vel_max = max(vel)

    n_vel = (vel - vel_min) / (vel_max - vel_min)
    return n_vel

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def calving_flux_km3yr(gdir, smb):
    """Calving mass-loss from specific MB equivalent.
    to Km^3/yr

    This is necessary to use RACMO to find k values.
    """

    if not gdir.is_tidewater:
        return 0.

    # Original units: mm. w.e a-1, to change to km3 a-1 (units of specific MB)
    rho = cfg.PARAMS['ice_density']

    q_calving = (smb * gdir.rgi_area_m2) / (1e9 * rho)

    return q_calving

def write_pickle_file(gdir, var, filename, filesuffix=''):
    """ Writes a variable to a pickle on disk.
    Parameters
    ----------
    var : object
        the variable to write to disk
    filename : str
        file name (must be listed in cfg.BASENAME)
    use_compression : bool
        whether or not the file ws compressed. Default is to use
        cfg.PARAMS['use_compression'] for this (recommended)
    filesuffix : str
        append a suffix to the filename (useful for experiments).
    """
    if filesuffix:
        filename = filename.split('.')
        assert len(filename) == 2
        filename = filename[0] + filesuffix + '.' + filename[1]

    fp = os.path.join(gdir.dir, filename)

    with open(fp, 'wb') as f:
        pickle.dump(var, f, protocol=-1)


def read_pickle_file(gdir, filename, filesuffix=''):
    """ Reads a variable to a pickle on disk.
    Parameters
    ----------
    filename : str
        file name
    filesuffix : str
        append a suffix to the filename (useful for experiments).
    """
    if filesuffix:
        filename = filename.split('.')
        assert len(filename) == 2
        filename = filename[0] + filesuffix + '.' + filename[1]

    fp = os.path.join(gdir.dir, filename)

    with open(fp, 'rb') as f:
        out = pickle.load(f)

    return out

def area_percentage(gdir):
    """ Calculates the lowest 5% of the glacier area from the rgi area
    (this is used in the velocity estimation)
    :param gdir: Glacier directory
    :return: area percentage and the index along the main flowline array
    where that lowest 5% is located.
    """
    rgi_area = gdir.rgi_area_m2
    area_percent = 0.05 * rgi_area

    inv = gdir.read_pickle('inversion_output')[-1]

    # volume in m3 and dx in m
    section = inv['volume'] / inv['dx']

    # Find the index where the lowest 5% percent of the rgi area is located
    index = (np.cumsum(section) <= area_percent).argmin()

    return area_percent, index


def find_flowline_elevation_threshold_index(gdir, ele):
    """ Finds the elevation index along all flowline that are equal to an
    elevation threshold
        :param gdir: Glacier directory
        :param ele: Elevation threshold in m from which we will calculate
        average velocities.
        :return: element index, flowline_number.
        TODO: make this so it can work with differnt flowlines
    """
    fl_no = []
    elem_index = []

    fls = gdir.read_pickle('inversion_flowlines')
    for i, fl in enumerate(fls):
        elevations = fl.surface_h[np.where(fl.surface_h <= ele)]
        if elevations.size != 0:
            inds = np.where(fl.surface_h <= ele)
            elem_index += inds
            fl_no.append(i)

    return np.asarray(fl_no), np.asarray(elem_index)

def read_rgi_ids_from_csv(file_path):
    """
    Function to read a csv file and get the glaciers ID's in that dataframe
    """
    data = pd.read_csv(file_path)
    rgi_ids = data.RGIId.values

    return rgi_ids


def calculate_study_area(ids, geo_df):
    """ Calculates the area for a selection of ids in a shapefile
    """
    keep_ids = [(i in ids) for i in geo_df.RGIId]
    rgi_ids = geo_df.iloc[keep_ids]
    area_sel = rgi_ids.Area.sum()

    return area_sel

def velocity_average_from_elevation(gdir, ele, filesuffix=''):
    """ Calculates the average velocity along the main flowline
    that falls below an elevation threshold
    :param gdir: Glacier directory
    :param ele: Elevation threshold in m (in this case from marcos data)
    :param with_sliding: set to False if you dont want a sliding velocity
    :return: cross-section velocity, surface velocity (m/yr)
    """
    num_fls, elev_indexes = find_flowline_elevation_threshold_index(gdir,
                                                                      ele)
    if filesuffix is None:
        vel = gdir.read_pickle('inversion_output')
    else:
        vel = gdir.read_pickle('inversion_output', filesuffix=filesuffix)

    surf_velocities = []
    cross_velocities = []

    for f, i in zip(num_fls, elev_indexes):
        selection = vel[int(f)]
        top_index = i[0]
        vel_surf_data = selection['u_surface'][top_index:len(selection['u_surface'])]
        vel_cross_data = selection['u_integrated'][top_index:len(selection['u_integrated'])]

        # print('surface', vel_surf_data)
        # print('cross', vel_cross_data)

        surf_velocities = np.append(surf_velocities, vel_surf_data)
        cross_velocities = np.append(cross_velocities, vel_cross_data)

    surf_final = np.nanmean(surf_velocities)
    cross_final = np.nanmean(cross_velocities)

    # print('mean surf', surf_final)
    # print('mean cross', cross_final)

    return surf_final, cross_final


def velocity_average_main_flowline(gdir, filesuffix=''):
    """ Calculates the average velocity along the main flowline
    in different parts (i) along the entire main flowline and at the
    calving front
    :param gdir: Glacier directory
    :param with_sliding: set to False if you dont want a sliding velocity

    :return: cross-section velocity along all the flowline (m/yr)
             surface velocity velocity along all the flowline (m/yr)
             cross-section velocity at the calving front (m/yr)
             surface velocity at the calving front (m/yr)
    """

    if filesuffix is None:
        vel = gdir.read_pickle('inversion_output')[-1]
    else:
        vel = gdir.read_pickle('inversion_output', filesuffix=filesuffix)[-1]

    vel_surf_data = vel['u_surface']
    vel_cross_data = vel['u_integrated']

    length_fls = len(vel_surf_data)/3

    surf_fls_vel = np.nanmean(vel_surf_data)
    cross_fls_vel = np.nanmean(vel_cross_data)

    surf_calving_front = np.nanmean(vel_surf_data[-np.int(length_fls):])
    cross_final = np.nanmean(vel_cross_data[-np.int(length_fls):])

    return surf_fls_vel, cross_fls_vel, surf_calving_front, cross_final


def init_glacier_dirs(workdir, RGI_file_path, error_file_path):

    cfg.initialize()
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['use_tar_shapefiles'] = False
    cfg.PATHS['working_dir'] = workdir
    cfg.PARAMS['border'] = 20

    # Read RGI file
    rgidf = gpd.read_file(RGI_file_path)

    # Run only for Lake Terminating and Marine Terminating
    glac_type = [0]
    keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
    rgidf = rgidf.iloc[keep_glactype]

    # Run only glaciers that have a week connection or are
    # not connected to the ice-sheet
    connection = [2]
    keep_connection = [(i not in connection) for i in rgidf.Connect]
    rgidf = rgidf.iloc[keep_connection]

    # Run glaciers without errors
    de = pd.read_csv(error_file_path)
    keep_indexes = [(i not in de.RGIId.values) for i in rgidf.RGIId]
    rgidf = rgidf.iloc[keep_indexes]

    return workflow.init_glacier_regions(rgidf)


def _get_flowline_lonlat(gdir):
    """Quick n dirty solution to write the flowlines as a shapefile"""

    cls = gdir.read_pickle('inversion_flowlines')
    olist = []
    for j, cl in enumerate(cls[::-1]):
        mm = 1 if j == 0 else 0
        gs = gpd.GeoSeries()
        gs['RGIID'] = gdir.rgi_id
        gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
        gs['MAIN'] = mm
        tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
        gs['geometry'] = shp_trafo(tra_func, cl.line)
        olist.append(gs)

    return olist


def write_flowlines_to_shape(gdir, filesuffix='', path=True):
    """Write the centerlines in a shapefile.

    Parameters
    ----------
    gdirs: the list of GlacierDir to process.
    filesuffix : str
        add suffix to output file
    path:
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """

    if path is True:
        path = os.path.join(cfg.PATHS['working_dir'],
                            'glacier_centerlines' + filesuffix + '.shp')

    olist = []

    olist.extend(_get_flowline_lonlat(gdir))

    odf = gpd.GeoDataFrame(olist)

    shema = dict()
    props = OrderedDict()
    props['RGIID'] = 'str:14'
    props['LE_SEGMENT'] = 'int:9'
    props['MAIN'] = 'int:9'
    shema['geometry'] = 'LineString'
    shema['properties'] = props

    crs = {'init': 'epsg:4326'}

    # some writing function from geopandas rep
    from shapely.geometry import mapping
    import fiona

    def feature(i, row):
        return {
            'id': str(i),
            'type': 'Feature',
            'properties':
                dict((k, v) for k, v in row.items() if k != 'geometry'),
            'geometry': mapping(row['geometry'])}

    with fiona.open(path, 'w', driver='ESRI Shapefile',
                    crs=crs, schema=shema) as c:
        for i, row in odf.iterrows():
            c.write(feature(i, row))

def k_calibration_with_observations(df_oggm, df_obs):
    """Calculates a k parameter per glacier that is
    close to surface velocity observations within a range of
     tolerance given by the observational data.

        Parameters
        ----------
        df_oggm: A data frame with oggm surface velocities per glacier
        calculated with different k values
        df_obs: A data frame with surface velocity observations for the same
        glacier
        rtol = default 0.1. Tolerance for comparing model to observed
        velocities
        :returns k value per glacier
        """
    # if tol is None:
    #     tol = 0.1
    # else:
    #     tol = tol

    var_names_oggm = ['velocity_cross', 'velocity_surf']

    var_names_obs = ['vel_calving_front',
                     'error_calving_front',
                     'rel_tol_calving_front']

    if df_obs.size == 0:
        k_value = None
        mu_star = None
        u_cross = None
        u_surf = None
        rel_tol = None
        length = None
    elif math.isnan(df_obs[var_names_obs[0]].values):
        k_value = None
        mu_star = None
        u_cross = None
        u_surf = None
        rel_tol = None
        length = None
    else:
        if (df_obs[var_names_obs[2]].values >= 0.99):
            tol = 0.98
        else:
            tol = df_obs[var_names_obs[2]].values

        index = df_oggm.index[np.isclose(df_oggm[var_names_oggm[1]],
                                         df_obs[var_names_obs[0]].values,
                                         rtol=tol, atol=0)].tolist()
        #print(index)
        df_oggm = df_oggm.loc[index]

        if df_oggm.empty:
            k_value = None
            mu_star = None
            u_cross = None
            u_surf = None
            rel_tol = tol
            length = None
        elif df_oggm['mu_star'].iloc[0] == 0:
            k_value = df_oggm['k_values'].iloc[0]
            mu_star = df_oggm['mu_star'].iloc[0]
            u_cross = df_oggm['velocity_cross'].iloc[0]
            u_surf = df_oggm['velocity_surf'].iloc[0]
            rel_tol = tol
            length = None
        else:
            k_value = np.mean(df_oggm['k_values'])
            mu_star = np.mean(df_oggm['mu_star'])
            u_cross = np.mean(df_oggm['velocity_cross'])
            u_surf = np.mean(df_oggm['velocity_surf'])
            rel_tol = tol
            length = len(df_oggm['k_values'])

    return k_value, mu_star, u_cross, u_surf, rel_tol, length

def k_calibration_with_mu_star(df_oggm):
    """Calculates a k parameter per glacier that is
       the mimimum k value before mu_star turns equal to zero

           Parameters
           ----------
           df_oggm: A data frame with oggm surface velocities and
            temperature sensitivities per glacier
           calculated with different k values
           :returns k value per glacier
           """
    if math.isnan(df_oggm['mu_star'].iloc[0]):
        k_value = None
        mu_star = None
        u_cross = None
        u_surf = None
    else:
        index = df_oggm.index[df_oggm['mu_star'] > 0.0].tolist()
        #print(df_oggm)
        k_value = df_oggm['k_values'].loc[index[-1]]
        mu_star = df_oggm['mu_star'].loc[index[-1]]
        u_cross = df_oggm['velocity_cross'].loc[index[-1]]
        u_surf = df_oggm['velocity_surf'].loc[index[-1]]

    return k_value, mu_star, u_cross, u_surf

## Tools to procress with RACMO
def open_racmo(netcdf_path, netcdf_mask_path=None):
    """Opens a netcdf from RACMO with a format PROJ (x, y, projection)
    and DATUM (lon, lat, time)

     Parameters:
    ------------
    netcdf_path: path to the data
    netcdf_mask_path: Must be given when openning SMB data else needs to be
    None.
    :returns
        out: xarray object with projection and coordinates in order
     """

    # RACMO open varaible file
    ds = xr.open_dataset(netcdf_path, decode_times=False)

    if netcdf_mask_path is not None:
        # open RACMO mask
        ds_geo = xr.open_dataset(netcdf_mask_path, decode_times=False)

        try:
            ds['x'] = ds_geo['x']
            ds['y'] = ds_geo['y']
            ds_geo.close()
        except KeyError as e:
            pass

    # Add the proj info to all variables
    proj = pyproj.Proj('EPSG:3413')
    ds.attrs['pyproj_srs'] = proj.srs
    for v in ds.variables:
        ds[v].attrs['pyproj_srs'] = proj.srs

    # Fix the time stamp
    ds['time'] = np.append(
        pd.period_range(start='2018.01.01', end='2018.12.01',
                        freq='M').to_timestamp(),
        pd.period_range(start='1958.01.01', end='2017.12.01',
                        freq='M').to_timestamp())

    out = ds
    ds.close()

    return out


def crop_racmo_to_glacier_grid(gdir, ds):
    """ Crops the RACMO data to the glacier grid

    Parameters
    -----------
    gdir: :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ds: xarray object
    :returns
        ds_sel_roi: xarray with the data cropt to the glacier outline """
    try:
        ds_sel = ds.salem.subset(grid=gdir.grid, margin=2)
    except ValueError:
        ds_sel = None

    if ds_sel is None:
        ds_sel_roi = None
    else:
        ds_sel = ds_sel.load().sortby('time')
        ds_sel_roi = ds_sel.salem.roi(shape=gdir.read_shapefile('outlines'))

    return ds_sel_roi


def get_racmo_time_series(ds_sel_roi, var_name,
                          dim_one, dim_two, dim_three,
                          time_start=None, time_end=None):
    """ Generates RACMO time series for a 31 year period centered
    in a t-star year, with the data already cropped to the glacier outline
    Parameters
    ------------
    ds_sel_roi: :xarray:obj already cropped to the glacier outline
    var_name: :ndarray: the variable name to extract the time series
    dim_one : 'x' or 'lon'
    dim_two: 'y' or 'lat'
    dim_three: 'time'
    time_start: a year where the RACMO time series should begin
    time_end: a year where the RACMO time series should end

    :returns
    ts_31yr: xarray object with a time series of the RACMO variable, monthly
    data for a 31 reference period according to X. Fettweis et al. 2017 and
    Eric Rignot and Pannir Kanagaratnam, 2006.
    """
    if ds_sel_roi is None:
        ts_31 = None
    elif ds_sel_roi[var_name].isnull().all():
        #print('the glacier has no RACMO data')
        ts_31 = None
    else:
        ts = ds_sel_roi[var_name].mean(dim=[dim_one, dim_two],
                        skipna=True).resample(time='AS').mean(dim=dim_three,
                                                              skipna=True)

        if time_start is None:
            ts_31 = ts
        else:
            ts_31 = ts.sel(time=slice(str(time_start), str(time_end)))

    return ts_31

def process_racmo_data(gdir, racmo_path,
                       time_start=None, time_end=None):
    """Processes and writes RACMO data in each glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    racmo_path: the main path to the RACMO data

    :returns
    writes an nc file in each glacier directory with the RACMO data
    time series of SMB, precipitation, run off and melt for a 31 reference
    period according to X. Fettweis et al. 2017,
     Eric Rignot and Pannir Kanagaratnam, 2006.
    """
    mask_file = os.path.join(racmo_path,
                             'Icemask_Topo_Iceclasses_lon_lat_average_1km.nc')

    smb_file = os.path.join(racmo_path,
                    'smb_rec.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    prcp_file = os.path.join(racmo_path,
                        'precip.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    run_off_file = os.path.join(racmo_path,
                        'runoff.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    melt_file = os.path.join(racmo_path,
                        'snowmelt.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    fall_file = os.path.join(racmo_path,
                        'snowfall.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    # Get files as xarray all units mm. w.e.
    # Surface Mass Balance
    ds_smb = open_racmo(smb_file, mask_file)
    # Total precipitation: solid + Liquid
    ds_prcp = open_racmo(prcp_file)
    # Run off
    ds_run_off = open_racmo(run_off_file)
    # water that result from snow and ice melting
    ds_melt = open_racmo(melt_file)
    # Solid precipitation
    ds_fall = open_racmo(fall_file)

    # crop the data to glacier outline
    smb_sel = crop_racmo_to_glacier_grid(gdir, ds_smb)
    prcp_sel = crop_racmo_to_glacier_grid(gdir, ds_prcp)
    run_off_sel = crop_racmo_to_glacier_grid(gdir, ds_run_off)
    melt_sel = crop_racmo_to_glacier_grid(gdir, ds_melt)
    fall_sel = crop_racmo_to_glacier_grid(gdir, ds_fall)

    # get RACMO time series in 31 year period centered in t*
    smb_31 = get_racmo_time_series(smb_sel,
                                   var_name='SMB_rec',
                                   dim_one='x',
                                   dim_two='y',
                                   dim_three='time',
                                   time_start=time_start, time_end=time_end)

    prcp_31 = get_racmo_time_series(prcp_sel,
                                    var_name='precipcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end)

    run_off_31 = get_racmo_time_series(run_off_sel,
                                       var_name='runoffcorr',
                                       dim_one='lon',
                                       dim_two='lat',
                                       dim_three='time',
                                       time_start=time_start, time_end=time_end)

    melt_31 = get_racmo_time_series(melt_sel,
                                    var_name='snowmeltcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end)

    fall_31 = get_racmo_time_series(fall_sel,
                                    var_name='snowfallcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end)

    fpath = gdir.dir + '/racmo_data.nc'
    if os.path.exists(fpath):
        os.remove(fpath)

    if smb_31 is None:
        return print('There is no RACMO file for this glacier ' + gdir.rgi_id)
    else:
        with ncDataset(fpath,
                       'w', format='NETCDF4') as nc:

            nc.createDimension('time', None)

            nc.author = 'B.M Recinos'
            nc.author_info = 'Open Global Glacier Model'

            timev = nc.createVariable('time', 'i4', ('time',))

            tatts = {'units': 'year'}

            calendar = 'standard'

            tatts['calendar'] = calendar

            timev.setncatts(tatts)
            timev[:] = smb_31.time

            v = nc.createVariable('smb', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'surface mass balance'
            v[:] = smb_31

            v = nc.createVariable('prcp', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly precipitation amount'
            v[:] = prcp_31

            v = nc.createVariable('run_off', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly run off amount'
            v[:] = run_off_31

            v = nc.createVariable('snow_melt', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly snowmelt amount'
            v[:] = melt_31

            v = nc.createVariable('snow_fall', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly snowfall amount'
            v[:] = fall_31


def get_smb31_from_glacier(gdir):
    """ Reads RACMO data and takes a mean over 31 year period of the
        surface mass balance

    Parameters
    ------------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    :returns
    smb_31 mean value in km^3/yr
    """
    fpath = gdir.dir + '/racmo_data.nc'

    if os.path.exists(fpath):
        with ncDataset(fpath, mode='r') as nc:
            smb = nc.variables['smb'][:]
            smb_mean = np.nanmean(smb)
            smb_cum = np.nansum(smb)
            smb_calving_mean = calving_flux_km3yr(gdir, smb_mean)
            smb_calving_cum = calving_flux_km3yr(gdir, smb_cum)
    else:
        print('This glacier has no racmo data ' + gdir.rgi_id)
        smb_mean = None
        smb_cum = None
        smb_calving_mean = None
        smb_calving_cum = None

    return smb_mean, smb_cum, smb_calving_mean, smb_calving_cum

def get_mu_star_from_glacier(gdir):
    """ Reads RACMO data and calculates the mean temperature sensitivity
    from RACMO SMB data and snow melt estimates. In a glacier wide average and
    a mean value of the entire RACMO time series. Based on the method described
    in Oerlemans, J., and Reichert, B. (2000).

        Parameters
        ------------
        gdir : :py:class:`oggm.GlacierDirectory`
            the glacier directory to process
        :returns
        Mu_star_racmo mean value in mm.w.e /K-1
        """
    fpath = gdir.dir + '/racmo_data.nc'

    if os.path.exists(fpath):
        with ncDataset(fpath, mode='r') as nc:
            smb = nc.variables['smb'][:]
            melt = nc.variables['snow_melt'][:]
            mu = smb / melt
            mean_mu = np.average(mu, weights=np.repeat(gdir.rgi_area_km2,
                                                       len(mu)))
    else:
        print('This glacier has no racmo data ' + gdir.rgi_id)
        mean_mu = None

    return mean_mu

# TODO: this function needs a lot of work still! we need to be able to tell
# the code what to do with different outcomes
def k_calibration_with_racmo(df_oggm,
                             df_racmo,
                             var_name,
                             rtol=None):
    """Calculates a k parameter per glacier that is close to the
    surface mass balance from RACMO within a range of tolerance.

    Parameters
    ----------
        df_oggm: A data frame with oggm calving fluxes per glacier
        calculated with different k values

        df_racmo: path to a data frame with the q_calving calculated with
        RACMO smb

        var_name: variable name from RACMO flux: q_calving_RACMO_cum or
        q_calving_RACMO_mean

        rtol = default 0.1. Tolerance for comparing model to RACMO
        estimates

    :returns k value per glacier and other variables for further
        analysis
        """

    if rtol is None:
        tol = 0.001
    else:
        tol = rtol

    if (df_racmo[var_name].values <= 0):
        k_value = 0
        mu_star = 0
        u_cross = 0
        u_surf = 0
        calving_flux = 0
        racmo_flux = df_racmo[var_name].values
    else:
        #df_oggm = df_oggm[df_oggm.calving_flux.values > 0]

        racmo_value = np.around(df_racmo[var_name].values, decimals=5)

        oggm_values = np.around(df_oggm['calving_flux'].values, decimals=5)

        if (oggm_values[0] > racmo_value):
            index = 0
            df_oggm_new = df_oggm.loc[index]
            k_value = df_oggm_new['k_values']
            mu_star = df_oggm_new['mu_star']
            u_cross = df_oggm_new['velocity_cross']
            u_surf = df_oggm_new['velocity_surf']
            calving_flux = np.around(df_oggm_new['calving_flux'], decimals=5)
            racmo_flux = racmo_value
            print('We pick the smallest k posible', df_racmo['RGI_ID'])
        else:

            # index = df_oggm.index[np.isclose(oggm_values,
            #                              racmo_value,
            #                              rtol=tol)].tolist()

            index = find_nearest(oggm_values, racmo_value)

            df_oggm_new = df_oggm.loc[index]

            k_value = df_oggm_new['k_values']
            mu_star = df_oggm_new['mu_star']
            u_cross = df_oggm_new['velocity_cross']
            u_surf = df_oggm_new['velocity_surf']
            calving_flux = np.around(df_oggm_new['calving_flux'], decimals=5)
            racmo_flux = racmo_value

    return k_value, mu_star, u_cross, u_surf, calving_flux, racmo_flux


def open_vel_raster(tiff_path):
    """Opens a tiff file from Greenland velocity observations
     and calculates a raster of velocities or uncertainties with the
     corresponding color bar

     Parameters:
    ------------
    tiff_path: path to the data
    :returns
        ds: xarray object with data already scaled
     """

    # Processing vel data
    src = rasterio.open(tiff_path)

    # Retrieve the affine transformation
    if isinstance(src.transform, Affine):
        transform = src.transform
    else:
        transform = src.affine

    dy = transform.e

    ds = salem.open_xr_dataset(tiff_path)

    data = ds.data.values

    # Read the image data, flip upside down if necessary
    data_in = data
    if dy < 0:
        data_in = np.flip(data_in, 0)

    # Scale the velocities by the log of the data.
    d = np.log(np.clip(data_in, 1, 3000))
    data_scale = (255 * (d - np.amin(d)) / np.ptp(d)).astype(np.uint8)

    ds.data.values = np.flip(data_scale, 0)

    return ds

def crop_vel_data_to_glacier_grid(gdir, vel, error):
    """
        Crop velocity data and uncertainty to the glacier grid
        for plotting only!

        :param:
            gdir: Glacier Directory
            vel: xarray data containing vel or vel errors from
                the whole Greenland
            error: xarray data containing the errors from
                the whole Greenland
        :return:
            ds_array: an array of velocity croped to the glacier grid
            dr_array: an array of velocity erros croped to the glacier
            grid
        """

    #Crop to glacier grid
    ds_glacier = vel.salem.subset(grid=gdir.grid, margin=2)
    dr_glacier = error.salem.subset(grid=gdir.grid, margin=2)

    return ds_glacier, dr_glacier


def crop_vel_data_to_flowline(vel, error, shp):
    """
    Crop velocity data and uncertainty to the glacier flowlines

    :param:
        vel: xarray data containing vel or vel errors from
            the whole Greenland
        error: xarray data containing the errors from
            the whole Greenland
        shp: Shape file containing the glacier flowlines
    :return:
        ds_array: an array of velocity croped to the glacier main flowline .
        dr_array: an array of velocity erros croped to the glacier
        main flowline.
    """

    #Crop to flowline
    ds_fls = vel.salem.roi(shape=shp.iloc[[0]])
    dr_fls = error.salem.roi(shape=shp.iloc[[0]])

    return ds_fls, dr_fls


def calculate_observation_vel_at_the_main_flowline(gdir,
                                                   ds_fls,
                                                   dr_fls):
    """
    Calculates the mean velocity and error at the end of the flowline
    exactly 5 pixels upstream of the last part of the glacier that contains
    a velocity measurements
    :param:
        gdir: Glacier directory
        ds_flowline: xarray data containing vel observations from the main
        flowline
        dr_flowline: xarray data containing errors in vel observations from
        the main flowline
    :return:
        ds_mean: a mean velocity value over the last parts of the flowline.
        dr_mean: a mean error of the velocity over the last parts of the
        main flowline.
    """

    coords = _get_flowline_lonlat(gdir)

    x, y = coords[0].geometry[3].coords.xy

    # We only want one third of the main centerline! kind of the end of the
    # glacier. For very long glaciers this might not be that ideal

    x_2 = x[-np.int(len(x) / 3):]
    y_2 = y[-np.int(len(x) / 3):]

    raster_proj = pyproj.Proj(ds_fls.attrs['pyproj_srs'])

    # For the entire flowline
    x_all, y_all = salem.gis.transform_proj(wgs84, raster_proj, x, y)
    vel_fls = ds_fls.interp(x=x_all, y=y_all, method='nearest')
    err_fls = dr_fls.interp(x=x_all, y=y_all, method='nearest')
    # Calculating means
    ds_mean = vel_fls.mean(skipna=True).data.values
    dr_mean = err_fls.mean(skipna=True).data.values

    # For the end of the glacier
    x_end, y_end = salem.gis.transform_proj(wgs84, raster_proj, x_2, y_2)
    vel_end = ds_fls.interp(x=x_end, y=y_end, method='nearest')
    err_end = dr_fls.interp(x=x_end, y=y_end, method='nearest')
    # Calculating means
    ds_mean_end = vel_end.mean(skipna=True).data.values
    dr_mean_end = err_end.mean(skipna=True).data.values

    vel_fls_all = np.around(ds_mean, decimals=2)
    err_fls_all = np.around(dr_mean, decimals=2)
    vel_fls_end = np.around(ds_mean_end, decimals=2)
    err_fls_end = np.around(dr_mean_end, decimals=2)

    return vel_fls_all, err_fls_all, vel_fls_end, err_fls_end, len(x)