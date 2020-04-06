import pandas as pd
import os
import geopandas as gpd
import numpy as np
from oggm import cfg, utils
from oggm import workflow
import warnings

cfg.initialize()
# Reading glacier directories per experiment

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

exp_dir_path = os.path.join(MAIN_PATH, 'output_data/12_volume_vsl/config')

full_dir_name_one = os.path.join(exp_dir_path, 'config_01_onlyMT/')
full_dir_name_two = os.path.join(exp_dir_path, 'config_02_onlyMT/')

# Reading RGI
RGI_FILE = os.path.join(MAIN_PATH,
'input_data/05_rgi61_GreenlandPeriphery_bea/05_rgi61_GreenlandPeriphery.shp')

#print(full_dir_name)

cfg.PATHS['working_dir'] = full_dir_name_two
cfg.PARAMS['border'] = 20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False

#Read RGI file
rgidf = gpd.read_file(RGI_FILE)

# Run only for Marine terminating
glac_type = [0]
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

error_file_path = os.path.join(MAIN_PATH,
        'output_data/1_Greenland_prepo/glaciers_with_prepro_errors.csv')
de = pd.read_csv(error_file_path)
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

no_solution = os.path.join(MAIN_PATH,
    'output_data/5_calibration_vel_results/glaciers_with_no_solution.csv')
d_no_sol = pd.read_csv(no_solution)
ids_rgi = d_no_sol.RGIId.values
keep_no_solution = [(i not in ids_rgi) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_solution]

# no_vel_data = os.path.join(MAIN_PATH,
#         'output_data/2_Process_vel_data/glaciers_with_no_velocity_data.csv')
# d_no_data = pd.read_csv(no_vel_data)
# ids_no_data = d_no_data.RGIId.values
# keep_no_data = [(i not in ids_no_data) for i in rgidf.RGIId]
# rgidf = rgidf.iloc[keep_no_data]

no_racmo_data = os.path.join(MAIN_PATH,
    'output_data/3_racmo/1960_1990/glaciers_with_no_racmo_data.csv')
d_no_data = pd.read_csv(no_racmo_data)
ids_no_data = d_no_data.RGIId.values
keep_no_data = [(i not in ids_no_data) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_data]

print(cfg.PATHS['working_dir'])


gdirs = workflow.init_glacier_regions(rgidf, reset=False)

vbsl_no_calving_per_dir = []
vbsl_calving_per_dir = []
ids = []

for gdir in gdirs:

    vbsl_no_calving_per_glacier = []
    vbsl_calving_per_glacier = []

    #Get the data that we need from each glacier
    map_dx = gdir.grid.dx

    #Get flowlines
    fls = gdir.read_pickle('inversion_flowlines')

    #Get inversion output
    inv = gdir.read_pickle('inversion_output', filesuffix='_without_calving_')
    inv_c = gdir.read_pickle('inversion_output')

    import matplotlib.pylab as plt
    for f, cl, cc, in zip(range(len(fls)), inv , inv_c):

        x = np.arange(fls[f].nx) * fls[f].dx * map_dx * 1e-3
        surface = fls[f].surface_h

        # Getting the thickness per branch
        thick = cl['thick']
        vol = cl['volume']

        thick_c = cc['thick']
        vol_c = cc['volume']

        bed = surface - thick
        bed_c = surface - thick_c

        # Find volume below sea level without calving in km³
        index_sl = np.where(bed < 0.0)
        vol_sl = sum(vol[index_sl]) / 1e9
        #print('before calving',vol_sl)

        index_sl_c = np.where(bed_c < 0.0)
        vol_sl_c = sum(vol_c[index_sl_c]) / 1e9
        #print('after calving',vol_sl_c)

        vbsl_no_calving_per_glacier = np.append(
         vbsl_no_calving_per_glacier, vol_sl)

        vbsl_calving_per_glacier = np.append(
         vbsl_calving_per_glacier, vol_sl_c)

        ids = np.append(ids, gdir.rgi_id)

    # We sum up all the volume below sea level in all branches
    vbsl_no_calving_per_glacier = sum(vbsl_no_calving_per_glacier)
    vbsl_calving_per_glacier = sum(vbsl_calving_per_glacier)

    vbsl_no_calving_per_dir = np.append(vbsl_no_calving_per_dir,
                                    vbsl_no_calving_per_glacier)

    vbsl_calving_per_dir = np.append(vbsl_calving_per_dir,
                                 vbsl_calving_per_glacier)

    np.set_printoptions(suppress=True)


d = {'RGIId': pd.unique(ids),
 'volume bsl': vbsl_no_calving_per_dir,
 'volume bsl with calving': vbsl_calving_per_dir}
data_frame = pd.DataFrame(data=d)
data_frame.to_csv(os.path.join(full_dir_name_two,'volume_below_sea_level.csv'))