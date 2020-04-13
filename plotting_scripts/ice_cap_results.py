import os
import salem
import xarray as xr
from affine import Affine
import numpy as np
import rasterio
import pyproj
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
os.getcwd()
import geopandas as gpd
from collections import OrderedDict

from oggm import workflow, cfg, utils

# PARAMS for plots
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
#rcParams['legend.fontsize'] = 12
sns.set_context('poster')

#Paths to data
MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

import sys
sys.path.append(MAIN_PATH)
# velocity module
from velocity_tools import utils_velocity as utils_vel
from oggm.graphics import _plot_map
import shapely.geometry as shpg
from matplotlib import cm as colormap

@_plot_map
def plot_inversion(gdirs, ax=None, smap=None, linewidth=3, vmax=None):
    """Plots the result of the inversion out of a glacier directory."""

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    toplot_th = np.array([])
    toplot_lines = []
    toplot_crs = []
    vol = []
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')
        inv = gdir.read_pickle('inversion_output')
        inv_no_calving = gdir.read_pickle('inversion_output',
                                          filesuffix='_without_calving_')
        # Plot boundaries
        poly_pix = geom['polygon_pix']
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                          linewidth=.2)
        for l in poly_pix.interiors:
            smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines')
        for l, c, d in zip(cls, inv, inv_no_calving):

            smap.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)
            toplot_th = np.append(toplot_th, c['thick'] - d['thick'])
            for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
                line = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                        shpg.Point(cur + wi / 2. * n2)])
                toplot_lines.append(line)
                toplot_crs.append(crs)
            vol.extend(c['volume'])

    dl = salem.DataLevels(cmap=colormap.get_cmap('YlOrRd'), nlevels=256,
                          data=toplot_th, vmin=0, vmax=400, extend='max')
    colors = dl.to_rgb()
    for l, c, crs in zip(toplot_lines, colors, toplot_crs):
        smap.set_geometry(l, crs=crs, color=c,
                          linewidth=linewidth, zorder=50)

    smap.plot(ax)
    return dict(cbar_label='thickness difference \n [m]',
                cbar_primitive=dl)



plot_path = os.path.join(MAIN_PATH, 'plots/')

RGI_FILE = os.path.join(MAIN_PATH,
'input_data/05_rgi61_GreenlandPeriphery_bea/05_rgi61_GreenlandPeriphery.shp')

mask_file = os.path.join(MAIN_PATH,
                'input_data/Icemask_Topo_Iceclasses_lon_lat_average_1km.nc')


ice_cap = os.path.join(MAIN_PATH,
                    'input_data/ice_cap/flade_isblink_mod_bea_no_errors.shp')

filename_coastline = os.path.join(MAIN_PATH,
                        'input_data/ne_10m_coastline/ne_10m_coastline.shp')

vel_file = os.path.join(MAIN_PATH,
                'input_data/velocity_tiff/greenland_vel_mosaic250_vv_v1.tif')

# OGGM Run
cfg.initialize()
cfg.PATHS['working_dir'] = utils.get_temp_dir('racmo')
gdirs = workflow.init_glacier_regions(['RGI60-05.10315'],
                                      from_prepro_level=3, prepro_border=10)

gdir = gdirs[0]

#Reading RACMO mask
# The mask and geo reference data
ds_geo = xr.open_dataset(mask_file, decode_times=False)
proj = pyproj.Proj('+init=EPSG:3413')
ds_geo.attrs['pyproj_srs'] = proj.srs

# Reading coastline
coast_line = salem.read_shapefile_to_grid(filename_coastline, gdir.grid)

# Selecting a zoom portion of the topo data fitting the ice cap
ds_geo_sel = ds_geo.salem.subset(grid=gdir.grid, margin=2)

# Reading ice cap outline and assinging the rgi grid
shape_cap = salem.read_shapefile_to_grid(ice_cap, gdir.grid)

sub_mar = shape_cap.loc[shape_cap['TermType']=='1']

print(sub_mar.RGIId)

# #################### Reading volume data #####################################
# Reading results with calving main volumes
output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')

df_vel = pd.read_csv(os.path.join(output_dir_path,
                                    'glacier_stats_vel_method.csv'))
df_vel = df_vel.loc[df_vel['rgi_id'].isin(sub_mar.RGIId)].copy()

df_racmo = pd.read_csv(os.path.join(output_dir_path,
                                      'glacier_stats_racmo_method.csv'))
df_racmo = df_racmo.loc[df_racmo['rgi_id'].isin(sub_mar.RGIId)].copy()

# Reading results without calving
prepo_dir_path = os.path.join(MAIN_PATH, 'output_data/1_Greenland_prepo/')
df_prepro = pd.read_csv(os.path.join(prepo_dir_path,
                'glacier_statistics_greenland_no_calving_with_sliding_.csv'))

df_prepro = df_prepro.loc[df_prepro['rgi_id'].isin(sub_mar.RGIId)].copy()

df_prepro = df_prepro[['rgi_id', 'inv_volume_km3']]
df_prepro.rename(columns={'inv_volume_km3': 'inv_volume_km3_no_calving'},
                 inplace=True)

# Reading configurations for volume below sea level
# Reading volume below sea level
out_vbsl_path = os.path.join(MAIN_PATH, 'output_data/12_volume_vsl/config/')

config_one_path = os.path.join(out_vbsl_path,
                               'config_01_onlyMT/volume_below_sea_level.csv')
config_two_path = os.path.join(out_vbsl_path,
                               'config_02_onlyMT/volume_below_sea_level.csv')
config_one = pd.read_csv(config_one_path)
config_two = pd.read_csv(config_two_path)

config_one.rename(columns={'RGIId': 'rgi_id'}, inplace=True)
config_one.rename(columns={'volume bsl': 'vol_bsl_MV'}, inplace=True)
config_one.rename(columns={'volume bsl with calving': 'vol_bsl_wc_MV'},
                  inplace=True)

config_two.rename(columns={'RGIId': 'rgi_id'}, inplace=True)
config_two.rename(columns={'volume bsl': 'vol_bsl_MR'}, inplace=True)
config_two.rename(columns={'volume bsl with calving': 'vol_bsl_wc_MR'},
                  inplace=True)

config_one = config_one.loc[config_one['rgi_id'].isin(sub_mar.RGIId)].copy()
config_two = config_two.loc[config_two['rgi_id'].isin(sub_mar.RGIId)].copy()


df_racmo_with_no_fa = pd.merge(left=df_racmo,
                    right=df_prepro,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

df_vel_with_no_fa = pd.merge(left=df_vel,
                    right=df_prepro,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

# Merge to main results
df_racmo_vbsl = pd.merge(left=df_racmo_with_no_fa,
                    right=config_two,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

df_vel_vbsl = pd.merge(left=df_vel_with_no_fa,
                    right=config_one,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

#############################################################################

#print(df_racmo_with_fari.columns)
Num_glacier = np.array([len(df_vel_vbsl.rgi_id),
                      len(df_vel_vbsl.rgi_id),
                      len(df_racmo_vbsl.rgi_id),
                      len(df_racmo_vbsl.rgi_id)])
print(Num_glacier)

vol_exp = np.array([df_vel_vbsl['inv_volume_km3_no_calving'].sum(),
                    df_vel_vbsl['inv_volume_km3'].sum(),
                    df_racmo_vbsl['inv_volume_km3'].sum()])


vol_bsl_exp = np.array([df_vel_vbsl['vol_bsl_MV'].sum(),
                    df_vel_vbsl['vol_bsl_wc_MV'].sum(),
                    df_racmo_vbsl['vol_bsl_wc_MR'].sum()])

vol_exp_sle = []
for vol in vol_exp:
    sle = utils_vel.calculate_sea_level_equivalent(vol)
    vol_exp_sle = np.append(vol_exp_sle, sle)

vol_bsl_exp_sle = []
for vol_bsl in vol_bsl_exp:
    sle = utils_vel.calculate_sea_level_equivalent(vol_bsl)
    vol_bsl_exp_sle = np.append(vol_bsl_exp_sle, sle)

print(vol_exp)
## TODO: CALCULATE ALL DIFF BETWEEN VOLUMES!!!
percentage_of_diff = [utils_vel.calculate_volume_percentage(vol_exp[0], vol_exp[1]),
                     utils_vel.calculate_volume_percentage(vol_exp[0], vol_exp[2]),
                     utils_vel.calculate_volume_percentage(vol_exp[2], vol_exp[1])]
                     # utils_vel.calculate_volume_percentage(vol_exp[4],  vol_exp[5]),
                     # utils_vel.calculate_volume_percentage(vol_exp[5], vol_exp[2])]
print(percentage_of_diff)

percentage_of_diff_vbsl = [utils_vel.calculate_volume_percentage(vol_bsl_exp[0], vol_bsl_exp[1]),
                     utils_vel.calculate_volume_percentage(vol_bsl_exp[0], vol_bsl_exp[2]),
                     utils_vel.calculate_volume_percentage(vol_bsl_exp[2], vol_bsl_exp[1])]
                     # utils_vel.calculate_volume_percentage(vol_bsl_exp[4],  vol_bsl_exp[5]),
                     # utils_vel.calculate_volume_percentage(vol_bsl_exp[5], vol_bsl_exp[2])]
print(percentage_of_diff_vbsl)


print(str(vol_exp_sle[0]) + 'increase to ' + str(vol_exp_sle[1])+ ' when using vel method')
print(str(vol_exp_sle[0]) + 'increase to ' + str(vol_exp_sle[2])+ ' when using RACMO method')

print(str(vol_bsl_exp_sle[0]) + 'increase to ' + str(vol_bsl_exp_sle[1])+ ' when using vel method')
print(str(vol_bsl_exp_sle[0]) + 'increase to ' + str(vol_bsl_exp_sle[2])+ ' when using RACMO method')
exit()
################# Initialise new dir ########################################
ids = df_racmo.rgi_id.values

full_dir_name_one = os.path.join(out_vbsl_path, 'config_01_onlyMT/')
full_dir_name_two = os.path.join(out_vbsl_path, 'config_02_onlyMT/')

cfg.initialize()
cfg.PATHS['working_dir'] = full_dir_name_one
cfg.PARAMS['border'] = 20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False

gdirs_one = workflow.init_glacier_regions(ids, reset=False)

cfg.initialize()
cfg.PATHS['working_dir'] = full_dir_name_two
cfg.PARAMS['border'] = 20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False

gdirs_two = workflow.init_glacier_regions(ids, reset=False)

##############################################################################

import matplotlib.gridspec as gridspec

# Plotting
fig2 = plt.figure(figsize=(16, 12), constrained_layout=True)

#wspace=0.2, hspace=0.2,
spec = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.4, width_ratios=[1, 1],
                         height_ratios=[1, 3])

ax1 = plt.subplot(spec[0, :])
color_palette = sns.color_palette("deep")
color_array = [color_palette[2], color_palette[0], color_palette[1]]

ax2= ax1.twiny()
# Example data
y_pos = np.arange(len(vol_exp))
y_pos = [0,0.1,0.2]

p1 = ax1.barh(y_pos, vol_bsl_exp*-1, align='center', color=sns.xkcd_rgb["grey"],
            height=0.1, edgecolor="white")

p2 = ax1.barh(y_pos, vol_exp, align='center', color=color_array, height=0.1)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([])
# labels read top-to-bottom
ax1.invert_yaxis()
ax1.set_xlabel('Volume [km³]')

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
array = ax1.get_xticks()

# Get the other axis on sea level equivalent
sle = []
for value in array:
    sle.append(np.round(abs(utils_vel.calculate_sea_level_equivalent(value)),2))

ax2.set_xticklabels(sle)
ax2.set_xlabel('Volume [mm SLE]')

# Shrink current axis's height by 10% on the bottom
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                  box.width * 0.8, box.height*0.8])

ax1.legend((p2[0], p2[1], p2[2]),
           ('Without $q_{calving}$',
            'With $q_{calving}$ \n - velocity',
            'With $q_{calving}$ \n - RACMO'),
            loc='center left', bbox_to_anchor=(1, 0.5),
            fancybox=True, shadow=True, fontsize=15)
at = AnchoredText('a', prop=dict(size=16), frameon=True, loc=2)
ax1.add_artist(at)

ax3 = plt.subplot(spec[-1, 0])
sm = ds_geo_sel.salem.get_map(countries=False);
sm.set_shapefile(shape_cap, color='black')
plot_inversion(gdirs_one, ax=ax3, smap=sm,
                        linewidth=1, add_scalebar=False,
                        title='')
sm.set_lonlat_contours(interval=3)
sm.visualize(ax=ax3, addcbar=False)
at = AnchoredText('b', prop=dict(size=16), frameon=True, loc=2)
ax3.add_artist(at)

ax4 = plt.subplot(spec[-1, 1])
sm = ds_geo_sel.salem.get_map(countries=False);
sm.set_shapefile(shape_cap, color='black')
plot_inversion(gdirs_two, ax=ax4, smap=sm,
                        linewidth=1, add_scalebar=False,
                        title='')
sm.set_lonlat_contours(interval=3)
sm.visualize(ax=ax4, addcbar=False)
at = AnchoredText('c', prop=dict(size=16), frameon=True, loc=2)
ax4.add_artist(at)

# make it nice
plt.tight_layout()
#plt.show()
#
plt.savefig(os.path.join(plot_path, 'ice_cap_results.pdf'),
             bbox_inches='tight')