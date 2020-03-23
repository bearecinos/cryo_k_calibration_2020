import os
import salem
import xarray as xr
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import pandas as pd
os.getcwd()
import geopandas as gpd

from oggm import cfg, utils, workflow, graphics

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')
import sys
sys.path.append(MAIN_PATH)
# velocity module
from velocity_tools import utils_velocity as utils_vel

# PARAMS for plots
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
#rcParams['legend.fontsize'] = 12
sns.set_context('poster')


# Data input for backgrounds #################################################
filename_coastline = os.path.join(MAIN_PATH,
                        'input_data/ne_10m_coastline/ne_10m_coastline.shp')
coast_line = salem.read_shapefile(filename_coastline)

# Opening files that we will need
# Projection
mask_file = os.path.join(MAIN_PATH,
            'input_data/Icemask_Topo_Iceclasses_lon_lat_average_1km.nc')
ds_geo = xr.open_dataset(mask_file, decode_times=False)
proj = pyproj.Proj('+init=EPSG:3413')
ds_geo.attrs['pyproj_srs'] = proj.srs

# Paths to vel data  ######################################################
vel_file = os.path.join(MAIN_PATH,
             'input_data/velocity_tiff/greenland_vel_mosaic250_vv_v1.tif')
err_vel_file = os.path.join(MAIN_PATH,
             'input_data/velocity_tiff/greenland_vel_mosaic250_ee_v1.tif')

dvel = utils_vel.open_vel_raster(vel_file)
derr = utils_vel.open_vel_raster(err_vel_file)

# Paths to RACMO data #####################################################
MAIN_PATH_racmo = os.path.expanduser('~/Documents/global_data_base/RACMO/')
smb_path = os.path.join(MAIN_PATH_racmo,
        'smb_rec.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

ds_smb = utils_vel.open_racmo(smb_path, mask_file)

dsc = xr.open_dataset(smb_path, decode_times=False)#.chunk({'time':20})
dsc.attrs['pyproj_srs'] = proj.srs

dsc['time'] = np.append(pd.period_range(start='2018.01.01',
                                    end='2018.12.01', freq='M').to_timestamp(),
                           pd.period_range(start='1958.01.01',
                                    end='2017.12.01', freq='M').to_timestamp())

ds_smb_two = dsc.isel(time=slice(36,12*34))
see = ds_smb_two.chunk({'time':2})
avg = see.SMB_rec.mean(dim='time').compute()

# OGGM run
cfg.initialize()
cfg.initialize(logging_level='WORKFLOW')
cfg.PATHS['working_dir'] = utils.gettempdir(dirname='OGGM-k-plots',
                                            reset=True)
cfg.PARAMS['border'] = 10

# Find a glacier with good coverage of both data
gdirs = workflow.init_glacier_regions(['RGI60-05.00800'],
                                      from_prepro_level=3)

gdir=gdirs[0]
utils_vel.write_flowlines_to_shape(gdir, path=gdir.dir)
shp_path = os.path.join(gdir.dir, 'RGI60-05.shp')
shp = gpd.read_file(shp_path)

# Crop velocity raster
dvel_sel, derr_sel = utils_vel.crop_vel_data_to_glacier_grid(gdir,
                                                             dvel, derr)

# Crop and plot racmo data
ds_sel = utils_vel.crop_racmo_to_glacier_grid(gdir, ds_smb)
# The time info is horrible
ds_sel['time'] = np.append(pd.period_range(start='2018.01.01',
                                    end='2018.12.01', freq='M').to_timestamp(),
                           pd.period_range(start='1958.01.01',
                                    end='2017.12.01', freq='M').to_timestamp())

# We select the time that we need 1960-1990
ds_smb_two_sel = ds_sel.isel(time=slice(36,12*34))
ds_smb_time_sel = ds_smb_two_sel.chunk({'time':2})

smb_avg_sel = ds_smb_time_sel.SMB_rec.mean(dim='time', skipna=True).compute()


#Now plotting
import matplotlib.gridspec as gridspec

# Plot Fig 1
fig1 = plt.figure(figsize=(18, 12), constrained_layout=True)

spec = gridspec.GridSpec(1, 2)

ax0 = plt.subplot(spec[0])
sm = dvel.salem.get_map(countries=False)
sm.set_shapefile(oceans=True)
sm.set_data(dvel.data)
sm.set_cmap('viridis')
x_conect, y_conect = sm.grid.transform(gdir.cenlon, gdir.cenlat)
ax0.scatter(x_conect, y_conect, s=80, marker="o", color='red')
ax0.text(x_conect, y_conect, s = gdir.rgi_id,
         color=sns.xkcd_rgb["white"],
         weight = 'black', fontsize=14)
sm.set_scale_bar()
sm.visualize(ax=ax0, cbar_title='Velocity m/yr')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
sm = ds_smb.salem.get_map(countries=False)
sm.set_shapefile(oceans=True)
sm.set_data(avg)
sm.set_cmap('RdBu')
x_conect, y_conect = sm.grid.transform(gdir.cenlon, gdir.cenlat)
ax1.scatter(x_conect, y_conect, s=80, marker="o", color='red')
ax1.text(x_conect, y_conect, s = gdir.rgi_id,
         color=sns.xkcd_rgb["black"],
         weight = 'black', fontsize=14)
sm.set_scale_bar()
sm.visualize(ax=ax1,  cbar_title='SMB (mm.w.e)')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
ax1.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(cfg.PATHS['working_dir'],
                         'data_input_all_greenland.pdf'),
                            bbox_inches='tight')
#
# Plot fig 2
fig2 = plt.figure(figsize=(14, 14), constrained_layout=False)

spec = gridspec.GridSpec(2, 2, wspace=0.6, hspace=0.05)

llkw = {'interval': 1}

ax0 = plt.subplot(spec[0])
graphics.plot_centerlines(gdirs[0], ax=ax0, title='', add_colorbar=True,
                          lonlat_contours_kwargs=llkw, add_scalebar=True)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
graphics.plot_catchment_width(gdirs[0], ax=ax1, title='', corrected=True,
                              lonlat_contours_kwargs=llkw,
                              add_colorbar=False, add_scalebar=False)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
sm = dvel_sel.salem.get_map(countries=False)
sm.set_shapefile(gdir.read_shapefile('outlines'))
sm.set_shapefile(shp, color='r')
sm.set_data(dvel_sel.data)
sm.set_cmap('viridis')
sm.set_scale_bar()
sm.set_lonlat_contours(interval=1)
sm.visualize(ax=ax2, cbar_title='Velocity m/yr')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
sm = ds_sel.salem.get_map(countries=False)
sm.set_shapefile(gdir.read_shapefile('outlines'))
sm.set_shapefile(shp, color='r')
sm.set_data(smb_avg_sel)
sm.set_cmap('RdBu')
sm.set_scale_bar()
sm.set_lonlat_contours(interval=1)
sm.visualize(ax=ax3,  cbar_title='SMB (mm.w.e)')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc=2)
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(cfg.PATHS['working_dir'],
                         'data_input_plot_glacier.pdf'),
                            bbox_inches='tight')