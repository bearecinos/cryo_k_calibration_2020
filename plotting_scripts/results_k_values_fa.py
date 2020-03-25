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
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
#rcParams['legend.fontsize'] = 12
sns.set_context('poster')

plot_path = os.path.join(MAIN_PATH, 'plots/')

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

## Paths to output data ####################################################
output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output')

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(output_dir_path):
    for file in f:
        files.append(os.path.join(r, file))

df_vel = pd.read_csv(files[1], index_col='Unnamed: 0')
df_racmo = pd.read_csv(files[0], index_col='Unnamed: 0')

## Only plot glaciers that calve
df_vel_to_plot = df_vel[df_vel['calving_flux'] > 0]
df_racmo_to_plot = df_racmo[df_racmo['calving_flux_x'] > 0]

## Get coordinates and data
lat_v = df_vel_to_plot.cenlat.values
lon_v = df_vel_to_plot.cenlon.values
rgi_index_v = df_vel_to_plot.index
k_v = df_vel_to_plot.k_value.values
fa_v = df_vel_to_plot.calving_flux.values

lat_r = df_racmo_to_plot.cenlat.values
lon_r = df_racmo_to_plot.cenlon.values
rgi_index_r = df_racmo_to_plot.index
k_r = df_racmo_to_plot.k_value.values
fa_r = df_racmo_to_plot.calving_flux_x.values

#Now plotting
import matplotlib.gridspec as gridspec

# Plot Fig 1
fig1 = plt.figure(figsize=(12, 19))

spec = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.1)

ax0 = plt.subplot(spec[0])
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon_v, lat_v)

ax0.scatter(xx, yy, 12**k_v, alpha=0.3, color=sns.xkcd_rgb["dark green"],
                                        edgecolor=sns.xkcd_rgb["green"])

# make legend with dummy points
for a in [1.5, 2.0, 2.5]:
    ax0.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=12**a,
                label=str(a) + 'yr$^{-1}$')
ax0.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower right', fontsize=10.5);
#sm.set_scale_bar(location=(0.87, 0.95))
sm.visualize(ax=ax0)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon_r, lat_r)

ax1.scatter(xx, yy, 12**k_r, alpha=0.3, color=sns.xkcd_rgb["dark red"],
                                        edgecolor=sns.xkcd_rgb["red"])

# make legend with dummy points
for a in [1.5, 2.0, 2.5]:
    ax1.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=12**a,
                label=str(a) + 'yr$^{-1}$')
ax1.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower right', fontsize=10.5);
#sm.set_scale_bar(location=(0.87, 0.95))
sm.visualize(ax=ax1)
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
ax1.add_artist(at)


ax2 = plt.subplot(spec[2])
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon_v, lat_v)

ax2.scatter(xx, yy, 5000*fa_v, alpha=0.3, color=sns.xkcd_rgb["light green"],
                                        edgecolor=sns.xkcd_rgb["green"])

# make legend with dummy points
for a in [0.001, 0.01, 0.1]:
    ax2.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=5000*a,
                label=str(a) + 'km$^{3}$/yr')
ax2.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower right', fontsize=11);
sm.visualize(ax=ax2)
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)


ax3 = plt.subplot(spec[3])
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon_r, lat_r)

ax3.scatter(xx, yy, 5000*fa_r, alpha=0.3, color=sns.xkcd_rgb["light red"],
                                        edgecolor=sns.xkcd_rgb["red"])

# make legend with dummy points
for a in [0.001, 0.01, 0.1]:
    ax3.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=5000*a,
                label=str(a) + 'km$^{3}$/yr')
ax3.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower right', fontsize=11);
sm.set_scale_bar(location=(0.85, 0.94))
sm.visualize(ax=ax3)
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc=2)
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'k_values_fa_result.pdf'),
            bbox_inches='tight')