import os
import salem
import xarray as xr
from affine import Affine
import numpy as np
import rasterio
import pyproj
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import pandas as pd
os.getcwd()
import geopandas as gpd
from collections import OrderedDict

from oggm import workflow, cfg, utils


#Paths to data
MAIN_PATH = os.path.expanduser('~/k_calibration/')

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
                        'input_data/velocity_tiff/vel_total.tif')

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


# Processing vel data
src = rasterio.open(vel_file)

# Retrieve the affine transformation
if isinstance(src.transform, Affine):
     transform = src.transform
else:
     transform = src.affine

N = src.width
M = src.height
dx = transform.a
dy = transform.e
minx = transform.c
maxy = transform.f

dvel = salem.open_xr_dataset(vel_file)

data = dvel.data.values

# Read the image data, flip upside down if necessary
data_in = data
if dy < 0:
  dy = -dy
  data_in = np.flip(data_in, 0)

# Generate X and Y grid locations
xdata = minx + dx/2 + dx*np.arange(N)
ydata = maxy - dy/2 - dy*np.arange(M-1,-1,-1)

# Scale the velocities by the log of the data.
d = np.log(np.clip(data_in, 1, 3000))
data_scale = (255*(d - np.amin(d))/np.ptp(d)).astype(np.uint8)

dvel.data.values = np.flip(data_scale, 0)

#Construct color scale
import matplotlib.colors as colors

# Construct an RGB table using a log scale between 1 and 3000 m/year.
vel = np.exp(np.linspace(np.log(1), np.log(3000), num=256))
hue = np.arange(256)/255.0
sat = np.clip(1./3 + vel/187.5, 0, 1)
value = np.zeros(256) + 0.75
hsv = np.stack((hue, sat, value), axis=1)
rgb = colors.hsv_to_rgb(hsv)
# Be sure the first color (the background) is white
rgb[0,:] = 1
cmap = colors.ListedColormap(rgb, name='velocity')



dve_sel = dvel.salem.subset(grid=gdir.grid, margin=2)

sub_mar = shape_cap.loc[shape_cap['TermType']=='1']


# Plotting
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 10))

sm = ds_geo_sel.salem.get_map(countries=False);
sm.set_shapefile(gdir.read_shapefile('outlines'), color='black')
#sm.set_shapefile(shp, color='r')
sm.set_data(ds_geo_sel.Topography)
sm.set_cmap('topo')
sm.set_scale_bar()
sm.visualize(ax=ax1, cbar_title='m. above s.l.');
at = AnchoredText('a', prop=dict(size=20), frameon=True, loc=2)
ax1.add_artist(at)

sm = ds_geo_sel.salem.get_map(countries=False);
sm.set_shapefile(shape_cap, color='black')
#sm.set_shapefile(shp, color='r')
sm.set_data(ds_geo_sel.Topography)
sm.set_cmap('topo')
sm.set_scale_bar()
sm.visualize(ax=ax2, cbar_title='m. above s.l.')
at = AnchoredText('b', prop=dict(size=20), frameon=True, loc=2)
ax2.add_artist(at)


sm = dve_sel.salem.get_map(countries=False);
sm.set_shapefile(shape_cap, color='black')
#sm.set_shapefile(shp, color='r')
sm.set_data(dve_sel.data)
sm.set_cmap(cmap)
sm.set_scale_bar()
sm.visualize(ax=ax3, cbar_title='m/yr')
at = AnchoredText('c', prop=dict(size=20), frameon=True, loc=2)
ax3.add_artist(at)

sm = dve_sel.salem.get_map(countries=False);
sm.set_shapefile(shape_cap, color='black', alpha=0.6)
sm.set_shapefile(sub_mar, color='r')
#sm.set_shapefile(shp, color='r')
sm.set_data(dve_sel.data)
sm.set_cmap(cmap)
sm.set_scale_bar()
sm.visualize(ax=ax3, cbar_title='m/yr')
at = AnchoredText('c', prop=dict(size=20), frameon=True, loc=2)
ax3.add_artist(at)

# make it nice
plt.tight_layout()
#plt.show()

plt.savefig(os.path.join(plot_path, 'ice_cap.pdf'),
            bbox_inches='tight')