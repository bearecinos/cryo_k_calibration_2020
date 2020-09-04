import os
import salem
import xarray as xr
import pyproj
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
os.getcwd()
import geopandas as gpd
from collections import OrderedDict
from matplotlib.offsetbox import AnchoredText
from matplotlib import rcParams

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

import sys
sys.path.append(MAIN_PATH)
# velocity module
from velocity_tools import utils_velocity as utils_vel

# PARAMS for plots
rcParams['axes.labelsize'] = 25
rcParams['xtick.labelsize'] = 25
rcParams['ytick.labelsize'] = 25
#rcParams['legend.fontsize'] = 8

sns.set_context('poster')

#Paths to data
plot_path = os.path.join(MAIN_PATH, 'plots/')

# Data input
RGI_FILE = os.path.join(MAIN_PATH,
'input_data/05_rgi61_GreenlandPeriphery_bea/05_rgi61_GreenlandPeriphery.shp')

mask_file = os.path.join(MAIN_PATH,
                'input_data/Icemask_Topo_Iceclasses_lon_lat_average_1km.nc')

filename_coastline = os.path.join(MAIN_PATH,
                        'input_data/ne_10m_coastline/ne_10m_coastline.shp')

#Reading RACMO mask
# The mask and geo reference data
ds_geo = xr.open_dataset(mask_file, decode_times=False)
proj = pyproj.Proj('+init=EPSG:3413')
ds_geo.attrs['pyproj_srs'] = proj.srs

coast_line = salem.read_shapefile(filename_coastline)

#RGI v6
df = gpd.read_file(RGI_FILE)

rgi_area_total = df['Area'].sum()

df.set_index('RGIId')
index = df.index.values

# Get the glaciers classified by Terminus type
sub_mar = df[df['TermType'].isin([1])]
sub_lan = df[df['TermType'].isin([0])]

# Classify Marine-terminating by connectivity
sub_no_conect = sub_mar[sub_mar['Connect'].isin([0, 1])]
sub_conect = sub_mar[sub_mar['Connect'].isin([2])]

## Make a table for the area distribution
area_per_reg = df[['Area', 'TermType']].groupby('TermType').sum()
area_per_reg['Area (% of all Alaska)'] = area_per_reg['Area'] / area_per_reg.Area.sum() * 100
area_per_reg['N Glaciers'] = df.groupby('TermType').count().RGIId


area_mar = sub_mar[['Area', 'Connect']].groupby('Connect').sum()
area_mar['Area (% of all Alaska)'] = area_mar['Area'] / area_per_reg.Area.sum() * 100

category = ['Land-terminating',
            'Tidewater strongly connected',
            'Tidewater weakly connected']

area = [area_per_reg.Area[0],
        area_mar.Area[2],
        area_mar.Area[0] + area_mar.Area[1]]

area_percent = area / df.Area.sum() * 100

d = {'Category': category,
     'Area (km²)': area,
     'Area (% of Greenland)': area_percent}
ds = pd.DataFrame(data=d)

print(ds)

############### analyse errors and data gaps #################################
output_dir_path = os.path.join(MAIN_PATH, 'output_data/')

full_exp_dir = []

exclude = {'10_plot', '4_k_exp_for_calibration', '7_calving_vel_calibrated',
           '8_calving_racmo_calibrated', '11_climate_stats', '12_volume_vsl',
           '13_non_calving', '14_ice_cap_prepo'}

for path, subdirs, files in os.walk(output_dir_path, topdown=True):
    subdirs[:] = [d for d in subdirs if d not in exclude]
    subdirs[:] = [d for d in subdirs if "rest" not in d]
    subdirs[:] = sorted(subdirs)

    for name in subdirs:
        full_exp_dir.append(os.path.join(path, name))

print(full_exp_dir)


prepro_erros = os.path.join(full_exp_dir[0],
                            'glaciers_with_prepro_errors.csv')
no_vel_data = os.path.join(full_exp_dir[1],
                           'glaciers_with_no_velocity_data.csv')
no_racmo_data = os.path.join(full_exp_dir[2],
                             '1960_1990/glaciers_with_no_racmo_data.csv')
no_solution =  os.path.join(full_exp_dir[3],
                            'glaciers_with_no_solution.csv')

prepro_ids = utils_vel.read_rgi_ids_from_csv(prepro_erros)
no_vel_ids = utils_vel.read_rgi_ids_from_csv(no_vel_data)
no_racmo_ids = utils_vel.read_rgi_ids_from_csv(no_racmo_data)
no_sol_ids = utils_vel.read_rgi_ids_from_csv(no_solution)


# Calculate study area precentage per error category
area_prepro = utils_vel.calculate_study_area(prepro_ids, sub_no_conect)
area_no_vel = utils_vel.calculate_study_area(no_vel_ids, sub_no_conect)
area_no_racmo = utils_vel.calculate_study_area(no_racmo_ids, sub_no_conect)
area_no_solution = utils_vel.calculate_study_area(no_sol_ids, sub_no_conect)

study_area = sub_no_conect.Area.sum()

category_two = ['OGGM pre-processing errors',
            'Glaciers with no velocity data',
            'Glaciers with no RACMO DATA',
            'Glaciers with no calving solution']

areas_two = [area_prepro, area_no_vel, area_no_racmo,
             area_no_solution]
area_percent_two = areas_two / study_area * 100

k = {'Category': category_two,
     'Area (km²)': areas_two,
     'Area (% of study area)': area_percent_two}
dk = pd.DataFrame(data=k)

print(dk)

print(study_area)
print(rgi_area_total)

world_area = 705738.793
study_area_percentage = study_area / world_area * 100
print(study_area_percentage)
print(study_area)

rgi_area_from_glims = 89717.066
print(rgi_area_from_glims-rgi_area_total)

drgi = gpd.read_file(rgi_original_path)
drgi_original_area = drgi['Area'].sum()
print(drgi_original_area/world_area*100)
#exit()
##############################################################################

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 8), constrained_layout=False)
spec = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1.5, 1.5])

ax0 = plt.subplot(spec[0])
# Define map projections and ext.
smap = ds_geo.salem.get_map(countries=False)

# Add coastline and Ice cap outline
smap.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.7)
#
# Land-terminating
smap.set_shapefile(sub_lan, facecolor=sns.xkcd_rgb["grey"],
                   label='Land-terminating',
                   edgecolor=None)

# Marine-terminating
# #   i: Not Connected
smap.set_shapefile(sub_no_conect, facecolor=sns.xkcd_rgb["medium blue"],
                   label='Tidewater weakly connected', linewidth=3.0,
                   edgecolor=sns.xkcd_rgb["medium blue"])
#
# #   ii: Connected
smap.set_shapefile(sub_conect, facecolor=sns.xkcd_rgb["navy blue"],
                   label='Tidewater strongly connected', linewidth=3.0,
                   edgecolor=sns.xkcd_rgb["navy blue"])

smap.set_scale_bar(location=(0.78, 0.04))
smap.visualize(ax=ax0)
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
ax0.add_artist(at)


ax1 = plt.subplot(spec[1])

# Plotting bar plot
N = 1
ind = np.arange(N)    # the x locations for the groups
width = 0.15       # the width of the bars: can also be len(x) sequence

Land_area = ds['Area (% of Greenland)'][0]
Marine_area = ds['Area (% of Greenland)'][1] + ds['Area (% of Greenland)'][2]
Tidewater_connected =  ds['Area (% of Greenland)'][1]
Tidewater_no_connected = ds['Area (% of Greenland)'][2]

# Heights of bars1 + bars2
bars = np.add(Land_area, Tidewater_connected).tolist()


p1 = ax1.bar(ind, Land_area, width,
             color=sns.xkcd_rgb["grey"], label='Land-terminating')
p2 = ax1.bar(ind, Tidewater_connected, width, bottom=Land_area,
             color=sns.xkcd_rgb["navy blue"],
             label='Tidewater connectivity level 2')
p3 = ax1.bar(ind, Tidewater_no_connected, width, bottom=bars,
             color=sns.xkcd_rgb["medium blue"],
             label='Tidewater connectivity level 0,1 (study area)')

ax1.set_ylabel('Area (% of Greenland PGs)')
#ax1.set_ylim(0, 100)
ax1.set_xticks(ind, ('1'))
ax1.set_xticks(ind)
ax1.set_xticklabels(['1'])

ax1.legend((p1[0], p2[0], p3[0]),
           ('Land-terminating',
            'Tidewater connectivity level 2',
            'Tidewater connectivity level 0,1 (study area)'))

ax1.get_legend().remove()

handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor= (0.55, 1.05), loc='upper center',
             ncol=3, fontsize=18)

at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
ax1.add_artist(at)

ax2 = plt.subplot(spec[2])
palette = sns.color_palette("colorblind")
# Plotting bar plot
N = 1
ind = np.arange(N)    # the x locations for the groups
width = 0.15       # the width of the bars: can also be len(x) sequence

prepro_area = dk['Area (% of study area)'][0]
no_vel_area = dk['Area (% of study area)'][1]
no_racmo_area =  dk['Area (% of study area)'][2]
no_sol_area = dk['Area (% of study area)'][3]

run_area = 100 - no_sol_area - prepro_area - no_vel_area - no_racmo_area

# Heights of bars1 + bars2
bars1 = np.add(run_area, no_sol_area).tolist()
bars2 = np.add(bars1, prepro_area).tolist()
bars3 = np.add(bars2, no_vel_area).tolist()

p5 = ax2.bar(ind, run_area, width, color=palette[0])

p1 = ax2.bar(ind, no_sol_area, width, bottom=run_area, color=palette[1])

p2 = ax2.bar(ind, prepro_area, width, bottom=bars1, color=palette[2])

p3 = ax2.bar(ind, no_vel_area, width, bottom=bars2, color=palette[3])

p4 = ax2.bar(ind, no_racmo_area, width, bottom=bars3, color=palette[4])

ax2.set_ylabel('Area (% of study area)')
ax2.set_xticks(ind)
ax2.set_xticklabels(['2'])

lgd = ax2.legend((p5[0], p1[0], p2[0], p3[0], p4[0]),
           ('Processed',
            'Without $q_{calving}$ solution',
            'OGGM errors',
            'Without velocity data',
            'Without RACMO data'),#, frameon=True,
            loc='lower right', bbox_to_anchor=(2.6, 0.0), fontsize=18)

at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)
ax2.add_artist(lgd)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'rgi_overview_test.png'),
            bbox_inches='tight', pad_inches=0.25)