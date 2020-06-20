import os
import salem
from salem import DataLevels
import xarray as xr
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import pandas as pd
os.getcwd()

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

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
output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')

df_both = pd.read_csv(os.path.join(output_dir_path,
                                'glacier_stats_both_methods.csv'),
                      index_col='Unnamed: 0')

print(df_both.columns)
#df_both  = df_both.loc[df_both.k_value_MR != 0]

df_both['diff_q'] = (df_both['calving_flux_MV'] - df_both['calving_flux_MR']).abs()

df_both['diff_k'] = (df_both['k_value_MV'] - df_both['k_value_MR']).abs()

df_both  = df_both.loc[df_both.diff_k != 0]

#df_both = df_both.loc[df_both.k_value_MR !=0]

# print(df_both.rgi_area_km2.sum() / 28515.391 * 100)
# exit()

## Get coordinates and data
lat = df_both.cenlat.values
lon = df_both.cenlon.values
rgi_index = df_both.index

diff_q = df_both.diff_q.values
diff_k = df_both.diff_k.values


from scipy import stats
print('k-value RACMO normality test: ',
      stats.shapiro(df_both.k_value_MR.values))
print('q_calving RACMO normality test: ',
      stats.shapiro(df_both.calving_flux_MR.values))
print('k-value vel normality test: ',
      stats.shapiro(df_both.k_value_MV.values))
print('q_calving vel normality test: ',
      stats.shapiro(df_both.calving_flux_MV.values))

r_kendal_k, p_kendal_k = stats.kendalltau(df_both.k_value_MR.values,
        df_both.k_value_MV.values)

r_kendal_q, p_kendal_q = stats.kendalltau(df_both.calving_flux_MR.values,
        df_both.calving_flux_MV.values)

r_pearson_k, p_pearson_k = stats.pearsonr(df_both.k_value_MR.values,
        df_both.k_value_MV.values)

r_pearson_q, p_pearson_q = stats.pearsonr(df_both.calving_flux_MR.values,
        df_both.calving_flux_MV.values)

if p_pearson_k > 0.05:
    print('k - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_k)
else:
    print('k - values are correlated (reject H0) p=%.3f' % p_pearson_k)
    print(p_pearson_k)

if p_pearson_q > 0.05:
    print('q - values are uncorrelated (fail to reject H0) p=%.4f' % p_pearson_q)
else:
    print('q - values are correlated (reject H0) p=%.3f' % p_pearson_q)
    print(p_pearson_k)
exit()
#Now plotting
import matplotlib.gridspec as gridspec

color_palette = sns.color_palette("muted")

# Plot Fig 1
fig1 = plt.figure(figsize=(14, 16), constrained_layout=False)

widths = [2, 2]
heights = [2, 4]

spec = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.3, width_ratios=widths,
                         height_ratios=heights)


ax0 = plt.subplot(spec[0])
sns.scatterplot(x='k_value_MR', y='k_value_MV', data=df_both, ax=ax0,
                color=color_palette[0])

ax0.plot([0, 2.5], [0, 2.5], c='grey', alpha=0.7)

ax0.set_xlabel('$k_{RACMO}$ \n [yr$^{-1}$]')
ax0.set_ylabel('$k_{velo}$ \n [yr$^{-1}$]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
test = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_k,
                    ".2f")) + '\np-value = ' + str(format(p_pearson_k,
                                                                ".3E")),
                    prop=dict(size=18), frameon=False, loc=4)
ax0.add_artist(at)
ax0.add_artist(test)

ax1 = plt.subplot(spec[1])
sns.scatterplot(x='calving_flux_MR', y='calving_flux_MV', data=df_both, ax=ax1,
                color=color_palette[1])
ax1.plot([0, 0.3], [0, 0.3], c='grey', alpha=0.7)
ax1.set_xlabel('$q_{calving-RACMO}$ \n [$km^3$yr$^{-1}$]')
ax1.set_ylabel('$q_{calving-velo}$ \n [$km^3$yr$^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
test = AnchoredText('$r_{s}$ = '+ str(format(r_pearson_q,
                    ".2f")) + '\np-value = ' + str(format(p_pearson_q,
                                                                ".3E")),
                    prop=dict(size=18), frameon=False, loc=1)
ax1.add_artist(at)
ax1.add_artist(test)

ax2 = plt.subplot(spec[2])
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon, lat)

ax2.scatter(xx, yy, 60**diff_k, alpha=0.6, color=color_palette[0],
                                        edgecolor=color_palette[0])

# make legend with dummy points
for a in [0.5, 1.0, 1.5]:
    ax2.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=60**a,
                label=str(a) + 'yr$^{-1}$')
ax2.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='center', fontsize=14,
           title='$k$ differences',
           title_fontsize=14);
sm.set_scale_bar(location=(0.17, 0.02))
sm.visualize(ax=ax2)
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
sm = ds_geo.salem.get_map(countries=False)
#sm.set_shapefile(oceans=True)
sm.set_shapefile(coast_line, countries=True, linewidth=1.0, alpha=0.8)
xx, yy = sm.grid.transform(lon, lat)

ax3.scatter(xx, yy, 2000*diff_q, alpha=0.5, color=color_palette[1],
                                        edgecolor=color_palette[1])

# make legend with dummy points
for a in [0.05, 0.1, 0.5]:
    ax3.scatter([], [], c=sns.xkcd_rgb["grey"], alpha=0.5, s=2000*a,
                label=str(a) + 'km$^{3}$yr$^{-1}$')
ax3.legend(scatterpoints=1, frameon=False,
           labelspacing=1.5, loc='center', fontsize=14,
           title='$q_{calving}$ differences',
           title_fontsize=14);
sm.visualize(ax=ax3)
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc=2)
ax3.add_artist(at)

plt.tight_layout()
#plt.show()
# plt.savefig(os.path.join(plot_path, 'k_values_fa_result.jpg'),
#              bbox_inches='tight')
plt.savefig(os.path.join(plot_path, 'k_values_fa_result.pdf'),
              bbox_inches='tight')
