import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
from oggm import utils
from scipy import stats
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
os.getcwd()

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

# PARAMS for plots
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
#rcParams['legend.fontsize'] = 12
sns.set_context('poster')

plot_path = os.path.join(MAIN_PATH, 'plots/')

## Paths to output data ###################################################
# Data input
RGI_FILE = os.path.join(MAIN_PATH,
'input_data/05_rgi61_GreenlandPeriphery_bea/05_rgi61_GreenlandPeriphery.shp')

#RGI v6
rgidf = gpd.read_file(RGI_FILE)
rgidf.set_index('RGIId')
index = rgidf.index.values

# Get the glaciers classified by Terminus type
sub_mar = rgidf[rgidf['TermType'].isin([1])]
sub_lan = rgidf[rgidf['TermType'].isin([0])]

# Classify Marine-terminating by connectivity
sub_no_conect = sub_mar[sub_mar['Connect'].isin([0, 1])]
sub_conect = sub_mar[sub_mar['Connect'].isin([2])]

## Get study area
study_area = sub_no_conect.Area.sum()
##############################################################################
vel_result_path = os.path.join(MAIN_PATH,
    'output_data/5_calibration_vel_results/k_calibration_velandmu_reltol.csv')

df_vel_result = pd.read_csv(vel_result_path, index_col='Unnamed: 0')

df_vel_result['error'] = df_vel_result['u_obs'] * df_vel_result['rtol']

df_obs_to_plot = df_vel_result[['RGIId',
                                'u_obs',
                                'rtol',
                                'error']].copy()


ids = df_vel_result.RGIId.values
keep_ids = [(i in ids) for i in rgidf.RGIId]
rgidf_vel_result = rgidf.iloc[keep_ids]

area_with_cal_result = rgidf_vel_result.Area.sum()

print('Area % coverage where we find a k value with velocities',
      area_with_cal_result / study_area * 100)

##############################################################################
no_solution = os.path.join(MAIN_PATH,
        'output_data/5_calibration_vel_results/glaciers_with_no_solution.csv')

df_no_solution = pd.read_csv(no_solution, index_col='Unnamed: 0')


data_gaps = os.path.join(MAIN_PATH,
        'output_data/2_Process_vel_data/glaciers_with_no_velocity_data.csv')

df_no_data = pd.read_csv(data_gaps, index_col='Unnamed: 0')

df_no_sol = pd.merge(left=df_no_solution,
                    right=df_no_data,
                    how='left',
                    left_on = 'RGIId',
                    right_on='RGIId')

ids = df_no_sol.RGIId.values
keep_ids = [(i in ids) for i in rgidf.RGIId]
rgidf_no_solution = rgidf.iloc[keep_ids]

area_no_solution = rgidf_no_solution.Area.sum()

print('Area % with no calving solution, no velocity',
      area_no_solution / study_area * 100)

print('sum',
(area_with_cal_result/study_area * 100) + (area_no_solution/study_area * 100))
#############################################################################

racmo_result_path = os.path.join(MAIN_PATH,
'output_data/6_racmo_calibration_results/k_calibration_racmo_reltol_q_calving_RACMO_meanNone_.csv')

df_racmo_result = pd.read_csv(racmo_result_path, index_col='Unnamed: 0')

################## racmo result to plot with vel observations ################
df_to_plot = pd.merge(left=df_racmo_result,
                    right=df_obs_to_plot,
                    how='left',
                    left_on = 'RGIId',
                    right_on='RGIId')

df_to_plot = df_to_plot.loc[df_to_plot.racmo_flux > 0]

nan_value = float("NaN")

df_to_plot.replace(" ", nan_value, inplace=True)

df_to_plot.dropna(subset =["u_obs"], inplace=True)
##############################################################################


df_racmo_negative = df_racmo_result.loc[df_racmo_result.racmo_flux < 0]

ids = df_racmo_result.RGIId.values
keep_ids = [(i in ids) for i in rgidf.RGIId]
rgidf_racmo_result = rgidf.iloc[keep_ids]

area_with_cal_result_racmo = rgidf_racmo_result.Area.sum()

print('Area % coverage where we find a k value with racmo data',
      area_with_cal_result_racmo / study_area * 100)

ids_negative = df_racmo_negative.RGIId.values
keep_ids_negative = [(i in ids_negative) for i in rgidf.RGIId]
rgidf_racmo_negative = rgidf.iloc[keep_ids_negative]

area_racmo_negative = rgidf_racmo_negative.Area.sum()

print('Area % coverage where racmo data is negative',
      area_racmo_negative/ study_area * 100)

negative_percet = np.round(area_racmo_negative/ study_area * 100)
##############################################################################

RMSD = utils.rmsd(df_vel_result.u_obs, df_vel_result.u_surf)

print('RMSD between observations and oggm',
      RMSD)

mean_dev = utils.md(df_vel_result.u_obs, df_vel_result.u_surf)

print('mean difference between observations and oggm',
      mean_dev)

slope, intercept, r_value, p_value, std_err = stats.linregress(df_vel_result.u_obs,
                                                               df_vel_result.u_surf)

Num = area_with_cal_result / study_area * 100

print('N = ', Num)
print('bo = ', slope)
print(intercept)
print(r_value)
print(p_value)
print(mean_dev)
print(RMSD)

#
test = AnchoredText(' Area % = '+ str(format(Num, ".2f")) +
                    '\n slope = '+ str(format(slope,".2f")) +
                    '\n intercept = '+ str(format(intercept, ".2f")) +
                    '\n r$^2$ = '+ str(format(r_value, ".2f")) +
                    '\n RMSD = ' + str(format(RMSD, ".2f")) + ' m$yr^{-1}$' +
                    '\n Bias = ' + str(format(mean_dev, ".2f")) + ' m$yr^{-1}$',
                    prop=dict(size=16), frameon=True, loc=1)

z = np.arange(0, len(df_vel_result), 1)
zline = slope*z+intercept
wline = 1*z+0


##############################################################################

RMSD_racmo = utils.rmsd(df_racmo_result.racmo_flux,
                        df_racmo_result.calving_flux)

print('RMSD between RACMO and oggm',
      RMSD_racmo)

mean_dev_racmo = utils.md(df_racmo_result.racmo_flux,
                          df_racmo_result.calving_flux)

print('mean difference between RACMO and oggm',
      mean_dev_racmo)

slope_r, intercept_r, r_value_r, p_value_r, std_err_r = stats.linregress(df_racmo_result.racmo_flux,
                                                               df_racmo_result.calving_flux)

Num_r = area_with_cal_result_racmo / study_area * 100

print('N = ', Num_r)
print('bo = ', slope_r)
print(intercept_r)
print(r_value_r)
print(p_value_r)
print(mean_dev_racmo)
print(RMSD_racmo)

#
test_racmo = AnchoredText(' Area % = '+ str(format(Num_r, ".2f")) +
                    '\n slope = '+ str(format(slope_r,".4f")) +
                    '\n intercept = '+ str(format(intercept_r, ".4f")) +
                    '\n r$^2$ = '+ str(format(r_value_r, ".4f")) +
                    '\n RMSD = ' + str(format(RMSD_racmo, ".4f")) + ' $km^{3}yr^{-1}$' +
                    '\n Bias = ' + str(format(mean_dev_racmo, ".4f")) + ' $km^{3}yr^{-1}$',
                    prop=dict(size=16), frameon=True, loc=1)

negative_area = AnchoredText('Area % = \n ' + str(format(negative_percet, ".2f")),
                             prop=dict(size=13, color='r', fontweight="bold"),
                             frameon=False, loc=6)



x_r = np.linspace(-0.01, max(df_racmo_result.racmo_flux), 100)
line_r = slope_r*x_r+intercept_r
line_w = 1*x_r+0

###### Test for correlation ###############################################

RMSD_racmo_obs = utils.rmsd(df_to_plot.u_obs, df_to_plot.u_surf)

print('RMSD between observations and oggm-racmo',
      RMSD_racmo_obs)

mean_dev_racmo_obs = utils.md(df_to_plot.u_obs, df_to_plot.u_surf)

print('mean difference between observations and oggm-racmo',
      mean_dev_racmo_obs)

slope_racmo_obs, intercept_racmo_obs, r_value_racmo_obs, p_value_racmo_obs, std_err_racmo_obs = stats.linregress(df_to_plot.u_obs,
                                                               df_to_plot.u_surf)

ids_racmo_obs = df_to_plot.RGIId.values
keep_ids_racmo_obs = [(i in ids_racmo_obs) for i in rgidf.RGIId]
rgidf_racmo_result_racmo_obs = rgidf.iloc[keep_ids_racmo_obs]

area_with_cal_result_racmo_obs = rgidf_racmo_result_racmo_obs.Area.sum()

Num_racmo_obs =  area_with_cal_result_racmo_obs/ study_area * 100

print('N = ', Num_racmo_obs)
print('bo = ', slope_racmo_obs)
print(intercept_racmo_obs)
print(r_value_racmo_obs)
print(p_value_racmo_obs)
print(mean_dev_racmo_obs)
print(RMSD_racmo_obs)

#
test_racmo_obs = AnchoredText(' Area % = '+ str(format(Num_racmo_obs, ".2f")) +
                    '\n slope = '+ str(format(slope_racmo_obs,".2f")) +
                    '\n intercept = '+ str(format(intercept_racmo_obs, ".2f")) +
                    '\n r$^2$ = '+ str(format(r_value_racmo_obs, ".2f")) +
                    '\n RMSD = ' + str(format(RMSD_racmo_obs, ".2f")) + ' m$yr^{-1}$' +
                    '\n Bias = ' + str(format(mean_dev_racmo_obs, ".2f")) + ' m$yr^{-1}$',
                    prop=dict(size=16), frameon=True, loc=1)

z_racmo_obs = np.arange(0, len(df_to_plot)+20, 1)
zline_racmo_obs = slope_racmo_obs*z_racmo_obs+intercept_racmo_obs

wline_racmo_obs = 1*z_racmo_obs+0

##############################################################################

fig1 = plt.figure(figsize=(18.5, 6), constrained_layout=False)

color_palette = sns.color_palette("muted")

spec = gridspec.GridSpec(1, 3, wspace=0.4)

ax0 = plt.subplot(spec[0])
ax0.errorbar(df_vel_result.u_obs, df_vel_result.u_surf,
            xerr=df_vel_result.error, fmt='o', alpha=0.7,
            color=color_palette[0], ecolor=sns.xkcd_rgb["light grey"],
            elinewidth=1.5)
ax0.plot(z, zline)
ax0.plot(z, wline, color='grey')
ax0.set_xlabel('Observed surface velocities \n [m $yr^{-1}$]')
ax0.set_ylabel('OGGM surface velocities')
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc=2)
ax0.add_artist(at)
ax0.add_artist(test)



ax1 = plt.subplot(spec[1])
ax1.errorbar(df_racmo_result.racmo_flux, df_racmo_result.calving_flux,
            xerr=None, fmt='o', alpha=0.5,
            color=color_palette[1])
ax1.plot(x_r, line_r, color=color_palette[1])
ax1.plot(x_r, line_w, color='grey')
ax1.axvline(0, color='k', linestyle='--')
ax1.set_xlabel('RACMO frontal ablation \n [$km^{3}$$yr^{-1}$]')
ax1.set_ylabel('OGGM frontal ablation')
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc=2)
ax1.add_artist(at)
ax1.add_artist(test_racmo)
ax1.add_artist(negative_area)

ax2 = plt.subplot(spec[2])
ax2.errorbar(df_to_plot.u_obs, df_to_plot.u_surf, xerr=df_to_plot.error,
             fmt='o', alpha=0.7, color=color_palette[1],
             ecolor=sns.xkcd_rgb["light grey"], elinewidth=1.5)
ax2.plot(z_racmo_obs, zline_racmo_obs, color=color_palette[1])
ax2.plot(z_racmo_obs, wline_racmo_obs, color='grey')

ax2.set_xlabel('Observed surface velocities \n [m $yr^{-1}$]')
ax2.set_ylabel('OGGM surface velocities')
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc=2)
ax2.add_artist(at)
ax2.add_artist(test_racmo_obs)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'k_values_result_stats_new.pdf'),
                bbox_inches='tight')

