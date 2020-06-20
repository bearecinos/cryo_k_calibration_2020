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
rcParams['axes.labelsize'] = 25
rcParams['xtick.labelsize'] = 25
rcParams['ytick.labelsize'] = 25
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

######################### Velocity calibration result ########################
vel_result_path = os.path.join(MAIN_PATH,
    'output_data/5_calibration_vel_results/k_calibration_velandmu_reltol.csv')

df_vel_result = pd.read_csv(vel_result_path, index_col='Unnamed: 0')

df_vel_result['error'] = df_vel_result['u_obs'] * df_vel_result['rtol']


df_obs = df_vel_result[['RGIId',
                        'u_obs',
                        'rtol',
                        'error']].copy()

print(len(df_obs))
####################### RACMO calibration result ##############################

racmo_result_path = os.path.join(MAIN_PATH,
'output_data/6_racmo_calibration_results/k_calibration_racmo_reltol_q_calving_RACMO_meanNone_.csv')

df_racmo_result = pd.read_csv(racmo_result_path, index_col='Unnamed: 0')


#### Merge data frames ######################################################

df_to_plot = pd.merge(left=df_racmo_result,
                    right=df_obs,
                    how='left',
                    left_on = 'RGIId',
                    right_on='RGIId')

df_to_plot = df_to_plot.loc[df_to_plot.racmo_flux > 0]

nan_value = float("NaN")

df_to_plot.replace(" ", nan_value, inplace=True)

df_to_plot.dropna(subset =["u_obs"], inplace=True)

###### Test for correlation ###############################################

RMSD = utils.rmsd(df_to_plot.u_obs, df_to_plot.u_surf)

print('RMSD between observations and oggm-racmo',
      RMSD)

mean_dev = utils.md(df_to_plot.u_obs, df_to_plot.u_surf)

print('mean difference between observations and oggm-racmo',
      mean_dev)

slope, intercept, r_value, p_value, std_err = stats.linregress(df_to_plot.u_obs,
                                                               df_to_plot.u_surf)

ids = df_to_plot.RGIId.values
keep_ids = [(i in ids) for i in rgidf.RGIId]
rgidf_racmo_result = rgidf.iloc[keep_ids]

area_with_cal_result_racmo = rgidf_racmo_result.Area.sum()

Num =  area_with_cal_result_racmo/ study_area * 100

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
                    '\n p-value = ' + str(format(p_value, ".3E"))+
                    '\n RMSD = ' + str(format(RMSD, ".2f")) + ' m $yr^{-1}$' +
                    '\n Bias = ' + str(format(mean_dev, ".2f")) + ' m $yr^{-1}$',
                    prop=dict(size=18), frameon=False, loc=1)

z = np.arange(0, len(df_to_plot), 1)
zline = slope*z+intercept

wline = 1*z+0


fig1, axs = plt.subplots(figsize=(7, 8))

color_palette = sns.color_palette("muted")

axs.errorbar(df_to_plot.u_obs, df_to_plot.u_surf, xerr=df_to_plot.error,
             fmt='o', alpha=0.7, color=color_palette[1],
             ecolor=sns.xkcd_rgb["light grey"], elinewidth=1.5)
axs.plot(z, zline, color=color_palette[1])
axs.plot(z, wline, color='grey')

axs.set_xlabel('Observed surface velocities \n [m $yr^{-1}$]')
axs.set_ylabel('OGGM modeled surface velocities \n [m $yr^{-1}$]')

axs.add_artist(test)

plt.show()