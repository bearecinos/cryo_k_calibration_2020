import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

import sys
sys.path.append(MAIN_PATH)
# velocity module
from velocity_tools import utils_velocity as utils_vel

# PARAMS for plots
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

sns.set_context('poster')

plot_path = os.path.join(MAIN_PATH, 'plots/')

farinotti_path = os.path.join(MAIN_PATH, 'input_data/rgi62_era5_itmix_df.csv')

# Reading results with calving main volumes
output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')
df_both = pd.read_csv(os.path.join(output_dir_path,
                                'glacier_stats_both_methods.csv'))

# Reading results without calving
prepo_dir_path = os.path.join(MAIN_PATH, 'output_data/1_Greenland_prepo/')
df_prepro = pd.read_csv(os.path.join(prepo_dir_path,
                'glacier_statistics_greenland_no_calving_with_sliding_.csv'))
df_prepro = df_prepro[['rgi_id', 'inv_volume_km3']]

out_vbsl_path = os.path.join(MAIN_PATH, 'output_data/12_volume_vsl/config/')
config_one_path = os.path.join(out_vbsl_path,
                               'config_01_onlyMT/volume_below_sea_level.csv')

config_two_path = os.path.join(out_vbsl_path,
                               'config_02_onlyMT/volume_below_sea_level.csv')

# Reading farinotti
fari = pd.read_csv(farinotti_path)
fari.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

fari_crop = fari[['rgi_id', 'vol_itmix_m3', 'vol_bsl_itmix_m3']]

vol_km3 = fari_crop.loc[:,'vol_itmix_m3']*1e-9
vol_bsl_km3 = fari_crop.loc[:,'vol_bsl_itmix_m3']*1e-9

fari_crop.loc[:, 'vol_itmix_km3'] = vol_km3
fari_crop.loc[:, 'vol_bsl_itmix_km3'] = vol_bsl_km3

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

# Merge to main results
df_common = pd.merge(left=config_one,
                    right=config_two,
                    how='inner',
                    left_on = 'rgi_id',
                    right_on='rgi_id')


#getting the common glaciers and making all a single data frame
df_both_plus_prepro = pd.merge(left=df_both,
                    right=df_prepro,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

df_all = pd.merge(left=df_both_plus_prepro,
                  right=df_common,
                  how='left',
                  left_on='rgi_id',
                  right_on='rgi_id')

df_all_plus_fari = pd.merge(left=df_all,
                  right=fari_crop,
                  how='inner',
                  left_on='rgi_id',
                  right_on='rgi_id')

no_calving_volume = df_all['inv_volume_km3'].sum()
no_calving_vol_bsl = df_all['vol_bsl_MV'].sum()

vel_volume_c = df_all['inv_volume_km3_MV'].sum()
vel_volume_bsl_c = df_all['vol_bsl_wc_MV'].sum()

racmo_volume_c = df_all['inv_volume_km3_MR'].sum()
racmo_volume_bsl_c = df_all['vol_bsl_wc_MR'].sum()

fari_volume = df_all_plus_fari['vol_itmix_km3'].sum()
fari_volume_bsl = df_all_plus_fari['vol_bsl_itmix_km3'].sum()

print(df_all_plus_fari.rgi_id)

exp_name = ['Farinotti et al. (2019)',
            'Without calving',
            '$q_{calving}$ with velocities',
            '$q_{calving}$ with RACMO']
exp_number = [1, 2, 3, 4]

vol_exp = np.array([fari_volume,
                    no_calving_volume,
                    vel_volume_c,
                    racmo_volume_c])

vol_bsl_exp = np.array([fari_volume_bsl,
                        no_calving_vol_bsl,
                        vel_volume_bsl_c,
                        racmo_volume_bsl_c])

vol_exp_sle = np.array([utils_vel.calculate_sea_level_equivalent(fari_volume),
                utils_vel.calculate_sea_level_equivalent(no_calving_volume),
                utils_vel.calculate_sea_level_equivalent(vel_volume_c),
                utils_vel.calculate_sea_level_equivalent(racmo_volume_c)])

vol_bsl_exp_sle = np.array([utils_vel.calculate_sea_level_equivalent(fari_volume_bsl),
                  utils_vel.calculate_sea_level_equivalent(no_calving_vol_bsl),
                  utils_vel.calculate_sea_level_equivalent(vel_volume_bsl_c),
                  utils_vel.calculate_sea_level_equivalent(racmo_volume_bsl_c)])

percentage = np.absolute(np.array([utils_vel.calculate_volume_percentage(vel_volume_c,
                                                                         racmo_volume_c),
    utils_vel.calculate_volume_percentage(vel_volume_c,
                                          no_calving_volume),
     utils_vel.calculate_volume_percentage(racmo_volume_c,
                                           no_calving_volume),
     utils_vel.calculate_volume_percentage(vel_volume_c,
                                           fari_volume),
    utils_vel.calculate_volume_percentage(racmo_volume_c,
                                          fari_volume)]))

percentage_diff = percentage

# Make a dataframe with each configuration output
d = {'Experiment No': exp_number,
     'Experiment Name': exp_name,
     'Volume in s.l.e': vol_exp_sle,
     'Volume bsl in s.l.e': vol_bsl_exp_sle,
     'Volume in km3': vol_exp,
     'Volume bsl in km3': vol_bsl_exp}
ds = pd.DataFrame(data=d)

print('FOR THE PAPER')
print('----------------')
print(exp_name)
print(vol_exp)
print(vol_bsl_exp)
print('Difference', np.diff(vol_exp))
print('Percentage', percentage)

print(ds)


fig = plt.figure(figsize=(12, 8))
sns.set(style="white", context="talk")

N = len(vol_exp)
ind = np.arange(N)    # the x locations for the groups
labelsxticks = exp_name

ax1=fig.add_subplot(111)
color_palette = sns.color_palette("deep")

color_array = [color_palette[3], color_palette[2],
               color_palette[0], color_palette[1]]

ax2= ax1.twiny()

# Example data
y_pos = np.arange(len(exp_name))

ax1.barh(y_pos, vol_bsl_exp*-1, align='center', color=sns.xkcd_rgb["grey"],
            height=0.5, edgecolor="white")

ax1.barh(y_pos, vol_exp, align='center', color=color_array, height=0.5)


ax1.set_yticks(y_pos)
ax1.set_yticklabels(labelsxticks)
# labels read top-to-bottom
ax1.invert_yaxis()
ax1.set_xlabel('Volume [km³]',fontsize=18)

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
array = ax1.get_xticks()

# Get the other axis on sea level equivalent
sle = []
for value in array:
    sle.append(np.round(abs(utils_vel.calculate_sea_level_equivalent(value)),2))

ax2.set_xticklabels(sle, fontsize=20)
ax2.set_xlabel('Volume [mm SLE]', fontsize=18)


plt.margins(0.05)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'volume_greenland.pdf'),
            bbox_inches='tight')