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

df_vel = pd.read_csv(os.path.join(output_dir_path,
                                    'glacier_stats_vel_method.csv'))


df_racmo = pd.read_csv(os.path.join(output_dir_path,
                                      'glacier_stats_racmo_method.csv'))

df_racmo = df_racmo.loc[df_racmo.calving_flux_y !=0]

# Reading results without calving
prepo_dir_path = os.path.join(MAIN_PATH, 'output_data/1_Greenland_prepo/')
df_prepro = pd.read_csv(os.path.join(prepo_dir_path,
                'glacier_statistics_greenland_no_calving_with_sliding_.csv'))
df_prepro = df_prepro[['rgi_id', 'inv_volume_km3']]
df_prepro.rename(columns={'inv_volume_km3': 'inv_volume_km3_no_calving'},
                 inplace=True)


# Reading farinotti
fari = pd.read_csv(farinotti_path)
fari.rename(columns={'RGIId': 'rgi_id'}, inplace=True)
fari_crop = fari[['rgi_id', 'vol_itmix_m3', 'vol_bsl_itmix_m3']]

vol_km3 = fari_crop.loc[:,'vol_itmix_m3']*1e-9
vol_bsl_km3 = fari_crop.loc[:,'vol_bsl_itmix_m3']*1e-9

fari_crop.loc[:, 'vol_itmix_km3'] = vol_km3
fari_crop.loc[:, 'vol_bsl_itmix_km3'] = vol_bsl_km3

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

### Merging calibration data frames with their respective volume without
# calving

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

## Merge the volume without calving per calibration method
df_racmo_with_fari = pd.merge(left=df_racmo_vbsl,
                    right=fari_crop,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

df_vel_with_fari = pd.merge(left=df_vel_vbsl,
                    right=fari_crop,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

# Merged glaciers
df_merged_with_fari = pd.merge(left=df_racmo_with_fari,
                    right=df_vel_with_fari,
                    how='inner',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

#print(df_merged_with_fari.columns.values)

#print(df_racmo_with_fari.columns)
Num_glacier = np.array([len(df_vel_with_fari.rgi_id),
                      len(df_vel_with_fari.rgi_id),
                      len(df_vel_with_fari.rgi_id),
                      len(df_racmo_with_fari.rgi_id),
                      len(df_racmo_with_fari.rgi_id),
                      len(df_racmo_with_fari.rgi_id),
                      len(df_merged_with_fari.rgi_id),
                      len(df_merged_with_fari.rgi_id),
                      len(df_merged_with_fari.rgi_id)])
print(Num_glacier)

vol_exp = np.array([df_vel_with_fari['vol_itmix_km3'].sum(),
                    df_vel_with_fari['inv_volume_km3_no_calving'].sum(),
                    df_vel_with_fari['inv_volume_km3'].sum(),
                    df_racmo_with_fari['vol_itmix_km3'].sum(),
                    df_racmo_with_fari['inv_volume_km3_no_calving'].sum(),
                    df_racmo_with_fari['inv_volume_km3'].sum(),
                    df_merged_with_fari['vol_itmix_km3_x'].sum(),
                    df_merged_with_fari['inv_volume_km3_no_calving_x'].sum(),
                    df_merged_with_fari['inv_volume_km3_y'].sum(),
                    df_merged_with_fari['inv_volume_km3_x'].sum()])


vol_bsl_exp = np.array([df_vel_with_fari['vol_bsl_itmix_km3'].sum(),
                        df_vel_with_fari['vol_bsl_MV'].sum(),
                        df_vel_with_fari['vol_bsl_wc_MV'].sum(),
                        df_racmo_with_fari['vol_bsl_itmix_km3'].sum(),
                        df_racmo_with_fari['vol_bsl_MR'].sum(),
                        df_racmo_with_fari['vol_bsl_wc_MR'].sum(),
                       df_merged_with_fari['vol_bsl_itmix_km3_x'].sum(),
                       df_merged_with_fari['vol_bsl_MR'].sum(),
                       df_merged_with_fari['vol_bsl_wc_MV'].sum(),
                       df_merged_with_fari['vol_bsl_wc_MR'].sum()])

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
percentage_of_diff = [utils_vel.calculate_volume_percentage(vol_exp[1], vol_exp[2]),
                     utils_vel.calculate_volume_percentage(vol_exp[4], vol_exp[5]),
                     utils_vel.calculate_volume_percentage(vol_exp[9], vol_exp[8])]
                     # utils_vel.calculate_volume_percentage(vol_exp[4],  vol_exp[5]),
                     # utils_vel.calculate_volume_percentage(vol_exp[5], vol_exp[2])]
print(percentage_of_diff)

percentage_of_diff_vbsl = [utils_vel.calculate_volume_percentage(vol_bsl_exp[1], vol_bsl_exp[2]),
                     utils_vel.calculate_volume_percentage(vol_bsl_exp[4], vol_bsl_exp[5]),
                     utils_vel.calculate_volume_percentage(vol_bsl_exp[9], vol_bsl_exp[8])]
                     # utils_vel.calculate_volume_percentage(vol_bsl_exp[4],  vol_bsl_exp[5]),
                     # utils_vel.calculate_volume_percentage(vol_bsl_exp[5], vol_bsl_exp[2])]
print(percentage_of_diff_vbsl)


print('For the paper check if the volume below sea level is bigger than diff among config.')
print('Differences in volume below sea level ')
print(abs(vol_bsl_exp_sle[9]-vol_bsl_exp_sle[8]))
print(abs(vol_bsl_exp_sle[7]-vol_bsl_exp_sle[8]))
print(abs(vol_bsl_exp_sle[7]-vol_bsl_exp_sle[9]))

print('Differences in volume')
print(abs(vol_exp_sle[9]-vol_exp_sle[8]))
print(abs(vol_exp_sle[7]-vol_exp_sle[8]))
print(abs(vol_exp_sle[7]-vol_exp_sle[9]))

print(str(vol_exp_sle[7]) + 'increase to ' + str(vol_exp_sle[8])+ ' when using vel method')
print(str(vol_exp_sle[7]) + 'increase to ' + str(vol_exp_sle[9])+ ' when using RACMO method')
print('farinotti '+str(vol_exp_sle[6]))

print(str(vol_bsl_exp_sle[7]) + 'increase to ' + str(vol_bsl_exp_sle[8])+ ' when using vel method')
print(str(vol_bsl_exp_sle[7]) + 'increase to ' + str(vol_bsl_exp_sle[9])+ ' when using RACMO method')
print('farinotti '+str(vol_bsl_exp_sle[6]))

print('Percentage farinotti compared to RACMO and Vel')
print('to vel method '+ str(utils_vel.calculate_volume_percentage(vol_exp[6],
                                                                  vol_exp[8])))
print('to RACMO method '+ str(utils_vel.calculate_volume_percentage(vol_exp[6],
                                                                  vol_exp[9])))

exit()
fig = plt.figure(figsize=(12, 8))
sns.set(style="white", context="talk")

ax1=fig.add_subplot(111)
color_palette = sns.color_palette("deep")

color_array = [color_palette[3], color_palette[2],
               color_palette[0], color_palette[3],
               color_palette[2], color_palette[1],
               color_palette[3], color_palette[2],
               color_palette[0], color_palette[1]]

ax2= ax1.twiny()

# Example data
y_pos = np.arange(len(vol_exp))
y_pos = [0,0.5,1,2,2.5,3,4,4.5,5,5.5]


p1 = ax1.barh(y_pos, vol_bsl_exp*-1, align='center', color=sns.xkcd_rgb["grey"],
            height=0.5, edgecolor="white")

p2 = ax1.barh(y_pos, vol_exp, align='center', color=color_array, height=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([])
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

plt.legend((p2[0], p2[1], p2[2], p2[5]),
           ('Farinotti et al. (2019)',
            'Without $q_{calving}$',
            'With $q_{calving}$ - velocity',
            'With $q_{calving}$ - RACMO'),
            frameon=True, bbox_to_anchor=(0.8, -0.2), ncol=2)
            #bbox_to_anchor=(1.1, -0.15), ncol=5, fontsize=15)

plt.margins(0.05)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'volume_greenland.pdf'),
             bbox_inches='tight')