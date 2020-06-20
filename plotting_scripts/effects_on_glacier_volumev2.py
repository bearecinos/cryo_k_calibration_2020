import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd

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

#### Read in RGI for study area ##############################################
RGI_FILE = os.path.join(MAIN_PATH,
'input_data/05_rgi61_GreenlandPeriphery_bea/05_rgi61_GreenlandPeriphery.shp')

#RGI v6
df = gpd.read_file(RGI_FILE)
df.set_index('RGIId')
index = df.index.values

# Get the glaciers classified by Terminus type
sub_mar = df[df['TermType'].isin([1])]
sub_lan = df[df['TermType'].isin([0])]

# Classify Marine-terminating by connectivity
sub_no_conect = sub_mar[sub_mar['Connect'].isin([0, 1])]
sub_conect = sub_mar[sub_mar['Connect'].isin([2])]

sub_no_conect_no_ice_cap = sub_no_conect[~sub_no_conect.RGIId.str.contains('RGI60-05.10315')]

study_area_no_ice_cap = sub_no_conect_no_ice_cap.Area.sum()

rgidf_ice_caps = df[df['RGIId'].str.match('RGI60-05.10315')]
ice_cap_ids = rgidf_ice_caps.RGIId.values

keep_indexes = [(i in ice_cap_ids) for i in df.RGIId]
ice_cap_rgi = df.iloc[keep_indexes]

ice_cap_area = ice_cap_rgi.Area.sum()

########################## Reading farinotti #################################
farinotti_path = os.path.join(MAIN_PATH, 'input_data/rgi62_era5_itmix_df.csv')
fari = pd.read_csv(farinotti_path)
print(len(fari))
# Selecting only the ice Cap
fari_ice_cap = fari[fari['RGIId'].str.match('RGI60-05.10315')].copy()
print(len(fari_ice_cap))

fari = fari.loc[fari['RGIId']!='RGI60-05.10315']
print(len(fari))

# Making RGI the same column name as output data and selecting what we need
fari.rename(columns={'RGIId': 'rgi_id'}, inplace=True)
fari = fari[['rgi_id', 'vol_itmix_m3', 'vol_bsl_itmix_m3']]

vol_km3 = fari.loc[:,'vol_itmix_m3'].copy()*1e-9
vol_bsl_km3 = fari.loc[:,'vol_bsl_itmix_m3'].copy()*1e-9

fari.loc[:, 'vol_itmix_km3'] = vol_km3
fari.loc[:, 'vol_bsl_itmix_km3'] = vol_bsl_km3

# Keeping the ice cap separate
fari_ice_cap.rename(columns={'RGIId': 'rgi_id'}, inplace=True)
fari_ice_cap = fari_ice_cap[['rgi_id', 'vol_itmix_m3', 'vol_bsl_itmix_m3']]

vol_ice_cap_km3 = fari_ice_cap.loc[:, 'vol_itmix_m3'].copy()*1e-9
vol_bsl_ice_cap_km3 = fari_ice_cap.loc[:, 'vol_bsl_itmix_m3'].copy()*1e-9

fari_ice_cap.loc[:, 'vol_itmix_km3'] = vol_ice_cap_km3
fari_ice_cap.loc[:, 'vol_bsl_itmix_km3'] = vol_bsl_ice_cap_km3

########### Reading results with calving main volumes ########################
output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')

# VELOCITY METHOD
df_vel = pd.read_csv(os.path.join(output_dir_path,
                                    'glacier_stats_vel_method.csv'))

print('Frontal ablation flux Velocity in km3/yr', df_vel['calving_flux'].sum())
print('Frontal ablation flux Velocity in Gt/yr', df_vel['calving_flux'].sum()/1.091)
print('Number of glaciers', len(df_vel))

df_vel_ice_cap = df_vel[df_vel['rgi_id'].str.match('RGI60-05.10315')]
df_vel_no_ice_cap = df_vel[~df_vel.rgi_id.str.contains('RGI60-05.10315')]

# RACMO METHOD
df_racmo = pd.read_csv(os.path.join(output_dir_path,
                                      'glacier_stats_racmo_method.csv'))

print('Frontal ablation flux RACMO in km3/yr', df_racmo['calving_flux_x'].sum())
print('Frontal ablation flux RACMO in Gt/yr', df_racmo['calving_flux_x'].sum()/1.091)
print('Number of glaciers', len(df_racmo))

df_racmo_ice_cap = df_racmo[df_racmo['rgi_id'].str.match('RGI60-05.10315')]
df_racmo_no_ice_cap = df_racmo[~df_racmo.rgi_id.str.contains('RGI60-05.10315')]

############  Reading results without calving  ###############################
prepo_dir_path = os.path.join(MAIN_PATH, 'output_data/1_Greenland_prepo/')
df_prepro = pd.read_csv(os.path.join(prepo_dir_path,
                'glacier_statistics_greenland_no_calving_with_sliding_.csv'))

df_prepro = df_prepro[['rgi_id', 'inv_volume_km3', 'rgi_area_km2']]
df_prepro.rename(columns={'inv_volume_km3': 'inv_volume_km3_no_calving'},
                 inplace=True)

# NO ICE CAP
df_prepro_no_ice_cap = df_prepro[~df_prepro.rgi_id.str.contains('RGI60-05.10315')]

studya_area_prepro_no_ice_cap = df_prepro_no_ice_cap['rgi_area_km2'].sum()

# Reading pre-pro ice cap
prepo_ice_cap_dir_path = os.path.join(MAIN_PATH, 'output_data/14_ice_cap_prepo/')
df_prepro_ice_cap = pd.read_csv(os.path.join(prepo_ice_cap_dir_path,
                'glacier_statistics_ice_cap_no_calving_with_sliding_.csv'))

df_prepro_ice_cap = df_prepro_ice_cap[['rgi_id', 'rgi_area_km2', 'inv_volume_km3', 'terminus_type']]
df_prepro_ice_cap.rename(columns={'inv_volume_km3': 'inv_volume_km3_no_calving'},
                 inplace=True)

df_prepro_ice_cap.to_csv(os.path.join(plot_path, 'ice_cap_prepro.csv'))

#### Reading configurations for volume below sea level #######################
# Reading volume below sea level
out_vbsl_path = os.path.join(MAIN_PATH, 'output_data/12_volume_vsl/config/')

config_one_path = os.path.join(out_vbsl_path,
                               'config_01_onlyMT/volume_below_sea_level.csv')

config_two_path = os.path.join(out_vbsl_path,
                               'config_02_onlyMT/volume_below_sea_level.csv')

config_one = pd.read_csv(config_one_path)
config_two = pd.read_csv(config_two_path)

# VELOCITY METHOD
config_one.rename(columns={'RGIId': 'rgi_id'}, inplace=True)
config_one.rename(columns={'volume bsl': 'vol_bsl_MV'}, inplace=True)
config_one.rename(columns={'volume bsl with calving': 'vol_bsl_wc_MV'},
                  inplace=True)
# RACMO METHOD
config_two.rename(columns={'RGIId': 'rgi_id'}, inplace=True)
config_two.rename(columns={'volume bsl': 'vol_bsl_MR'}, inplace=True)
config_two.rename(columns={'volume bsl with calving': 'vol_bsl_wc_MR'},
                  inplace=True)

# Filter ice cap
config_one_rest = config_one[~config_one.rgi_id.str.contains('RGI60-05.10315')]
config_one_ice_cap = config_one[config_one['rgi_id'].str.match('RGI60-05.10315')]


config_two_rest = config_two[~config_two.rgi_id.str.contains('RGI60-05.10315')]
config_two_ice_cap = config_two[config_two['rgi_id'].str.match('RGI60-05.10315')]


### Merging output data with their respective volume without calving
racmo_rest_and_prepro = pd.merge(left=df_racmo_no_ice_cap,
                    right=df_prepro_no_ice_cap,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

vel_rest_and_prepro = pd.merge(left=df_vel_no_ice_cap,
                    right=df_prepro_no_ice_cap,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')


# Merge with vbsl results
racmo_rest_and_prepro_and_vbsl = pd.merge(left=racmo_rest_and_prepro,
                                          right=config_two_rest,
                                          how='left',
                                          left_on = 'rgi_id',
                                          right_on='rgi_id')

vel_rest_and_prepro_and_vbsl = pd.merge(left=vel_rest_and_prepro,
                                          right=config_one_rest,
                                          how='left',
                                          left_on = 'rgi_id',
                                          right_on='rgi_id')


## Merge now with Farinotti 2019
racmo_rest = pd.merge(left=racmo_rest_and_prepro_and_vbsl,
                    right=fari,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

racmo_rest.to_csv(os.path.join(plot_path, 'ramo_rest.csv'))

vel_rest = pd.merge(left=vel_rest_and_prepro_and_vbsl,
                    right=fari,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

vel_rest.to_csv(os.path.join(plot_path, 'vel_rest.csv'))

# Merged glaciers
common_rest = pd.merge(left=racmo_rest,
                    right=vel_rest,
                    how='inner',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

common_rest.to_csv(os.path.join(plot_path, 'common_rest.csv'))

##### Add up the volume of the Ice cap #######################################
ice_cap_tw_no_calving = df_prepro_ice_cap.loc[df_prepro_ice_cap['terminus_type'] == 'Marine-terminating']
ice_cap_land_no_calving = df_prepro_ice_cap.loc[df_prepro_ice_cap['terminus_type'] == 'Land-terminating']

print(ice_cap_tw_no_calving.rgi_id)
#check which id's can be run in pre-processing but not in the calving module
missing_ids = ['RGI60-05.10315_d329', 'RGI60-05.10315_d328']
keep_index = [(i in missing_ids) for i in ice_cap_tw_no_calving.rgi_id]
missing_glaciers = ice_cap_tw_no_calving.loc[keep_index]

ice_cap_area_tw = ice_cap_tw_no_calving['rgi_area_km2'].sum()
ice_cap_area_land = ice_cap_land_no_calving['rgi_area_km2'].sum()

# Volume before calving
ICE_CAP_volume_before_calving = ice_cap_land_no_calving['inv_volume_km3_no_calving'].sum() + ice_cap_tw_no_calving['inv_volume_km3_no_calving'].sum()

# Volume after calving
ICE_CAP_volume_RACMO = df_racmo_ice_cap['inv_volume_km3'].sum() + ice_cap_land_no_calving['inv_volume_km3_no_calving'].sum() + missing_glaciers['inv_volume_km3_no_calving'].sum()
ICE_CAP_volume_VEL = df_vel_ice_cap['inv_volume_km3'].sum() + ice_cap_land_no_calving['inv_volume_km3_no_calving'].sum() + missing_glaciers['inv_volume_km3_no_calving'].sum()

# volume below sea level BEFORE calving
vol_ice_cap_bsl_RACMO = config_two_ice_cap['vol_bsl_MR'].sum()
# volume below sea level AFTER calving
vol_ice_cap_bsl_wc_RACMO = config_two_ice_cap['vol_bsl_wc_MR'].sum()

# volume below sea level BEFORE calving
vol_ice_cap_bsl_VEL = config_one_ice_cap['vol_bsl_MV'].sum()

if vol_ice_cap_bsl_RACMO == vol_ice_cap_bsl_VEL:
    print(True)

# volume below sea level AFTER calving
vol_ice_cap_bsl_wc_VEL = config_one_ice_cap['vol_bsl_wc_MV'].sum()

fari_vol_ice_cap = fari_ice_cap['vol_itmix_km3'].sum()
fari_vol_bsl_ice_cap = fari_ice_cap['vol_bsl_itmix_km3'].sum()

#### Add up volumes for the rest of the Glaciers #############################

#RACMO
RACMO_volume_before_calving = racmo_rest['inv_volume_km3_no_calving'].sum()
RACMO_volume_after_calving = racmo_rest['inv_volume_km3'].sum()
#below sea level
RACMO_volume_bsl_before_calving = racmo_rest['vol_bsl_MR'].sum()
RACMO_volume_bsl_after_calving = racmo_rest['vol_bsl_wc_MR'].sum()
# farinotti volumes
Farinotti_RACMO = racmo_rest['vol_itmix_km3'].sum()
Farinotti_RACMO_bsl = racmo_rest['vol_bsl_itmix_km3'].sum()

#VELOCITY
VEL_volume_before_calving = vel_rest['inv_volume_km3_no_calving'].sum()
VEL_volume_after_calving = vel_rest['inv_volume_km3'].sum()
#below sea level
VEL_volume_bsl_before_calving = vel_rest['vol_bsl_MV'].sum()
VEL_volume_bsl_after_calving = vel_rest['vol_bsl_wc_MV'].sum()
# farinotti volumes
Farinotti_VEL = vel_rest['vol_itmix_km3'].sum()
Farinotti_VEL_bsl = vel_rest['vol_bsl_itmix_km3'].sum()

## Common
common_volume_before_calving = common_rest['inv_volume_km3_no_calving_x'].sum()
common_volume_after_calving_RACMO = common_rest['inv_volume_km3_x'].sum()
common_volume_after_calving_VEL = common_rest['inv_volume_km3_y'].sum()

if common_rest['vol_bsl_MR'].sum() == common_rest['vol_bsl_MV'].sum():
    print(True)

if vol_ice_cap_bsl_RACMO == vol_ice_cap_bsl_VEL:
    print(True)

if common_rest['vol_itmix_km3_x'].sum() == common_rest['vol_itmix_km3_y'].sum():
    print(True)

common_volume_bsl_before_calving = common_rest['vol_bsl_MR'].sum()
common_RACMO_volume_bsl_after_calving = common_rest['vol_bsl_wc_MR'].sum()
common_VEL_volume_bsl_after_calving = common_rest['vol_bsl_wc_MV'].sum()

Farinotti_common = common_rest['vol_itmix_km3_x'].sum()
Farinotti_common_bsl = common_rest['vol_bsl_itmix_km3_x'].sum()

### Making a data frame
order = ['Velocity-Farinotti', 'Velocity-No-calving', 'Velocity-after-calving',
         'RACMO-Farinotti', 'RACMO-No-calving', 'RACMO-after-calving',
         'Common-Farinotti', 'Common-No-calving', 'Common-Velocity', 'Common-RACMO']

vol_exp = np.array([Farinotti_VEL,
                    VEL_volume_before_calving,
                    VEL_volume_after_calving,
                    Farinotti_RACMO,
                    RACMO_volume_before_calving,
                    RACMO_volume_after_calving,
                    Farinotti_common,
                    common_volume_before_calving,
                    common_volume_after_calving_VEL,
                    common_volume_after_calving_RACMO])

vol_bsl_exp = np.array([Farinotti_VEL_bsl,
                        VEL_volume_bsl_before_calving,
                        VEL_volume_bsl_after_calving,
                        Farinotti_RACMO_bsl,
                        RACMO_volume_bsl_before_calving,
                        RACMO_volume_bsl_after_calving,
                        Farinotti_common_bsl,
                        common_volume_bsl_before_calving,
                        common_VEL_volume_bsl_after_calving,
                        common_RACMO_volume_bsl_after_calving])

ice_cap_vol_exp = np.array([fari_vol_ice_cap, ICE_CAP_volume_before_calving, ICE_CAP_volume_VEL,
                            fari_vol_ice_cap, ICE_CAP_volume_before_calving, ICE_CAP_volume_RACMO,
                            fari_vol_ice_cap, ICE_CAP_volume_before_calving, ICE_CAP_volume_VEL, ICE_CAP_volume_RACMO])

ice_cap_vol_bsl_exp = np.array([fari_vol_bsl_ice_cap,
                                vol_ice_cap_bsl_RACMO,
                                vol_ice_cap_bsl_wc_VEL,
                                fari_vol_bsl_ice_cap,
                                vol_ice_cap_bsl_RACMO,
                                vol_ice_cap_bsl_wc_RACMO,
                                fari_vol_bsl_ice_cap,
                                vol_ice_cap_bsl_RACMO,
                                vol_ice_cap_bsl_wc_VEL,
                                vol_ice_cap_bsl_wc_RACMO])



print(vel_rest['rgi_area_km2'].sum())
print(racmo_rest['rgi_area_km2'].sum())
print(common_rest['rgi_area_km2_x'].sum())
print('Number of glaciers common',len(common_rest))

study_area = study_area_no_ice_cap + ice_cap_area_tw

vel_area = vel_rest['rgi_area_km2'].sum() + ice_cap_area_tw
racmo_area = racmo_rest['rgi_area_km2'].sum()  + ice_cap_area_tw
common_area = common_rest['rgi_area_km2_x'].sum()  + ice_cap_area_tw

Area_coverage_percent = [(vel_area/study_area)*100, (racmo_area/study_area)*100, (common_area/study_area)*100]
print(Area_coverage_percent)

# ## TODO: CALCULATE ALL DIFF BETWEEN VOLUMES!!!
total_vol = vol_exp + ice_cap_vol_exp
total_vol_bsl = vol_bsl_exp + ice_cap_vol_bsl_exp
print(total_vol)

vol_exp_sle = []
for vol in total_vol:
    sle = utils_vel.calculate_sea_level_equivalent(vol)
    vol_exp_sle = np.append(vol_exp_sle, sle)

vol_bsl_exp_sle = []
for vol_bsl in total_vol_bsl:
    sle = utils_vel.calculate_sea_level_equivalent(vol_bsl)
    vol_bsl_exp_sle = np.append(vol_bsl_exp_sle, sle)

print('Percentage of difference around OGGM config')
percentage_of_diff_nofari = [utils_vel.calculate_volume_percentage(total_vol[1], total_vol[2]),
                      utils_vel.calculate_volume_percentage(total_vol[4], total_vol[5]),
                      utils_vel.calculate_volume_percentage(total_vol[7], total_vol[8]),
                      utils_vel.calculate_volume_percentage(total_vol[7],  total_vol[9])]
print(percentage_of_diff_nofari)

print('Percentage between RACMO and vel vol')
config_percentage_diff = utils_vel.calculate_volume_percentage(total_vol[9], total_vol[8])
print(config_percentage_diff )

print('Percentage of difference between OGGM after calving and fari')
percentage_of_diff_fari = [utils_vel.calculate_volume_percentage(total_vol[0], total_vol[2]),
                      utils_vel.calculate_volume_percentage(total_vol[3], total_vol[5]),
                      utils_vel.calculate_volume_percentage(total_vol[6], total_vol[8]),
                      utils_vel.calculate_volume_percentage(total_vol[6],  total_vol[9])]
print(percentage_of_diff_fari)

#exit()

print('For the paper check if the volume below sea level is bigger than diff among config.')
print('Differences in volume below sea level ')
print(abs(vol_bsl_exp_sle[9]-vol_bsl_exp_sle[8]))
print(abs(vol_bsl_exp_sle[7]-vol_bsl_exp_sle[8]))
print(abs(vol_bsl_exp_sle[7]-vol_bsl_exp_sle[9]))
#
print('Differences in volume')
print(abs(vol_exp_sle[9]-vol_exp_sle[8]))
print(abs(vol_exp_sle[7]-vol_exp_sle[8]))
print(abs(vol_exp_sle[7]-vol_exp_sle[9]))
#
print(str(vol_exp_sle[7]) + 'increase to ' + str(vol_exp_sle[8])+ ' when using vel method')
print(str(vol_exp_sle[7]) + 'increase to ' + str(vol_exp_sle[9])+ ' when using RACMO method')
print('farinotti '+str(vol_exp_sle[6]))
#
print(str(vol_bsl_exp_sle[7]) + 'increase to ' + str(vol_bsl_exp_sle[8])+ ' when using vel method')
print(str(vol_bsl_exp_sle[7]) + 'increase to ' + str(vol_bsl_exp_sle[9])+ ' when using RACMO method')
print('farinotti '+str(vol_bsl_exp_sle[6]))
#
print('Percentage farinotti compared to RACMO and Vel')
print('to vel method '+ str(utils_vel.calculate_volume_percentage(total_vol[6],
                                                                  total_vol[8])))
print('to RACMO method '+ str(utils_vel.calculate_volume_percentage(total_vol[6],
                                                                  total_vol[9])))

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

# Example data
y_pos = np.arange(len(vol_exp))
y_pos = [0,0.5,1,2,2.5,3,4,4.5,5,5.5]

p0 = ax1.barh(y_pos, (ice_cap_vol_bsl_exp+vol_bsl_exp)*-1, align='center', color=sns.xkcd_rgb["grey"],
            height=0.5, edgecolor="white", hatch="/")

p1 = ax1.barh(y_pos, vol_bsl_exp*-1, align='center', color=sns.xkcd_rgb["grey"],
            height=0.5, edgecolor="white")

p2 = ax1.barh(y_pos, vol_exp+ice_cap_vol_exp, align='center', color=color_array, height=0.5, hatch="/")

p3 = ax1.barh(y_pos, vol_exp, align='center', color=color_array, height=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([])
# labels read top-to-bottom
ax1.invert_yaxis()
ax1.set_xlabel('Volume [km³]',fontsize=18)

ax1.set_xticks([-4000, -2000, 0, 2000, 4000, 6000])

ax2= ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
array = ax1.get_xticks()

#Get the other axis on sea level equivalent
sle = []
for value in array:
    sle.append(np.round(abs(utils_vel.calculate_sea_level_equivalent(value)),2))
print(sle)

ax2.set_xticklabels(sle, fontsize=20)
ax2.set_xlabel('Volume [mm SLE]', fontsize=18)

plt.legend((p3[0], p3[1], p3[2], p3[5]),
           ('Farinotti et al. (2019)',
            'Without $q_{calving}$',
            'With $q_{calving}$ - velocity',
            'With $q_{calving}$ - RACMO'),
            frameon=True, bbox_to_anchor=(0.8, -0.2), ncol=2)
            #bbox_to_anchor=(1.1, -0.15), ncol=5, fontsize=15)

plt.margins(0.05)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'volume_greenland_with_ice_cap.pdf'),
               bbox_inches='tight')

