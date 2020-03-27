import os
import numpy as np
import glob
import pandas as pd

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

import sys
sys.path.append(MAIN_PATH)
# velocity module
from velocity_tools import utils_velocity as utils_vel

# Reading glacier directories per experiment
output_dir_path = os.path.join(MAIN_PATH, 'output_data/')

output_path = os.path.join(MAIN_PATH,
                        'output_data/9_summary_output/')

if not os.path.exists(output_path):
    os.makedirs(output_path)

## Reading the directories that we need
full_exp_dir = []

exclude = {'10_plot', '4_k_exp_for_calibration',
           '9_summary_output'}

for path, subdirs, files in os.walk(output_dir_path, topdown=True):
    subdirs[:] = [d for d in subdirs if d not in exclude]
    subdirs[:] = [d for d in subdirs if "rest" not in d]
    subdirs[:] = sorted(subdirs)

    for name in subdirs:
        full_exp_dir.append(os.path.join(path, name))


#print(full_exp_dir)


# Data Errors and gaps
prepro_erros = os.path.join(full_exp_dir[0],
                            'glaciers_with_prepro_errors.csv')
no_vel_data = os.path.join(full_exp_dir[3],
                           'glaciers_with_no_vel_data.csv')
no_racmo_data = os.path.join(full_exp_dir[4],
                             'glaciers_with_no_racmo_data.csv')
no_solution =  os.path.join(full_exp_dir[4],
                            'glaciers_with_no_solution.csv')

# # Results k values
k_value_vel = os.path.join(full_exp_dir[3],
                           'k_calibration_velandmu_reltol.csv')

k_value_racmo = os.path.join(full_exp_dir[4],
                'k_calibration_racmo_reltol_q_calving_RACMO_meanNone_.csv')

dk_vel = pd.read_csv(k_value_vel)

# dk_vel_rest = dk_vel.loc[dk_vel.method != 'calibrated with velocities']
# dk_vel_rest.to_csv(os.path.join(output_path,
#                                     'glacier_stats_no_matching_vel.csv'))
#
# # K results filtered
# dk_vel = dk_vel.loc[dk_vel.method == 'calibrated with velocities']

dk_racmo = pd.read_csv(k_value_racmo)

# Get correct id's!
prepro_ids = utils_vel.read_rgi_ids_from_csv(prepro_erros)
no_vel_ids = utils_vel.read_rgi_ids_from_csv(no_vel_data)
no_racmo_ids = utils_vel.read_rgi_ids_from_csv(no_racmo_data)
no_sol_ids = utils_vel.read_rgi_ids_from_csv(no_solution)

## Gather statistics on glaciers that dont calve make a single dataframe
## for them
glac_stats_no_calving = pd.read_csv(os.path.join(full_exp_dir[0],
                'glacier_statistics_greenland_no_calving_with_sliding_.csv'))

keep_glaciers = [(i in no_sol_ids) for i in glac_stats_no_calving.rgi_id]

df_no_sol_stats = glac_stats_no_calving.iloc[keep_glaciers]

vel_obs = pd.read_csv(os.path.join(full_exp_dir[1],
                                   'velocity_observations.csv'))

racmo_obs = pd.read_csv(os.path.join(full_exp_dir[2],
                                     '1960_1990/racmo_data_19601960_.csv'))

print(len(df_no_sol_stats), len(no_sol_ids))



vel_obs.rename(columns={'RGI_ID': 'rgi_id'}, inplace=True)
racmo_obs.rename(columns={'RGI_ID': 'rgi_id'}, inplace=True)

df_no_sol = pd.merge(left=df_no_sol_stats,
                     right=vel_obs,
                     how='left',
                     left_on='rgi_id',
                     right_on='rgi_id')

df_no_sol = df_no_sol.drop(['Unnamed: 0'], axis=1)

df_no_sol = pd.merge(left=df_no_sol,
                    right=racmo_obs,
                    how='left',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

df_no_sol = df_no_sol.drop(['Unnamed: 0'], axis=1)

df_no_sol.to_csv(os.path.join(output_path,
                              'glacier_stats_no_solution.csv'))

## Gather glacier statistics for glaciers that do calve per method

glac_stats_vel = pd.read_csv(os.path.join(full_exp_dir[5],
    'glacier_statistics_greenland_calving_with_sliding_k_vel_calibrated.csv'))

glac_stats_racmo =  pd.read_csv(os.path.join(full_exp_dir[6],
'glacier_statistics_greenland_calving_with_sliding_k_racmo_calibrated.csv'))

print(len(glac_stats_vel), len(glac_stats_racmo))
print(len(dk_vel), len(dk_racmo))


dk_vel.rename(columns={'RGIId': 'rgi_id'}, inplace=True)
dk_racmo.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

glac_stats_vel_plus = pd.merge(left=glac_stats_vel, right=dk_vel,
                               how='left', left_on = 'rgi_id',
                               right_on='rgi_id')

glac_stats_racmo_plus = pd.merge(left=glac_stats_racmo, right=dk_racmo,
                                 how='left', left_on = 'rgi_id',
                                 right_on='rgi_id')

print(len(glac_stats_vel_plus), len(glac_stats_racmo_plus))


glac_stats_vel_plus = glac_stats_vel_plus.drop(['Unnamed: 0'], axis=1)
glac_stats_racmo_plus = glac_stats_racmo_plus.drop(['Unnamed: 0'], axis=1)

glac_stats_vel_plus.to_csv(os.path.join(output_path,
                                        'glacier_stats_vel_method.csv'))

glac_stats_racmo_plus.to_csv(os.path.join(output_path,
                                          'glacier_stats_racmo_method.csv'))


## Gather common glaciers data set for comparision between methods

glac_stats_racmo_plus.rename(columns={'inv_volume_km3': 'inv_volume_km3_MR',
                                      'inv_thickness_m': 'inv_thickness_m_MR',
                                      'vas_volume_km3': 'vas_volume_km3_MR',
                                      'vas_thickness_m': 'vas_thickness_m_MR',
                                      'calving_flux_x': 'calving_flux_MR',
                                      'calving_mu_star': 'calving_mu_star_MR',
                                      'calving_law_flux': 'calving_law_flux_MR',
                                      'calving_thick': 'calving_thick_MR',
                                      'calving_water_depth': 'calving_water_depth_MR',
                                      'k_value': 'k_value_MR',
                                      'u_cross': 'u_cross_MR',
                                      'u_surf': 'u_surf_MR'}, inplace=True)

glac_stats_racmo_plus_filtered = glac_stats_racmo_plus[['rgi_id',
                                                        'rgi_region',
                                                        'rgi_subregion',
                                                        'name',
                                                        'cenlon', 'cenlat',
                                                        'rgi_area_km2',
                                                        'glacier_type',
                                                        'terminus_type',
                                                        'status',
                                                        'inv_volume_km3_MR',
                                                        'inv_thickness_m_MR',
                                                        'vas_volume_km3_MR',
                                                        'vas_thickness_m_MR',
                                                        'calving_flux_MR',
                                                        'calving_mu_star_MR',
                                                        'calving_law_flux_MR',
                                                        'calving_slope',
                                                        'calving_thick_MR',
                                                        'calving_water_depth_MR',
                                                        'calving_free_board',
                                                        'calving_front_width',
                                                        'dem_mean_elev',
                                                        'dem_min_elev',
                                                        'dem_min_elev_on_ext',
                                                        'flowline_mean_elev',
                                                        'flowline_min_elev',
                                                        'flowline_avg_width',
                                                        'flowline_avg_slope',
                                                        't_star',
                                                        'k_value_MR',
                                                        'u_cross_MR',
                                                        'u_surf_MR',
                                                        'racmo_flux']]

glac_stats_vel_plus.rename(columns={'inv_volume_km3': 'inv_volume_km3_MV',
                                      'inv_thickness_m': 'inv_thickness_m_MV',
                                      'vas_volume_km3': 'vas_volume_km3_MV',
                                      'vas_thickness_m': 'vas_thickness_m_MV',
                                      'calving_flux': 'calving_flux_MV',
                                      'calving_mu_star': 'calving_mu_star_MV',
                                      'calving_law_flux': 'calving_law_flux_MV',
                                      'calving_thick': 'calving_thick_MV',
                                      'calving_water_depth': 'calving_water_depth_MV',
                                      'k_value': 'k_value_MV',
                                      'u_cross': 'u_cross_MV',
                                      'u_surf': 'u_surf_MV'}, inplace=True)

glac_stats_vel_plus_filtered = glac_stats_vel_plus[['rgi_id',
                                                    'inv_volume_km3_MV',
                                                    'inv_thickness_m_MV',
                                                    'vas_volume_km3_MV',
                                                    'vas_thickness_m_MV',
                                                    'calving_flux_MV',
                                                    'calving_mu_star_MV',
                                                    'calving_law_flux_MV',
                                                    'calving_thick_MV',
                                                    'calving_water_depth_MV',
                                                    'k_value_MV',
                                                    'u_cross_MV',
                                                    'u_surf_MV',
                                                    'u_obs',
                                                    'rtol',
                                                    'No of k']]

df_common = pd.merge(left=glac_stats_racmo_plus_filtered,
                    right=glac_stats_vel_plus_filtered,
                    how='inner',
                    left_on = 'rgi_id',
                    right_on='rgi_id')

df_common.to_csv(os.path.join(output_path,
                                    'glacier_stats_both_methods.csv'))