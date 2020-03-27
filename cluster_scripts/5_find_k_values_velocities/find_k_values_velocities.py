import os
import numpy as np
import glob
import pandas as pd

# Reading glacier directories per experiment
MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')
output_k_exp_path = os.path.join(MAIN_PATH,
                                'output_data/4_k_exp_for_calibration/')

import sys
sys.path.append(MAIN_PATH)
# velocity module
from velocity_tools import utils_velocity as utils_vel

# Read output from k experiment
WORKING_DIR = os.path.join(output_k_exp_path, '*.csv')

# Read  marcos data# Read  marcos data
obs_path = os.path.join(MAIN_PATH,
                   'output_data/2_Process_vel_data/velocity_observations.csv')

d_obs = pd.read_csv(obs_path)

output_path = os.path.join(MAIN_PATH,
                        'output_data/5_calibration_vel_results/')


if not os.path.exists(output_path):
    os.makedirs(output_path)

#Sort files
filenames = sorted(glob.glob(WORKING_DIR))

files_no_calving = []
files_no_vel_data = []

ids = []
k_values = []
mu_star = []
u_cross = []
u_surf = []
message = []
rtol = []
no_k_values = []
u_obs = []

for j, f in enumerate(filenames):
    glacier = pd.read_csv(f)
    glacier = glacier.drop_duplicates(subset=('calving_flux'), keep=False)
    if glacier.empty:
        base = os.path.basename(f)
        name = os.path.splitext(base)[0]
        files_no_calving = np.append(files_no_calving, name)
    else:
        glacier = pd.read_csv(f)
        base = os.path.basename(f)
        rgi_id = os.path.splitext(base)[0]

        # Get observations for that glacier
        index = d_obs.index[d_obs['RGI_ID'] == rgi_id].tolist()

        if len(index) == 0:
            print('There is no Velocity data for this glacier'+ rgi_id)
            files_no_vel_data = np.append(files_no_vel_data, rgi_id)
            continue
        else:
            data_obs = d_obs.iloc[index]

            ## TODO : fix this according to the output that we have!!!
            #print(rgi_id)
            out = utils_vel.k_calibration_with_observations(glacier, data_obs)

            if out[0] is None:
                #print(rgi_id)
                out_2 = utils_vel.k_calibration_with_mu_star(glacier, data_obs)
                ids = np.append(ids, rgi_id)
                k_values = np.append(k_values, out_2[0])
                mu_star = np.append(mu_star, out_2[1])
                u_cross = np.append(u_cross, out_2[2])
                u_surf = np.append(u_surf, out_2[3])
                u_obs = np.append(u_obs, out_2[6])
                rtol = np.append(rtol, out_2[4])
                message = np.append(message, out_2[5])
                no_k_values = np.append(no_k_values, 1)
            else:
                #print(rgi_id)
                ids = np.append(ids, rgi_id)
                k_values = np.append(k_values, out[0])
                mu_star = np.append(mu_star, out[1])
                u_cross = np.append(u_cross, out[2])
                u_surf = np.append(u_surf, out[3])
                u_obs = np.append(u_obs, out[6])
                message = np.append(message, 'calibrated with velocities')
                rtol = np.append(rtol, out[4])
                no_k_values = np.append(no_k_values, out[5])

d = {'RGIId': files_no_calving}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(output_path,
                       'glaciers_with_no_solution.csv'))

s = {'RGIId': files_no_vel_data}
ds = pd.DataFrame(data=s)
ds.to_csv(os.path.join(output_path,
                       'glaciers_with_no_vel_data.csv'))

dk = {'RGIId': ids,
      'k_value': k_values,
      'mu_star': mu_star,
      'u_cross': u_cross,
      'u_surf': u_surf,
      'u_obs': u_obs,
      'method': message,
      'rtol': rtol,
      'No of k': no_k_values}

df2 = pd.DataFrame(data=dk)
df2.to_csv(os.path.join(output_path,
              'k_calibration_velandmu_reltol.csv'))


# ds = {'RGIId': ids_x,
#       'k_value': k_values_x,
#       'mu_star': mu_star_x,
#       'u_cross': u_cross_x,
#       'u_surf': u_surf_x}
#
# df3 = pd.DataFrame(data=ds)
# df3.to_csv(os.path.join(output_path,
#               'miss_matching_glaciers_'+str(tol_r)+'.csv'))