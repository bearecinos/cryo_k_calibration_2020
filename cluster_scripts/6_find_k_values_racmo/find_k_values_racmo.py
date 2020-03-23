import os
import numpy as np
import glob
import pandas as pd
import math
from oggm import cfg

# Reading glacier directories per experiment
MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')
import sys
sys.path.append(MAIN_PATH)

# velocity module
from velocity_tools import utils_velocity as utils_vel

# Read RACMO data path
racmo_output = os.path.join(MAIN_PATH,
        'output_data/3_Process_RACMO_data/1960_1990/racmo_data_19601960_.csv')
df_racmo = pd.read_csv(racmo_output)

# Reading OGGM calibration results
output_k_exp_path = os.path.join(MAIN_PATH,
                                'output_data/4_k_exp_for_calibration/')

# Read output from k experiment
WORKING_DIR = os.path.join(output_k_exp_path, '*.csv')

# Writing output path
output_path = os.path.join(MAIN_PATH,
                        'output_data/6_racmo_calibration_results/')
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Sort files
filenames = sorted(glob.glob(WORKING_DIR))

files_no_data = []
files_no_racmo = []

ids = []
k_values = []
mu_star = []
u_cross = []
u_surf = []
q_calving = []
racmo_calving = []
message = []

for j, f in enumerate(filenames):
    glacier = pd.read_csv(f)
    glacier = glacier.drop_duplicates(subset=('calving_flux'), keep=False)
    if glacier.empty:
        base = os.path.basename(f)
        name = os.path.splitext(base)[0]
        files_no_data = np.append(files_no_data, name)
    else:
        glacier = pd.read_csv(f)
        base = os.path.basename(f)
        rgi_id = os.path.splitext(base)[0]

        #Get observations for that glacier
        index = df_racmo.index[df_racmo['RGI_ID'] == rgi_id].tolist()

        if len(index) == 0:
            print('There is no Racmo data for this glacier'+ rgi_id)
            files_no_racmo = np.append(files_no_racmo, rgi_id)
            continue
        else:
            data_racmo = df_racmo.iloc[index]

            tol_r = None

            var_name = 'q_calving_RACMO_mean'
            out = utils_vel.k_calibration_with_racmo(glacier, data_racmo,
                                                     var_name, rtol=tol_r)

            #print(rgi_id)
            ids = np.append(ids, rgi_id)
            k_values = np.append(k_values, out[0])
            mu_star = np.append(mu_star, out[1])
            u_cross = np.append(u_cross, out[2])
            u_surf = np.append(u_surf, out[3])
            q_calving = np.append(q_calving, out[4])
            racmo_calving = np.append(racmo_calving, out[5])

            if tol_r is None:
                message = np.append(message,
                                'calibrated with Racmo, neareast values fun')
            else:
                message = np.append(message,
                                'calibrated with Racmo, reltol_'+str(tol_r))

# #
d = {'RGIId': files_no_data}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(output_path,
                       'glaciers_with_no_solution'+'.csv'))

ds = {'RGIId': files_no_racmo}
df1 = pd.DataFrame(data=ds)
df1.to_csv(os.path.join(output_path,
                       'glaciers_with_no_racmo_data'+'.csv'))

dk = {'RGIId': ids,
      'k_value': k_values,
      'mu_star': mu_star,
      'u_cross': u_cross,
      'u_surf': u_surf,
      'calving_flux': q_calving,
      'racmo_flux': racmo_calving,
      'method': message}

df2 = pd.DataFrame(data=dk)
df2.to_csv(os.path.join(output_path,
              'k_calibration_racmo_reltol_'+var_name+str(tol_r)+'_.csv'))

