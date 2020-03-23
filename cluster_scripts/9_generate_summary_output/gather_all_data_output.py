import os
import numpy as np
import glob
import pandas as pd

# Reading glacier directories per experiment
MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/')

full_exp_dir = []

for path, subdirs, files in os.walk(output_dir_path, topdown=True):
    subdirs[:] = sorted(subdirs)
    for name in subdirs:
        full_exp_dir.append(os.path.join(path, name))




dk = {'RGIId': ids,
      'k_value': k_values,
      'mu_star': mu_star,
      'u_cross': u_cross,
      'u_surf': u_surf,
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