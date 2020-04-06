# This will run OGGM and obtain surface mass balance data from RACMO
# to calibrate the k parameter in Greenland

from __future__ import division

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import numpy as np
import geopandas as gpd
import pandas as pd

print(np.__version__)

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils
from oggm.core import inversion

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

import sys
sys.path.append(MAIN_PATH)
# velocity module
from velocity_tools import utils_velocity as utils_vel

# Time
import time
start = time.time()

# Regions:
# Greenland
rgi_region = '05'

# Initialize OGGM and set up the run parameters
# ---------------------------------------------

cfg.initialize()
rgi_version = '61'

SLURM_WORKDIR = os.environ["WORKDIR"]

# Local paths (where to write output and where to download input)
WORKING_DIR = SLURM_WORKDIR
cfg.PATHS['working_dir'] = WORKING_DIR

RGI_FILE = os.path.join(MAIN_PATH,
'input_data/05_rgi61_GreenlandPeriphery_bea/05_rgi61_GreenlandPeriphery.shp')

# RACMO data
racmo_path = os.path.join(MAIN_PATH, 'input_data/RACMO/')

# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True

# We make the border 20 so we can use the Columbia itmix DEM
cfg.PARAMS['border'] = 20
# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['min_mu_star'] = 0.0
cfg.PARAMS['inversion_fs'] = 5.7e-20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False

# We use intersects
path = utils.get_rgi_intersects_region_file(rgi_region, version=rgi_version)
cfg.set_intersects_db(path)

# RGI file
rgidf = gpd.read_file(RGI_FILE)

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
p = utils.get_cru_file(var='pre')
print('CRU file: ' + p)

# Run only for Lake Terminating and Marine Terminating
glac_type = [0]
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

# Run only glaciers that have a week connection or are
# not connected to the ice-sheet
connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

# Run glaciers without errors
error_file_path = os.path.join(MAIN_PATH,
'output_data/1_Greenland_prepo/glaciers_with_prepro_errors.csv')
de = pd.read_csv(error_file_path)
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

#print(len(rgidf))
# Run a single id for testing
# glacier = ['RGI60-05.00304', 'RGI60-05.08443']
# keep_indexes = [(i in glacier) for i in rgidf.RGIId]
# rgidf = rgidf.iloc[keep_indexes]

# one = True
# if one:
#     #Run glaciers in two groups
#     # Group 1: small glaciers
#     glac_sel = rgidf.loc[(rgidf.Area < 5.0)]
#     ids_sel = glac_sel.RGIId.values
#     keep_glac = [(i in ids_sel) for i in rgidf.RGIId]
#     rgidf = rgidf.iloc[keep_glac]
# else:
#     #Group 2: big glaciers
#     glac_sel = rgidf.loc[(rgidf.Area < 5.0)]
#     ids_sel = glac_sel.RGIId.values
#     keep_glac = [(i not in ids_sel) for i in rgidf.RGIId]
#     rgidf = rgidf.iloc[keep_glac]
#
# print(len(rgidf))

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

execute_entity_task(tasks.glacier_masks, gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM dirs finished! Time needed: %02d:%02d:%02d" %
         (h, m, s))

ids = []
smb_avg = []
smb_cum = []
racmo_calving_avg = []
racmo_calving_cum = []

files_no_data = []

for gdir in gdirs:

    utils_vel.process_racmo_data(gdir, racmo_path,
                                 time_start=1961,
                                 time_end=1990)

    # We compute a calving flux from RACMO data
    out = utils_vel.get_smb31_from_glacier(gdir)

    if out[0] is None:
        print('There is no RACMO data for this glacier')
        files_no_data = np.append(files_no_data, gdir.rgi_id)
    else:
        # We append everything
        ids = np.append(ids, gdir.rgi_id)
        smb_avg = np.append(smb_avg, out[0])
        smb_cum = np.append(smb_cum, out[1])
        racmo_calving_avg = np.append(racmo_calving_avg, out[2])
        racmo_calving_cum = np.append(racmo_calving_cum, out[3])

d = {'RGIId': files_no_data}
df = pd.DataFrame(data=d)

df.to_csv(cfg.PATHS['working_dir'] + 'glaciers_with_no_racmo_data.csv')

dr = {'RGI_ID': ids,
      'smb_mean': smb_avg,
      'smb_cum': smb_cum,
      'q_calving_RACMO_mean': racmo_calving_avg,
      'q_calving_RACMO_cum': racmo_calving_cum}

df_r = pd.DataFrame(data=dr)
df_r.to_csv(cfg.PATHS['working_dir']+'racmo_data_'+str(1961)+str(1990)+'_.csv')
