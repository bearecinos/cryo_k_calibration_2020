# This will run OGGM preprocessing task and the inversion with calving
# For Greenland already calibrated with default DEM

from __future__ import division

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import numpy as np
import geopandas as gpd
import pandas as pd

import shutil

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils
from oggm.core import inversion

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

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

OUTPUT_DATA = os.path.join(MAIN_PATH,
                    'output_data/5_calibration_vel_results/')

data_gap = os.path.join(MAIN_PATH,
                        'output_data/2_Process_vel_data/')

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

no_solution = os.path.join(OUTPUT_DATA,
                           'glaciers_with_no_solution.csv')
d_no_sol = pd.read_csv(no_solution)
ids_rgi = d_no_sol.RGIId.values
keep_no_solution = [(i not in ids_rgi) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_solution]

no_vel_data = os.path.join(data_gap,
                           'glaciers_with_no_velocity_data.csv')
d_no_data = pd.read_csv(no_vel_data)
ids_no_data = d_no_data.RGIId.values
keep_no_data = [(i not in ids_no_data) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_no_data]

calibration_results = os.path.join(OUTPUT_DATA,
                'k_calibration_velandmu_reltol.csv')

dc = pd.read_csv(calibration_results)

# index_vel = np.where(dc['method'] == 'calibrated with mu_star')
# ids_mu_star = dc.loc[index_vel].RGIId.values
#
# keep_no_mu_star = [(i not in ids_mu_star) for i in rgidf.RGIId]
# rgidf = rgidf.iloc[keep_no_mu_star]


# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

execute_entity_task(tasks.glacier_masks, gdirs)

# Prepro tasks
task_list = [
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Climate tasks -- we make sure that calving is = 0 for all tidewater
for gdir in gdirs:
    gdir.inversion_calving_rate = 0

execute_entity_task(tasks.process_cru_data, gdirs)
execute_entity_task(tasks.local_t_star, gdirs)
execute_entity_task(tasks.mu_star_calibration, gdirs)

# Inversion tasks
execute_entity_task(tasks.prepare_for_inversion, gdirs, add_debug_var=True)
execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                    filesuffix='_without_calving_')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM is done! Time needed: %02d:%02d:%02d" % (h, m, s))

dc = pd.read_csv(calibration_results, index_col='RGIId')

# Compute a calving flux
for gdir in gdirs:
    sel = dc[dc.index == gdir.rgi_id]
    k_value = sel.k_value.values

    cfg.PARAMS['k_calving'] = float(k_value)
    out = inversion.find_inversion_calving(gdir)
    if out is None:
        continue