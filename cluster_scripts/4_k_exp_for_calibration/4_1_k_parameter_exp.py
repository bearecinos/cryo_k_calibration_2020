# This will run k x factors experiment only for MT
# and compute a surface velocity per glacier in
# For Greenland

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

# Read  marcos data
obs = os.path.join(MAIN_PATH,
                   'output_data/2_Process_vel_data/velocity_observations.csv')
d_obs = pd.read_csv(obs)

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

print(len(rgidf))


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
execute_entity_task(tasks.mass_conservation_inversion, gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM without calving is done! Time needed: %02d:%02d:%02d" %
         (h, m, s))

k_factors = np.arange(0.01,3.01,0.01)

for gdir in gdirs:
    cross = []
    surface = []
    flux = []
    mu_star = []
    k_used = []

    for k in k_factors:

        # Find a calving flux.
        cfg.PARAMS['k_calving'] = k
        out = inversion.find_inversion_calving(gdir)
        if out is None:
            continue

        calving_flux = out['calving_flux']
        calving_mu_star = out['calving_mu_star']

        inversion.compute_velocities(gdir)

        vel_out = utils_vel.velocity_average_main_flowline(gdir)

        vel_surface = vel_out[2]
        vel_cross = vel_out[3]

        cross = np.append(cross, vel_cross)
        surface = np.append(surface, vel_surface)
        flux = np.append(flux, calving_flux)
        mu_star = np.append(mu_star, calving_mu_star)
        k_used = np.append(k_used, k)

        if mu_star[-1] == 0:
            break

    d = {'k_values': k_used,
         'velocity_cross': cross,
         'velocity_surf': surface,
         'calving_flux': flux,
         'mu_star': mu_star}

    df = pd.DataFrame(data=d)

    df.to_csv(os.path.join(cfg.PATHS['working_dir'], gdir.rgi_id + '.csv'))