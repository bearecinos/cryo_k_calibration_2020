# This will run OGGM preprocessing task and the inversion with calving
# For Greenland with default MB calibration and DEM: Glims

from __future__ import division

# Module logger
import logging
log = logging.getLogger(__name__)

# Python imports
import os
import geopandas as gpd

# Locals
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils
from oggm.core import inversion

import numpy as np
import pandas as pd

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

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

RGI_FILE = os.path.join(MAIN_PATH,
'input_data/05_rgi61_GreenlandPeriphery_bea/05_rgi61_GreenlandPeriphery.shp')

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

# Here we select the ice cap ids that will have to be ran
# without taking into account the RGI area
#rgidf_ice_caps = rgidf[rgidf['RGIId'].str.match('RGI60-05.10315')]
#ice_cap_ids = rgidf_ice_caps.RGIId.values

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

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

# Compile output
utils.compile_glacier_statistics(gdirs,
                        filesuffix='_greenland_no_calving_with_sliding_')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM is done! Time needed: %02d:%02d:%02d" % (h, m, s))

cfg.PARAMS['continue_on_error'] = False

glac_errors = []
glac_dont_calve = []

# Compute a calving flux
for gdir in gdirs:
    try:
        out = inversion.find_inversion_calving(gdir)
    except:
        print('there was an error in calving', gdir.rgi_id)
        glac_errors = np.append(glac_errors, gdir.rgi_id)
        pass
    if out is None:
        glac_dont_calve = np.append(glac_dont_calve, gdir.rgi_id)
        pass

d = {'RGIId': glac_errors}
df = pd.DataFrame(data=d)
df.to_csv(os.path.join(WORKING_DIR, 'glaciers_with_prepro_errors'+'.csv'))

s = {'RGIId': glac_dont_calve}
ds = pd.DataFrame(data=s)
ds.to_csv(os.path.join(WORKING_DIR,
                       'glaciers_dont_calve_with_cgf_params'+'.csv'))

cfg.PARAMS['continue_on_error'] = True

# Compile output
utils.compile_glacier_statistics(gdirs,
                                 filesuffix='_greenland_calving_with_sliding')