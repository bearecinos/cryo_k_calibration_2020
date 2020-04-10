import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

import oggm
from oggm import cfg, utils, workflow, tasks, graphics
from oggm.core.inversion import find_inversion_calving, calving_flux_from_depth
from oggm.core.inversion import sia_thickness

# PARAMS for plots
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
sns.set_context('poster')


MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

exp_dir_path = os.path.join(MAIN_PATH, 'output_data/13_non_calving/')

plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')

df = pd.read_csv(os.path.join(output_dir_path,
                                'glacier_stats_no_solution.csv'))

rgi_ids = df['rgi_id'].values

cfg.initialize(logging_level='WORKFLOW')
cfg.PARAMS['min_mu_star'] = 0.0
cfg.PARAMS['inversion_fs'] = 5.7e-20
cfg.PATHS['working_dir'] = exp_dir_path
cfg.PARAMS['border'] = 20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False

gdirs = workflow.init_glacier_regions(rgi_ids)

from collections import OrderedDict


k_value = AnchoredText('$k$ = ' + str(0.01) + 'yr$^{-1}$',
                    prop=dict(size=18, color='r', fontweight="bold"),
                       frameon=False, loc=7)

width_cm = 12
height_cm = 7

fig1 = plt.figure(figsize=(18, 10))

color_palette = sns.color_palette("muted")

ax = plt.subplot(111)

for gdir in gdirs:
    cls = gdir.read_pickle('inversion_input')[-1]
    slope = cls['slope_angle'][-1]
    width = cls['width'][-1]


    def to_minimize(wd):
        fl = calving_flux_from_depth(gdir, water_depth=wd, k=0.01)
        oggm = sia_thickness([slope], [width],
                             np.array([fl['flux'] * 1e9 / cfg.SEC_IN_YEAR]))[0]
        return fl['thick'] - oggm


    wd = np.linspace(0.1, 400)
    out = []
    for w in wd:
        out.append(to_minimize(w))

    plt.plot(wd, out)
    ax.set_yscale('symlog')
    plt.hlines([0], 0, 400, linewidth=2);
    plt.xlabel('Water depth [m]');
    plt.ylabel('Difference Calving Law - OGGM');

ax.add_artist(k_value)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'non_calving.pdf'),
             bbox_inches='tight')