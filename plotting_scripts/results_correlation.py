import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
os.getcwd()

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

# PARAMS for plots
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14
sns.set_context('poster')

plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')

df_both = pd.read_csv(os.path.join(output_dir_path,
                                'glacier_stats_both_methods.csv'))

## Add total precipitation over the glacier
df_climate = pd.read_csv(os.path.join(MAIN_PATH,
'output_data/11_climate_stats/glaciers_PDM_temp_at_the_freeboard_height.csv'),
                         index_col='Unnamed: 0')

df_climate.rename(columns={'RGIId': 'rgi_id'}, inplace=True)

df = pd.merge(left=df_both,
              right=df_climate,
              how='inner',
              left_on = 'rgi_id',
              right_on='rgi_id')

df_vel = df[['k_value_MV', 'calving_flux_MV','calving_mu_star_MV',
             'total_prcp_top', 'calving_slope',
             'calving_free_board', 'calving_front_width']].copy()

df_vel.rename(columns={'k_value_MV': 'k_value'}, inplace=True)
df_vel.rename(columns={'calving_flux_MV': 'calving_flux'}, inplace=True)
df_vel.rename(columns={'calving_mu_star_MV': 'calving_mu_star'}, inplace=True)

# Set NaN to zero ... must probably those glaciers had to clip the mu
df_vel['calving_mu_star'] = df_vel['calving_mu_star'].fillna(0)
df_vel['Method'] = np.repeat('Velocity', len(df_vel.k_value))


df_racmo = df[['k_value_MR', 'calving_flux_MR','calving_mu_star_MR',
               'total_prcp_top', 'calving_slope',
               'calving_free_board', 'calving_front_width']].copy()

df_racmo.rename(columns={'k_value_MR': 'k_value'}, inplace=True)
df_racmo.rename(columns={'calving_flux_MR': 'calving_flux'}, inplace=True)
df_racmo.rename(columns={'calving_mu_star_MR':
                             'calving_mu_star'}, inplace=True)

df_racmo['Method'] = np.repeat('RACMO', len(df_racmo.k_value))

df_racmo['k_diff'] = (df_vel.k_value - df_racmo.k_value).abs()
df_vel['k_diff'] = (df_vel.k_value - df_racmo.k_value).abs()

df_racmo['q_diff'] = (df_vel.calving_flux - df_racmo.calving_flux).abs()
df_vel['q_diff'] = (df_vel.calving_flux - df_racmo.calving_flux).abs()


data_all = pd.concat([df_vel, df_racmo], sort=False)

data_all.rename(columns={'k_value': '$k$ \n [$yr^{-1}$]',
                        'calving_mu_star': '$\mu^{*}$ \n [mm $yr^{-1}K^{-1}$]',
                         'total_prcp_top': 'precipitation \n [mm $yr^{-1}$]',
                         'calving_slope': 'slope angle \n [$^\circ$]',
                         'calving_free_board': 'free board \n [m a.s.l]',
                         'calving_front_width': 'calving front width \n [m]'},
                inplace=True)

g = sns.PairGrid(data_all, diag_sharey=False, height=6, aspect=1,
                 hue = 'Method', palette="muted", dropna=True)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
g.add_legend(bbox_to_anchor=(1.0,0.93))

g.fig.set_figheight(24)
g.fig.set_figwidth(24.5)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'correlation_matrix_plot.pdf'),
            bbox_inches='tight')
