import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
os.getcwd()

MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

# PARAMS for plots
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
sns.set_context('poster')

plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')

df_both = pd.read_csv(os.path.join(output_dir_path,
                                'glacier_stats_both_methods.csv'))

df_both['diff_mu_star'] = (df_both['calving_mu_star_MV'] - df_both['calving_mu_star_MR']).abs()

df_both  = df_both.loc[df_both.diff_mu_star != 0]

df_both['diff_fluxes_MR'] = df_both['calving_flux_MR'] - df_both['calving_law_flux_MR']
df_both['diff_fluxes_MV'] = df_both['calving_flux_MV'] - df_both['calving_law_flux_MV']

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

## Select data to plot
df  = df.loc[df.k_value_MR != 0]

r_pearson_mu, p_pearson_mu = stats.kendalltau(df.calving_mu_star_MR.values,
        df.calving_mu_star_MV.values)


# corr_matrix = df.corr(method='pearson')
# corr_matrix.to_csv(os.path.join(plot_path,
#                                 'correlation_matrix_all.csv'))

df_vel = df[['k_value_MV',
             'calving_slope',
             'calving_free_board',
             'calving_front_width',
             'calving_mu_star_MV',
             'diff_fluxes_MV'
             ]].copy()

df_vel.rename(columns={'k_value_MV': 'k_value'}, inplace=True)
df_vel.rename(columns={'calving_mu_star_MV': 'mu_star'}, inplace=True)
df_vel.rename(columns={'diff_fluxes_MV': 'diff_fluxes'}, inplace=True)

df_vel['Method'] = np.repeat('Velocity', len(df_vel.k_value))

df_racmo = df[['k_value_MR',
               'calving_slope',
               'calving_free_board',
               'calving_front_width',
               'calving_mu_star_MR',
                'diff_fluxes_MR'
             ]].copy()

df_racmo.rename(columns={'k_value_MR': 'k_value'}, inplace=True)
df_racmo.rename(columns={'calving_mu_star_MR': 'mu_star'}, inplace=True)
df_racmo.rename(columns={'diff_fluxes_MR': 'diff_fluxes'}, inplace=True)

df_racmo['Method'] = np.repeat('RACMO', len(df_racmo.k_value))

df_racmo['k_diff'] = (df_vel.k_value - df_racmo.k_value).abs()
df_vel['k_diff'] = (df_vel.k_value - df_racmo.k_value).abs()

data_all = pd.concat([df_vel, df_racmo], sort=False)

data_all['calving_front_width'] = data_all.loc[:,'calving_front_width']*1e-3

import matplotlib.gridspec as gridspec

fig1 = plt.figure(figsize=(10, 12))
color_palette = sns.color_palette("muted")
spec = gridspec.GridSpec(3, 1)

ax0 = plt.subplot(spec[0])
data = data_all[['mu_star', 'Method']]
_, bins = np.histogram(data["mu_star"])
g = sns.FacetGrid(data, hue="Method")
g = g.map(sns.distplot, "mu_star", bins=bins*0.3, ax=ax0)
plt.close(2)
ax0.axvline(linewidth=2, color='k',  linestyle='--')
ax0.set_xlabel('$\mu^{*}$ [mm $yr^{-1}K^{-1}$]')
ax0.set_ylabel('Density')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
corr, p = stats.pearsonr(data_all.mu_star, data_all.k_value)
sns.scatterplot(x='k_value', y='mu_star', data=data_all, hue='Method',
                ax=ax1, alpha=0.5)
ax1.set_xlabel('$k$ [yr$^{-1}$]')
ax1.set_ylabel('$\mu^{*}$ [mm $yr^{-1}K^{-1}$]')
ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
                    decimals=5)) + '\np-value = ' + str(format(p, ".3E")),
                    prop=dict(size=14), frameon=True, loc=1,
                    bbox_transform=ax1.transAxes)
test.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.add_artist(test)
ax1.legend(loc=4)

ax1.get_legend().remove()

handles, labels = ax1.get_legend_handles_labels()
ax0.legend(handles, labels, loc=1, fontsize=14)

## Density plot to show fabi if mu_star if being clip or not
ax2 = plt.subplot(spec[2])
data = data_all[['diff_fluxes', 'Method']]
_, bins = np.histogram(data["diff_fluxes"])
g = sns.FacetGrid(data, hue="Method")
g = g.map(sns.distplot, "diff_fluxes", bins=bins*0.3, ax=ax2)
plt.close(2)
ax2.axvline(linewidth=2, color='k',  linestyle='--')
ax2.set_xlabel('Difference $q$ - $q_{calving}$')
ax2.set_ylabel('Density')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)



plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'correlation_plot_mustar.pdf'),
              bbox_inches='tight')
