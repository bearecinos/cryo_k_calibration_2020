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

# corr_matrix = df.corr(method='pearson')
# corr_matrix.to_csv(os.path.join(plot_path,
#                                 'correlation_matrix_all.csv'))

df_vel = df[['k_value_MV',
             'calving_slope',
             'calving_free_board',
             'calving_front_width',
             'calving_mu_star_MV'
             ]].copy()

df_vel.rename(columns={'k_value_MV': 'k_value'}, inplace=True)
df_vel.rename(columns={'calving_mu_star_MV': 'mu_star'}, inplace=True)

df_vel['Method'] = np.repeat('Velocity', len(df_vel.k_value))

df_racmo = df[['k_value_MR',
               'calving_slope',
               'calving_free_board',
               'calving_front_width',
               'calving_mu_star_MR'
             ]].copy()

df_racmo.rename(columns={'k_value_MR': 'k_value'}, inplace=True)
df_racmo.rename(columns={'calving_mu_star_MR': 'mu_star'}, inplace=True)

df_racmo['Method'] = np.repeat('RACMO', len(df_racmo.k_value))

df_racmo['k_diff'] = (df_vel.k_value - df_racmo.k_value).abs()
df_vel['k_diff'] = (df_vel.k_value - df_racmo.k_value).abs()

data_all = pd.concat([df_vel, df_racmo], sort=False)

#print(data_all)


data_all['calving_front_width'] = data_all.loc[:,'calving_front_width']*1e-3



# data_all.rename(columns={'k_value': '$k$ \n [$yr^{-1}$]',
#                         'calving_mu_star': '$\mu^{*}$ \n [mm $yr^{-1}K^{-1}$]',
#                          'total_prcp_top': 'precipitation \n [mm $yr^{-1}$]',
#                          'calving_slope': 'slope angle \n [$^\circ$]',
#                          'calving_free_board': 'free board \n [m a.s.l]',
#                          'calving_front_width': 'calving front width \n [m]'},
#                 inplace=True)

#Now plotting
import matplotlib.gridspec as gridspec

fig1 = plt.figure(figsize=(12, 6))
color_palette = sns.color_palette("muted")
spec = gridspec.GridSpec(1, 2)

ax0 = plt.subplot(spec[0])
data = data_all[['mu_star', 'Method']]
_, bins = np.histogram(data["mu_star"])
g = sns.FacetGrid(data, hue="Method")
g = g.map(sns.distplot, "mu_star", bins=bins*0.3, ax=ax0)
plt.close(2)
ax0.axvline(linewidth=2, color='k',  linestyle='--')
ax0.set_xlabel('Temperature sensitivity \n $\mu^{*}$ [mm $yr^{-1}K^{-1}$]')
ax0.set_ylabel('Density')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
corr, p = stats.pearsonr(data_all.mu_star, data_all.k_value)
sns.scatterplot(x='mu_star', y='k_value', data=data_all, hue='Method',
                ax=ax1, alpha=0.5)
ax1.set_xlabel('Temperature sensitivity \n $\mu^{*}$ [mm $yr^{-1}K^{-1}$]')
ax1.set_ylabel('$k$ \n [yr$^{-1}$]')
ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
                    decimals=5)) + '\np-value = ' + str(format(p, ".3E")),
                    prop=dict(size=18), frameon=True, loc=1,
                    bbox_transform=ax1.transAxes)
test.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.add_artist(test)
ax1.legend(loc=4)

ax1.get_legend().remove()

handles, labels = ax1.get_legend_handles_labels()
ax0.legend(handles, labels, loc=1, fontsize=18)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'correlation_plot_mustar.pdf'),
             bbox_inches='tight')
