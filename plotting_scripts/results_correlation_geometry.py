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
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
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

corr_matrix = df.corr(method='pearson')
corr_matrix.to_csv(os.path.join(plot_path,
                                'correlation_matrix_all.csv'))
exit()
df_vel = df[['k_value_MV',
             'calving_slope',
             'calving_free_board',
             'calving_front_width',
             'total_prcp_top'
             ]].copy()

df_vel.rename(columns={'k_value_MV': 'k_value'}, inplace=True)
df_vel['Method'] = np.repeat('Velocity', len(df_vel.k_value))

df_racmo = df[['k_value_MR',
               'calving_slope',
               'calving_free_board',
               'calving_front_width',
               'total_prcp_top'
             ]].copy()

df_racmo.rename(columns={'k_value_MR': 'k_value'}, inplace=True)
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

color_palette = sns.color_palette("muted")

# Plot Fig 1
fig1 = plt.figure(figsize=(16, 14), constrained_layout=True)

spec = gridspec.GridSpec(2, 2)

ax0 = plt.subplot(spec[0])
corr, p = stats.pearsonr(data_all.calving_slope, data_all.k_value)
sns.scatterplot(x='calving_slope', y='k_value', data=data_all, hue='Method',
                ax=ax0)
ax0.set_xlabel('calving front slope angle \n [$^{o}$]')
ax0.set_ylabel('$k$ \n [yr$^{-1}$]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
                    decimals=3)) + '\np-value = ' + str(format(p, ".3E")),
                    prop=dict(size=16), frameon=True, loc=6,
                    bbox_transform=ax0.transAxes)
test.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

ax0.add_artist(at)
ax0.add_artist(test)
ax0.legend()
ax0.get_legend().remove()

ax1 = plt.subplot(spec[1])
corr, p = stats.pearsonr(data_all.calving_free_board, data_all.k_value)
sns.scatterplot(x='calving_free_board', y='k_value', data=data_all, hue='Method',
                ax=ax1)
ax1.set_xlabel('freeboard \n [m]')
ax1.set_ylabel('$k$ \n [yr$^{-1}$]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
                    decimals=5)) + '\np-value = ' + str(format(p, ".3E")),
                    prop=dict(size=16), frameon=True, loc=1,
                    bbox_transform=ax1.transAxes)
test.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.add_artist(test)
ax1.get_legend().remove()

ax2 = plt.subplot(spec[2])
corr, p = stats.pearsonr(data_all.calving_front_width, data_all.k_value)
sns.scatterplot(x='calving_front_width', y='k_value', data=data_all, hue='Method',
                ax=ax2)
ax2.set_xlabel('calving front width \n [km]')
ax2.set_ylabel('$k$ \n [yr$^{-1}$]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
                    decimals=3)) + '\np-value = ' + str(format(p, ".3E")),
                    prop=dict(size=16), frameon=True, loc=1,
                    bbox_transform=ax2.transAxes)
test.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)
ax2.add_artist(test)
ax2.get_legend().remove()

ax3 = plt.subplot(spec[3])
corr, p = stats.pearsonr(data_all.total_prcp_top, data_all.k_value)
sns.scatterplot(x='total_prcp_top', y='k_value', data=data_all, hue='Method',
                ax=ax3)
ax3.set_xlabel('Avg. total solid prcp \n [mm/yr]')
ax3.set_ylabel('$k$ \n [yr$^{-1}$]')
ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc=2)
test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
                    decimals=5)) + '\np-value = ' + str(format(p, ".3E")),
                    prop=dict(size=16), frameon=True, loc=1,
                    bbox_transform=ax3.transAxes)
test.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

ax3.add_artist(at)
ax3.add_artist(test)
ax3.get_legend().remove()

handles, labels = ax0.get_legend_handles_labels()
fig1.legend(handles, labels, loc='center', ncol=3, fontsize=18,
            bbox_to_anchor= (0.5, 0.99),
            fancybox=False, framealpha=1, shadow=True, borderpad=1)

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_path, 'correlation_plot.jpg'),
            bbox_inches='tight')
plt.savefig(os.path.join(plot_path, 'correlation_plot.pdf'),
            bbox_inches='tight')

# # Plot Fig 1
# fig2 = plt.figure(figsize=(14, 12), constrained_layout=True)
#
# spec = gridspec.GridSpec(2, 2)
#
# ax0_0 = plt.subplot(spec[0])
# corr, p = stats.pearsonr(data_all.calving_slope, data_all.k_diff)
# sns.scatterplot(x='calving_slope', y='k_diff', data=data_all,
#                 ax=ax0_0)
# ax0_0.set_xlabel('calving front slope angle \n [$^{o}$]')
# ax0_0.set_ylabel('$k$ differences \n [yr$^{-1}$]')
# at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
# test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
#                     decimals=3)) + '\np-value = ' + str(format(p, ".3E")),
#                     prop=dict(size=16), frameon=False, loc=6)
# ax0_0.add_artist(at)
# ax0_0.add_artist(test)
#
#
# ax0_1 = plt.subplot(spec[1])
# corr, p = stats.pearsonr(data_all.calving_free_board, data_all.k_diff)
# sns.scatterplot(x='calving_free_board', y='k_diff', data=data_all,
#                 ax=ax0_1)
# ax0_1.set_xlabel('freeboard \n [m]')
# ax0_1.set_ylabel('$k$ differences \n [yr$^{-1}$]')
# at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
# test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
#                     decimals=3)) + '\np-value = ' + str(format(p, ".3E")),
#                     prop=dict(size=16), frameon=False, loc=1)
# ax0_1.add_artist(at)
# ax0_1.add_artist(test)
#
#
#
# ax1_0 = plt.subplot(spec[2])
# corr, p = stats.pearsonr(data_all.calving_front_width, data_all.k_diff)
# sns.scatterplot(x='calving_front_width', y='k_diff', data=data_all,
#                 ax=ax1_0)
# ax1_0.set_xlabel('calving front width \n [km]')
# ax1_0.set_ylabel('$k$ differences \n [yr$^{-1}$]')
# at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
# test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
#                     decimals=3)) + '\np-value = ' + str(format(p, ".3E")),
#                     prop=dict(size=16), frameon=False, loc=7)
# ax1_0.add_artist(at)
# ax1_0.add_artist(test)
#
#
# ax1_1 = plt.subplot(spec[3])
# corr, p = stats.pearsonr(data_all.total_prcp_top, data_all.k_diff)
# sns.scatterplot(x='total_prcp_top', y='k_diff', data=data_all,
#                 ax=ax1_1)
# ax1_1.set_xlabel('total solid prcp \n [mm/yr]')
# ax1_1.set_ylabel('$k$ differences \n [yr$^{-1}$]')
# ax1_1.xaxis.set_major_locator(plt.MaxNLocator(3))
# at = AnchoredText('d', prop=dict(size=18), frameon=True, loc=2)
# test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
#                     decimals=3)) + '\np-value = ' + str(format(p, ".3E")),
#                     prop=dict(size=16), frameon=False, loc=9)
# ax1_1.add_artist(at)
# ax1_1.add_artist(test)
#
# plt.tight_layout()
# plt.savefig(os.path.join(plot_path, 'correlation_plot_diff.pdf'),
#             bbox_inches='tight')