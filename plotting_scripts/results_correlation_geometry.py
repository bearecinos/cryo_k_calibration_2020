import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import optimize
os.getcwd()

def func(x, a, b):
    return a*np.exp(b*x)

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
df = df.loc[df.k_value_MR != 0]

# corr_matrix = df.corr(method='pearson')
# corr_matrix.to_csv(os.path.join(plot_path,
#                                 'correlation_matrix_all.csv'))
# exit()
df_vel = df[['k_value_MV',
             'calving_slope',
             'calving_free_board',
             'calving_front_width',
             'total_prcp_top',
             'calving_mu_star_MV'
             ]].copy()

## Correlation test's
corr_vel_k_s, p_vel_k_s = stats.pearsonr(df_vel.k_value_MV,
                                 df_vel.calving_slope)
corr_vel_k_f, p_vel_k_f = stats.pearsonr(df_vel.k_value_MV,
                                 df_vel.calving_free_board)
corr_vel_k_m, p_vel_k_m = stats.pearsonr(df_vel.k_value_MV,
                                 df_vel.calving_mu_star_MV)
corr_vel_k_p, p_vel_k_p = stats.pearsonr(df_vel.k_value_MV,
                                 df_vel.total_prcp_top)

### poly fit for slope
a_v, b_v, c_v = np.polyfit(df_vel.calving_slope, df_vel.k_value_MV, 2)
x_v = np.linspace(0, max(df_vel.calving_slope), 100)
# y = ax^2 + bx + c
y_v = a_v*(x_v**2) + b_v*x_v + c_v

eq_v = '\ny = ' + str(np.around(a_v, decimals=2))+'x^2 '+ str(np.around(b_v,
                            decimals=2))+'x +'+ str(np.around(c_v, decimals=2))

## exponential fit for slope
initial_guess = [0.1, 1]
popt_v, pcov_v = optimize.curve_fit(func, df_vel.calving_slope, df_vel.k_value_MV, initial_guess)
x_fit_v = np.linspace(0, max(df_vel.calving_slope), 100)

eq_exp_v = '\ny = ' + str(np.around(popt_v[0], decimals=2))+'e$^{'+ str(np.around(popt_v[1],
                            decimals=2))+'x}$'

df_vel.rename(columns={'k_value_MV': 'k_value'}, inplace=True)
df_vel.rename(columns={'calving_mu_star_MV': 'calving_mu_star'}, inplace=True)
df_vel['Method'] = np.repeat('Velocity', len(df_vel.k_value))

df_racmo = df[['k_value_MR',
               'calving_slope',
               'calving_free_board',
               'calving_front_width',
               'total_prcp_top',
               'calving_mu_star_MR'
             ]].copy()

corr_racmo_k_s, p_racmo_k_s = stats.pearsonr(df_racmo.k_value_MR,
                                 df_racmo.calving_slope)
corr_racmo_k_f, p_racmo_k_f = stats.pearsonr(df_racmo.k_value_MR,
                                 df_racmo.calving_free_board)
corr_racmo_k_m, p_racmo_k_m = stats.pearsonr(df_racmo.k_value_MR,
                                 df_racmo.calving_mu_star_MR)
corr_racmo_k_p, p_racmo_k_p = stats.pearsonr(df_racmo.k_value_MR,
                                 df_racmo.total_prcp_top)

### poly fit for slope
a_r, b_r, c_r = np.polyfit(df_racmo.calving_slope, df_racmo.k_value_MR, 2)
x_r = np.linspace(0, max(df_racmo.calving_slope), 100)
# y = ax^2 + bx + c
y_r = a_r*(x_r**2) + b_r*x_r + c_r

eq_r = 'y = ' + str(np.around(a_r, decimals=2))+'x^2 '+ str(np.around(b_r,
                            decimals=2))+'x +'+ str(np.around(c_r, decimals=2))

## exponential fit for slope
popt_r, pcov_r = optimize.curve_fit(func, df_racmo.calving_slope, df_racmo.k_value_MR, initial_guess)
x_fit_r = np.linspace(0, max(df_racmo.calving_slope), 100)

eq_exp_r = 'y = ' + str(np.around(popt_r[0], decimals=2))+'e$^{'+ str(np.around(popt_r[1],
                            decimals=2))+'x}$'

df_racmo.rename(columns={'k_value_MR': 'k_value'}, inplace=True)
df_racmo.rename(columns={'calving_mu_star_MR': 'calving_mu_star'}, inplace=True)
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
fig1 = plt.figure(figsize=(19, 5.5), constrained_layout=True)

spec = gridspec.GridSpec(1, 4)

ax0 = plt.subplot(spec[0])
sns.scatterplot(x='calving_slope', y='k_value', data=data_all, hue='Method',
                ax=ax0, alpha=0.7)
# ax0.plot(x_v, y_v) # y = ax^2 + bx + c
# ax0.plot(x_r, y_r)
ax0.plot(x_fit_v, func(x_fit_v, *popt_v))
ax0.plot(x_fit_r, func(x_fit_r, *popt_r))
ax0.set_xlabel('calving front slope angle \n [rad]')
ax0.set_ylabel('$k$ \n [yr$^{-1}$]')
ax0.set_ylim(-0.2,4.5)
at = AnchoredText('a', prop=dict(size=14), frameon=True, loc=2)
test1 = AnchoredText(eq_exp_r +
                     eq_exp_v,
                    prop=dict(size=16),
                    frameon=True, loc=9,
                    bbox_transform=ax0.transAxes)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")

ax0.add_artist(at)
ax0.add_artist(test1)
ax0.legend()
ax0.get_legend().remove()

ax1 = plt.subplot(spec[1])
sns.scatterplot(x='calving_free_board', y='k_value', data=data_all, hue='Method',
                ax=ax1, alpha=0.7)
ax1.set_xlabel('freeboard \n [m]')
ax1.set_ylabel('')
ax1.set_ylim(-0.2,4.5)
at = AnchoredText('b', prop=dict(size=14), frameon=True, loc=2)
test1 = AnchoredText('$r_{s}$ = '+ str(np.around(corr_racmo_k_f, decimals=3)) +
                     '\np-value = ' + str(format(p_racmo_k_f, ".3E")) +
                     '\n$r_{s}$ = ' + str(np.around(corr_vel_k_f, decimals=3)) +
                     '\np-value = ' + str(format(p_vel_k_f, ".3E")),
                    prop=dict(size=16),
                    frameon=True, loc=1,
                    bbox_transform=ax1.transAxes)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)
ax1.add_artist(test1)
ax1.get_legend().remove()

ax2 = plt.subplot(spec[2])
sns.scatterplot(x='calving_mu_star', y='k_value', data=data_all, hue='Method',
                ax=ax2, alpha=0.7)
ax2.set_xlabel('$\mu^{*}$ [mm $yr^{-1}K^{-1}$]')
ax2.set_ylabel('')
ax2.set_ylim(-0.2,4.5)
at = AnchoredText('c', prop=dict(size=14), frameon=True, loc=2)
test1 = AnchoredText('$r_{s}$ = '+ str(np.around(corr_racmo_k_m, decimals=3)) +
                     '\np-value = ' + str(format(p_racmo_k_m, ".3E")) +
                     '\n$r_{s}$ = ' + str(np.around(corr_vel_k_m, decimals=3)) +
                     '\np-value = ' + str(format(p_vel_k_m, ".3E")),
                    prop=dict(size=16),
                    frameon=True, loc=1,
                    bbox_transform=ax2.transAxes)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)
ax2.add_artist(test1)
ax2.get_legend().remove()

ax3 = plt.subplot(spec[3])
sns.scatterplot(x='total_prcp_top', y='k_value', data=data_all, hue='Method',
                ax=ax3, alpha=0.7)
ax3.set_xlabel('Avg. total solid prcp \n [kg m$^{-2}$ yr$^{-1}$]')
ax3.set_ylabel('')
ax3.set_ylim(-0.2,4.5)
ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
at = AnchoredText('d', prop=dict(size=14), frameon=True, loc=2)
test1 = AnchoredText('$r_{s}$ = '+ str(np.around(corr_racmo_k_p, decimals=3)) +
                     '\np-value = ' + str(format(p_racmo_k_p, ".3E")) +
                     '\n$r_{s}$ = ' + str(np.around(corr_vel_k_p, decimals=3)) +
                     '\np-value = ' + str(format(p_vel_k_p, ".3E")),
                    prop=dict(size=16),
                    frameon=True, loc=1,
                    bbox_transform=ax3.transAxes)
test1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax3.add_artist(at)
ax3.add_artist(test1)
ax3.get_legend().remove()

handles, labels = ax0.get_legend_handles_labels()
fig1.legend(handles, labels, loc='center', ncol=3, fontsize=20,
            bbox_to_anchor= (0.5, 0.99),
            fancybox=False, framealpha=1, shadow=True, borderpad=1)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(plot_path, 'correlation_plot_exp_fit.pdf'),
             bbox_inches='tight')
# plt.savefig(os.path.join(plot_path, 'correlation_plot.pdf'),
#               bbox_inches='tight')

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
# at = AnchoredText('a', prop=dict(size=20), frameon=True, loc=2)
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
# at = AnchoredText('b', prop=dict(size=20), frameon=True, loc=2)
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
# at = AnchoredText('c', prop=dict(size=20), frameon=True, loc=2)
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
# at = AnchoredText('d', prop=dict(size=20), frameon=True, loc=2)
# test = AnchoredText('$r_{s}$ = '+ str(np.around(corr,
#                     decimals=3)) + '\np-value = ' + str(format(p, ".3E")),
#                     prop=dict(size=16), frameon=False, loc=9)
# ax1_1.add_artist(at)
# ax1_1.add_artist(test)
#
# plt.tight_layout()
# plt.savefig(os.path.join(plot_path, 'correlation_plot_diff.pdf'),
#             bbox_inches='tight')