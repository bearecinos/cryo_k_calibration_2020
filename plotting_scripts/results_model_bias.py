import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np


MAIN_PATH = os.path.expanduser('~/cryo_k_calibration_2020/')

# PARAMS for plots
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

sns.set_context('poster')

plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')

df_k_vel = pd.read_csv(os.path.join(output_dir_path,
                                    'glacier_stats_vel_method.csv'))


df_k_racmo = pd.read_csv(os.path.join(output_dir_path,
                                      'glacier_stats_racmo_method.csv'))

# Calculate Bias
df_k_racmo['bias'] = df_k_racmo.calving_flux_x - df_k_racmo.racmo_flux
df_k_vel['bias'] = df_k_vel.u_surf - df_k_vel.u_obs

# Re-calculate error just for not having to call a different data set
# and re-organize glaciers we calculated multiplying the velocity observation
# times the relative tolerance caculated before . rtol = obs_error / u_surface
df_k_vel['obs_error'] = df_k_vel.u_obs*df_k_vel.rtol


fig1 = plt.figure(figsize=(12, 12))

color_palette = sns.color_palette("muted")

spec = gridspec.GridSpec(2, 2)


ax0 = plt.subplot(spec[0])
sns.distplot(df_k_vel.bias, bins=40, kde=False,
                color=color_palette[0],
                ax=ax0, label='Bias velocity method')
ax0_c = ax0.twinx()
sns.distplot(df_k_vel.bias, bins=40, kde=True,
                    hist=False, color=color_palette[0], ax=ax0_c)
ax0_c.set_yticks([])
ax0.axvline(linewidth=2, color='k',  linestyle='--')
ax0.legend(fontsize=15)
ax0.set_xlabel('Velocity [m/yr]')
at = AnchoredText('a', prop=dict(size=18), frameon=True, loc=2)
n_value = AnchoredText('N = ' + str(len(df_k_vel.bias)),
                       prop=dict(size=18), frameon=False, loc=4)
ax0.add_artist(at)
ax0.add_artist(n_value)


ax1 = plt.subplot(spec[1])
sns.distplot(df_k_racmo.bias, bins=40, kde=False,
                  color=color_palette[1],
                 ax=ax1, label='Bias RACMO method')
ax1_c = ax1.twinx()
sns.distplot(df_k_racmo.bias, bins=40, kde=True,
                    hist=False, color=color_palette[1], ax=ax1_c)
ax1_c.set_yticks([])
ax1.axvline(linewidth=2, color='k', linestyle='--')
ax1.legend(fontsize=15)
ax1.set_xlabel('Frontal ablation [$km^{3}$/yr]')
at = AnchoredText('b', prop=dict(size=18), frameon=True, loc=2)
n_value = AnchoredText('N = ' + str(len(df_k_racmo.bias)),
                       prop=dict(size=18), frameon=False, loc=4)
ax1.add_artist(at)
ax1.add_artist(n_value)

ax2 = plt.subplot(spec[2])
bins_one=np.histogram(np.hstack((df_k_racmo.k_value, df_k_vel.k_value)),
                  bins=20)[1]

sns.distplot(df_k_racmo.k_value, bins=bins_one, kde=False,
                  color=color_palette[1], ax=ax2,
                  label='$k$ - RACMO, N = ' + str(len(df_k_racmo.k_value)))

ax2_c = ax2.twinx()
sns.distplot(df_k_racmo.k_value, bins=bins_one, kde=True, hist=False,
             color=color_palette[1],
             ax=ax2_c)
ax2_c.set_yticks([])

sns.distplot(df_k_vel.k_value, bins=bins_one, kde=False,
                  color=color_palette[0], ax=ax2,
                  label='$k$ - velocity, N = ' + str(len(df_k_vel.k_value)))
ax2_c_c = ax2.twinx()
sns.distplot(df_k_vel.k_value, bins=bins_one, kde=True,
             hist=False, color=color_palette[0], ax=ax2_c_c)
ax2_c_c.set_yticks([])
ax2.axvline(linewidth=2, color='k', linestyle='--')
ax2.legend(fontsize=15)
ax2.set_xlabel('$k$ [$yr^{-1}$]')
at = AnchoredText('c', prop=dict(size=18), frameon=True, loc=2)
ax2.add_artist(at)

ax3 = plt.subplot(spec[3])
bins_two=np.histogram(np.hstack((df_k_racmo.calving_flux_x,
                                 df_k_vel.calving_flux)),
                  bins=20)[1]

sns.distplot(df_k_racmo.calving_flux_x, bins=bins_two, kde=False,
                  color=color_palette[1], ax=ax3,
                  label='$q_{calving}$ - RACMO, N = ' + str(len(df_k_racmo.calving_flux_x)))

ax3_c = ax3.twinx()

sns.distplot(df_k_racmo.calving_flux_x, bins=bins_two, kde=True, hist=False,
             color=color_palette[1],
             ax=ax3_c)
ax3_c.set_yticks([])

sns.distplot(df_k_vel.calving_flux, bins=bins_two, kde=False,
                  color=color_palette[0], ax=ax3,
                  label='$q_{calving}$ - velocity, N = ' + str(len(df_k_vel.calving_flux)))

ax3_c_c = ax3.twinx()

sns.distplot(df_k_vel.calving_flux, bins=bins_two, kde=True, hist=False,
             color=color_palette[0],
             ax=ax3_c_c)
ax3_c_c.set_yticks([])

ax3.axvline(linewidth=2, color='k', linestyle='--')
ax3.legend(fontsize=15)
ax3.set_xlabel('$q_{calving}$ [$km^{3}$/yr]')
at = AnchoredText('d', prop=dict(size=18), frameon=True, loc=2)
ax3.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'model_bias.pdf'),
            bbox_inches='tight')
