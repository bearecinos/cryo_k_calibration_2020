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

fig1 = plt.figure(figsize=(12, 6))

spec = gridspec.GridSpec(1, 2)

ax0 = plt.subplot(spec[0])

bins=np.histogram(np.hstack((df_k_vel.bias,df_k_vel.obs_error)), bins=40)[1]

p1=sns.distplot(df_k_vel.bias, bins=bins, kde=True, color=sns.xkcd_rgb["dark green"],
                ax=ax0, label='Bias velocity method')
p2=sns.distplot(df_k_vel.obs_error, bins=bins, kde=True, color=sns.xkcd_rgb["dark red"],
                ax=ax0, label='Observational error')
ax0.axvline(linewidth=2, color='k',  linestyle='--')
ax0.legend(fontsize=12)
ax0.set_xlabel('Velocity [m/yr]')
at = AnchoredText('a', prop=dict(size=12), frameon=True, loc=2)
ax0.add_artist(at)

ax1 = plt.subplot(spec[1])
k = sns.distplot(df_k_racmo.bias, bins=50, kde=True, color=sns.xkcd_rgb["dark purple"],
                 ax=ax1, label='Bias RACMO method')
ax1.axvline(linewidth=2, color='purple', linestyle='--')
ax1.legend(fontsize=12)
ax1.set_xlabel('Frontal ablation [$km^{3}$/yr]')
at = AnchoredText('b', prop=dict(size=12), frameon=True, loc=2)
ax1.add_artist(at)

plt.tight_layout()
plt.savefig(os.path.join(plot_path, 'model_bias.pdf'),
            bbox_inches='tight')
