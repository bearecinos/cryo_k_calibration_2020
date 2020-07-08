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
rcParams['axes.labelsize'] = 25
rcParams['xtick.labelsize'] = 25
rcParams['ytick.labelsize'] = 25
sns.set_context('poster')

plot_path = os.path.join(MAIN_PATH, 'plots/')

output_dir_path = os.path.join(MAIN_PATH, 'output_data/9_summary_output/')

df_both = pd.read_csv(os.path.join(output_dir_path,
                                'glacier_stats_both_methods.csv'))

df_both['diff_k'] = (df_both['k_value_MV'] - df_both['k_value_MR']).abs()

df_both  = df_both.loc[df_both.diff_k != 0]

df_both = df_both.loc[df_both.k_value_MR !=0]

#df_both = df_both.loc[df_both.k_value_MR !=0]

# Classify the glaciers by area classes
df_both["area_class"] = np.digitize(df_both["rgi_area_km2"],
                                    [0, 5, 15, 50, 1300],
                                 right=True)

to_plot_q = df_both[['calving_flux_MV', 'calving_flux_MR', 'area_class']]
to_plot_k = df_both[['k_value_MV', 'k_value_MR', 'area_class']]

to_plot_q = to_plot_q.melt('area_class',
                           var_name='Method',  value_name='calving_flux')
to_plot_k = to_plot_k.melt('area_class',
                           var_name='Method',  value_name='k_value')

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 8))

g0 = sns.catplot(x="area_class", y="k_value", hue='Method',
                 data=to_plot_k, kind='box', ax=ax0, legend=True)
ax0.set_xticklabels(labels=['0-5', '5-15', '15-50', '50-1300'])
ax0.set_xlabel('Area class [$km^2$]')
ax0.set_ylabel('$k$ [$yr^{-1}$]')
at = AnchoredText('a', prop=dict(size=22), frameon=True, loc=2)
ax0.add_artist(at)

# replace labels
ax0.get_legend().remove()
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, ['$k_{velo}$',
                     '$k_{RACMO}$'], loc=6, fontsize=22)

ax1.set_yscale("log")
g1 = sns.catplot(x="area_class", y="calving_flux", hue='Method',
                 data=to_plot_q, kind='box', ax=ax1, legend=True)
ax1.set_xticklabels(labels=['0-5', '5-15', '15-50', '50-1300'])
ax1.set_xlabel('Area class [$km^2$]')
ax1.set_ylabel('$q_{calving}$ [$km^3yr^{-1}$]')
at = AnchoredText('b', prop=dict(size=22), frameon=True, loc=2)
ax1.add_artist(at)

# replace labels
ax1.get_legend().remove()
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, ['$q_{calving-velo}$',
                     '$q_{calving-RACMO}$'], loc=9, fontsize=22)

plt.close(2)
plt.close(3)

plt.tight_layout()
# plt.show()
# plt.savefig(os.path.join(plot_path, 'box_plot.jpg'),
#             bbox_inches='tight')
plt.savefig(os.path.join(plot_path, 'box_plot.png'),
            bbox_inches='tight')