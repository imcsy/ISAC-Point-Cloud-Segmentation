'''
compare accuracy w.r.t. Chamfer Distance
for classfication under perturbation and injection attacks
Vanilla + Baseline + PointGuard
'''
#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import numpy as np
import json

#%%
per_data_path = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\pointguard\pointguard_classification_mix\epoch_10_npoint_16_bsize_64\numerical result\per_sensitivity.json"

baseline_ls = ["SOR"] #, "SPR", "Instance-Dis", "PointGuard"]
baseline_jsonname_ls = ["sor", "pointguard"]
baseline_colors = ["#1f77b4"] #, "#2ca02c", "#d62728", "#ff7f0e"]
N_basline = len(baseline_ls)

#%%
per_intensity = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
per_x_pos = np.arange(len(per_intensity))

with open(per_data_path, "r") as f:
    per_json_data = json.load(f)


#%%
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for i in range(N_basline):
        auc20 = []
        auc20_min = []
        auc20_max = []

        for val in per_intensity:
                item = next((d for d in per_json_data
                        if d["percentage"] == 0.2 and d["baseline"] == "sor" and d["intensity"] == 5))
                auc20.append(item["mean_auc"])
                auc20_min.append(item["min_auc"])
                auc20_max.append(item["max_auc"])

        ax.plot(per_x_pos, auc20, label='PointGuard', color=baseline_colors[3], markersize=8, linewidth=2.5, zorder=3)
        yerr = [per_auc20 - per_auc20_min,  per_auc20_max - per_auc20]
ax.errorbar(
    per_x_pos,
    per_auc20,
    yerr=yerr,
    label='PointGuard',
    color=baseline_colors[3],
    markersize=8,
    linewidth=2.5,
    capsize=6,
    zorder=3
)

ax.set_xticks(per_x_pos)
ax.set_xticklabels(per_intensity)

ax.grid(axis='y', color="#D9D9D9B9", linestyle='-', linewidth=3, zorder=1)
ax.grid(axis='x', linewidth=0) # Explicitly turn off vertical lines

for spine in ['right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#D9D9D9B9')
ax.spines['top'].set_color('#D9D9D9B9')
ax.spines['bottom'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)

ax.set_ylabel('AUC', fontsize=18, labelpad=15)
ax.set_xlabel('Perturbation Intensity', fontsize=18, labelpad=15)

# ax.set_ylim(50, 100)
ax.tick_params(axis='both', which='major', labelsize=14, color='white')

# model_legend = [
#     Line2D([0], [0], color=vanilla_color[0], marker=vanilla_color[1], lw=3, label='Vanilla'),
#     Line2D([0], [0], color=noitrain_color[0], marker=noitrain_color[1], lw=3, label='NoiseTrain'),
# #     Line2D([0], [0], color=advtrain_color[0], marker=advtrain_color[1], lw=3, label='AdvTrain'),
#     Line2D([0], [0], color=pointnet_color[0], marker=pointnet_color[1], lw=3, label='PointGuard'),
# ]
# attack_legend = [
#     Line2D([0], [0], color='#000000', lw=2.5, label='Perturbation', linestyle='-'), 
#     Line2D([0], [0], color='#000000', lw=2.5, label='Injection', linestyle='--'),
# ]

# leg1 = ax.legend(handles=model_legend, loc='lower left', ncol=1, prop={'size': 14},
#                   frameon=True, edgecolor='grey', bbox_to_anchor=(0.02, 0.02))
# ax.add_artist(leg1)
# leg2 = ax.legend(handles=attack_legend, loc='lower left', ncol=1, prop={'size': 14},
#                   frameon=True, edgecolor='grey', bbox_to_anchor=(0.2, 0.02))

plt.tight_layout()
plt.show()