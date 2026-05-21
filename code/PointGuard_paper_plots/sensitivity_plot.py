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

baseline_ls = ["SOR", "SPR", "Instance-Dis",  "PointGuard"] 
baseline_jsonname_ls =["sor", "spr", "discriminator", "pointguard"]
baseline_colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"] 
N_basline = len(baseline_ls)
perturbation_percentage_ls = [0.2, 0.5]
linestyle_ls = ['-', '--']

#%%
per_intensity = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
per_x_pos = np.arange(len(per_intensity))

with open(per_data_path, "r") as f:
    per_json_data = json.load(f)


#%%
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for j in range(2):
        for i in range(N_basline):
                auc20 = []
                auc20_min = []
                auc20_max = []

                for val in per_intensity:
                        item = next((d for d in per_json_data
                                if d["percentage"] == perturbation_percentage_ls[j] 
                                and d["baseline"] == baseline_jsonname_ls[i] and d["intensity"] == val))
                        auc20.append(item["mean_auc"])
                        auc20_min.append(item["min_auc"])
                        auc20_max.append(item["max_auc"])
                auc20, auc20_min, auc20_max = np.array(auc20), np.array(auc20_min), np.array(auc20_max), 

                ax.plot(per_x_pos, auc20, linestyle_ls[j], label=baseline_ls[i], color=baseline_colors[i], 
                        markersize=8, linewidth=2.5, zorder=3)

                ax.errorbar(per_x_pos, auc20, yerr=[auc20-auc20_min, auc20_max-auc20], 
                        fmt='none', color=baseline_colors[i], capsize=6, linewidth=2, zorder=2)


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

ax.set_ylabel('AUC', fontsize=32, labelpad=15)
ax.set_xlabel('Perturbation Intensity', fontsize=32, labelpad=15)

ax.set_ylim(0.45, 0.95)
ax.tick_params(axis='both', which='major', labelsize=22, color='white')

model_legend = [
    Line2D([0], [0], color=baseline_colors[0], lw=3, label='SOR'),
    Line2D([0], [0], color=baseline_colors[1], lw=3, label='SPR'),
    Line2D([0], [0], color=baseline_colors[2], lw=3, label='Instance-Dis'),
    Line2D([0], [0], color=baseline_colors[3], lw=3, label='PointGuard'),
]
attack_legend = [
    Line2D([0], [0], color='#000000', lw=2.5, label='20%', linestyle='-'), 
    Line2D([0], [0], color='#000000', lw=2.5, label='60%', linestyle='--'),
]

leg1 = ax.legend(handles=model_legend, loc='lower left', ncol=1, prop={'size': 20},
                  frameon=True, edgecolor='grey', bbox_to_anchor=(0.01, 0.72))
ax.add_artist(leg1)
leg2 = ax.legend(handles=attack_legend, loc='lower left', ncol=1, prop={'size': 20},
                  frameon=True, edgecolor='grey', bbox_to_anchor=(0.28, 0.85))

plt.tight_layout()
plt.show()


#%%
inj_data_path = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\pointguard\pointguard_classification_mix\epoch_10_npoint_16_bsize_64\numerical result\inj_sensitivity.json"

N_basline = 4

inj_clutter_size = np.array([1, 2, 4, 6, 8])
inj_x_pos = np.arange(len(inj_clutter_size))

inj_surface = ["off", "on"]

with open(inj_data_path, "r") as f:
    inj_json_data = json.load(f)

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for j in range(2):
        for i in range(N_basline):
                auc20 = []
                auc20_min = []
                auc20_max = []

                for val in inj_clutter_size:
                        item = next((d for d in inj_json_data
                                if d["surface"] == inj_surface[j] 
                                and d["baseline"] == baseline_jsonname_ls[i] and d["inj_clutter_size"] == val))
                        auc20.append(item["mean_auc"])
                        auc20_min.append(item["min_auc"])
                        auc20_max.append(item["max_auc"])
                auc20, auc20_min, auc20_max = np.array(auc20), np.array(auc20_min), np.array(auc20_max), 

                ax.plot(inj_x_pos, auc20, linestyle_ls[j], label=baseline_ls[i], color=baseline_colors[i], 
                        markersize=8, linewidth=2.5, zorder=3)

                ax.errorbar(inj_x_pos, auc20, yerr=[auc20-auc20_min, auc20_max-auc20], 
                        fmt='none', color=baseline_colors[i], capsize=6, linewidth=2, zorder=2)


ax.set_xticks(inj_x_pos)
ax.set_xticklabels(inj_clutter_size)

ax.grid(axis='y', color="#D9D9D9B9", linestyle='-', linewidth=3, zorder=1)
ax.grid(axis='x', linewidth=0) # Explicitly turn off vertical lines

for spine in ['right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#D9D9D9B9')
ax.spines['top'].set_color('#D9D9D9B9')
ax.spines['bottom'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)

ax.set_ylabel('AUC', fontsize=32, labelpad=15)
ax.set_xlabel('Injection Clutter Size', fontsize=32, labelpad=15)

ax.set_ylim(0.4, 1)
ax.tick_params(axis='both', which='major', labelsize=22, color='white')

model_legend = [
    Line2D([0], [0], color=baseline_colors[0], lw=3, label='SOR'),
    Line2D([0], [0], color=baseline_colors[1], lw=3, label='SPR'),
    Line2D([0], [0], color=baseline_colors[2], lw=3, label='Instance-Dis'),
    Line2D([0], [0], color=baseline_colors[3], lw=3, label='PointGuard'),
]
attack_legend = [
    Line2D([0], [0], color='#000000', lw=2.5, label='off surface', linestyle='-'), 
    Line2D([0], [0], color='#000000', lw=2.5, label='on surface', linestyle='--'),
]

leg1 = ax.legend(handles=model_legend, loc='lower left', ncol=1, prop={'size': 20},
                  frameon=True, edgecolor='grey', bbox_to_anchor=(0.72, 0.01))
ax.add_artist(leg1)
leg2 = ax.legend(handles=attack_legend, loc='lower left', ncol=1, prop={'size': 20},
                  frameon=True, edgecolor='grey', bbox_to_anchor=(0.47, 0.01))

plt.tight_layout()
plt.show()
