'''
compare mIoU w.r.t. Chamfer Distance
for segmnetation under perturbation and injection attacks
Vanilla + Baseline + PointGuard
'''
#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

#%%
vanilla_color = ["#9D9090", 's']   # Vanilla
noitrain_color = ['#4472C4', '^']   # NoiseTrain
# advtrain_color = ["#44C451", '*']    # AdvTrain
pointnet_color = ['#ED7D31', 'o']    # PointNet 

#%%
VAN_PER_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\sem_seg\pointnet2_sem_seg\epoch_10_npoint_1024_bsize_16\per_attack_channel_[0, 1, 2, 3]_eps_1.5.csv"
van_per_df = pd.read_csv(VAN_PER_MODEL)
NOI_PER_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\sem_seg\pointnet2_sem_seg\epoch_10_npoint_1024_bsize_16_dropout_shift\per_attack_channel_[0, 1, 2, 3]_eps_1.5.csv"
noi_per_df = pd.read_csv(NOI_PER_MODEL)
# ADV_PER_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\AdvTrain_epoch_5_npoint_16_bsize_64\per_attack_comparison_channel_[0, 1, 2, 3]_eps_1.0.csv"
# adv_per_df = pd.read_csv(ADV_PER_MODEL)
MY_PER_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\sem_seg\pointnet2_sem_seg_pointguard\epoch_10_npoint_1024_bsize_16\per_attack_channel_[0, 1, 2, 3]_eps_1.5_perfect_scores.csv"
my_per_df = pd.read_csv(MY_PER_MODEL)

VAN_INJ_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\sem_seg\pointnet2_sem_seg\epoch_10_npoint_1024_bsize_16\inj_attack_npoint_200_cluttersize_10.csv"
van_inj_df = pd.read_csv(VAN_INJ_MODEL)
NOI_INJ_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\sem_seg\pointnet2_sem_seg\epoch_10_npoint_1024_bsize_16_dropout_shift\inj_attack_npoint_200_cluttersize_10.csv"
noi_inj_df = pd.read_csv(NOI_INJ_MODEL)
# # ADV_INJ_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\AdvTrain_epoch_5_npoint_16_bsize_64\inj_attack_comparison_npointsinj_4_cluttersizeinj_2.csv"
# # adv_inj_df = pd.read_csv(ADV_INJ_MODEL)
MY_INJ_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\sem_seg\pointnet2_sem_seg_pointguard\epoch_10_npoint_1024_bsize_16\inj_attack_npoint_200_cluttersize_10.csv"
my_inj_df = pd.read_csv(MY_INJ_MODEL)

#%%
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# perturbation
ax.plot(van_per_df['cd_lower'], van_per_df['mIoU'] * 100, 
        label='Vanilla (per)', color=vanilla_color[0], 
        marker=vanilla_color[1], markersize=8, linewidth=2.5, zorder=3)
ax.plot(noi_per_df['cd_lower'], noi_per_df['mIoU'] * 100, 
        label='Baseline (per)', color=noitrain_color[0], 
        marker=noitrain_color[1], markersize=8, linewidth=2.5, zorder=3)
# ax.plot(adv_per_df['cd_lower'], adv_per_df['mIoU'] * 100, 
#         label='Baseline (per)', color=advtrain_color[0], 
#         marker=advtrain_color[1], markersize=8, linewidth=2.5, zorder=3)
ax.plot(my_per_df['cd_lower'], my_per_df['mIoU'] * 100, 
        label='PointGuard (per)', color=pointnet_color[0], 
        marker=pointnet_color[1], markersize=8, linewidth=2.5, zorder=3)

# injection
ax.plot(van_inj_df['cd_lower'], van_inj_df['mIoU'] * 100, 
        label='Vanilla (inj)', color=vanilla_color[0], 
        marker=vanilla_color[1], markersize=8, linewidth=2.5, zorder=3, linestyle='--')
ax.plot(noi_inj_df['cd_lower'], noi_inj_df['mIoU'] * 100, 
        label='Baseline (inj)', color=noitrain_color[0], 
        marker=noitrain_color[1], markersize=8, linewidth=2.5, zorder=3, linestyle='--')
# # ax.plot(adv_inj_df['cd_lower'], adv_inj_df['class_acc'] * 100, 
# #         label='Baseline (per)', color=advtrain_color[0], 
# #         marker=advtrain_color[1], markersize=8, linewidth=2.5, zorder=3, linestyle='--')
ax.plot(my_inj_df['cd_lower'], my_inj_df['mIoU'] * 100, 
        label='PointNet (inj)', color=pointnet_color[0], 
        marker=pointnet_color[1], markersize=8, linewidth=2.5, zorder=3, linestyle='--')

ax.grid(axis='y', color="#D9D9D9B9", linestyle='-', linewidth=3, zorder=1)
ax.grid(axis='x', linewidth=0) # Explicitly turn off vertical lines

for spine in ['right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#D9D9D9B9')
ax.spines['top'].set_color('#D9D9D9B9')
ax.spines['bottom'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)

ax.set_ylabel('mIoU (%)', fontsize=18, labelpad=15)
ax.set_xlabel('Chamfer Distance', fontsize=18, labelpad=15)

ax.set_ylim(40, 100)
ax.tick_params(axis='both', which='major', labelsize=14, color='white')

model_legend = [
    Line2D([0], [0], color=vanilla_color[0], marker=vanilla_color[1], lw=3, label='Vanilla'),
    Line2D([0], [0], color=noitrain_color[0], marker=noitrain_color[1], lw=3, label='NoiseTrain'),
#     Line2D([0], [0], color=advtrain_color[0], marker=advtrain_color[1], lw=3, label='AdvTrain'),
    Line2D([0], [0], color=pointnet_color[0], marker=pointnet_color[1], lw=3, label='PointGuard'),
]
attack_legend = [
    Line2D([0], [0], color='#000000', lw=2.5, label='Perturbation', linestyle='-'), 
    Line2D([0], [0], color='#000000', lw=2.5, label='Injection', linestyle='--'),
]

leg1 = ax.legend(handles=model_legend, loc='lower left', ncol=1, prop={'size': 14},
                  frameon=True, edgecolor='grey', bbox_to_anchor=(0.02, 0.02))
ax.add_artist(leg1)
leg2 = ax.legend(handles=attack_legend, loc='lower left', ncol=1, prop={'size': 14},
                  frameon=True, edgecolor='grey', bbox_to_anchor=(0.2, 0.02))

plt.tight_layout()
plt.show()