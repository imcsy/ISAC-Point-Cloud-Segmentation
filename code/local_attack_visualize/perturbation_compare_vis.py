#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
VAN_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_10_npoint_16_bsize_64\per_attack_comparison_channel_[0, 1, 2, 3]_eps_1.0.csv"
van_df = pd.read_csv(VAN_MODEL)
NOI_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_10_npoint_16_bsize_64_dropout_shift\per_attack_comparison_channel_[0, 1, 2, 3]_eps_1.0.csv"
noi_df = pd.read_csv(NOI_MODEL)
ADV_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\AdvTrain_epoch_10_npoint_16_bsize_64\per_attack_comparison_channel_[0, 1, 2, 3]_eps_1.0.csv"
adv_df = pd.read_csv(ADV_MODEL)
MY_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet_defense\epoch_10_npoint_16_bsize_32\per_attack_comparison_channel_[0, 1, 2, 3]_eps_1.0.csv"
my_df = pd.read_csv(MY_MODEL)

#%%
plt.figure(figsize=(12, 8))

plt.plot(van_df['cd_upper'], van_df['class_acc'] * 100, 
         label='Vanilla', color='#ff7f0e', markersize=4, linewidth=2)
# plt.plot(bas_df['cd_upper'], bas_df['class_acc'] * 100, 
#          label='Baseline', color="#0e6aff", markersize=4, linewidth=2)
# plt.plot(my_df['cd_upper'], my_df['class_acc'] * 100, 
#          label='PointGuard', color="#d62728", markersize=4, linewidth=2)

# plt.title('Vulnerability Analysis under Perturbation Attack', fontsize=14, pad=15)
plt.xlabel('Chamfer Distance', fontsize=20)
plt.ylabel('Class Accuracy (%)', fontsize=20)
plt.legend(fontsize=16)
plt.grid()

#%%
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

pointnet_color = '#ED7D31'    # PointNet 
vanilla_color = '#A5A5A5'    # Vanilla
baseline_color = '#4472C4'   # Baseline

ax.plot(van_df['cd_upper'], van_df['class_acc'] * 100, 
        label='Vanilla (per)', color=vanilla_color, 
        marker='o', markersize=8, linewidth=2.5, zorder=3)

ax.plot(bas_df['cd_upper'], bas_df['class_acc'] * 100, 
        label='Baseline (per)', color=baseline_color, 
        marker='o', markersize=8, linewidth=2.5, zorder=3)

ax.plot(my_df['cd_upper'], my_df['class_acc'] * 100, 
        label='PointGuard (per)', color=pointnet_color, 
        marker='o', markersize=8, linewidth=2.5, zorder=3)

ax.grid(axis='y', color="#D9D9D9B9", linestyle='-', linewidth=3, zorder=1)
ax.grid(axis='x', linewidth=0) # Explicitly turn off vertical lines

for spine in ['right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#D9D9D9B9')
ax.spines['top'].set_color('#D9D9D9B9')

ax.set_ylabel('Class Accuracy (%)', fontsize=18, labelpad=15)
ax.set_xlabel('Attack Magnitude ($\epsilon$)', fontsize=18, labelpad=15)
# ax.set_title('Vulnerability Analysis', fontsize=20, pad=20)


# Set Y-axis limit and remove the tick marks themselves for a cleaner look
ax.set_ylim(60, 100)
ax.tick_params(axis='both', which='major', labelsize=14, color='white')

# 6. Legend Style: Borderless and Horizontal-ish
ax.legend(loc='lower left', 
          bbox_to_anchor=(0, 0.1), # Position it slightly above the x-axis
          fontsize=16, 
          frameon=False, 
          ncol=3) # Aligns them horizontally like in the image

plt.tight_layout()
plt.show()

#%%
print(van_df['cd_upper'])

#   OLD
# ==================================================
#%%
# INI_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_10_npoint_16_bsize_64\attack_comparison.csv"
INI_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_5_npoint_16\attack_comparison.csv"
ini_df = pd.read_csv(INI_MODEL)
DROP_SHIFT_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_10_npoint_16_bsize_64_dropout_shift\attack_comparison_perturb_eps0-4.csv"
drop_shift_df = pd.read_csv(DROP_SHIFT_MODEL)
MYMODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet_defense\epoch_10_npoint_16_bsize_64\attack_comparison_pointguard_eps1n3_channel0123.csv"
my_df = pd.read_csv(MYMODEL)

#%%
#   Attack Analysis of VANILLA model
# ==================================================
plt.figure(figsize=(12, 8))

plt.plot(ini_df['epsilon'], ini_df['accuracy_perturb_pos'] * 100, 
         label='Spatial Attack', color='#1f77b4', markersize=4, linewidth=2)
plt.plot(ini_df['epsilon'], ini_df['accuracy_perturb_vel'] * 100, 
         label='Doppler Attack', color='#d62728', markersize=4, linewidth=2)
plt.plot(ini_df['epsilon'], ini_df['accuracy_perturb_pos_vel'] * 100, 
         label='Combined Attack', color='#ff7f0e', markersize=4, linewidth=2)

plt.title('Vulnerability Analysis under Perturbation Attack', fontsize=14, pad=15)
plt.xlabel('Attack Magnitude ($\epsilon$)', fontsize=12)
plt.ylabel('Class Accuracy (%)', fontsize=12)
plt.legend()
plt.grid()


#%%
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 2. Plotting the lines
# Using the specific style colors from your image
spatial_color = '#4472C4'    # PointNet Blue
doppler_color = '#A5A5A5'    # DGCNN Grey
combined_color = '#ED7D31'   # PointNet++ Orange

ax.plot(ini_df['epsilon'], ini_df['accuracy_perturb_pos'] * 100, 
        label='Spatial Attack', color=spatial_color, 
        marker='o', markersize=8, linewidth=2.5, zorder=3)

ax.plot(ini_df['epsilon'], ini_df['accuracy_perturb_vel'] * 100, 
        label='Doppler Attack', color=doppler_color, 
        marker='o', markersize=8, linewidth=2.5, zorder=3)

ax.plot(ini_df['epsilon'], ini_df['accuracy_perturb_pos_vel'] * 100, 
        label='Combined Attack', color=combined_color, 
        marker='o', markersize=8, linewidth=2.5, zorder=3)

# 3. Configure Grid: Horizontal Lines ONLY
ax.grid(axis='y', color="#D9D9D9B9", linestyle='-', linewidth=3, zorder=1)
ax.grid(axis='x', linewidth=0) # Explicitly turn off vertical lines

# 4. Remove Axis Spines (The black box around the plot)
# The style in the image usually only shows the bottom line or no lines
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#D9D9D9')

# 5. Labels and Ticks
ax.set_ylabel('Class Accuracy (%)', fontsize=18, labelpad=15)
ax.set_xlabel('Attack Magnitude ($\epsilon$)', fontsize=18, labelpad=15)
# ax.set_title('Vulnerability Analysis', fontsize=20, pad=20)


# Set Y-axis limit and remove the tick marks themselves for a cleaner look
ax.set_ylim(50, 100)
ax.tick_params(axis='both', which='major', labelsize=14, color='white')

# 6. Legend Style: Borderless and Horizontal-ish
ax.legend(loc='lower left', 
          bbox_to_anchor=(0, 0.1), # Position it slightly above the x-axis
          fontsize=16, 
          frameon=False, 
          ncol=3) # Aligns them horizontally like in the image

plt.tight_layout()
plt.show()

#%%
#   Robustness comparison (vanilla, baseline, pointguard)
# ==================================================
plt.figure(figsize=(12, 8))

plt.plot(ini_df['epsilon'], ini_df['accuracy_perturb_pos'] * 100, 
         label='Spatial Attack (Vanilla)', color='#1f77b4', markersize=4, linewidth=2, linestyle=':')
plt.plot(ini_df['epsilon'], ini_df['accuracy_perturb_vel'] * 100, 
         label='Doppler Attack (Vanilla)', color='#d62728', markersize=4, linewidth=2, linestyle=':')
plt.plot(ini_df['epsilon'], ini_df['accuracy_perturb_pos_vel'] * 100, 
         label='Combined Attack (Vanilla)', color='#ff7f0e', markersize=4, linewidth=2, linestyle=':')

plt.plot(drop_shift_df['epsilon'], drop_shift_df['accuracy_perturb_pos'] * 100, 
         label='Spatial Attack (Baseline)', color='#1f77b4', markersize=4, linewidth=2, linestyle='--')
plt.plot(drop_shift_df['epsilon'], drop_shift_df['accuracy_perturb_vel'] * 100, 
         label='Doppler Attack (Baseline)', color='#d62728', markersize=4, linewidth=2, linestyle='--')
plt.plot(drop_shift_df['epsilon'], drop_shift_df['accuracy_perturb_pos_vel'] * 100, 
         label='Combined Attack (Baseline)', color='#ff7f0e', markersize=4, linewidth=2, linestyle='--')

plt.plot(my_df['epsilon'], my_df['accuracy_perturb_pos'] * 100, 
         label='Spatial Attack (PointGuard)', color='#1f77b4', markersize=4, linewidth=2, linestyle='-')
plt.plot(my_df['epsilon'], my_df['accuracy_perturb_vel'] * 100, 
         label='Doppler Attack (PointGuard)', color='#d62728', markersize=4, linewidth=2, linestyle='-')
plt.plot(my_df['epsilon'], my_df['accuracy_perturb_pos_vel'] * 100, 
         label='Combined Attack (PointGuard)', color='#ff7f0e', markersize=4, linewidth=2, linestyle='-')


plt.title('Robustness Comparison: Classification Performance under Perturbations', fontsize=14, pad=15)
plt.xlabel('Attack Magnitude ($\epsilon$)', fontsize=12)
plt.ylabel('Class Accuracy (%)', fontsize=12)
plt.legend()
plt.grid()

#%%
index = 30
print(my_df['epsilon'][index], ini_df['accuracy_perturb_vel'][index], 
      drop_shift_df['accuracy_perturb_vel'][index], my_df['accuracy_perturb_vel'][index])
