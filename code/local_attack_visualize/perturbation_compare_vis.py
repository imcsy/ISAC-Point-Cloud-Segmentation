#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
INI_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_5_npoint_16\attack_comparison.csv"
ini_df = pd.read_csv(INI_MODEL)
DROP_SHIFT_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_10_npoint_16_bsize_64_dropout_shift\attack_comparison.csv"
drop_shift_df = pd.read_csv(DROP_SHIFT_MODEL)
MYMODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet_defense\epoch_10_npoint_16_bsize_64\attack_comparison.csv"
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
index = 0
print(my_df['epsilon'][index], ini_df['accuracy_perturb_vel'][index], 
      drop_shift_df['accuracy_perturb_vel'][index], my_df['accuracy_perturb_vel'][index])
