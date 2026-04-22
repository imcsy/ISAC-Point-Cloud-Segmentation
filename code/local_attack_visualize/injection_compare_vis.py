#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
VAN_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_10_npoint_16_bsize_64\inj_attack_comparison_npointsinj_4_cluttersizeinj_2.csv"
van_df = pd.read_csv(VAN_MODEL)
BAS_MODEL =  r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_10_npoint_16_bsize_64_dropout_shift\inj_attack_comparison_npointsinj_4_cluttersizeinj_2.csv"
bas_df = pd.read_csv(BAS_MODEL)
MY_MODEL = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet_defense\epoch_10_npoint_16_bsize_32\inj_attack_comparison_npointsinj_4_cluttersizeinj_2.csv"
my_df = pd.read_csv(MY_MODEL)

#%%
#   Attack Analysis of VANILLA model
# ==================================================
plt.figure(figsize=(12, 8))

plt.plot(van_df['cd_upper'], van_df['class_acc'] * 100, 
         label='Vanilla', color='#ff7f0e', markersize=4, linewidth=2)
plt.plot(van_df['cd_upper'], bas_df['class_acc'] * 100, 
         label='Baseline', color="#0e6aff", markersize=4, linewidth=2)
plt.plot(my_df['cd_upper'], my_df['class_acc'] * 100, 
         label='PointGuard', color="#d62728", markersize=4, linewidth=2)

# plt.title('Vulnerability Analysis under Injection Attack', fontsize=14, pad=15)
plt.xlabel('Chamfer Distance', fontsize=20)
plt.ylabel('Class Accuracy (%)', fontsize=20)
plt.legend(fontsize=16)
plt.grid()


