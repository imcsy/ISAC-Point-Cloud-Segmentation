#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
csv_path = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet_cls_mymodelnet\epoch_5_npoint_16\attack_comparison.csv"
df = pd.read_csv(csv_path)

#%%
plt.figure(figsize=(10, 6))
plt.plot(df['epsilon'], df['accuracy_perturb_pos'] * 100, 
         label='Spatial Attack (Position)', color='#1f77b4', markersize=4, linewidth=2)

plt.plot(df['epsilon'], df['accuracy_perturb_vel'] * 100, 
         label='Doppler Attack (Velocity)', color='#d62728', markersize=4, linewidth=2)

plt.plot(df['epsilon'], df['accuracy_perturb_pos_vel'] * 100, 
         label='Combined Attack (Pos + Vel)', color='#ff7f0e', markersize=4, linewidth=2)

plt.title('Vulnerability Analysis under Random Attack', fontsize=14, pad=15)
plt.xlabel('Attack Magnitude ($\epsilon$)', fontsize=12)
plt.ylabel('Class Accuracy (%)', fontsize=12)
plt.legend()
plt.grid()
