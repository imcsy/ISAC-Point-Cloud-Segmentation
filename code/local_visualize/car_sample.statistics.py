#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
MYMODELNET_CAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyModelNet_cls\clutter" # (x, y, z, v)
MYMODELNET_CAR_STATISTICS_PATH = r"G:\我的云端硬盘\THESIS\code\local_visualize\data\cls_statistics"
num_samples = 4640 # 3552, 4640
index_car = np.arange(1,num_samples+1)

#%% 
# ==================================================
# number of points (histogram)
# avg. velocity (histogram)
# ==================================================
num_ps_vec = np.zeros(num_samples, dtype=int)
ave_vel_vec = np.zeros(num_samples, dtype=float)
spread_vec = np.zeros(num_samples, dtype=float)
for i in tqdm(index_car):
    path = os.path.join(MYMODELNET_CAR_PATH, f"clutter_{i:05d}.txt")
    ps = np.loadtxt(path, delimiter=',')
    ps = np.atleast_2d(ps)

    num_ps_vec[i-1] = ps.shape[0]
    ave_vel_vec[i-1] = ps[:,3].mean()

    dist = np.linalg.norm(ps[:,:3], axis=1)
    spread_vec[i-1] = np.mean(dist)

#%%
path = os.path.join(MYMODELNET_CAR_STATISTICS_PATH, "num_ps_vec_clutter.npy")
num_ps_vec = np.load(path)
num_ps_vec = num_ps_vec[num_ps_vec < 50]
path = os.path.join(MYMODELNET_CAR_STATISTICS_PATH, "ave_vel_vec_clutter.npy")
ave_vel_vec = np.load(path)
path = os.path.join(MYMODELNET_CAR_STATISTICS_PATH, "spread_vec_clutter.npy")
spread_vec = np.load(path)

#%%
plt.figure()
plt.hist(num_ps_vec, bins=50)
plt.xlabel("Number of points per sample")
plt.ylabel("Frequency")
plt.title("Point Cloud Size Distribution (Clutter)")
plt.show()

plt.figure()
plt.hist(ave_vel_vec, bins=30)
plt.xlabel("Average Velocity per sample")
plt.ylabel("Frequency")
plt.title("Average Velocity Distribution (Clutter)")
plt.show()

plt.figure()
plt.hist(spread_vec, bins=30)
plt.xlabel("Average Spread Distance per sample")
plt.ylabel("Frequency")
plt.title("Average Spread Distance Distribution (Clutter)")
plt.show()

#%%
# save 30 mins work
path = os.path.join(MYMODELNET_CAR_STATISTICS_PATH, "num_ps_vec_clutter.npy")
np.save(path, num_ps_vec)

path = os.path.join(MYMODELNET_CAR_STATISTICS_PATH, "ave_vel_vec_clutter.npy")
np.save(path, ave_vel_vec)

path = os.path.join(MYMODELNET_CAR_STATISTICS_PATH, "spread_vec_clutter.npy")
np.save(path, spread_vec)