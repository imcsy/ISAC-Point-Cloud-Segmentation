#%%
import numpy as np
import os
# import open3d as o3d
# from utils.create_3d_bbox import create_3d_bbox
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

#%%
# path
MYDATASET_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\radar" # (x, y, z, v, label)
MYS3DIS_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyS3DIS_seg" # output
MYMODELNET_FRAME_STATISTICS_PATH = r"G:\我的云端硬盘\THESIS\code\local_visualize\data\seg_statistics"

COLOR_MAP = {
    0: (255, 128, 0),     # orange                # car
    1: (0, 128, 255),     # blue                  # buildings
    2: (255, 102, 255),   # pink                  # pole
    3: (192, 192, 192),  # gray                  # unlabeled
}
COLOR_MAP = {k: tuple(np.array(c)/255.0) for k, c in COLOR_MAP.items()}

num_frames = 1200
start_idx = 16653
index_frame = np.arange(start_idx,start_idx+num_frames) 

#%%
num_ps_vec = np.zeros(num_frames, dtype=int)
for i, idx_frame in enumerate(tqdm(index_frame)):
    path = os.path.join(MYDATASET_RADAR_PATH, f"{idx_frame:06d}.txt")
    ps = np.loadtxt(path, delimiter=',')
    ps[ps[:, 4] == -1, 4] = 3
    
    # store number of points here
    num_ps_vec[i] = ps.shape[0]

    # save npy file like S3DIS (x,y,z,v,l)
    if random.random() < 0.3:
      save_path = os.path.join(MYS3DIS_RADAR_PATH, f"Test_{idx_frame:06d}.npy")
      np.save(save_path, ps)
    else:
      save_path = os.path.join(MYS3DIS_RADAR_PATH, f"Train_{idx_frame:06d}.npy")
      np.save(save_path, ps)

#%%
path = os.path.join(MYMODELNET_FRAME_STATISTICS_PATH, "num_ps_vec_frame.npy")
np.save(path, num_ps_vec)
# np.load(path, num_ps_vec)

#%%
unique_values, counts = np.unique(num_ps_vec, return_counts=True)
plt.figure()
plt.hist(num_ps_vec, bins=len(unique_values))
plt.xlabel("Number of points per frame")
plt.ylabel("Frequency")
plt.title("Point Cloud Size Distribution (Frame)")
plt.show()

#%%
print("mean:", num_ps_vec.mean())
print("min:", num_ps_vec.min())
print("max:", num_ps_vec.max())
# mean: 1648.9441666666667
# min: 1591
# max: 1703