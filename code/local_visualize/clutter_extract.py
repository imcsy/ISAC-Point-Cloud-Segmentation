#%%
import numpy as np
import os
import open3d as o3d
from utils.create_3d_bbox import create_3d_bbox
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

#%%
# path
# MYDATASET_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\radar" # (x, y, z, v, label)
MYS3DIS_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyS3DIS_seg" # (x, y, z, v, label)

COLOR_MAP = {
    0: (255, 128, 0),     # orange                # car
    1: (0, 128, 255),     # blue                  # buildings
    2: (255, 102, 255),   # pink                  # pole
    3: (192, 192, 192)    # gray                  # clutter
}
COLOR_MAP = {k: tuple(np.array(c)/255.0) for k, c in COLOR_MAP.items()}

#%%
# Load the file
file_list = [os.path.join(MYS3DIS_RADAR_PATH, f) for f in os.listdir(MYS3DIS_RADAR_PATH) if f.endswith('.npy')]
f = file_list[99]
data = np.load(f)
# index = Path(f).stem.split('_')[1]

unlabeled_points = data[data[:,-1] == 3]

#%%
print(data.shape[0], unlabeled_points.shape[0])

#%%
# cluster using DBSCAN
clustering = DBSCAN(eps=2.0, min_samples=2).fit(unlabeled_points[:, :3])
dbscan_labels = clustering.labels_
max_label = dbscan_labels.max()

#%%
clutter_ls = []
prob = 0.2
for i in range(max_label):
    ps = unlabeled_points[dbscan_labels == i]

    # find clutter above the ground with prob of 15%
    if ps.size > 0 and 0.2 < np.mean(ps[:, 2], axis=0) < 3 and np.random.rand() < prob:
            centroid = np.mean(ps[:, :3], axis=0)
            ps[:, 0:3] = ps[:, 0:3] - centroid
            clutter_ls.append(ps[:, :5])

print(len(clutter_ls))

#%%
print(clutter_ls[2])

