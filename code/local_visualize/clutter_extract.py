#%%
import numpy as np
import os
import open3d as o3d
from utils.create_3d_bbox import create_3d_bbox
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#%%
# path
MYDATASET_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\radar" # (x, y, z, v, label)

COLOR_MAP = {
    -1: (192, 192, 192),  # gray                  # unlabeled
    0: (255, 128, 0),     # orange                # car
    1: (0, 128, 255),     # blue                  # buildings
    2: (255, 102, 255),   # pink                  # pole
}
COLOR_MAP = {k: tuple(np.array(c)/255.0) for k, c in COLOR_MAP.items()}

#%%
# Load the file
index = "016838"
radar_txt_path = os.path.join(MYDATASET_RADAR_PATH, index + ".txt")
radar_points_labels = np.loadtxt(radar_txt_path, delimiter=',')
unlabeled_points = radar_points_labels[radar_points_labels[:,-1] == -1]

#%%
# cluster using DBSCAN
clustering = DBSCAN(eps=2.0, min_samples=2).fit(unlabeled_points[:, :3])
dbscan_labels = clustering.labels_
max_label = dbscan_labels.max()

#%%
clutter_ls = []
for i in range(max_label):
    ps = unlabeled_points[dbscan_labels == i]
    if 0.2 < np.mean(ps[:, 2], axis=0) < 3:
        clutter_ls.append(ps[:, :4])

print(clutter_ls[14])

#%%
# # box = create_3d_bbox(-6, -3, -16, -13, 0.4, 2)

# max_label = labels.max()
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels == -1] = [0.8, 0.8, 0.8, 1]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(unlabeled_points[:,:3])
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
# o3d.visualization.draw_geometries([pcd, axis],
#                                   window_name="Radar Point Cloud",
#                                     zoom=0.1,
#                                     front=[0.3, 0.3, 0.3],
#                                     lookat=[-10, -10, 1],
#                                     up=[0, 0, 1])
