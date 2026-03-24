#%%
import numpy as np
import open3d as o3d
import sys

#%%
COLOR_MAP = {
    0: (255, 128, 0),     # orange                # car
    1: (0, 128, 255),     # blue                  # buildings
    2: (255, 102, 255),   # pink                  # pole
    3: (192, 192, 192),  # gray                  # unlabeled
    4: (255, 0, 0)      # red                       # wrong prediction
}
COLOR_MAP = {k: tuple(np.array(c)/255.0) for k, c in COLOR_MAP.items()}


#%%
path = r"G:\我的云端硬盘\THESIS\Pointnet_Pointnet2_pytorch-master\log\sem_seg\pointnet2_sem_seg\visual\eval_36.npy"
data = np.load(path)
sample = data[30]

#%%
points = sample[:, :3]
true_label = sample[:, 4]
pred_label = sample[:, 5]

npoints = 1024
label_compare = np.zeros(npoints, dtype=int)
for i in range(npoints):
    if true_label[i] == pred_label[i]:
        label_compare[i] = true_label[i]
    else:
        label_compare[i] = 4

#%%
colors = np.array([COLOR_MAP[l] for l in label_compare])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
o3d.visualization.draw_geometries([pcd, axis],
                                  window_name="Radar Point Cloud")