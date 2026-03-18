#%%
import numpy as np
import open3d as o3d
import json
import os
from sklearn.neighbors import KDTree
from utils.get_rotation_matrix import get_rotation_matrix
from utils.create_bbox_lineset import create_bbox_lineset
from utils.car_box2label import is_car
from utils.pole_box2label import is_pole

#%%
# path
MYDATASET_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyModelNet_cls\clutter" # (x, y, z, v)

#%%
index = 1
path = os.path.join(MYDATASET_RADAR_PATH, f"clutter_{index:05d}.txt")
radar_points_labels = np.loadtxt(path, delimiter=',')

xyz_ls = radar_points_labels[:,:3]

#%%
# visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_ls)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
o3d.visualization.draw_geometries([pcd],
                                  window_name="Radar Point Cloud")


