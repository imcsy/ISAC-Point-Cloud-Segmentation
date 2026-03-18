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
MYDATASET_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\radar" # (x, y, z, v, label)

#%%
# setup4
COLOR_MAP = {
    -1: (192, 192, 192),  # gray                  # unlabeled
    0: (255, 128, 0),     # orange                # car
    1: (0, 128, 255),     # blue                  # buildings
    2: (255, 102, 255),   # pink                  # pole
}
COLOR_MAP = {k: tuple(np.array(c)/255.0) for k, c in COLOR_MAP.items()}

# radar location and angle
radar_loc = [1.5, 4.670000076293945, 4.099999904632568]
radar_rot = [0.0, 0.0, -135.0]

#%%
index = "016653"
path = os.path.join(MYDATASET_RADAR_PATH, index + ".txt")
radar_points_labels = np.loadtxt(path, delimiter=',')

xyz_ls = radar_points_labels[:,:3]
label_ls = radar_points_labels[:,4]

#%%
# visualize
point_colors_car = np.array([COLOR_MAP[l] for l in label_ls])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_ls)
pcd.colors = o3d.utility.Vector3dVector(point_colors_car)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
o3d.visualization.draw_geometries([pcd, axis],
                                  window_name="Radar Point Cloud",
                                    zoom=0.1,
                                    front=[0.3, 0.3, 0.3],
                                    lookat=[-10, -10, 1],
                                    up=[0, 0, 1])


