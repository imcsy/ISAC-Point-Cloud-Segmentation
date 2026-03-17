#%%
import numpy as np
import os
import open3d as o3d
from utils.get_rotation_matrix import get_rotation_matrix
from utils.create_bbox_lineset import create_bbox_lineset
from utils.car_box2label import is_car

#%%
# path
MYDATASET_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\radar" # (x, y, z, v, label)

#%%
# Load the file
index = "016653"
radar_txt_path = os.path.join(MYDATASET_RADAR_PATH, index + ".txt")
radar_points_labels = np.loadtxt(radar_txt_path, delimiter=',')
xyz_ls = radar_points_labels[:,0:3]
xyzv_ls = radar_points_labels[:,0:4]

#%%
# ==================================================
#   car
# ==================================================
# import car box corners and create line set
car_corners_list = np.load(rf"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\car_box\{index}.npy")

# delete boxes that are out of sight
car_box_centers_ls = car_corners_list.mean(axis=1)
mask = (car_box_centers_ls[:,0] < 0) & (car_box_centers_ls[:,1] < 0)
car_corners_list = car_corners_list[mask]

#%%
one_car_points = []
one_car_corners = car_corners_list[1]
one_car_corners = np.array([one_car_corners])

for i, point in enumerate(xyz_ls):
    if is_car(point, one_car_corners):
        one_car_points.append(xyz_ls[i])

one_car_points = np.array(one_car_points)
centroid = np.mean(one_car_points[:, :3], axis=0)
one_car_points[:, 0:3] = one_car_points[:, 0:3] - centroid
