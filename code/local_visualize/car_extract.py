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
MYDATASET_CAR_BOX_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\car_box"
MYMODELNET_CAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyModelNet_cls\car" # (x, y, z, v)

#%%
# Load the file
index = "016653"
radar_txt_path = os.path.join(MYDATASET_RADAR_PATH, index + ".txt")
radar_points_labels = np.loadtxt(radar_txt_path, delimiter=',')
xyz_ls = radar_points_labels[:,0:3]
xyzv_ls = radar_points_labels[:,0:4]

#%%
# load car box file
car_corners_list_path = os.path.join(MYDATASET_CAR_BOX_PATH, index + ".npy")
car_corners_list = np.load(car_corners_list_path)
# delete boxes that are out of sight
car_box_centers_ls = car_corners_list.mean(axis=1)
mask = (car_box_centers_ls[:,0] < 0) & (car_box_centers_ls[:,1] < 0)
car_corners_list = car_corners_list[mask]
num_car = car_corners_list.shape[0]

#%%
counter_car = 1
for i_car in range(num_car):
    ps = []
    corners = car_corners_list[i_car]
    corners = np.array([corners])

    for i, point in enumerate(xyz_ls):
        if is_car(point, corners):
            ps.append(xyzv_ls[i])

    ps = np.array(ps)
    centroid = np.mean(ps[:, :3], axis=0)
    ps[:, 0:3] = ps[:, 0:3] - centroid

    car_save_pth = os.path.join(MYMODELNET_CAR_PATH, f"car_{counter_car:05d}.txt")
    np.savetxt(car_save_pth, ps, fmt='%.6f', delimiter=',')
    counter_car += 1
