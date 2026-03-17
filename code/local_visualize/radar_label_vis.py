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
DATASET_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\rsu_1"
MYDATASET_LIDAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\lidar"
MYDATASET_LIDAR_LABEL_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\labels_lidar"
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
# read raw radar point cloud file
index = '016699'
# original data (DATASET)
radar_json_path = os.path.join(DATASET_PATH, index + ".json")
# new data (MYDATASET)
radar_txt_path = os.path.join(MYDATASET_RADAR_PATH, index + ".txt")

# carlibrate coordinates
# xyzv (N, 4)
with open(radar_json_path, "r") as f:
    points = json.load(f)

xyz_ls = []
v_ls = []
for p in points:
    r, az, alt, v = p["depth"], p["azimuth"], p["altitude"], p["velocity"]
    x = r * np.cos(alt) * np.cos(az)
    y = r * np.cos(alt) * np.sin(az)
    z = r * np.sin(alt)
    xyz_ls.append([x, y, z])
    v_ls.append([v])

xyz_ls = np.array(xyz_ls)
v_ls = np.array(v_ls)

# transform
t = np.array(radar_loc)
R = get_rotation_matrix(radar_rot)
xyz_ls = (R @ xyz_ls.T).T + t

# create label vector for each point
label_ls = -1 * np.ones(xyz_ls.shape[0], dtype=int)

#%%
# ==================================================
#   buildings
# ==================================================
# label buildings (lidar point cloud labels --> radar point cloud labels)

# get lidar points
lidar_pcd_path = os.path.join(MYDATASET_LIDAR_PATH, index + ".pcd")
lidar_pcd = o3d.io.read_point_cloud(lidar_pcd_path)
lidar_points = np.asarray(lidar_pcd.points)
# get lidar labels
lidar_label_path = os.path.join(MYDATASET_LIDAR_LABEL_PATH, index + ".npy")
lidar_labels = np.load(lidar_label_path)
lidar_labels[lidar_labels != 1] = -1

#%%
tree = KDTree(lidar_points)
dist, ind = tree.query(xyz_ls, k=5)  # ind.shape = (num_radar_points, 5)
dist_threshold = 10

for i in range(xyz_ls.shape[0]):
  close_lidar_labels = lidar_labels[ind[i]].copy()
  close_lidar_labels = close_lidar_labels[dist[i]<dist_threshold]
  if len(close_lidar_labels) == 0:
    label_ls[i] = -1
  else:
    close_lidar_labels += 1
    label_ls[i] = np.bincount(close_lidar_labels).argmax() - 1

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

# generate box line set
bbox_list = create_bbox_lineset(car_corners_list)

#%%
# ==================================================
#   pole 
# ==================================================
# pole space range
pole_range_box = np.array([[-12, -10, -11, -9, 0, 7],
                       [1, 3, -14, -12, 0, 7],
                       [-6, 3, -16, -12, 5, 9],
                       [-14, -10, -11, 3, 5, 9]])

#%%
# label points as car within the car boxes
for i, point in enumerate(xyz_ls):
    if is_car(point, car_corners_list):
        label_ls[i] = 0
    elif is_pole(point, pole_range_box):
        label_ls[i] = 2

#%%
# visualize
point_colors_car = np.array([COLOR_MAP[l] for l in label_ls])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_ls)
pcd.colors = o3d.utility.Vector3dVector(point_colors_car)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
o3d.visualization.draw_geometries([pcd, axis] + bbox_list,
                                  window_name="Radar Point Cloud",
                                    zoom=0.1,
                                    front=[0.3, 0.3, 0.3],
                                    lookat=[-10, -10, 1],
                                    up=[0, 0, 1])

#%%
# save it to txt file
# (x, y, z, v, label)   xyz after transform
label_ls = label_ls.reshape(-1,1)
radar_points_labels = np.hstack((xyz_ls, v_ls, label_ls))

radar_txt_path = os.path.join(MYDATASET_RADAR_PATH, index + ".txt")
np.savetxt(radar_txt_path, radar_points_labels, fmt='%.6f', delimiter=',')

