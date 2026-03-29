#%%
import numpy as np
import os
import open3d as o3d
from utils.create_3d_bbox import create_3d_bbox
from utils.create_bbox_lineset import create_bbox_lineset
import tqdm as tqdm

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
index = "016653"
radar_txt_path = os.path.join(MYDATASET_RADAR_PATH, index + ".txt")
radar_points_labels = np.loadtxt(radar_txt_path, delimiter=',')
xyz_ls = radar_points_labels[:,0:3]
label_ls = radar_points_labels[:, -1]
point_colors = np.array([COLOR_MAP[l] for l in label_ls])

#%%
car_corners_list = np.load(rf"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\car_box\{index}.npy")

# delete boxes that are out of sight
car_box_centers_ls = car_corners_list.mean(axis=1)
mask = (car_box_centers_ls[:,0] < 0) & (car_box_centers_ls[:,1] < 0)
car_corners_list = car_corners_list[mask]

# generate box line set
bbox_list = create_bbox_lineset(car_corners_list)

#%%
# box = create_3d_bbox(-6, 3, -16, -12, 5, 9)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_ls)
pcd.colors = o3d.utility.Vector3dVector(point_colors)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
o3d.visualization.draw_geometries([pcd, axis] + bbox_list,
                                  window_name="Radar Point Cloud",
                                    zoom=0.2,
                                    front=[0.25, 0.25, 0.2],
                                    lookat=[-20, -20, 1],
                                    up=[0, 0, 1])
