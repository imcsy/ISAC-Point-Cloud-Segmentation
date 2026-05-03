#   correct car labels (y-axis++)
# ==================================================
#%%
import numpy as np
import os
import open3d as o3d
from utils.create_3d_bbox import create_3d_bbox
from utils.create_bbox_lineset import create_bbox_lineset
from utils.car_box2label import is_car
from pathlib import Path

#%%
# path
MYS3DIS_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyS3DIS_seg" # (x, y, z, v, label)
MYDATASET_CAR_BOX_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyDataset_rsu1\car_box"
MYS3DIS_VIS_PATH = r"D:\thesis\report\fig\MyS3DIS_seg_vis"

COLOR_MAP = {
    0: (255, 128, 0),     # orange                # car
    1: (0, 128, 255),     # blue                  # buildings
    2: (255, 102, 255),   # pink                  # pole
    3: (192, 192, 192)    # gray                  # unlabeled
}
COLOR_MAP = {k: tuple(np.array(c)/255.0) for k, c in COLOR_MAP.items()}

#%%
# Load the file
file_list = [os.path.join(MYS3DIS_RADAR_PATH, f) for f in os.listdir(MYS3DIS_RADAR_PATH) if f.endswith('.npy')]

#%%
#   correct the car labels
# ==================================================
# for f in tqdm(file_list):
#     data = np.load(f)
#     index = Path(f).stem.split('_')[1]

#     # load old labels
#     xyz_ls = data[:,0:3]
#     label_ls = data[:, -1]    

#     # get car box
#     path = os.path.join(MYDATASET_CAR_BOX_PATH, index + ".npy")
#     car_corners_list = np.load(path)
#     # delete boxes that are out of sight
#     car_box_centers_ls = car_corners_list.mean(axis=1)
#     mask = (car_box_centers_ls[:,0] < 0) & (car_box_centers_ls[:,1] < 5)    # revise y-axis range to (-inf, 5)
#     car_corners_list = car_corners_list[mask]
#     # generate box line set
#     bbox_list = create_bbox_lineset(car_corners_list)

#     # label points as car within the car boxes
#     for i, point in enumerate(xyz_ls):
#         if is_car(point, car_corners_list):
#             label_ls[i] = 0

#     data[:,-1] = label_ls

#     # save
#     np.save(f, data)

#%%
#   visualize for checking
# ==================================================
f = file_list[0]
data = np.load(f)
index = Path(f).stem.split('_')[1]

# load old labels
xyz_ls = data[:,0:3]
label_ls = data[:, -1]

# get car box
path = os.path.join(MYDATASET_CAR_BOX_PATH, index + ".npy")
car_corners_list = np.load(path)
# delete boxes that are out of sight
car_box_centers_ls = car_corners_list.mean(axis=1)
mask = (car_box_centers_ls[:,0] < 0) & (car_box_centers_ls[:,1] < 5)    # revise y-axis range to (-inf, 5)
car_corners_list = car_corners_list[mask]
# generate box line set
bbox_list = create_bbox_lineset(car_corners_list)

# box = create_3d_bbox(-6, 3, -16, -12, 5, 9)
point_colors = np.array([COLOR_MAP[l] for l in label_ls])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_ls)
pcd.colors = o3d.utility.Vector3dVector(point_colors)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
o3d.visualization.draw_geometries([pcd, axis] + bbox_list,
                                  window_name="Radar Point Cloud",
                                    zoom=0.25,
                                    front=[0.25, 0.25, 0.2],
                                    lookat=[-20, -20, 1],
                                    up=[0, 0, 1])

#%%
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='S3DIS Capture', width=1200, height=900) # 3:4 Ratio
vis.add_geometry(pcd)

vis.add_geometry(pcd)
vis.add_geometry(axis)
for bbox in bbox_list:
    vis.add_geometry(bbox)

view_ctl = vis.get_view_control()
view_ctl.set_zoom(0.25)
view_ctl.set_front([0.25, 0.25, 0.2])
view_ctl.set_lookat([-20, -20, 1])
view_ctl.set_up([0, 0, 1])

vis.poll_events()
vis.update_renderer()

#%%
# Save
image_path = os.path.join(MYS3DIS_VIS_PATH, f"seg_{index}.png")
vis.capture_screen_image(image_path)
print(f"Screenshot saved to: {image_path}")

vis.destroy_window()