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
}
COLOR_MAP = {k: tuple(np.array(c)/255.0) for k, c in COLOR_MAP.items()}

#%%
def visualize_s3dis(file_path):
    print(f"Loading {file_path}...")
    data = np.load(file_path)
    
    xyz = data[:, :3]        # First 3 columns
    labels = data[:, 4]
    colos = np.array([COLOR_MAP[l] for l in labels])

    # rgb = data[:, 3:6]        # Next 3 columns (RGB)
    # if rgb.max() > 1.0:
    #     rgb = rgb / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colos)
    
    print(f"Visualizing {len(xyz)} points. Close the window to exit.")
    
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="S3DIS Room Visualization",
                                      width=1200, height=800)

#%%
path = "G:\我的云端硬盘\THESIS_dataset\mmw\MyS3DIS_seg\Test_016789.npy"
# "G:\我的云端硬盘\THESIS_dataset\S3DIS\stanford_indoor3d\Area_1_hallway_1.npy"
# "G:\我的云端硬盘\THESIS_dataset\mmw\MyS3DIS_seg\Train_016745.npy"
visualize_s3dis(path)

#%%
data = np.load(path)
print(data.shape)
print(data[:10,:])