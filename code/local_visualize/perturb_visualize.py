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
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

#%%
# FIGURE_SAVE_PATH = r"D:\thesis\figure\Attack_vis\cls"
# MYDATASET_RADAR_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyModelNet_cls\car" # (x, y, z, v)
FIGURE_SAVE_PATH = r"D:\thesis\figure\Attack_vis\seg"
MYDATASET_PATH = r"G:\我的云端硬盘\THESIS_dataset\mmw\MyS3DIS_seg\Train_017843.npy" # (x, y, z, v)

#%%
def Chamfer_Dist(S, Sp, weights=[1,1,1,5]):
    """
    S: Clean point set (N, 4) -> (x, y, z, v)
    Sp: Attacked point set (N + npoints_inj, 4) under injection or (N, 4) under perturbation
    weights: units and also importance
    
    Returns:
        cd: Weighted Chamfer Distance
        d_SSp: Nearest neighbor distances for each point in S (N)
        d_SpS: Nearest neighbor distances for each point in Sp (N + npoints_inj)
    """
    # pairwise squared distances
    diff = S[:, None, :] - Sp[None, :, :]   # (N, N + npoints_inj, D)
    dist2 = np.sum((diff ** 2) * weights, axis=2)      # (N, N + npoints_inj)

    # nearest neighbor distances
    d_SSp = np.min(dist2, axis=1)  # (N + npoints_inj)
    d_SpS = np.min(dist2, axis=0)  # (N)
    cd = (np.mean(d_SSp) + np.mean(d_SpS)) / 2   # (1)

    return cd, d_SSp, d_SpS

def perturb_attack(clean_points, channels=[0,1,2,3], eps_max=1):
    '''
    input: a clean point set,    # (N,4)
           channels and eps to perturb

    output: an augumented perturbed point set (with reliability score),      # (N,5)
            chamfer distance
    '''
    eps = random.uniform(0, eps_max)
    per_points = clean_points.copy()       # (N, 4)
    sigma = np.array([0.5, 0.5, 0.5, 3])          # [0.7221, 0.6430, 0.3123, 4.4498]

    noise = np.random.randn(clean_points.shape[0], len(channels)) 
    jitter = noise * sigma[channels] * eps
    per_points[:, channels] += jitter                # (N, 4)
    cd, _, d_SpS = Chamfer_Dist(clean_points, per_points)   # cd (1);  d_SpS (N)

    # caculate reliability score
    lam = 1
    ad_channel = np.exp(-lam * d_SpS).reshape(-1, 1)        # (N)

    per_points_aug = np.concatenate([per_points, ad_channel], axis=1)

    return per_points_aug, cd

def split_evenly(total, k):
    base = total // k
    remainder = total % k
    sizes = np.full(k, base)
    sizes[:remainder] += 1  # distribute leftovers
    
    return sizes

def inject_attack(clean_points, npoints_inj, clutter_size_inj):
    '''
    input: a clean point set, 
           npoints_inj: number of points injected
           clutter_size_inj: the approximate number od points for the injected clutter

    output: an augumented injected point set (with reliability score),
            chamfer distance
    '''
    N = clean_points.shape[0]
    inj_points_aug = np.column_stack((clean_points, np.ones(N)))

    clutter_sizes = split_evenly(npoints_inj, clutter_size_inj)
    xmin, ymin, zmin, vmin = clean_points.min(axis=0)
    xmax, ymax, zmax, vmax = clean_points.max(axis=0)
    xyzscale = ((xmax - xmin) + (ymax - ymin) + (zmax - zmin)) * 0.1 / 3
    vscale = 1.0
    
    clutter_ls = []
    for s in clutter_sizes:
        xyz_cen = np.random.uniform(low=[xmin, ymin, zmin], high=[xmax, ymax, zmax])
        v_cen = np.random.uniform(2, 8)
        
        xyz = np.random.normal(loc=xyz_cen, scale=xyzscale, size=(s,3))
        v = np.random.normal(loc=v_cen, scale=vscale, size=(s, 1))
        a = np.zeros((s, 1))
        clutter = np.concatenate([xyz, v, a], axis=1)      # (s, 5)
        clutter_ls.append(clutter)
    inj_points_aug = np.concatenate([inj_points_aug] + clutter_ls, axis=0).astype(np.float32)
        
    cd, _, _ = Chamfer_Dist(clean_points, inj_points_aug[:, :4])
    return inj_points_aug, cd       # inj_points_aug (N + npoints_inj, 5);  cd (1)


#%%
# cls
# index = 919
# path = os.path.join(MYDATASET_PATH, f"car_{index:05d}.txt")
# points_clean = np.loadtxt(path, delimiter=',')
# seg
points = np.load(MYDATASET_PATH)
points_clean = points[:,:4]

# #   perturbation
# # ==================================================
# cd = 0
# while cd < 2.95 or cd > 3.05:
#     points, cd = perturb_attack(points_clean)

#   injection
# ==================================================
cd = 0
while cd < 2.95 or cd > 3.05:
    points, cd = inject_attack(points_clean, npoints_inj=10, clutter_size_inj=5)
    print(cd)

xyz_ls = points[:,:3]
v = points[:, 3]
v_range = [-8, 8]
v_clipped = np.clip(v, v_range[0], v_range[1])
v_norm = (v_clipped - v_range[0]) / (v_range[1] - v_range[0])

#%%
# visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_ls)
# color
cmap = plt.get_cmap('plasma')   # strong contrast
colors = cmap(v_norm)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=7, origin=[0,0,0])  #(size=0.55, origin=[0,0,0]) 

#%%
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Radar Point Cloud", width=1600, height=1200)
vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
vis.add_geometry(pcd)
vis.add_geometry(axis)
# grid = create_3d_grid(x_range=[-2,2], y_range=[-2,2], z_range=[-1,1], step=1)
# vis.add_geometry(grid)

vis.poll_events()
vis.update_renderer()

render_option = vis.get_render_option()
render_option.point_size = 5           # 25 cls
render_option.light_on = False

view_control = vis.get_view_control()
view_control.set_front([0.25, 0.25, 0.2]) 
view_control.set_lookat([-20, -20, 1])     
view_control.set_up([0, 0, 1])         
view_control.set_zoom(0.25)            

vis.poll_events()
vis.update_renderer()

image_path = os.path.join(FIGURE_SAVE_PATH, f"seg_inj_cd3.png")
vis.capture_screen_image(image_path)
print(f"Screenshot saved to: {image_path}")

vis.destroy_window()

#%%
o3d.visualization.draw_geometries([pcd,axis],
                                  window_name="Radar Point Cloud",
                                  zoom=0.25, front=[0.25, 0.25, 0.2], lookat=[-20, -20, 1], up=[0, 0, 1])  # seg
                                    # zoom=0.8, front=[0.25, 0.25, 0.1], lookat=[0, 0, -0.3], up=[0, 0, 1]) # cls
#%%
cmap = plt.get_cmap('plasma')

fig, ax = plt.subplots(figsize=(6, 0.4))

norm = mpl.colors.Normalize(vmin=v_range[0], vmax=v_range[1])
cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal' )

cbar.set_label('Velocity (m/s)')

plt.show()


# #%%
# def create_3d_grid(x_range, y_range, z_range, step=1.0):
#     points = []
#     lines = []

#     x_vals = np.arange(x_range[0], x_range[1] + step, step)
#     y_vals = np.arange(y_range[0], y_range[1] + step, step)
#     z_vals = np.arange(z_range[0], z_range[1] + step, step)

#     idx = 0

#     # Lines along X direction
#     for y in y_vals:
#         for z in z_vals:
#             points.append([x_range[0], y, z])
#             points.append([x_range[1], y, z])
#             lines.append([idx, idx + 1])
#             idx += 2

#     # Lines along Y direction
#     for x in x_vals:
#         for z in z_vals:
#             points.append([x, y_range[0], z])
#             points.append([x, y_range[1], z])
#             lines.append([idx, idx + 1])
#             idx += 2

#     # Lines along Z direction
#     for x in x_vals:
#         for y in y_vals:
#             points.append([x, y, z_range[0]])
#             points.append([x, y, z_range[1]])
#             lines.append([idx, idx + 1])
#             idx += 2

#     grid = o3d.geometry.LineSet()
#     grid.points = o3d.utility.Vector3dVector(points)
#     grid.lines = o3d.utility.Vector2iVector(lines)

#     # Light gray color
#     colors = [[0.85, 0.85, 0.85] for _ in lines]
#     grid.colors = o3d.utility.Vector3dVector(colors)

#     return grid