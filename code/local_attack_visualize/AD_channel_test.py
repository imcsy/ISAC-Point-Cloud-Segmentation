#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

#%%
object = "car"
MYMODELNET_PATH = rf"G:\我的云端硬盘\THESIS_dataset\mmw\MyModelNet_cls\{object}"

#%%
def perturbation_attack(points, channels, eps):
    """
    Adds Gaussian jitter to specified channels of a point cloud tensor.
    
    Args:
        points: Input tensor of shape (batch_size, npoints, dim_input)
        channels: List of indices to perturb, e.g., [0, 1, 2] for XYZ
        delta: The standard deviation of the Gaussian noise
        
    Returns:
        perturbed_points: A new tensor with noise added
    """
    perturbed_points = points.clone()
    target_data = points[:, :, channels].reshape(-1, len(channels)) 

    sigma = torch.std(target_data, axis=0)

    noise_shape = (points.shape[0], points.shape[1], len(channels))
    jitter = torch.randn(noise_shape, device=points.device) * eps * sigma
    
    perturbed_points[:, :, channels] += jitter
    return perturbed_points, sigma

#%%
# calculate the Weighted Euclidean Distance
def weighted_dist_per(ps_ref, ps_att, weights):
    ps_ref, ps_att = ps_ref[0], ps_att[0]       # add loop for batch
    dist_vec = np.zeros(ps_ref.shape[0], dtype=float)
    for i, p_ref in enumerate(ps_ref):
        p_att = ps_att[i] 
        diff = (p_ref - p_att) ** 2
        dist_vec[i] = torch.dot(diff, weights)

        # dist_vec[i] = ws*((p_ref[0]-p_att[0])**2 + (p_ref[1]-p_att[1])**2 + (p_ref[2]-p_att[2])**2) + wv*(p_ref[3]-p_att[3])**2

    # dist_vec = dist_vec.reshape(ps_ref.shape)
    dist_vec = torch.tensor(dist_vec)
    return dist_vec

#%%
index = 2068
path = os.path.join(MYMODELNET_PATH, f"{object}_{index:05d}.txt")
points = np.loadtxt(path, delimiter=',')
points = points.reshape(1, points.shape[0], 4)
points = torch.tensor(points)

#%%
channel= [3]
eps = 4
per_points, sigma = perturbation_attack(points, channels=channel, eps=eps)
# print("clean data:", points[:,:,channel].reshape(-1))
# print("pertubed data:", per_points[:,:,channel].reshape(-1))

print(points.shape)
# print(per_points-points)
# print("channel:", channel, ",eps:" ,eps)

weights = torch.zeros(4, dtype=float)
weights[channel] = 1 / (sigma**2)  # **2 or not
weights[-1] = weights[-1] * 3
dist = weighted_dist_per(points, per_points, weights)
# print("dist", dist)
print("dist mean:", dist.reshape(-1).mean())
lam = 1
ad_channel = torch.exp(-lam*dist)
# print("AD channel:", ad_channel)
print("AD mean:", ad_channel.reshape(-1).mean())