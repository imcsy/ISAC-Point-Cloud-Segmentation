import torch

def perturbation_attack(points, channels, eps):
    """
    Adds Gaussian jitter to specified channels of a point cloud tensor.
    
    Args:
        points: Input tensor of shape (batch_size, npoints, dim_input)
        channels: List of indices to perturb, e.g., [0, 1, 2] for XYZ
        eps: The standard deviation of the Gaussian noise
        
    Returns:
        perturbed_points: A new tensor with noise added
    """
    perturbed_points = points.clone()
    target_data = points[:, :, channels].reshape(-1, len(channels)) 

    sigma = torch.std(target_data, axis=0).to(points.device)
    sigma = sigma.clamp(min=1e-4) 

    noise_shape = (points.shape[0], points.shape[1], len(channels))
    jitter = torch.randn(noise_shape, device=points.device) * eps * sigma
    
    perturbed_points[:, :, channels] += jitter
    return perturbed_points, sigma

def weighted_dist_per(clean_points, per_points, weights):
    '''
    calculate the Weighted Euclidean Distance between clean_points and per_points
    input: clean_points / per_points # (batch_size, npoints, 4)
           weights # (4)
    output: weighted distance between clean_point and per_points # ((batch_size, npoints, 1)
    '''
    # ps_ref, ps_att = ps_ref[0], ps_att[0]       # add loop for batch
    ps_ref = clean_points.reshape(-1,4)
    ps_att = per_points.reshape(-1,4)
    dist_vec = torch.zeros(ps_ref.shape[0], dtype=torch.float32)
    for i, p_ref in enumerate(ps_ref):
        p_att = ps_att[i] 
        diff = (p_ref - p_att) ** 2
        dist_vec[i] = torch.dot(diff, weights)

    dist_vec = dist_vec.reshape(clean_points.shape[0], clean_points.shape[1], 1)
    dist_vec = torch.tensor(dist_vec)
    return dist_vec

def add_ADchannel(clean_points, is_perturbed, channels=[0,1,2,3], eps=0):
    '''
    Add 5-th channel (abnormal detector) to the points data using generative model

    Input: clean points # (batch_size, npoints, 4)
    Return: clean/perturbed points + abnormal detetcor # (batch_size, npoints, 5)
    '''
    if is_perturbed:
        per_points, sigma = perturbation_attack(clean_points, channels, eps)
        weights = torch.zeros(4, dtype=torch.float32, device=clean_points.device)
        weights[channels] = 1 / (sigma**2)  # **2 or not
        weights[-1] = weights[-1] * 3
        dist = weighted_dist_per(clean_points, per_points, weights)         # (B,N,1)
        lam = 1
        ad_channel = torch.exp(-lam*dist).to(clean_points.device)           # (B,N,1)
        out = torch.cat([per_points, ad_channel], dim=2)                    # (B,N,5)
    else:
        ad_channel = torch.ones(clean_points.shape[0], clean_points.shape[1], 1, device=clean_points.device)
        out = out = torch.cat([clean_points, ad_channel], dim=2)
        
    return out
