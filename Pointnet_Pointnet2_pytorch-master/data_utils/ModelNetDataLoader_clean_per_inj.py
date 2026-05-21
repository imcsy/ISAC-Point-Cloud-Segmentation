'''
a data loader for point set with mix of clean, perturbed and injected ones
return 5 channels directly (x, y, z, v, a)
'a' is either 0 (clean) or 1 (injected), or in between for perturbed
'''
import os
import numpy as np
import warnings
import pickle
import torch
import random

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_downsample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
        ------ N > npoint---------
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def repeat_point_upsample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
        ------ N < npoint---------
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    num_old_point = point.shape[0]
    num_repeat = npoint // num_old_point
    remainder = npoint % num_old_point

    new_points = np.tile(point, (num_repeat, 1))
    if remainder > 0:
        extra = point[:remainder, :]
        new_points = np.vstack((new_points, extra))

    return new_points

def split_evenly(total, k):
    base = total // k
    remainder = total % k
    sizes = np.full(k, base)
    sizes[:remainder] += 1  # distribute leftovers
    
    return sizes

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


#   Clean
# ==================================================
def no_attack(clean_points):
    '''
    input: a clean point set

    output: an augumented clean point set,
            chamfer distance
    '''
    N = clean_points.shape[0]
    clean_points_aug = np.column_stack((clean_points, np.ones(N)))
    cd = 0

    return clean_points_aug, cd

#  Injection
# ==================================================
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
        v_cen = np.random.uniform(vmin, vmax)
        
        xyz = np.random.normal(loc=xyz_cen, scale=xyzscale, size=(s,3))
        v = np.random.normal(loc=v_cen, scale=vscale, size=(s, 1))
        a = np.zeros((s, 1))
        clutter = np.concatenate([xyz, v, a], axis=1)      # (s, 5)
        clutter_ls.append(clutter)
    inj_points_aug = np.concatenate([inj_points_aug] + clutter_ls, axis=0).astype(np.float32)
        
    cd, _, _ = Chamfer_Dist(clean_points, inj_points_aug[:, :4])
    return inj_points_aug, cd       # inj_points_aug (N + npoints_inj, 5);  cd (1)


def inject_attack_onoff_surface(clean_points, npoints_inj, clutter_size_inj):
    '''
    input: a clean point set, 
           npoints_inj: number of points injected
           clutter_size_inj: the approximate number od points for the injected clutter

    output: an augumented injected point set (with reliability score),
            chamfer distance
    '''
    N = clean_points.shape[0]
    clutter_sizes = split_evenly(npoints_inj, clutter_size_inj)
    xmin, ymin, zmin, vmin = clean_points.min(axis=0)
    xmax, ymax, zmax, vmax = clean_points.max(axis=0)
    xyzscale = ((xmax - xmin) + (ymax - ymin) + (zmax - zmin)) * 0.1 / 3
    vscale = 1.0

    cd = 100
    if npoints_inj < 4:
        cd_upper = 1
    elif npoints_inj < 8:
        cd_upper = 1.5
    else:
        cd_upper = 2
        
    while cd > cd_upper:
        inj_points_aug = np.column_stack((clean_points, np.ones(N)))

        clutter_ls = []
        for s in clutter_sizes:
            xyz_cen = np.random.uniform(low=[xmin, ymin, zmin], high=[xmax, ymax, zmax])
            v_cen = np.random.uniform(vmin, vmax)
            
            xyz = np.random.normal(loc=xyz_cen, scale=xyzscale, size=(s,3))
            v = np.random.normal(loc=v_cen, scale=vscale, size=(s, 1))
            a = np.zeros((s, 1))
            clutter = np.concatenate([xyz, v, a], axis=1)      # (s, 5)
            clutter_ls.append(clutter)
        inj_points_aug = np.concatenate([inj_points_aug] + clutter_ls, axis=0).astype(np.float32)
            
        cd, _, _ = Chamfer_Dist(clean_points, inj_points_aug[:, :4])

    return inj_points_aug, cd       # inj_points_aug (N + npoints_inj, 5);  cd (1)



#  Perturbation
# ==================================================
def perturb_attack(clean_points, channels=[0,1,2,3], eps_max=1):
    '''
    input: a clean point set,    # (N,4)
           channels and eps to perturb

    output: an augumented perturbed point set (with reliability score),      # (N,5)
            chamfer distance
    '''
    eps = random.uniform(0, eps_max)
    per_points = clean_points.copy()       # (N, 4)
    sigma = np.array([0.5, 0.5, 0.5, 1])          # [0.7221, 0.6430, 0.3123, 4.4498]

    noise = np.random.randn(clean_points.shape[0], len(channels)) 
    jitter = noise * sigma[channels] * eps
    per_points[:, channels] += jitter                # (N, 4)
    cd, _, d_SpS = Chamfer_Dist(clean_points, per_points)   # cd (1);  d_SpS (N)

    # caculate reliability score
    lam = 1
    ad_channel = np.exp(-lam * d_SpS).reshape(-1, 1)        # (N)

    per_points_aug = np.concatenate([per_points, ad_channel], axis=1)

    return per_points_aug, cd

def perturb_partial_attack(clean_points, channels=[0,1,2,3], eps_max=1.5):
    '''
    input: a clean point set,    # (N,4)
           channels and eps to perturb

    output: an augumented perturbed point set (with reliability score),      # (N,5)
            chamfer distance
    '''
    # eps = random.uniform(0, eps_max)
    eps = eps_max
    per_points = clean_points.copy()       # (N, 4)
    sigma = np.array([0.5, 0.5, 0.5, 1])          # [0.7221, 0.6430, 0.3123, 4.4498]

    noise = np.random.randn(clean_points.shape[0], len(channels)) 
    jitter = noise * sigma[channels] * eps
    N = clean_points.shape[0]
    mask = (np.random.rand(N, 1) < 0.6).astype(np.float32)
    per_points[:, channels] += jitter * mask                # (N, 4)

    cd, _, d_SpS = Chamfer_Dist(clean_points, per_points)   # cd (1);  d_SpS (N)

    # caculate reliability score
    lam = 1
    ad_channel = np.exp(-lam * d_SpS).reshape(-1, 1)        # (N)

    per_points_aug = np.concatenate([per_points, ad_channel], axis=1)

    return per_points_aug, cd

#   Removal
# ==================================================
def removal_attack(clean_points, max_dropout_ratio=0.6):
    '''
    input: a clean point set

    output: remove random points
    '''
    # dropout_ratio =  np.random.random()*max_dropout_ratio 
    drop_idx = np.where(np.random.random((clean_points.shape[0]))<=max_dropout_ratio)[0]
    drop_points = clean_points.copy() 
    if len(drop_idx)>0:
            drop_points[drop_idx,:] = clean_points[0,:] # set to the first point

    N = drop_points.shape[0]
    drop_points_aug = np.column_stack((drop_points, np.ones(N)))
    cd = -1 ################ wrong but anyway

    return drop_points_aug, cd

#   Scale attack
# ==================================================
def scale_attack(clean_points, scale_factor=2):
    '''
    return a scaled point set
    '''
    sca_points = clean_points.copy()
    sca_points = sca_points * scale_factor

    N = sca_points.shape[0]
    sca_points_aug = np.column_stack((sca_points, np.ones(N)))
    cd = -1 ################ wrong but anyway

    return sca_points_aug, cd


class ModelNetDataLoader_clean_per_inj(Dataset):
    def __init__(self, root, args, split='train', process_data=False, per_prob=0, inject_prob=0, removal_prob=0, scale_prob=0):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.num_channel = args.num_channel
        self.num_category = args.num_category
        # probs
        self.per_prob = per_prob
        self.inject_prob = inject_prob
        self.removal_prob = removal_prob
        self.scale_prob = scale_prob
        # param for injection attack
        self.npoints_inj = args.npoints_inj
        self.clutter_size_inj = args.clutter_size_inj
        # param for perturbation attack
        self.channels_per = args.channels_per
        self.eps_per = args.eps_per

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        elif self.num_category == 2:
            self.catfile = os.path.join(self.root, 'modelnet2_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        elif self.num_category == 2:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet2_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet2_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            point_set = np.atleast_2d(point_set)

            probs = [self.per_prob, self.inject_prob, self.removal_prob, self.scale_prob, 1-self.per_prob-self.inject_prob-self.removal_prob-self.scale_prob]
            idx = np.random.choice(len(probs), p=probs)
            if idx == 0:
                point_set_aug, cd = perturb_partial_attack(point_set, channels=self.channels_per, eps_max=self.eps_per)
            elif idx == 1:
                point_set_aug, cd = inject_attack_onoff_surface(point_set, npoints_inj=self.npoints_inj, clutter_size_inj=self.clutter_size_inj)
            elif idx == 2:
                point_set_aug, cd = removal_attack(point_set)
            elif idx == 3:
                point_set_aug, cd = scale_attack(point_set)
            elif idx == 4:
                point_set_aug, cd = no_attack(point_set)

            if point_set_aug.shape[0] < self.npoints:
                point_set_aug = repeat_point_upsample(point_set_aug, self.npoints)
            elif point_set_aug.shape[0] > self.npoints:
                point_set_aug = farthest_point_downsample(point_set_aug, self.npoints)

        return point_set_aug, label[0], cd     # point_set (N / N+npoints_inj, 5)
    
    def __getitem__(self, index):
        return self._get_item(index)

