'''
a customized S3DIS loader that gives clean / perturbed / injected samples
return 5 channels directly (x, y, z, v, a)
'a' is either 0 (clean) or 1 (injected), or in between for perturbed
'''

import os
import numpy as np
import random

from tqdm import tqdm
from torch.utils.data import Dataset

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
    sigma = np.array([0.5, 0.5, 0.5, 1])          

    noise = np.random.randn(clean_points.shape[0], len(channels)) 
    jitter = noise * sigma[channels] * eps
    per_points[:, channels] += jitter                # (N, 4)
    cd, _, d_SpS = Chamfer_Dist(clean_points, per_points)   # cd (1);  d_SpS (N)

    # caculate reliability score
    lam = 1
    ad_channel = np.exp(-lam * d_SpS).reshape(-1, 1)        # (N)

    per_points_aug = np.concatenate([per_points, ad_channel], axis=1)

    return per_points_aug, cd


class S3DISDataset_mix(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=1024, sample_rate=1.0, transform=None, samples_per_frame=4, num_classes=4, dim_input=4, per_prob=0, inj_prob=0):
        '''
        - MyS3DIS
            - Train_frame_016653.npy
              Train_frame_017753.npy
              ...
              Test_frame_016698.npy
              Test_frame_016699.npy
        '''
        super().__init__()
        self.num_point = num_point
        self.transform = transform
        self.per_prob = per_prob
        self.inj_prob = inj_prob
        frames = sorted(os.listdir(data_root))
        if split == 'train':
            file_list = [frame for frame in frames if 'Train' in frame]
        else:
            file_list = [frame for frame in frames if 'Test' in frame]

        self.frame_points, self.frame_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(num_classes) # 4

        for room_name in tqdm(file_list, total=len(file_list)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzvl, N*5
            points, labels = room_data[:, 0:dim_input], room_data[:, dim_input]  # xyzv, N*4; L, N
            tmp, _ = np.histogram(labels, range(num_classes+1))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.frame_points.append(points), self.frame_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        # higher importance to rare classes
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        frame_idxs = []
        for index in range(len(file_list)):
            frame_idxs.extend([index] * samples_per_frame)
        self.frame_idxs = np.array(frame_idxs)

        print("Totally {} samples in {} set.".format(len(self.frame_idxs), split))

    def __getitem__(self, idx):
        frame_idx = self.frame_idxs[idx]
        points = self.frame_points[frame_idx]   # N * 4
        labels = self.frame_labels[frame_idx]   # N
        N_points_ini = points.shape[0]

        # generate mix of samples according to probs
        probs = [self.per_prob, self.inj_prob, 1-self.per_prob-self.inj_prob]
        idx_mix = np.random.choice(len(probs), p=probs)
        if idx_mix == 0:
            print("perturb")
            points_aug, cd = perturb_attack(points, channels=[0,1,2,3], eps_max=1)      # points_aug (M,5); cd (1,)
        elif idx_mix == 1:
            print("inject but perturb")
            points_aug, cd = perturb_attack(points, channels=[0,1,2,3], eps_max=1)
        elif idx_mix == 2:
            points_aug, cd = no_attack(points)

        # random centering
        center = points_aug[np.random.choice(N_points_ini)][:3]
        point_idxs = np.arange(N_points_ini)
        points_aug[:, 0] = points_aug[:, 0] - center[0]
        points_aug[:, 1] = points_aug[:, 1] - center[1]

        if N_points_ini >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        selected_points = points_aug[selected_point_idxs, :]  # num_point * 5
        selected_labels = labels[selected_point_idxs]

        return selected_points, selected_labels, cd

    def __len__(self):
        return len(self.frame_idxs)

