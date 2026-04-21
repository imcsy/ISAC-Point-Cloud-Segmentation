'''
a data loader for point injection attacks
return 5 channels directly (x, y, z, v, a)
a is either 0 (clean) or 1 (injected)
'''
import os
import numpy as np
import warnings
import pickle

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
    S: (N, 4)
    Sp: (N + npoints_inj, 4)
    return: weighted chamfer distance
    """
    # pairwise squared distances
    diff = S[:, None, :] - Sp[None, :, :]   # (N, N + npoints_inj, D)
    dist2 = np.sum((diff ** 2) * weights, axis=2)      # (N, N + npoints_inj)

    # nearest neighbor distances
    d_SSp = np.min(dist2, axis=0)  # (N + npoints_inj)

    cd = np.mean(d_SSp)
    return cd

def inject_attack(clean_points, npoints_inj, clutter_size_inj, is_attack=True):
    '''
    input: a clean point set, 
           npoints_inj: number of points injected
           clutter_size_inj: the approximate number od points for the injected clutter

    output: an augumented injected point set (with reliability score),
            chamfer distance
    '''
    N = clean_points.shape[0]
    inj_points_aug = np.column_stack((clean_points, np.ones(N)))

    # if no attack
    if not is_attack:
        return inj_points_aug, 0

    # if attack
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
        
    cd = Chamfer_Dist(clean_points, inj_points_aug[:, :4])
    return inj_points_aug, cd       # inj_points_aug (N + npoints_inj, 5);  cd (1)

class ModelNetDataLoader_Injection(Dataset):
    def __init__(self, root, args, split='train', process_data=False, inject_prob=0.9):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.num_channel = args.num_channel
        self.num_category = args.num_category
        self.inject_prob = inject_prob
        self.npoints_inj = args.npoints_inj
        self.clutter_size_inj = args.clutter_size_inj

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
            
            if np.random.rand() < self.inject_prob:
                point_set_aug, cd = inject_attack(point_set, npoints_inj=self.npoints_inj, clutter_size_inj=self.clutter_size_inj, is_attack=True)
            else: 
                point_set_aug, cd = inject_attack(point_set, 0, 0, is_attack=False)

            if point_set_aug.shape[0] < self.npoints:
                point_set_aug = repeat_point_upsample(point_set_aug, self.npoints)
            elif point_set_aug.shape[0] > self.npoints:
                point_set_aug = farthest_point_downsample(point_set_aug, self.npoints)

        return point_set_aug, label[0], cd     # point_set (N / N+npoints_inj, 5)
    def __getitem__(self, index):
        return self._get_item(index)

