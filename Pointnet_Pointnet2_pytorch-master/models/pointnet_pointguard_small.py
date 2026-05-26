import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder_small, feature_transform_reguliarzer

'''
scorer
'''
class get_model(nn.Module):
    def __init__(self, num_channels=4):
        super(get_model, self).__init__()
        self.feat = PointNetEncoder_small(global_feat=False, channel=num_channels)
        self.conv1 = torch.nn.Conv1d(160, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = torch.nn.Conv1d(64, 1, 1)

    def forward(self, x):
        batchsize = x.size()[0]                 # x: (B,4,N)
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)     # (B,160,N)
        x = F.relu(self.bn1(self.conv1(x)))     # (B,128,N)
        x = F.relu(self.bn2(self.conv2(x)))     # (B,64,N)
        latent_64d = x

        x = torch.sigmoid(self.conv3(x))        # (B,1,N) values between 0 and 1
        x = x.squeeze(1)                        # (B, N)
        return x, trans_feat, latent_64d        # x:(B,N)    latent_64d:(B,64,N)

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # pred: (B,N)   target: (B,N)
        mse_loss = F.mse_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = mse_loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss