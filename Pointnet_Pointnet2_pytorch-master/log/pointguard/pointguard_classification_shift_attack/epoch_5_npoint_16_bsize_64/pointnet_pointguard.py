import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, num_channels=4):
        super(get_model, self).__init__()
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=num_channels)
        # if global_feat == False, return 
        # global feature(copy) + local per-point feature
        # (B,1024,N) + (B,64,N)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, 1, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]                 # x: (B,4,N)
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)     # (B,1088,N)
        x = F.relu(self.bn1(self.conv1(x)))     # (B,512,N)
        x = F.relu(self.bn2(self.conv2(x)))     # (B,256,N)
        x = F.relu(self.bn3(self.conv3(x)))     # (B,128,N)
        x = F.relu(self.bn4(self.conv4(x)))     # (B,64,N)
        x = torch.sigmoid(self.conv5(x))        # (B,1,N) values between 0 and 1
        x = x.squeeze(1)                        # (B, N)
        return x, trans_feat                    # x:(B,N)   trans_feat:(B,64,64)

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