import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder_PointGuard, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, num_channel=5):
        '''
        input: (B, 5, N)
        output: (B, N, 2)
        '''
        super(get_model, self).__init__()
        self.conv1 = nn.Conv1d(num_channel, 32, 1) 
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 16, 1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 2, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x): 
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))     # (B, 32, N)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))     # (B, 16, N)
        x = self.dropout(x)
        x = self.conv3(x)                       # (B, 2, N)
        x = x.transpose(2,1).contiguous()       # (B, N, 2)
        x = F.log_softmax(x.view(-1,2), dim=-1)
        x = x.view(batchsize, n_pts, 2)         # (B, N, 2)
        return x


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, weight):
        loss = F.nll_loss(pred, target,  weight = weight)

        return loss
