import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class EntropyLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(EntropyLoss, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, ypred):
        entropy = -torch.sum(ypred * torch.log(ypred + self.eps), dim=-1)
        if self.reduction == 'mean':
            return torch.mean(entropy)
        if self.reduction == 'batchmean':
            return torch.mean(entropy)

        else:
            return entropy
        
class ClassLogitContrastiveLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(ClassLogitContrastiveLoss, self).__init__()

        self.reduction = reduction

    def forward(self, ypred, xyz):
        # ypred: B L J N, xyz: B L N 3
        # B: batch size
        # L: number of frames
        # J: number of joints
        # N: number of points in the point cloud
        # minimize the distance between the class logits of the same joint in different frames
        # and maximize the distance between the class logits of different joints in the same frame
        B, L, J, N = ypred.shape
        xyz_ = xyz.reshape(B*L, N, 3)
        dist_xyz = torch.cdist(xyz_, xyz_, p=2)
        dist_xyz_max_idx = torch.max(dist_xyz, dim=2, keepdim=True)[1]
        dist_xyz[dist_xyz == 0] = 1e6
        dist_xyz_min_idx = torch.min(dist_xyz, dim=2, keepdim=True)[1]
        # dist_xyz_zero_idx = torch.where(dist_xyz == 0)
        # dist_xyz_min_idx = 

        ypred_ = ypred.permute(0, 1, 3, 2).reshape(B*L, N, J)
        ypred_sim = torch.bmm(ypred_, ypred_.permute(0, 2, 1))
        ypred_sim_at_max = torch.gather(ypred_sim, dim=2, index=dist_xyz_max_idx).squeeze(2)
        ypred_sim_at_min = torch.gather(ypred_sim, dim=2, index=dist_xyz_min_idx).squeeze(2)

        return torch.mean(ypred_sim_at_max - ypred_sim_at_min)
        # if self.reduction == 'mean':
        # if self.reduction == 'batchmean':
        #     return torch.mean(torch.sum(ypred_sim_at_max, dim=1) - torch.sum(ypred_sim_at_min, dim=1))

if __name__ == '__main__':
    ypred = torch.randn(16, 7, 17, 1024)
    xyz = torch.randn(16, 7, 1024, 3)

    l = ClassLogitContrastiveLoss(reduction='mean')
    loss = l(ypred, xyz)
    print(loss)