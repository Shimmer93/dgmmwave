import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class GeodesicLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(GeodesicLoss, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,m1,m2):
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, m1.new(np.ones(batch)))
        cos = torch.max(cos, m1.new(np.ones(batch)) * -1)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            # breakpoint()
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))

        else:
            return theta
        
class SymmetryLoss(nn.Module):
    def __init__(self, left_bones, right_bones, weights=None, reduction='batchmean'):
        super(SymmetryLoss, self).__init__()

        self.reduction = reduction
        self.left_bones = left_bones
        self.right_bones = right_bones
        self.weights = weights

    def forward(self, ypred):
        if self.weights is None:
            self.weights = torch.ones(ypred.shape[-2], device=ypred.device)

        loss = 0
        for i, (l, r) in enumerate(zip(self.left_bones, self.right_bones)):
            l_dist_sq = torch.sqrt(((ypred[..., l[0], :] - ypred[..., l[1], :]) ** 2).sum(dim=-1))
            r_dist_sq = torch.sqrt(((ypred[..., r[0], :] - ypred[..., r[1], :]) ** 2).sum(dim=-1))
            loss += self.weights[i] * (l_dist_sq - r_dist_sq) ** 2

        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'batchmean':
            return torch.mean(loss)
        
class ReferenceBoneLoss(nn.Module):
    def __init__(self, bones, threshold=0, weights=None, reduction='batchmean'):
        super(ReferenceBoneLoss, self).__init__()

        self.reduction = reduction
        self.bones = bones
        self.threshold = threshold
        self.weights = weights

    def forward(self, ypred, yref):
        # print(ypred.shape, yref.shape)
        if self.weights is None:
            self.weights = torch.ones(ypred.shape[-2], device=ypred.device)

        # print(self.weights.shape)

        loss = 0
        for i, b in enumerate(self.bones):
            dist_sq = ((ypred[..., b[0], :] - ypred[..., b[1], :]) ** 2).sum(dim=-1)
            ref_dist_sq = ((yref[..., b[0], :] - yref[..., b[1], :]) ** 2).sum(dim=-1)
            loss += self.weights[i] * (dist_sq - ref_dist_sq) ** 2

        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'batchmean':
            return torch.mean(loss)