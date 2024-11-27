import torch
import sys 
import os
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from .point_4d_convolution import *
from .transformer import *
# from torchvision.models import resnet18

class CorrelationDiscriminator(nn.Module):
    def __init__(self, num_points, dim):
        super().__init__()

        # self.mlp1 = nn.Sequential(
        #     nn.Linear(num_points, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU(),
        #     nn.Linear(dim, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )

        self.transformer = Transformer(dim, depth=1, heads=8, dim_head=dim//8, mlp_dim=dim*2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 1)
        )

    def corr(self, x):
        B, N, D = x.shape
        y = x @ x.transpose(1, 2) / sqrt(N)
        return y

    def forward(self, x):
        # corr = self.corr(x)
        # y = self.mlp1(corr)
        y = self.transformer(x)
        y = self.avg_pool(y.permute(0, 2, 1)).squeeze(-1)
        y = self.mlp2(y)

        return y

class P4TransformerDA7(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_joints=13, features=3, mem_size=1024):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mem = nn.Parameter(torch.FloatTensor(1, dim, mem_size).normal_(0.0, 1.0))
        self.disc = CorrelationDiscriminator(num_points=256//8*5, dim=dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints*3),
        )

    def forward_mem(self, x):
        B, N, D = x.shape
        _, _, M = self.mem.shape

        m = self.mem.repeat(B, 1, 1)
        m_key = m.transpose(1, 2)
        x_ = x.permute(0, 2, 1)
        logits = torch.bmm(m_key, x_) / sqrt(D)
        x_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        x_new_ = x_new.permute(0, 2, 1)

        return x_new_

    def forward(self, input, mode='inference'):     
        assert mode in ['train', 'adapt', 'inference']                                                                                                # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input[:,:,:,:3], input[:,:,:,3:].permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

        # print('xyzs: ', xyzs.max().item(), xyzs.min().item())
        # print('features: ', features.max().item(), features.min().item())

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output0 = self.transformer(embedding)
        # print('output after transformer: ', output.max().item(), output.min().item())

        if mode == 'adapt':
            # with torch.no_grad():
            self.mem.requires_grad_(False)
            output = self.forward_mem(output0)
        else:
            output = self.forward_mem(output0)

        l_rec = F.mse_loss(output, output0)

        if mode == 'adapt':
            # with torch.no_grad():
            self.disc.requires_grad_(False)
            domain = self.disc(output)
        else:
            domain = self.disc(output)

        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        if mode == 'adapt':
            self.mlp_head.requires_grad_(False)
        output = self.mlp_head(output)
        output = output.reshape(output.shape[0], 1, output.shape[-1]//3, 3) # B 1 J 3
        # print('output after mlp_head: ', output.max().item(), output.min().item())
        if mode == 'train':
            return output, domain, l_rec
        elif mode == 'adapt':
            return output, domain, l_rec
        else:
            return output
