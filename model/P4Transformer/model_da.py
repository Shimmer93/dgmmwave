import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
import os
from math import sqrt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from .point_4d_convolution import *
from .transformer import *
# from torchvision.models import resnet18


class P4TransformerDA(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, mem_size, features=3):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mem = nn.Parameter(torch.FloatTensor(1, dim, mem_size).normal_(0.0, 1.0))

        self.deconv = P4DTransConv(in_planes=dim, mlp_planes=[dim], mlp_activation=[True], mlp_batch_norm=[True], original_planes=features)
        self.seg_head = nn.Conv2d(in_channels=dim, out_channels=output_dim//3, kernel_size=1, stride=1, padding=0)
        self.pc_head = nn.Conv2d(in_channels=dim, out_channels=dim//8, kernel_size=1, stride=1, padding=0)

        self.skl_head = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Linear(dim, mlp_dim),
            # nn.GELU(),
            nn.Linear(dim, output_dim//3 * dim//8),
        )

        self.num_joints = output_dim//3
        self.final_dim = dim//8

        self.final_attn = QueryAttention(self.final_dim, heads=8, dim_head=64, dropout=0.1)
        self.final_head = nn.Sequential(
            nn.LayerNorm(self.final_dim),
            nn.Linear(self.final_dim, self.final_dim),
            nn.GELU(),
            nn.Linear(self.final_dim, 3),
        )

    def forward_mem(self, y):
        b, l, d = y.shape
        m = self.mem.repeat(b, 1, 1)
        m_key = m.transpose(1, 2)
        y_ = y.permute(0, 2, 1)
        logits = torch.bmm(m_key, y_) / sqrt(d)
        y_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        y_new_ = y_new.permute(0, 2, 1)

        return y_new_

    def forward(self, input, update_memory=True):                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs0, features0 = self.tube_embedding(input[:,:,:,:3], input[:,:,:,3:].permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

        B, L, _, N = features0.size()

        xyzts = []
        xyzs = torch.split(tensor=xyzs0, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features0.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        if update_memory:       
            output_rec = self.forward_mem(output)
        else:
            with torch.no_grad():
                self.mem.requires_grad = False
                output_rec = self.forward_mem(output)
                self.mem.requires_grad = True
        loss_rec = F.mse_loss(output_rec, output)

        # torch.Size([100, 7x16, 1024])
        output_feat = torch.reshape(input=output_rec, shape=(B, L, N, output_rec.shape[2]))
        output_feat = output_feat.permute(0, 1, 3, 2)
        # print(xyzs0.shape, input[:,:,:,:3].shape, output_seg.shape, features0.shape)
        # torch.Size([100, 7, 16, 3]) torch.Size([100, 7, 128, 3]) torch.Size([100, 7, 1024, 16]) torch.Size([100, 7, 1024, 16])
        _, output_feat = self.deconv(xyzs0, input[:,:,:,:3], output_feat.float(), input[:,:,:,3:].permute(0,1,3,2).float())
        output_seg = self.seg_head(output_feat.transpose(1,2)).transpose(1,2)
        # print('output_seg: ', output_seg.size())
        # torch.Size([128, 7, 17, 128])
        output_pc = self.pc_head(output_feat.transpose(1,2)).transpose(1,2) # B L D N
        output_pc = output_pc[:, (output_pc.shape[1]-1)//2, ...] # B D N
        # output_pc = torch.max(input=output_pc, dim=1, keepdim=False, out=None)[0] # B N D
        output_pc = output_pc.transpose(1, 2) # B N D
        # print('output_pc: ', output_pc.size())

        output_skl = torch.max(input=output_rec, dim=1, keepdim=False, out=None)[0]
        output_skl = self.skl_head(output_skl)
        # print('output_skl: ', output_skl.size())
        output_skl = output_skl.reshape(output_skl.shape[0], self.num_joints, self.final_dim) # B J D

        output_skl = self.final_attn(output_skl, output_pc)
        output_skl = self.final_head(output_skl)
        output_skl = torch.unsqueeze(input=output_skl, dim=1)

        return output_skl, output_seg, loss_rec
