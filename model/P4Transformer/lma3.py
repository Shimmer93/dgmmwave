import torch
import sys 
import os
import torch.nn.functional as F
from math import sqrt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from .point_4d_convolution import *
from .transformer import *

class QueueMemoryBank(nn.Module):
    def __init__(self, dim, mem_size):
        super().__init__()
        self.mem = nn.Parameter(torch.FloatTensor(mem_size, dim).normal_(0.0, 1.0))
        self.mem.requires_grad_(False)
        self.mem_size = mem_size

    def forward(self, x, use_mem=True):
        x = x.squeeze(1)
        B, D = x.size()
        if self.training:
            self.mem = torch.cat([self.mem, x.clone().detach()], dim=0)[B:, ...]

        if use_mem:
            m = self.mem.unsqueeze(0).repeat(B, 1, 1)
            m_key = m
            x_ = x.unsqueeze(1)
            logits = torch.bmm(m_key, x_.transpose(1, 2))
            x_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        else:
            x_new = x.unsqueeze(1)

        return x_new
        

    # def forward(self, x):
    #     # x: [B, 1, D] a batch of input vectors
    #     # we want each memory vector to be as independent to each other as possible
    #     # we compute the correlation matrix between the memory vectors
    #     # find the memory vector that has the most correlation with the memory bank
    #     # then we remove that memory vector from the memory bank
    #     # and add the input vector to the memory bank
    #     # if the new memory bank has lower correlation than the previous one, we keep the new memory bank
    #     # otherwise we keep the previous memory bank

    #     x = x.squeeze(1)
    #     if self.training:
    #         for i in range(x.size(0)):
    #             mem_norm = self.mem - self.mem.mean(dim=0, keepdim=True)
    #             mem_cov = mem_norm.T @ mem_norm
    #             mem_corr = mem_cov / (self.mem_size - 1)

    #             corr_sums = torch.sum(mem_corr, dim=1)
    #             max_corr_idx = torch.argmax(corr_sums)

    #             new_mem = self.mem.clone()
    #             new_mem[max_corr_idx] = x[i].clone()

    #             new_mem_norm = new_mem - new_mem.mean(dim=0, keepdim=True)
    #             new_mem_cov = new_mem_norm.T @ new_mem_norm
    #             new_mem_corr = new_mem_cov / (self.mem_size - 1)

    #             new_corr_sums = torch.sum(new_mem_corr, dim=1)

    #             if torch.sum(new_corr_sums) < torch.sum(corr_sums):
    #                 self.mem.data = new_mem.data

    #     m = self.mem.unsqueeze(0).repeat(x.size(0), 1, 1)
    #     m_key = m
    #     x_ = x.unsqueeze(1)
    #     logits = torch.bmm(m_key, x_.transpose(1, 2)) / sqrt(x.size(-1))
    #     x_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))

    #     return x_new
        

class AttentionMemoryBank(nn.Module):
    def __init__(self, dim, mem_size):
        super().__init__()
        self.mem = nn.Parameter(torch.FloatTensor(1, dim, mem_size).normal_(0.0, 1.0))

    def forward(self, x):
        B, N, D = x.shape
        _, _, M = self.mem.shape

        m = self.mem.repeat(B, 1, 1)
        m_key = m.transpose(1, 2)
        x_ = x.permute(0, 2, 1)
        logits = torch.bmm(m_key, x_) / sqrt(D)
        x_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        x_new_ = x_new.permute(0, 2, 1)

        return x_new_

class P4ConvEncoder(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu, dim):
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=3, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

    def forward(self, x):
        device = x.get_device()
        xyzs0, features = self.tube_embedding(x[:,:,:,:3], x[:,:,:,:3].clone().permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs0, split_size_or_sections=1, dim=1)
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

        return embedding
    
class MLPDecoder(nn.Module):
    def __init__(self, dim, mlp_dim, out_dim):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp_head(x)
        # x = x.reshape(x.shape[0], 1, x.shape[-1]//3, 3)
        return x
    
class AttentionAggregator(nn.Module):
    # input: [B, N, D]
    # output: [B, D]
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, N, D]
        # out: [B, D]
        attn = F.softmax(self.proj(x), dim=1)
        out = torch.sum(attn * x, dim=1)
        return out
    
class EmbAggregator(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.emb = nn.Parameter(torch.FloatTensor(1, dim).normal_(0.0, 1.0))
        self.attn = Attention(dim, heads, dim_head)
        self.ff = FeedForward(dim, dim*2)

    def forward(self, x):
        B, N, D = x.shape
        emb = self.emb.repeat(B, 1, 1)
        x = torch.cat([emb, x], dim=1)
        x = self.attn(x)[:, 0:1]
        x = self.ff(x)
        return x
    
class LMA3_P4T(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 dim_proposal, heads_proposal, dim_head_proposal, num_proposal,
                 mlp_dim, num_joints, mem_size, num_points):   # output
        super().__init__()

        self.enc = P4ConvEncoder(radius, nsamples, spatial_stride, 
                                     temporal_kernel_size, temporal_stride, 
                                     emb_relu, dim)
        
        self.mixer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.agg_pose = EmbAggregator(dim, heads, dim_head)
        self.mem_pose = QueueMemoryBank(dim, mem_size)
        self.dec_pose = MLPDecoder(dim, mlp_dim, num_joints*3)

        self.agg_loc = EmbAggregator(dim, heads, dim_head)
        self.dec_loc = MLPDecoder(dim, mlp_dim, 3)

    def forward(self, input, mode='inference'):
        assert mode in ['inference', 'train']
        if mode == 'inference':
            return self.forward_inference(input)
        else:
            return self.forward_train(input)
        
    def forward_inference(self, x):
        emb = self.enc(x)
        feat = self.mixer(emb)
        feat_pose = self.agg_pose(feat)
        feat_pose_mem = self.mem_pose(feat_pose)
        y_pose = self.dec_pose(feat_pose_mem)
        feat_loc = self.agg_loc(feat)
        y_loc = self.dec_loc(feat_loc)
        y_pose = y_pose.reshape(y_pose.shape[0], 1, y_pose.shape[-1]//3, 3)
        y_loc = y_loc.reshape(y_loc.shape[0], 1, 1, 3)
        y = y_pose + y_loc
        return y

    def forward_debug(self, start_idx, end_idx):
        m = self.mem_pose.mem
        m = m[:, :, start_idx:end_idx].permute(2, 0, 1)
        y_pose = self.dec_pose(m)
        y_pose = y_pose.reshape(y_pose.shape[0], 1, y_pose.shape[-1]//3, 3)
        return y_pose

    def forward_train(self, x):
        x_lidar, x_mmwave = x

        emb_lidar = self.enc(x_lidar)
        emb_mmwave = self.enc(x_mmwave)

        feat_lidar = self.mixer(emb_lidar)
        feat_mmwave = self.mixer(emb_mmwave)

        feat_lidar_pose = self.agg_pose(feat_lidar)
        feat_lidar_pose_mem = self.mem_pose(feat_lidar_pose)
        feat_mmwave_pose = self.agg_pose(feat_mmwave)
        feat_mmwave_pose_mem = self.mem_pose(feat_mmwave_pose)
        l_rec_lidar = F.mse_loss(feat_lidar_pose_mem, feat_lidar_pose.detach())
        l_rec_mmwave = F.mse_loss(feat_mmwave_pose_mem, feat_mmwave_pose.detach())

        feat_lidar_loc = self.agg_loc(feat_lidar)

        l_ortho = torch.square((feat_lidar_pose.squeeze(1) @ feat_lidar_loc.squeeze(1).T)).sum()

        y_pose_lidar = self.dec_pose(feat_lidar_pose_mem)
        y_loc_lidar = self.dec_loc(feat_lidar_loc)

        y_pose_lidar = y_pose_lidar.reshape(y_pose_lidar.shape[0], 1, y_pose_lidar.shape[-1]//3, 3)
        y_loc_lidar = y_loc_lidar.reshape(y_loc_lidar.shape[0], 1, 1, 3)

        return y_pose_lidar, y_loc_lidar, l_rec_lidar, l_rec_mmwave, l_ortho