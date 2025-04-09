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

class SpatialAttention(Attention):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__(dim, heads, dim_head, dropout=dropout)

    def forward(self, x):
        # x: B T J D
        # output: B T J D
        B, T, J, D = x.shape
        x = x.reshape(B*T, J, D)
        y = super().forward(x)
        y = y.reshape(B, T, J, D)
        return y
    
class TemporalAttention(Attention):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__(dim, heads, dim_head, dropout=dropout)

    def forward(self, x):
        # x: B T J D
        # output: B T J D
        B, T, J, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*J, T, D)
        y = super().forward(x)
        y = y.reshape(B, J, T, D).permute(0, 2, 1, 3)
        return y

class SpatialTemporalTransformer(nn.Module):
    def __init__(self, depth, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, SpatialAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                Residual(PreNorm(dim, TemporalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        # x: B T J D
        # output: B T J D
        for spatial, mlp1, temporal, mlp2 in self.layers:
            x = spatial(x)
            x = mlp1(x)
            x = temporal(x)
            x = mlp2(x)
        return x

class SpatialTemporalJointTransformer(nn.Module):
    def __init__(self, seq_len, num_joints, depth, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.time_emb = nn.Parameter(torch.randn(1, seq_len, 1, dim))
        self.joint_emb = nn.Parameter(torch.randn(1, 1, num_joints, dim))
        self.encoder = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        # self.pos_encoder = nn.Sequential(
        #     nn.Linear(3, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        self.transformer = SpatialTemporalTransformer(depth, dim, heads, dim_head, mlp_dim, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 3)
        )

    def forward(self, input):
        # input: B T J 3 (B: batch size, T: number of frames, J: number of joints, 3: x, y, z)
        # pos: B T 1 3
        # output: B T J 3
        B, T, J, _ = input.shape
        x = self.encoder(input)
        x = x + self.time_emb
        x = x + self.joint_emb
        # pos = self.pos_encoder(pos)
        # x = torch.cat([x, pos], dim=2)
        x = self.transformer(x)
        x = self.decoder(x)
        return x

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
        return x        

class PCAdapter(nn.Module):
    def __init__(self, dim, depth, heads, dim_head):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.mixer = Transformer(dim, depth, heads, dim_head, dim * 2, dropout=0.1)

        self.dec_point = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3)
        )

        self.dec_conf = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        B, L, N, C = x.shape
        x = x.reshape(B*L, N, C)
        x_new = self.enc(x)
        x_new = self.mixer(x_new)
        conf = self.dec_conf(x_new)
        x_new = self.dec_point(x_new)
        conf_ = F.sigmoid(conf).detach()
        x_new = x_new * (1 - conf_) + x * conf_
        x_new = x_new.reshape(B, L, N, C)
        conf = conf.reshape(B, L, N, 1)
        return x_new, conf

class P4TransformerFlowDA(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, mem_size, features=3):                                                 # output
        super().__init__()

        self.mod = PCAdapter(dim//8, 3, heads, dim_head//8)

        self.enc = P4ConvEncoder(radius, nsamples, spatial_stride, temporal_kernel_size, 
                                 temporal_stride, emb_relu, dim)
        
        self.mixer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mem_lidar = AttentionMemoryBank(dim, mem_size)
        self.mem_mmwave = AttentionMemoryBank(dim, mem_size)

        self.flow_dec_lidar = MLPDecoder(dim, mlp_dim, output_dim)
        self.loc_dec_lidar = MLPDecoder(dim, mlp_dim, 3)
        # self.pose_dec_lidar = MLPDecoder(dim, mlp_dim, output_dim)
        
        self.flow_dec_mmwave = MLPDecoder(dim, mlp_dim, output_dim)
        self.loc_dec_mmwave = MLPDecoder(dim, mlp_dim, 3)
        # self.pose_dec_mmwave = MLPDecoder(dim, mlp_dim, output_dim)

    def forward(self, input, mode='inference'):
        assert mode in ['inference', 'train']
        if mode == 'inference':
            return self.forward_inference(input)
        else:
            return self.forward_train(input)
        
    def forward_train(self, x):
        x_lidar, x_mmwave = x
        B, T, N, C = x_lidar.shape

        x_lidar[..., :3], conf_lidar = self.mod(x_lidar[..., :3])
        x_mmwave[..., :3], _ = self.mod(x_mmwave[..., :3])

        l_conf = F.binary_cross_entropy_with_logits(conf_lidar[x_lidar[...,-1:] > 0], torch.ones_like(conf_lidar[x_lidar[...,-1:] > 0])) + \
                    F.binary_cross_entropy_with_logits(conf_lidar[x_lidar[...,-1:] == 0], torch.zeros_like(conf_lidar[x_lidar[...,-1:] == 0]))

        x_lidar = x_lidar[..., :3]
        x_mmwave = x_mmwave[..., :3]

        emb_lidar = self.enc(x_lidar)
        emb_mmwave = self.enc(x_mmwave)

        feat_lidar = self.mixer(emb_lidar)
        feat_mmwave = self.mixer(emb_mmwave)

        self.mem_lidar.requires_grad_(True)
        self.mem_mmwave.requires_grad_(True)
        feat_mem_lidar = self.mem_lidar(feat_lidar)
        feat_mem_mmwave = self.mem_mmwave(feat_mmwave)
        l_rec_lidar = F.mse_loss(feat_mem_lidar, feat_lidar)
        l_rec_mmwave = F.mse_loss(feat_mem_mmwave, feat_mmwave)

        self.mem_lidar.requires_grad_(False)
        self.mem_mmwave.requires_grad_(False)
        feat_mem_trans = self.mem_mmwave(feat_lidar.clone())

        feat_mem_lidar = feat_mem_lidar.reshape(B, T, -1, feat_mem_lidar.shape[-1])
        feat_mem_lidar = torch.max(input=feat_mem_lidar, dim=2, keepdim=False, out=None)[0]
        feat_mem_lidar0 = feat_mem_lidar[:, 0:1, ...] # B D

        feat_mem_trans = feat_mem_trans.reshape(B, T, -1, feat_mem_trans.shape[-1])
        feat_mem_trans = torch.max(input=feat_mem_trans, dim=2, keepdim=False, out=None)[0]
        feat_mem_trans0 = feat_mem_trans[:, 0:1, ...] # B D

        flow_lidar = self.flow_dec_lidar(feat_mem_lidar)
        flow_lidar = flow_lidar.reshape(B, T, flow_lidar.shape[-1]//3, 3) # B 1 J 3
        loc_lidar = self.loc_dec_lidar(feat_mem_lidar0)
        loc_lidar = loc_lidar.reshape(B, 1, 1, 3) # B 1 1 3
        # pose_lidar = self.pose_dec_lidar(feat_mem_lidar0)
        # pose_lidar = pose_lidar.reshape(B, 1, pose_lidar.shape[-1]//3, 3) # B 1 J 3

        flow_trans = self.flow_dec_mmwave(feat_mem_trans)
        flow_trans = flow_trans.reshape(B, T, flow_trans.shape[-1]//3, 3) # B 1 J 3
        loc_trans = self.loc_dec_mmwave(feat_mem_trans0)
        loc_trans = loc_trans.reshape(B, 1, 1, 3) # B 1 1 3
        # pose_trans = self.pose_dec_mmwave(feat_mem_trans0)
        # pose_trans = pose_trans.reshape(B, 1, pose_trans.shape[-1]//3, 3) # B 1 J 3

        return flow_lidar, loc_lidar, flow_trans, loc_trans, \
               l_rec_lidar, l_rec_mmwave, l_conf

    def forward_inference(self, x):
        B, T, N, C = x.shape
        x[..., :3], _ = self.mod(x[..., :3])
        x = x[..., :3]
        emb = self.enc(x)
        feat = self.mixer(emb)
        feat_mem = self.mem_mmwave(feat)

        feat_mem = feat_mem.reshape(B, T, -1, feat_mem.shape[-1])
        feat_mem = torch.max(input=feat_mem, dim=2, keepdim=False, out=None)[0]
        feat_mem0 = feat_mem[:, 0:1, ...] # B D

        flow = self.flow_dec_mmwave(feat_mem)
        flow = flow.reshape(B, T, flow.shape[-1]//3, 3) # B 1 J 3
        loc = self.loc_dec_mmwave(feat_mem0)
        loc = loc.reshape(B, 1, 1, 3) # B 1 1 3

        return flow, loc