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
    def __init__(self, dim, mlp_dim, num_joints):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints*3)
        )

    def forward(self, x):
        x = torch.max(input=x, dim=1, keepdim=False, out=None)[0]
        x = self.mlp_head(x)
        x = x.reshape(x.shape[0], 1, x.shape[-1]//3, 3)
        return x

class VisDecoder(nn.Module):
    def __init__(self, dim, mlp_dim, num_joints):
        super().__init__()
        self.vis_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints)
        )
        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints*3)
        )
        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints*3)
        )

    def forward(self, x):
        x = torch.max(input=x, dim=1, keepdim=False, out=None)[0]
        vis = self.vis_head(x)
        vis = vis.reshape(vis.shape[0], 1, vis.shape[-1], 1)
        bvis = F.sigmoid(vis).detach()
        y1 = self.mlp_head1(x)
        y2 = self.mlp_head2(x)
        
        y1 = y1.reshape(y1.shape[0], 1, y1.shape[-1]//3, 3)
        y2 = y2.reshape(y2.shape[0], 1, y2.shape[-1]//3, 3)
        y = y1 * (1 - bvis) + y2 * bvis
        return y, vis

# class VisDecoder(nn.Module):
#     def __init__(self, dim, mlp_dim, num_joints, num_points):
#         super().__init__()
#         self.vis_cls = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, 1)
#         )

#         self.temp_attn = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, 1)
#         )

#         self.joint_attn = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, num_joints)
#         )

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, 3)
#         )

#         self.lc_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, num_points)
#         )

#     def forward(self, feat, pcs):
#         B, L, N, C = pcs.shape
#         _, Ln, D = feat.shape

#         feat_temp = feat.reshape(B, L, -1, D).permute(0, 2, 1, 3).reshape(-1, L, D)
#         attn_temp = F.softmax(self.temp_attn(feat_temp), dim=1)
#         feat = (feat_temp * attn_temp).sum(dim=1).reshape(B, -1, D)

#         attn_joint = F.softmax(self.joint_attn(feat), dim=1).permute(0, 2, 1) # B, J, n
#         feat_joint = torch.bmm(attn_joint, feat).reshape(B, -1, D) # B, J, D

#         vis = self.vis_cls(feat_joint)
#         bvis = F.sigmoid(vis).detach() # B, J, 1

#         y_invisible = self.mlp_head(feat_joint)
#         y_visible = F.softmax(self.lc_head(feat_joint), dim=2) # B, J, N
#         pc = pcs[:, -1, :, :3] # B, N, 3
#         # print(pc.shape, y_visible.shape)
#         y_visible = torch.bmm(y_visible, pc) # B, J, 3

#         y_final = y_invisible * (1 - bvis) + y_visible * bvis
#         return y_final, vis

    # def decode(self, output):
    #     output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
    #     output = self.mlp_head(output)
    #     output = output.reshape(output.shape[0], 1, output.shape[-1]//3, 3) # B 1 J 3
    #     return output
    
    # def forward(self, input):
    #     embedding = self.encode(input)
    #     output0 = self.transformer(embedding)
    #     output = self.mem(output0) # [B, L*n, C]
    #     output = self.decode(output)

    #     return output

class VisibilityClassifier(nn.Module):
    def __init__(self, dim, num_joints):
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_joints)
        )

    def forward(self, x):
        x = self.mlp_head(x)
        return x

class LMA_P4T(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 dim_proposal, heads_proposal, dim_head_proposal, num_proposal,
                 mlp_dim, num_joints, mem_size, num_points):   # output
        super().__init__()

        self.mod = PCAdapter(dim_proposal, 3, heads_proposal, dim_head_proposal)

        self.enc = P4ConvEncoder(radius, nsamples, spatial_stride, 
                                     temporal_kernel_size, temporal_stride, 
                                     emb_relu, dim)
        
        self.mixer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mem_lidar = AttentionMemoryBank(dim, mem_size)
        self.mem_mmwave = AttentionMemoryBank(dim, mem_size)

        # self.dec_lidar = MLPDecoder(dim, mlp_dim, num_joints)
        # self.dec_mmwave = MLPDecoder(dim, mlp_dim, num_joints)
        self.dec_lidar = VisDecoder(dim, mlp_dim, num_joints)
        self.dec_mmwave = VisDecoder(dim, mlp_dim, num_joints)

    def forward(self, input, mode='inference'):
        assert mode in ['inference', 'train']
        if mode == 'inference':
            return self.forward_inference(input)
        else:
            return self.forward_train(input)

    def forward_train(self, x):
        x_lidar, x_mmwave = x # B, L, N, 4
        x_lidar[..., :3], conf_lidar = self.mod(x_lidar[..., :3])
        x_mmwave[..., :3], _ = self.mod(x_mmwave[..., :3])

        l_conf = F.binary_cross_entropy_with_logits(conf_lidar[x_lidar[...,-1:] > 0], torch.ones_like(conf_lidar[x_lidar[...,-1:] > 0])) + \
                    F.binary_cross_entropy_with_logits(conf_lidar[x_lidar[...,-1:] == 0], torch.zeros_like(conf_lidar[x_lidar[...,-1:] == 0]))

        # print(x_lidar.shape, x_mmwave.shape)
        x_lidar = x_lidar.clone()[:, 1:, ...] #torch.cat([x_lidar[..., :-1, :], x_lidar.clone()[..., 1:, :]], dim=0)
        x_mmwave = torch.cat([x_mmwave[:, :-1, ...], x_mmwave.clone()[:, 1:, ...]], dim=0)

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
        feat_mem_transferred = self.mem_mmwave(feat_lidar.clone())

        y_lidar, vis_lidar = self.dec_lidar(feat_mem_lidar)
        y_mmwave, _ = self.dec_mmwave(feat_mem_mmwave)
        y_transferred, vis_transferred = self.dec_mmwave(feat_mem_transferred)

        # y_lidar = self.dec_lidar(feat_mem_lidar)
        # y_mmwave = self.dec_mmwave(feat_mem_mmwave)
        # y_transferred = self.dec_mmwave(feat_mem_transferred)

        return y_lidar, y_transferred, y_mmwave, \
               l_rec_lidar, l_rec_mmwave, l_conf, \
               vis_lidar, vis_transferred
    
    def forward_inference(self, x):
        x[..., :3], _ = self.mod(x[..., :3])
        x = x[:, 1:, ...]
        emb = self.enc(x)
        feat = self.mixer(emb)
        feat_mem = self.mem_mmwave(feat)
        # y = self.dec_mmwave(feat_mem)
        y, _ = self.dec_mmwave(feat_mem)
        return y




        