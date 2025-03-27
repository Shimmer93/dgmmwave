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
            nn.Linear(dim, 3)
        )

    def forward(self, x):
        # B, N, C = x.shape
        x_new = self.enc(x)
        x_new = self.mixer(x_new)
        x_new = self.dec_point(x)
        conf = self.dec_conf(x)
        conf_ = F.sigmoid(conf).detach()
        x_new = x_new * (1 - conf_) + x * conf_
        return x_new, conf

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
    
class JointAttentionMemoryBank(nn.Module):
    def __init__(self, dim, mem_size_per_joint, num_joints):
        super().__init__()
        self.mem = nn.Parameter(torch.FloatTensor(1, dim, mem_size_per_joint * num_joints).normal_(0.0, 1.0))
        self.mem_size_per_joint = mem_size_per_joint
        self.num_joints = num_joints

    def forward(self, x, idx=-1):
        B, N, D = x.shape
        _, _, M = self.mem.shape

        if idx == -1:
            m = self.mem.repeat(B, 1, 1)
        else:
            m = self.mem[:, :, idx*self.mem_size_per_joint:(idx+1)*self.mem_size_per_joint].repeat(B, 1, 1)
        m = self.mem.repeat(B, 1, 1)
        m_key = m.transpose(1, 2)
        x_ = x.permute(0, 2, 1)
        logits = torch.bmm(m_key, x_) / sqrt(D)
        x_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        x_new_ = x_new.permute(0, 2, 1)

        return x_new_

# class PointModifier(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
        
#         self.enc = nn.Sequential(
#             nn.Linear(3, dim),
#             nn.LayerNorm(dim),
#             nn.GELU(),
#             nn.Linear(dim, dim)
#         )

#         self.


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
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 3)
        )

    def forward(self, x):
        return self.mlp_head(x).unsqueeze(1)

class JointMLPDecoder(nn.Module):
    def __init__(self, dim, mlp_dim, num_joints):
        super().__init__()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, mlp_dim),
        #     nn.GELU(),
        #     nn.Linear(mlp_dim, 3)
        # )
        self.heads = nn.ModuleList()

        for i in range(num_joints):
            self.heads.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, 3)
            ))

        self.num_joints = num_joints

    def forward(self, x):
        y = []
        for i in range(self.num_joints):
            y.append(self.heads[i](x[:, i, :]))
        y = torch.stack(y, dim=1).unsqueeze(1)
        return y
    
# class MLPDecoder(nn.Module):
#     def __init__(self, dim, mlp_dim, num_joints):
#         super().__init__()
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, num_joints*3)
#         )

#     def forward(self, x):
#         x = torch.max(input=x, dim=1, keepdim=False, out=None)[0]
#         x = self.mlp_head(x)
#         x = x.reshape(x.shape[0], 1, x.shape[-1]//3, 3)
#         return x

class VisDecoder(nn.Module):
    def __init__(self, dim, mlp_dim, num_joints, num_points):
        super().__init__()
        self.vis_cls = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1)
        )

        self.temp_attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1)
        )

        self.joint_attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints)
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 3)
        )

        self.lc_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_points)
        )

    def forward(self, feat, pcs):
        B, L, N, C = pcs.shape
        _, Ln, D = feat.shape

        feat_temp = feat.reshape(B, L, -1, D).permute(0, 2, 1, 3).reshape(-1, L, D)
        attn_temp = F.softmax(self.temp_attn(feat_temp), dim=1)
        feat = (feat_temp * attn_temp).sum(dim=1).reshape(B, -1, D)

        attn_joint = F.softmax(self.joint_attn(feat), dim=1).permute(0, 2, 1) # B, J, n
        feat_joint = torch.bmm(attn_joint, feat).reshape(B, -1, D) # B, J, D

        vis = self.vis_cls(feat_joint)
        bvis = F.sigmoid(vis) # B, J, 1

        y_invisible = self.mlp_head(feat_joint)
        y_visible = F.softmax(self.lc_head(feat_joint), dim=2) # B, J, N
        pc = pcs[:, -1, :, :3] # B, N, 3
        # print(pc.shape, y_visible.shape)
        y_visible = torch.bmm(y_visible, pc) # B, J, 3

        y_final = y_invisible * (1 - bvis) + y_visible * bvis
        return y_final, vis

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

class LMA2_P4T(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 dim_proposal, heads_proposal, dim_head_proposal, num_proposal,
                 mlp_dim, num_joints, mem_size_per_joint, num_points, depth_joint):   # output
        super().__init__()

        # self.mod = 

        self.enc = P4ConvEncoder(radius, nsamples, spatial_stride, 
                                     temporal_kernel_size, temporal_stride, 
                                     emb_relu, dim)
        
        self.mixer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.mem_lidar = JointAttentionMemoryBank(dim, mem_size_per_joint, num_joints)
        self.mem_mmwave = JointAttentionMemoryBank(dim, mem_size_per_joint, num_joints)
        # self.joint_mixer = Transformer(dim, depth_joint, heads, dim_head, mlp_dim)
        self.dec_lidar = JointMLPDecoder(dim, mlp_dim, num_joints)
        self.dec_mmwave = JointMLPDecoder(dim, mlp_dim, num_joints)

        self.num_joints = num_joints

    def forward(self, input, mode='inference'):
        assert mode in ['inference', 'train']
        if mode == 'inference':
            return self.forward_inference(input)
        else:
            return self.forward_train(input)

    def forward_train(self, x):
        x_lidar, x_mmwave = x # B, L, N, 4
        # print(x_lidar.shape, x_mmwave.shape)
        # x_lidar = x_lidar.clone()[:, 1:, ...] #torch.cat([x_lidar[..., :-1, :], x_lidar.clone()[..., 1:, :]], dim=0)
        # x_mmwave = torch.cat([x_mmwave[:, :-1, ...], x_mmwave.clone()[:, 1:, ...]], dim=0)

        emb_lidar = self.enc(x_lidar)
        emb_mmwave = self.enc(x_mmwave)

        feat_lidar_ = self.mixer(emb_lidar) # B, Ln, D
        feat_mmwave = self.mixer(emb_mmwave)

        self.mem_mmwave.requires_grad_(True)
        # feat_mem_lidar = self.mem_lidar(feat_lidar)
        feat_mem_mmwave = self.mem_mmwave(feat_mmwave)
        # l_rec_lidar = F.mse_loss(feat_mem_lidar, feat_lidar)
        l_rec_mmwave = F.mse_loss(feat_mem_mmwave, feat_mmwave)
        
        # self.mem.requires_grad_(False)

        feat_lidar = feat_lidar_.unsqueeze(0).repeat(self.num_joints, 1, 1, 1)
        feat_mem_lidar = []
        for i in range(self.num_joints):
            feat_mem_lidar.append(self.mem_lidar(feat_lidar[i], idx=i))
        feat_mem_lidar = torch.stack(feat_mem_lidar, dim=1)
        feat_mem_lidar = torch.max(input=feat_mem_lidar, dim=2, keepdim=False, out=None)[0]

        self.mem_mmwave.requires_grad_(False)
        feat_transferred = feat_lidar_.clone().unsqueeze(0).repeat(self.num_joints, 1, 1, 1)
        feat_mem_transferred = []
        for i in range(self.num_joints):
            feat_mem_transferred.append(self.mem_mmwave(feat_transferred[i], idx=i))
        feat_mem_transferred = torch.stack(feat_mem_transferred, dim=1) # B, J, Ln, D
        feat_mem_transferred = torch.max(input=feat_mem_transferred, dim=2, keepdim=False, out=None)[0]
        
        # feat_mem_transferred = self.joint_mixer(feat_mem_transferred)
        
        y_lidar = self.dec_lidar(feat_mem_lidar)
        y_transferred = self.dec_mmwave(feat_mem_transferred)

        return y_lidar, y_transferred, l_rec_mmwave
    
    def forward_inference(self, x):
        emb = self.enc(x)
        feat = self.mixer(emb)
        # feat_mem = self.mem_mmwave(feat)
        feat_mem_new = []
        for i in range(self.num_joints):
            feat_mem_new.append(self.mem_mmwave(feat, idx=i))
        feat_mem_new = torch.stack(feat_mem_new, dim=1)
        feat_mem_new = torch.max(input=feat_mem_new, dim=2, keepdim=False, out=None)[0]
        # feat_mem_new = self.joint_mixer(feat_mem_new)
        y = self.dec_mmwave(feat_mem_new)
        return y




        