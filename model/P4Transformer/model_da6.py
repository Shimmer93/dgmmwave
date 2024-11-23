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
# from torchvision.models import resnet18


class JointAttention(nn.Module):
    def __init__(self, num_joints=13, dim=64):
        super(JointAttention, self).__init__()
        self.num_joints = num_joints
        self.dim = dim
        self.joint_emb = nn.Parameter(torch.randn(1, num_joints, dim))
        self.to_kv = nn.Linear(dim, dim*2)

    def forward(self, x):
        B, N, D = x.shape

        k, v = self.to_kv(x).chunk(2, dim=-1)

        joint_emb = self.joint_emb.expand(B, -1, -1) # [B, J, D]
        attn = k @ joint_emb.transpose(1, 2) # [B, N, J]
        attn = F.softmax(attn, dim=1) # [B, N, J]
        x = attn.transpose(1, 2) @ v # [B, J, D]

        return x
    
class MultiHeadJointAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, num_joints=13):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.joint_emb = nn.Parameter(torch.randn(1, num_joints, dim))
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        J = self.joint_emb.shape[1]

        q = self.joint_emb.expand(B, -1, -1)
        q = q.reshape(B, J, self.heads, self.dim_head).permute(0, 2, 1, 3)

        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: t.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3), kv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v).reshape(B, self.heads, J, self.dim_head)
        out = out.permute(0, 2, 1, 3).reshape(B, J, -1)
        out = self.to_out(out)

        return out

class PointProposer(nn.Module):
    def __init__(self, dim, heads, dim_head, features, num_proposal):
        super().__init__()
        
        self.m = nn.Sequential(
            nn.Linear(3+features, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            MultiHeadJointAttention(dim=dim, heads=heads, dim_head=dim_head, num_joints=num_proposal),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3+features)
        )

    def forward(self, x):
        return self.m(x)

class CorrelationDiscriminator(nn.Module):
    def __init__(self, num_points, dim):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(num_points, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

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
        corr = self.corr(x)
        y = self.mlp1(corr)
        y = self.transformer(y)
        y = self.avg_pool(y.permute(0, 2, 1)).squeeze(-1)
        y = self.mlp2(y)

        return y


class P4TransformerDA6(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 dim_proposal, heads_proposal, dim_head_proposal,
                 mlp_dim, num_joints=13, features=3, num_proposal=64, mem_size=1024):                                                 # output
        super().__init__()

        self.num_proposal = num_proposal
        self.point_proposer = PointProposer(dim_proposal, heads_proposal, dim_head_proposal, features, num_proposal)

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mem = nn.Parameter(torch.FloatTensor(1, dim, mem_size).normal_(0.0, 1.0))
        self.disc = CorrelationDiscriminator(num_points=(1024+64)//8*5, dim=256)

        self.deconv = P4DTransConv(in_planes=dim, mlp_planes=[dim], mlp_activation=[True], mlp_batch_norm=[True], original_planes=features)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, num_joints),
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

    def feature_encode(self, input, mode='train'):
        B, L, N, C = input.shape
        input = input.reshape(B*L, N, C)
        new_input = self.point_proposer(input)
        input = torch.cat((input, new_input), dim=1)
        input = input.reshape(B, L, -1, C)

        device = input.get_device()
        xyzs0, features = self.tube_embedding(input[:,:,:,:3], input[:,:,:,3:].permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 
        xyzs_ = input[:,:,:,:3].clone()
        # print('xyzs: ', xyzs.max().item(), xyzs.min().item())
        # print('features: ', features.max().item(), features.min().item())

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

        # print('xyzts: ', xyzts.max().item(), xyzts.min().item())

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output0 = self.transformer(embedding)

        if mode == 'adapt':
            with torch.no_grad():
                self.mem.requires_grad_(False)
                output = self.forward_mem(output0)
        else:
            output = self.forward_mem(output0)

        l_rec = F.mse_loss(output, output0)

        if mode == 'adapt':
            with torch.no_grad():
                self.disc.requires_grad_(False)
                domain = self.disc(output)
        else:
            domain = self.disc(output)

        output = torch.reshape(input=output, shape=(B, L, -1, output.shape[2])) # B L n D
        output = output.permute(0, 1, 3, 2) # B L D n
        xyzs, output = self.deconv(xyzs0, input[:,:,:,:3], output.float(), input[:,:,:,3:].permute(0,1,3,2).float()) # B L D n

        return output, xyzs, l_rec, domain

    def forward(self, input, mode='inference'):
        assert mode in ['train', 'adapt', 'inference']
        output, xyzs, l_rec, domain = self.feature_encode(input, mode=mode)
        output = output[:, output.size(1)//2, :, :].permute(0, 2, 1) # B N D
        xyz = xyzs[:, xyzs.size(1)//2, :, :]

        output = self.mlp_head(output).permute(0, 2, 1) # B 13 N
        output = F.softmax(output, dim=-1)
        output = torch.bmm(output, xyz).unsqueeze(1)

        if mode == 'train':
            return output, domain, l_rec
        elif mode == 'adapt':
            return output, domain
        else:
            return output
        