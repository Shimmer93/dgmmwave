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

class P4ConvEncoder(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, features=3):                                                      # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

    def forward(self, input):
        device = input.get_device()
        xyzs0, features = self.tube_embedding(input[:,:,:,:3], input[:,:,:,3:].permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

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

        return embedding, xyzs0
    
class AttentionMemoryBank(nn.Module):
    def __init__(self, dim, mem_size=1024):
        super().__init__()

        self.mem = nn.Parameter(torch.FloatTensor(1, dim, mem_size).normal_(0.0, 1.0))

    def forward(self, x, mode='train'):
        if mode == 'adapt':
            self.mem.requires_grad_(False)
        B, N, D = x.shape
        _, _, M = self.mem.shape

        m = self.mem.repeat(B, 1, 1)
        m_key = m.transpose(1, 2)
        x_ = x.permute(0, 2, 1)
        logits = torch.bmm(m_key, x_) / sqrt(D)
        x_new = torch.bmm(m_key.transpose(1, 2), F.softmax(logits, dim=1))
        x_new_ = x_new.permute(0, 2, 1)

        return x_new_

class MemTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, mem_size=1024):
        super().__init__()

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.mem = AttentionMemoryBank(dim, mem_size)
    
    def forward(self, input, mode='train'):
        x = self.transformer(input)
        x_ = self.mem(x, mode)
        l_rec = F.mse_loss(x, x_)
        return x_, l_rec
    
class P4DeConvDecoder(nn.Module):
    def __init__(self, dim, num_joints=13, features=3):                                                 # output
        super().__init__()

        self.deconv = P4DTransConv(in_planes=dim, mlp_planes=[dim], mlp_activation=[True], 
                                   mlp_batch_norm=[True], original_planes=features)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, num_joints),
        )

    def forward(self, embedding, xyzs0, input):
        B, L, _, _ = input.shape
        output = torch.reshape(input=embedding, shape=(B, L, -1, embedding.shape[2])) # B L n D
        output = output.permute(0, 1, 3, 2) # B L D n

        xyzs, output = self.deconv(xyzs0, input[:,:,:,:3], output.float(), input[:,:,:,3:].permute(0,1,3,2).float()) # B L D n
        output = output[:, output.size(1)//2, :, :].permute(0, 2, 1) # B N D
        xyz = xyzs[:, xyzs.size(1)//2, :, :]

        output = self.mlp_head(output).permute(0, 2, 1) # B 13 N
        output = F.softmax(output, dim=-1)
        output = torch.bmm(output, xyz).unsqueeze(1)

        return output
    
class MLPDecoder(nn.Module):
    def __init__(self, dim, mlp_dim, num_joints=13):                                                 # output
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints*3),
        )

    def forward(self, embedding, xyzs0, input):
        output = torch.max(input=embedding, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        output = output.reshape(output.shape[0], 1, output.shape[-1]//3, 3) # B 1 J 3

        return output

class MemPointcloudUpdater(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,
                 temporal_kernel_size, temporal_stride,
                 emb_relu,
                 dim, depth, heads, dim_head,
                 mlp_dim, mem_size, features=3):
        super().__init__()

        self.enc = P4ConvEncoder(radius, nsamples, spatial_stride,
                                 temporal_kernel_size, temporal_stride,
                                 emb_relu, dim, features)
        self.transformer = MemTransformer(dim, depth, heads, dim_head, mlp_dim, mem_size)
        self.dec = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 3),
        )

    def forward(self, input, mode='train'):
        embed, coords = self.enc(input)
        x, l_rec = self.transformer(embed, mode)
        x = self.dec(x)

        return x, l_rec

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

class PointProposalUpdater(nn.Module):
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
        B, L, N, D = x.shape
        x = x.reshape(B*L, N, D)
        pp = self.m(x)
        x = torch.cat((x, pp), dim=1)
        x = x.reshape(B, L, -1, D)

        return x
    
class MemProposalUpdater(nn.Module):
    def __init__(self, dim, heads, dim_head, features, num_proposal,
                 radius, nsamples, spatial_stride,
                 temporal_kernel_size, temporal_stride,
                 emb_relu, depth, mlp_dim, mem_size):
        super().__init__()
        
        self.proposal = PointProposalUpdater(dim, heads, dim_head, features, num_proposal)
        self.mem = MemPointcloudUpdater(radius, nsamples, spatial_stride,
                                        temporal_kernel_size, temporal_stride,
                                        emb_relu, dim, depth, heads, dim_head,
                                        mlp_dim, mem_size, features=features)
        
    def forward(self, x, mode='train'):
        x = self.proposal(x)
        x, l_rec = self.mem(x, mode)
        
        return x, l_rec
    
class MemSkeletonUpdater(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, mem_size=1024):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(3, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim)
        )

        self.transformer = MemTransformer(dim=dim, depth=depth, heads=heads,
                                          dim_head=dim_head, mlp_dim=mlp_dim, mem_size=mem_size)

        self.dec = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, 3)
        )

    def forward(self, input, mode='train'):
        x = self.enc(input)
        x, l_rec = self.transformer(x, mode)
        x = self.dec(x)

        return x, l_rec

class PointcloudDiscriminator(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,
                    temporal_kernel_size, temporal_stride,
                    emb_relu,
                    dim, depth, heads, dim_head,
                    mlp_dim, features=3):
            super().__init__()
    
            self.enc = P4ConvEncoder(radius, nsamples, spatial_stride,
                                    temporal_kernel_size, temporal_stride,
                                    emb_relu, dim, features)
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
            self.dec = nn.Sequential(
                nn.Linear(dim, dim//2),
                nn.GELU(),
                nn.Linear(dim//2, 1)
            )

    def forward(self, input):
        embed, coords = self.enc(input)
        x = self.transformer(embed)
        x = self.dec(x)

        return x

class FeatureDiscriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 1)
        )

    def forward(self, x):
        # B, L, N, D = x.shape
        # x = x.shape(B, L*N, D)
        x_mean = torch.mean(x, dim=1)
        x_std = torch.std(x, dim=1)
        x = torch.cat((x_mean, x_std), dim=1)

        return self.mlp(x)

class SkeletonDiscriminator(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(3, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim)
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.dec = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, 1)
        )

    def forward(self, input):
        x = self.enc(input)
        x = self.transformer(x)
        x = self.dec(x)

        return x

class P4TransformerMeta(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,
                 temporal_kernel_size, temporal_stride,
                 emb_relu,
                 dim, depth, heads, dim_head,
                 mlp_dim, num_joints=13, features=3, mem_size=1024, num_proposal=0, 
                 enc='p4t', mixer='mem', dec='p4t', pc_update='mem', skl_update='mem',
                 dim_pc_up=128, depth_pc_up=3, heads_pc_up=8, dim_head_pc_up=16, mlp_dim_pc_up=256, mem_size_pc_up=1024,
                 dim_pc_disc=128, depth_pc_disc=3, heads_pc_disc=8, dim_head_pc_disc=16, mlp_dim_pc_disc=256,
                 dim_skl_up=128, depth_skl_up=3, heads_skl_up=8, dim_head_skl_up=16, mlp_dim_skl_up=256, mem_size_skl_up=1024,
                 dim_skl_disc=128, depth_skl_disc=3, heads_skl_disc=8, dim_head_skl_disc=16, mlp_dim_skl_disc=256):
        super().__init__()

        if num_proposal > 0:
            self.pc_proposer = PointProposalUpdater(dim_pc_up, heads_pc_up, dim_head_pc_up, features, num_proposal)
        else:
            self.pc_proposer = None

        if pc_update:
            assert pc_update in ['mem', 'prop', 'mem+prop']
            if pc_update == 'mem':
                self.pc_updater = MemPointcloudUpdater(radius, nsamples, spatial_stride,
                                                       temporal_kernel_size, temporal_stride,
                                                       emb_relu, dim_pc_up, depth_pc_up, heads_pc_up, dim_head_pc_up,
                                                       mlp_dim_pc_up, mem_size_pc_up, features)
            elif pc_update == 'prop':
                self.pc_updater = PointProposalUpdater(dim_pc_up, heads_pc_up, dim_head_pc_up, features, num_proposal)
            elif pc_update == 'mem+prop':
                self.pc_updater = MemProposalUpdater(dim_pc_up, heads_pc_up, dim_head_pc_up, features, num_proposal,
                                                     radius, nsamples, spatial_stride,
                                                     temporal_kernel_size, temporal_stride,
                                                     emb_relu, depth_pc_up, mlp_dim_pc_up, mem_size_pc_up)
            self.pc_disc = PointcloudDiscriminator(radius, nsamples, spatial_stride,
                                                   temporal_kernel_size, temporal_stride,
                                                   emb_relu, dim_pc_disc, depth_pc_disc, heads_pc_disc, dim_head_pc_disc,
                                                   mlp_dim_pc_disc, features)
        else:
            self.pc_updater = None
            self.pc_disc = None

        assert enc in ['p4t']
        self.enc = P4ConvEncoder(radius, nsamples, spatial_stride,
                                 temporal_kernel_size, temporal_stride,
                                 emb_relu, dim, features)
        
        assert mixer in ['mem']
        self.mixer = MemTransformer(dim, depth, heads, dim_head, mlp_dim, mem_size)
        self.feat_disc = FeatureDiscriminator(dim)

        assert dec in ['p4t', 'mlp']
        if dec == 'p4t':
            self.dec = P4DeConvDecoder(dim, num_joints, features)
        elif dec == 'mlp':
            self.dec = MLPDecoder(dim, mlp_dim, num_joints)
        else:
            raise ValueError('Invalid decoder type')
        
        if skl_update:
            assert skl_update in ['mem']
            self.skl_updater = MemSkeletonUpdater(dim_skl_up, depth_skl_up, heads_skl_up, dim_head_skl_up, mlp_dim_skl_up, mem_size_skl_up)
            self.skl_disc = SkeletonDiscriminator(dim_skl_disc, depth_skl_disc, heads_skl_disc, dim_head_skl_disc, mlp_dim_skl_disc)
        else:
            self.skl_updater = None
            self.skl_disc = None

    def forward_pc(self, input):
        output, loss_pc = self.pc_updater(input, 'train')
        d_pc = self.pc_disc(input)
        return output, loss_pc, d_pc
    
    def forward_skl(self, input):
        output, loss_skl = self.skl_updater(input, 'train')
        d_skl = self.skl_disc(input)
        return output, loss_skl, d_skl

    def forward(self, input, mode='inference'):
        assert mode in ['train', 'adapt', 'inference', 'eval']
        # input: [B, L, N, 3+F]

        if self.pc_proposer:
            input = self.pc_proposer(input)

        aux_losses = []
        disc_scores = []
        if not (self.pc_updater is None or mode in ['train', 'eval']):
            input, loss_input = self.pc_updater(input, mode)
            d_pc = self.pc_disc(input)
            aux_losses.append(loss_input)
            disc_scores.append(d_pc)
        
        embed, coords = self.enc(input) # [B, L*n, D], [B, L, n, 3]
        feat, loss_feat = self.mixer(embed, mode) # [B, L*n, D]
        d_feat = self.feat_disc(feat)
        aux_losses.append(loss_feat)
        disc_scores.append(d_feat)
        output = self.dec(feat, coords, input) # [B, 1, J, 3]

        if not (self.skl_updater is None or mode in ['train', 'eval']):
            output, loss_output = self.skl_updater(output, mode)
            d_skl = self.skl_disc(output)
            aux_losses.append(loss_output)
            disc_scores.append(d_skl)

        if mode in ['inference', 'eval']:
            return output
        return output, aux_losses, disc_scores