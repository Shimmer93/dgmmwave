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
        # self.joint_emb = nn.Parameter(torch.randn(1, num_joints, dim))
        self.to_kv = nn.Linear(dim, dim*2)

    def forward(self, x, joint_emb):
        B, N, D = x.shape

        k, v = self.to_kv(x).chunk(2, dim=-1)

        joint_emb = joint_emb.expand(B, -1, -1) # [B, J, D]
        attn = k @ joint_emb.transpose(1, 2) # [B, N, J]
        attn = F.softmax(attn, dim=1) # [B, N, J]
        x = attn.transpose(1, 2) @ v # [B, J, D]

        return x
    
class PointAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # self.joint_emb = nn.Parameter(torch.randn(1, num_joints, dim))
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, joint_emb):
        B, N, C = x.shape
        J = joint_emb.shape[1]

        q = joint_emb.expand(B, -1, -1)
        q = q.reshape(B, J, self.heads, self.dim_head).permute(0, 2, 1, 3)

        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: t.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3), kv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v).reshape(B, self.heads, J, self.dim_head)
        out = out.permute(0, 2, 1, 3).reshape(B, J, -1)
        out = self.to_out(out)

        return out

class PointReplacer(nn.Module):
    def __init__(self, dim, heads, dim_head, num_to_add):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.trans = Transformer(dim, 4, heads, dim_head, dim*2)
        self.attn = PointAttention(dim, heads, dim_head)

        self.dec_point = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3)
        )

        self.dec_conf = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, points, proposal_embed, conf_gt=None, apply_conf=True):
        B, L, N, C = points.shape
        points = points.reshape(B*L, N, C)
        points_orig = points

        feat = self.enc(points)
        feat = self.trans(feat)
        feat_new = self.attn(feat, proposal_embed)
        conf = self.dec_conf(feat)
        points_new = self.dec_point(feat_new) 

        if conf_gt is not None:
            conf_ = conf_gt.reshape(B*L, N, 1).detach()
        else:
            conf_ = (F.sigmoid(conf) > 0.5).to(points.dtype).detach()

        points = conf_ * points_orig + (1 - conf_) * points_new

        # self.enc.requires_grad_(False)
        # self.transformer.requires_grad_(False)
        # self.conf_dec.requires_grad_(False)
        # feat_new = self.enc(points_new)
        # feat_new = self.transformer(feat_new)
        # conf_new = self.conf_dec(feat_new)

        # self.enc.requires_grad_(True)
        # self.transformer.requires_grad_(True)
        # self.conf_dec.requires_grad_(True)

        points = points.reshape(B, L, -1, C).float()
        return points, conf.reshape(B, L, -1, 1), points_new.reshape(B, L, -1, C) #, conf_new.reshape(B, L, -1, 1)

class PointRemover(nn.Module):
    def __init__(self, dim, heads, dim_head, num_to_remove):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.trans = Transformer(dim, 4, heads, dim_head, dim*2)

        self.dec = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

        self.num_to_remove = num_to_remove

    def forward(self, points):
        B, L, N, C = points.shape
        points = points.reshape(B*L, N, C)

        conf = self.dec(self.trans(self.enc(points)))
        conf_ = conf.detach().clone()

        for i in range(self.num_to_remove):
            min_idx = conf.argmin(dim=1, keepdim=True)
            points = torch.cat([points[:, :min_idx], points[:, min_idx+1:]], dim=1)
            conf_ = torch.cat([conf_[:, :min_idx], conf_[:, min_idx+1:]], dim=1)

        points = points.reshape(B, L, -1, C)

        return points, conf.reshape(B, L, -1, 1)
    
class PointReplacer(nn.Module):
    def __init__(self, dim, heads, dim_head, num_to_add):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.attn_source = SelfAttentionMemoryBank(dim, num_to_add)
        # self.attn_target = SelfAttentionMemoryBank(dim, num_to_add)

        self.dec_pc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3)
        )

        self.dec_conf = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

        self.num_to_add = num_to_add

    def forward(self, points, target=False):
        B, L, N, C = points.shape
        points = points.reshape(B*L, N, C)

        feat = self.enc(points)
        # if target:
        #     feat_mem = self.attn_target(feat)
        # else:
        feat_mem = self.attn_source(feat)

        conf = self.dec_conf(feat_mem)
        conf_ = conf.detach().clone()

        points_rec = self.dec_pc(feat_mem)
        # print(points_rec.shape, points.shape)
        l_prec = F.mse_loss(points_rec[:, :N, :], points)

        masks_keep = []
        for i in range(B*L):
            idx_to_remove = conf_[i, :, 0].topk(self.num_to_add, largest=False).indices
            mask_keep = torch.ones(N+self.num_to_add, dtype=torch.bool, device=points.device)
            mask_keep[idx_to_remove] = False
            masks_keep.append(mask_keep)
        masks_keep = torch.stack(masks_keep, dim=0).unsqueeze(-1)
        points_new = torch.cat([points, points_rec[:, N:, :]], dim=1)
        points_masked = points_new * masks_keep

        # points_masked = points_rec[:, :N, :]

        points_masked = points_masked.reshape(B, L, -1, C)
        conf = conf[:, :N, :].reshape(B, L, -1, 1)

        # print(points_masked.shape, conf.shape)

        return points_masked, conf.reshape(B, L, -1, 1), l_prec


class SelfAttentionMemoryBank(nn.Module):
    def __init__(self, dim, mem_size):
        super().__init__()
        self.mem = nn.Parameter(torch.FloatTensor(1, dim, mem_size).normal_(0.0, 1.0))

        self.to_qkv = nn.Linear(dim, dim*3, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        _, _, M = self.mem.shape

        m = self.mem.repeat(B, 1, 1)
        m_key = m.transpose(1, 2)

        xm = torch.cat([x, m_key], dim=1)
        q, k, v = self.to_qkv(xm).chunk(3, dim=-1) # [B, N+M, D]

        logits = torch.bmm(k, q.transpose(1, 2)) / sqrt(D) # [B, N+M, N+M]
        x_new = torch.bmm(F.softmax(logits, dim=1), v) # [B, N+M, D]

        return x_new
    
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
    
    
class Model(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 dim_proposal, heads_proposal, dim_head_proposal,
                 mlp_dim, num_joints=13, features=3, num_proposal=16, mem_size=1024):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mem = AttentionMemoryBank(dim, mem_size)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints*3),
        )

    def encode(self, input):
        device = input.get_device()
        xyzs0, features = self.tube_embedding(input[:,:,:,:3], input[:,:,:,:3].clone().permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

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

    def decode(self, output):
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        output = output.reshape(output.shape[0], 1, output.shape[-1]//3, 3) # B 1 J 3
        return output
    
    def forward(self, input):
        embedding = self.encode(input)
        output0 = self.transformer(embedding)
        output = self.mem(output0) # [B, L*n, C]
        output = self.decode(output)

        return output

class P4TransformerDA11(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 dim_proposal, heads_proposal, dim_head_proposal,
                 mlp_dim, num_joints=13, features=3, num_proposal=16, mem_size=1024):   # output
        super().__init__()

        self.point_replacer = PointReplacer(dim_proposal, heads_proposal, dim_head_proposal, num_proposal)

        # self.proposal_embed_s = nn.Parameter(torch.randn(1, num_proposal, dim_proposal))
        # self.proposal_embed_t = nn.Parameter(torch.randn(1, num_proposal, dim_proposal))

        self.model_s = Model(radius, nsamples, spatial_stride, temporal_kernel_size, temporal_stride, 
                             emb_relu, dim, depth, heads, dim_head, dim_proposal, heads_proposal, 
                             dim_head_proposal, mlp_dim, num_joints, features, num_proposal, mem_size)
        
        self.model_t = Model(radius, nsamples, spatial_stride, temporal_kernel_size, temporal_stride, 
                             emb_relu, dim, depth, heads, dim_head, dim_proposal, heads_proposal, 
                             dim_head_proposal, mlp_dim, num_joints, features, num_proposal, mem_size)
        
    def forward(self, input, mode='inference'):
        assert mode in ['inference', 'inference_source', 'train']
        if mode == 'inference':
            return self.forward_inference(input)
        elif mode == 'inference_source':
            return self.model_s(input)
        else:
            return self.forward_train(input)
        
    def forward_inference(self, input):
        input, _, _ = self.point_replacer(input[...,:3], target=True)
        embedding = self.model_s.encode(input)
        output0 = self.model_s.transformer(embedding)
        output = self.model_t.mem(output0) # [B, L*n, C]
        output = self.model_t.decode(output)
        return output
        
    def forward_train(self, input):
        input_s, input_t = input
        # print(input_s.shape, input_t.shape)
        # self.proposal_embed_s.requires_grad_(True)
        # self.point_proposer.mem.requires_grad_(True)
        points_s, conf_s, l_prec_s = self.point_replacer(input_s[...,:3], target=False)
        # self.point_proposer.mem.requires_grad_(False)
        points_t, _, l_prec_t = self.point_replacer(input_t[...,:3], target=True)
        # self.proposal_embed_s.requires_grad_(False)
        # print(conf_s.shape, input_s.shape, (input_s[...,-1:] > 0).sum().item(), (input_s[...,-1:] == 0).sum().item())
        # print(F.sigmoid(conf_s).max().item(), F.sigmoid(conf_s).min().item(), F.sigmoid(conf_s).mean().item())
        l_conf = F.binary_cross_entropy_with_logits(conf_s[input_s[...,-1:] > 0], torch.ones_like(conf_s[input_s[...,-1:] > 0])) + \
                    F.binary_cross_entropy_with_logits(conf_s[input_s[...,-1:] == 0], torch.zeros_like(conf_s[input_s[...,-1:] == 0]))#+ \
                    # F.binary_cross_entropy_with_logits(conf_new_s, torch.ones_like(conf_new_s)) + \
                    # F.binary_cross_entropy_with_logits(conf_new_t, torch.zeros_like(conf_new_t))
        # l_prec_s = F.mse_loss(points_new_s, input_s[...,:3])
        # l_prec_t = F.mse_loss(points_new_t, input_t[...,:3])

        self.model_s.mem.requires_grad_(True)
        self.model_t.mem.requires_grad_(True)
        embedding_s = self.model_s.encode(points_s)
        output0_s = self.model_s.transformer(embedding_s)
        output0_s2 = output0_s.clone()
        mem_s = self.model_s.mem(output0_s) # [B, L*n, C]
        l_rec_s = F.mse_loss(mem_s[:, :output0_s.shape[1], :], output0_s)
        output_s = self.model_s.decode(mem_s)

        embedding_t = self.model_s.encode(points_t)
        output0_t = self.model_s.transformer(embedding_t)
        output0_t2 = output0_t.clone()
        output_t = self.model_t.mem(output0_t) # [B, L*n, C]
        l_rec_t = F.mse_loss(output_t[:, :output0_t.shape[1], :], output0_t)
        output_t = self.model_t.decode(output_t)

        self.model_s.mem.requires_grad_(False)
        self.model_t.mem.requires_grad_(False)
        mem_s2 = self.model_t.mem(output0_s2)
        l_mem = F.mse_loss(mem_s2, mem_s.detach())

        output_s2 = self.model_t.decode(mem_s2)
        output_t2 = self.model_s.mem(output0_t2)
        output_t2 = self.model_s.decode(output_t2)

        return output_s, output_s2, l_conf, l_rec_s, l_rec_t, l_mem, l_prec_s, l_prec_t #, points_new_s, points_new_t