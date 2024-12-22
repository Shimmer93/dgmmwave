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

        self.deconv = P4DTransConv(in_planes=dim, mlp_planes=[dim], mlp_activation=[True], mlp_batch_norm=[True], original_planes=features)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_joints*3),
        )

    def encode(self, input):
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

class P4TransformerDA8(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 dim_proposal, heads_proposal, dim_head_proposal,
                 mlp_dim, num_joints=13, features=3, num_proposal=16, mem_size=1024):   # output
        super().__init__()

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
        embedding = self.model_s.encode(input)
        output0 = self.model_s.transformer(embedding)
        output = self.model_t.mem(output0) # [B, L*n, C]
        output = self.model_t.decode(output)
        return output
        
    def forward_train(self, input):
        input_s, input_t = input

        self.model_s.mem.requires_grad_(True)
        self.model_t.mem.requires_grad_(True)
        embedding_s = self.model_s.encode(input_s)
        output0_s = self.model_s.transformer(embedding_s)
        output0_s2 = output0_s.clone()
        mem_s = self.model_s.mem(output0_s) # [B, L*n, C]
        l_rec_s = F.mse_loss(mem_s, output0_s)
        output_s = self.model_s.decode(mem_s)

        embedding_t = self.model_s.encode(input_t)
        output0_t = self.model_t.transformer(embedding_t)
        output0_t2 = output0_t.clone()
        output_t = self.model_t.mem(output0_t) # [B, L*n, C]
        l_rec_t = F.mse_loss(output_t, output0_t)
        output_t = self.model_t.decode(output_t)

        self.model_s.mem.requires_grad_(False)
        self.model_t.mem.requires_grad_(False)
        mem_s2 = self.model_t.mem(output0_s2)
        l_mem = F.mse_loss(mem_s2, mem_s.detach())

        output_s2 = self.model_t.decode(mem_s2)
        output_t2 = self.model_s.mem(output0_t2)
        output_t2 = self.model_s.decode(output_t2)

        return output_s, output_t, output_s2, output_t2, l_rec_s, l_rec_t, l_mem