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

class P4TransformerDG2(nn.Module):
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

        self.joint_emb = nn.Parameter(torch.FloatTensor(1, output_dim//3, dim).normal_(0.0, 1.0))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mem = AttentionMemoryBank(dim, mem_size)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 3)
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
        output = output[:, :15, :]
        output = torch.cat([output, self.joint_emb.repeat(output.shape[0], 1, 1)], dim=-1)
        output = self.mlp_head(output)
        output = output.reshape(output.shape[0], 1, 15, 3) # B 1 J 3
        return output
    
    def forward_train(self, input):
        input0, input1 = input
        embedding0 = self.encode(input0)
        embedding1 = self.encode(input1)
        embedding0 = torch.cat([embedding0, self.joint_emb.repeat(embedding0.shape[0], 1, 1)], dim=1)
        embedding1 = torch.cat([embedding1, self.joint_emb.repeat(embedding1.shape[0], 1, 1)], dim=1)
        output0_ = self.transformer(embedding0)
        output1_ = self.transformer(embedding1)

        output0__ = self.mem(output0_) # [B, L*n, C]
        output1__ = self.mem(output1_) # [B, L*n, C]

        l_rec = F.mse_loss(output0__, output1__)

        output0 = self.decode(output0__)
        output1 = self.decode(output1__)

        return output0, output1, l_rec
    
    def forward_inference(self, input):
        embedding = self.encode(input)
        embedding = torch.cat([embedding, self.joint_emb.repeat(embedding.shape[0], 1, 1)], dim=1)
        output0 = self.transformer(embedding)
        output = self.mem(output0) # [B, L*n, C]
        output = self.decode(output)

        return output
    
    def forward(self, input):
        if self.training:
            return self.forward_train(input)
        else:
            return self.forward_inference(input)