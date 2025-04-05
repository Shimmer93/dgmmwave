import torch
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from ..P4Transformer.transformer import *
# from torchvision.models import resnet18

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module)
            nn.init.zeros_(module)            

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