import torch
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from .point_4d_convolution import *
from .transformer import *
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
        

class P4TransformerFlow(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features=3):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.loc_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 3),
        )

        self.pose_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

        self.flow_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(self, input): 
        B, T, N, C = input.shape                                                                                                          # [B, L, N, 3]
        device = input.get_device()
        xyzs, features = self.tube_embedding(input[:,:,:,:3], input[:,:,:,:3].clone().permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

        # print('xyzs: ', xyzs.max().item(), xyzs.min().item())
        # print('features: ', features.max().item(), features.min().item())

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
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

        output = self.transformer(embedding)
        output_ = output.reshape(B, T, -1, output.shape[-1])
        output_ = torch.max(input=output_, dim=2, keepdim=False, out=None)[0]
        output0 = output_[:, 0:1, ...] # B D
        flow = self.flow_head(output_)
        flow = flow.reshape(B, T, flow.shape[-1]//3, 3) #[:, -1] # B 1 J 3
        pose = self.pose_head(output0)
        pose = pose.reshape(B, 1, pose.shape[-1]//3, 3) #[:, -1] # B 1 J 3
        loc = self.loc_head(output0) # 
        loc = loc.reshape(B, 1, 1, 3) # B 1 J 3
        # print(flow.shape, pose.shape, loc.shape)

        # pose = self.flow2pose(flow.detach()) # B T J 3
        # print('output after mlp_head: ', output.max().item(), output.min().item())
        if self.training:
            return pose, loc, flow
        else:
            y0 = pose + loc
            accum_flow = torch.cumsum(flow[:, :-1, ...], dim=1)
            
            y = torch.cat([y0, y0 + accum_flow], dim=1)
            return y, pose, loc, flow
