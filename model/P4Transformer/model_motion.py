import torch
import sys 
import os
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from .point_4d_convolution import *
from .transformer import *
# from torchvision.models import resnet18


class P4TransformerMotion(nn.Module):
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

        self.deconv = P4DTransConv(in_planes=dim, mlp_planes=[dim], mlp_activation=[True], mlp_batch_norm=[True], original_planes=features)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(self, input):
        B, L, N, C = input.shape                                                                                                               # [B, L, N, 3]
        device = input.get_device()
        xyzs0, features = self.tube_embedding(input[:,:,:,:3], input[:,:,:,2:3].clone().permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

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

        embedding = xyzts + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        output = torch.reshape(input=output, shape=(B, L, -1, output.shape[2])) # B L n D
        output = output.permute(0, 1, 3, 2) # B L D n
        # print('output after transformer: ', output.max().item(), output.min().item())

        xyzs, output = self.deconv(xyzs0, input[:,:,:,:3], output.float(), input[:,:,:,2:3].clone().permute(0,1,3,2)) # B L D n

        output = output[:, -1, :, :].permute(0, 2, 1) # B N D

        # output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        output = output.unsqueeze(1) # B 1 N 3
        # print('output after mlp_head: ', output.max().item(), output.min().item())
        return output
