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

class Encoder(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features):                                                             # embedding: relu
        super().__init__()
        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')
        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

    def forward(self, input): 
        device = input.get_device()
        xyzs, features = self.tube_embedding(input[:,:,:,:3], input[:,:,:,2:3].clone().permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n] 

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
            
        return embedding                                                                                                         # [B, L, N, 3]

class P4TransformerLidarMMWave(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features=3):                                                 # output
        super().__init__()

        self.enc0 = Encoder(radius, nsamples, spatial_stride,                                # P4DConv: spatial
                      temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                      emb_relu,                                                              # embedding: relu
                      dim, depth, heads, dim_head,                                           # transformer
                      mlp_dim, output_dim, features)
        self.enc1 = Encoder(radius, nsamples, spatial_stride,                                # P4DConv: spatial
                      temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                      emb_relu,                                                              # embedding: relu
                      dim, depth, heads, dim_head,                                           # transformer
                      mlp_dim, output_dim, features)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        # self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(self, input0, input1):  
        if input0 is None:
            embedding = self.enc1(input1)
        elif input1 is None:
            embedding = self.enc0(input0)
        else:                                                                                                         # [B, L, N, 3]
            embedding0 = self.enc0(input0)                                                                                                         # [B, L, N, 64]
            embedding1 = self.enc1(input1)
            embedding = torch.cat((embedding0, embedding1), dim=1)                                                                                                        # [B, L, N, 64]

        output = self.transformer(embedding)
            # output1 = self.transformer1(embedding1)

            # output = torch.cat(tensors=(output0, output1), dim=1)
            # output = output0 + output1
        # print('output after transformer: ', output.max().item(), output.min().item())

        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)
        output = output.reshape(output.shape[0], 1, output.shape[-1]//3, 3) # B 1 J 3
        # print('output after mlp_head: ', output.max().item(), output.min().item())
        return output
