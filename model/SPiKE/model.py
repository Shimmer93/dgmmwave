"""
Module for the SPiKE model.
"""

import sys
import os
import torch
from torch import nn

# Local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "modules"))
from .point_spat_convolution import PointSpatialConv
from .transformer import Transformer


class SPiKE(nn.Module):
    """
    SPiKE model class.
    """

    def __init__(
        self,
        radius,
        nsamples,
        spatial_stride,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        num_coord_joints,
        dropout1=0.0,
        dropout2=0.0,
    ):
        super().__init__()
        self.stem = self._build_stem(radius, nsamples, spatial_stride, dim)
        self.pos_embed = self._build_pos_embed(dim)
        self.transformer = self._build_transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout1
        )
        self.mlp_head = self._build_mlp_head(dim, mlp_dim, num_coord_joints, dropout2)

    def _build_stem(self, radius, nsamples, spatial_stride, dim):
        return PointSpatialConv(
            in_channels=0,
            mlp_channels=[dim],
            spatial_kernel_size=radius,
           nsamples=nsamples,
            spatial_stride=spatial_stride,
        )

    def _build_pos_embed(self, dim):
        return nn.Conv1d(
            in_channels=4,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def _build_transformer(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        return Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)

    def _build_mlp_head(self, dim, mlp_dim, num_coord_joints, dropout):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_coord_joints),
        )

    def forward(self, x):  # [B, L, N, 3]
        """
        Forward pass of the SPiKE model.
        """
        device = x.device
        xyzs, features = self.stem(x)  # [B, L, n, 3], [B, L, C, n]

        batch_size, seq_len, n, _ = xyzs.shape
        t = torch.arange(seq_len, device=device).view(1, seq_len, 1, 1).expand(batch_size, -1, n, -1) + 1
        xyzts = torch.cat((xyzs, t), dim=-1)
        xyzts = xyzts.view(batch_size, -1, 4)  # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)  # [B, L, n, C]
        features = features.reshape(batch_size, -1, features.shape[3])  # [B, L*n, C]

        xyzts = self.pos_embed(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
        embedding = xyzts + features

        output = self.transformer(embedding)
        output = torch.max(output, dim=1, keepdim=False)[0]
        joints_coord = self.mlp_head(output)

        return joints_coord