"""
Module for point spatial convolution.
"""

from typing import List
import os
import sys
import torch
from torch import nn
import pointnet2_utils

# Local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


class PointSpatialConv(nn.Module):
    """
    Point spatial convolution layer.
    """

    def __init__(
        self,
        in_channels: int,
        mlp_channels: List[int],
        spatial_kernel_size: float,
        nsamples: int,
        spatial_stride: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mlp_channels = mlp_channels
        self.spatial_kernel_size = spatial_kernel_size
        self.nsamples = nsamples
        self.spatial_stride = spatial_stride

        self.conv_d = self._build_conv_d()
        self.mlp = self._build_mlp()

    def _build_conv_d(self):
        conv_d = [
            nn.Conv2d(
                in_channels=4,
                out_channels=self.mlp_channels[0],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        ]
        return nn.Sequential(*conv_d)

    def _build_mlp(self):
        mlp = []
        for i in range(1, len(self.mlp_channels)):
            if self.mlp_channels[i] != 0:
                mlp.append(
                    nn.Conv2d(
                        in_channels=self.mlp_channels[i - 1],
                        out_channels=self.mlp_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                )
            if self.mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=self.mlp_channels[i]))
            if self.mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        return nn.Sequential(*mlp)

    def forward(self, xyzs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass for the point spatial convolution layer.

        Args:
            xyzs (torch.Tensor): Input point cloud coordinates.

        Returns:
            torch.Tensor: New point cloud coordinates.
            torch.Tensor: New features.
        """
        device = xyzs.get_device()
        xyzs = [xyz.squeeze(dim=1).contiguous() for xyz in torch.split(xyzs, 1, dim=1)]
        new_xyzs, new_features = [], []

        for xyz in xyzs:
            reference_idx = pointnet2_utils.furthest_point_sample(
                xyz, xyz.size(1) // self.spatial_stride
            )
            reference_xyz_flipped = pointnet2_utils.gather_operation(
                xyz.transpose(1, 2).contiguous(), reference_idx
            )
            reference_xyz = reference_xyz_flipped.transpose(1, 2).contiguous()

            idx = pointnet2_utils.ball_query(
                self.spatial_kernel_size, self.nsamples, xyz, reference_xyz
            )
            neighbor_xyz_grouped = pointnet2_utils.grouping_operation(
                xyz.transpose(1, 2).contiguous(), idx
            )
            displacement = torch.cat(
                (
                    neighbor_xyz_grouped - reference_xyz_flipped.unsqueeze(3),
                    torch.zeros(
                        (
                            xyz.size(0),
                            1,
                            xyz.size(1) // self.spatial_stride,
                            self.nsamples,
                        ),
                        device=device,
                    ),
                ),
                dim=1,
            )
            displacement = self.conv_d(displacement)

            feature = torch.max(self.mlp(displacement), dim=-1, keepdim=False)[0]
            new_features.append(
                torch.max(torch.stack([feature], dim=1), dim=1, keepdim=False)[0]
            )
            new_xyzs.append(reference_xyz)

        return torch.stack(new_xyzs, dim=1), torch.stack(new_features, dim=1)