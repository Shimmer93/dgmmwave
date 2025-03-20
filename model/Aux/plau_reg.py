import torch
import torch.nn as nn
import torch.nn.functional as F

class PlausibilityRegressor(nn.Module):
    def __init__(self, num_bones, depth_enc=2, depth_dec=5, dim=6):
        super(PlausibilityRegressor, self).__init__()
        self.depth_enc = depth_enc
        self.depth_dec = depth_dec
        self.dim = dim

        encoder = []
        encoder.append(nn.Linear(3, dim))
        encoder.append(nn.ReLU())
        for _ in range(depth_enc - 1):
            encoder.append(nn.Linear(dim, dim))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for _ in range(depth_dec - 1):
            decoder.append(nn.Linear(dim * num_bones, dim * num_bones))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(dim * num_bones, 1))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x