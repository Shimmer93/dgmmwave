import torch
import torch.nn as nn
import torch.nn.functional as F

class DebugModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # batch size, sequence length, num points, features
        # x: B L N 3
        # out: B L J 3
        B, L, N, D = x.size()
        x = x.permute(0, 1, 3, 2).reshape(B*L*D, N)
        x = self.fc(x)
        x = x.reshape(B, L, -1, 3)
        return x