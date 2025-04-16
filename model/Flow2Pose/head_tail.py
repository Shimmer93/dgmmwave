import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class HeadTailNet(nn.Module):
    def __init__(self, model_head, model_tail):
        self.model_head = model_head
        self.model_tail = model_tail

    def forward(self, x):
        # x: B T J D
        x_head = x
        x_tail = -x[:, ::-1, :, :]

        y_head = self.model_head(x_head)
        y_tail = self.model_tail(x_tail)
        y_tail = y_tail[:, ::-1, :, :]
        weight = torch.arange(0, y_head.shape[1], 1).reshape(1, -1, 1, 1).to(y_head.device)
        weight = weight / (y_head.shape[1] - 1)
        weight = torch.clamp(weight, 0, 1)
        y = weight * y_head + (1 - weight) * y_tail

        if self.training:
            return y_head, y_tail, y
        else:
            return y