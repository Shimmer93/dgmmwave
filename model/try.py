from chamfer_distance import ChamferDistance
import torch
import torch.nn as nn

a = torch.randn(8, 100, 3).to('cuda')
b = torch.randn(8, 140, 3).to('cuda')

chamfer = ChamferDistance().to('cuda')

d0, d1 = chamfer(a, b)
print(d0.shape, d1.shape)