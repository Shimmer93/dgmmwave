import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
from copy import deepcopy

from dataset.temporal_dataset import TemporalDataset

class PosNegDataset(TemporalDataset):
    def __init__(self, data_path, transform=None, split='train'):
        super().__init__(data_path, transform, split)
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        for key in ['point_clouds', 'keypoints', 'centroid', 'radius']:
            batch_data[key] = torch.stack([sample[0][key] for sample in batch], dim=0)
            batch_data['neg_'+key] = torch.stack([sample[1][key] for sample in batch], dim=0)

        return batch_data