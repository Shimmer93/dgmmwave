import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain

from dataset.temporal_dataset import TemporalDataset

class SklOnlyDataset(TemporalDataset):
    def __init__(self, data_path, transform=None, split='train'):
        super().__init__(data_path, transform, split)

    def __getitem__(self, idx):
        seq_idx = 0
        while idx >= self.seq_lens[seq_idx]:
            idx -= self.seq_lens[seq_idx]
            seq_idx += 1
        sample = deepcopy(self.data[seq_idx])

        sample['dataset_name'] = self.data_path.split('/')[-1].split('.')[0]
        sample['index'] = idx

        sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        keys = ['keypoints']
        if 'bone_dirs' in batch[0].keys():
            keys.append('bone_dirs')
        if 'bone_motions' in batch[0].keys():
            keys.append('bone_motions')
        if 'joint_motions' in batch[0].keys():
            keys.append('joint_motions')
        for key in keys:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)

        return batch_data


        
        