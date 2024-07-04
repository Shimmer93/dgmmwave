import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
from copy import deepcopy

from dataset.temporal_dataset import TemporalDataset

class ReferenceDataset(TemporalDataset):
    def __init__(self, data_path, ref_data_path, transform=None, split='train', ref_split='train'):
        super().__init__(data_path, transform, split)
        self.ref_data_path = ref_data_path

        with open(ref_data_path, 'rb') as f:
            self.ref_all_data = pickle.load(f)

        self.ref_split = self.ref_all_data['splits'][ref_split]
        self.ref_data = [self.ref_all_data['sequences'][i] for i in self.split]
        self.ref_seq_lens = [len(seq['point_clouds']) for seq in self.ref_data]
        self.ref_len = np.sum(self.ref_seq_lens)
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        ref_idx = random.randint(0, self.ref_len - 1)
        ref_seq_idx = 0
        while ref_idx >= self.ref_seq_lens[ref_seq_idx]:
            ref_idx -= self.ref_seq_lens[ref_seq_idx]
            ref_seq_idx += 1
        ref_sample = deepcopy(self.ref_data[ref_seq_idx])
        ref_sample['index'] = ref_idx
        # sample = self.data[idx]
        ref_sample = self.transform(ref_sample)
        return sample, ref_sample
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        for key in ['point_clouds', 'keypoints', 'centroid', 'radius']:
            batch_data[key] = torch.stack([sample[0][key] for sample in batch], dim=0)
            batch_data['ref_'+key] = torch.stack([sample[1][key] for sample in batch], dim=0)

        return batch_data