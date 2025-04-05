import torch
import pickle
import random
import numpy as np
from copy import deepcopy
from itertools import chain

from dataset.temporal_dataset import TemporalDataset

class ReferenceDataset(TemporalDataset):
    def __init__(self, data_path, ref_data_path, transform=None, ref_transform=None, split='train', ref_split='train'):
        super().__init__(data_path, transform, split)
        self.ref_data_path = ref_data_path
        self.ref_transform = ref_transform

        with open(ref_data_path, 'rb') as f:
            self.ref_all_data = pickle.load(f)

        if isinstance(ref_split, str):
            self.ref_split = self.ref_all_data['splits'][ref_split]
        elif isinstance(ref_split, list):
            self.ref_split = [self.ref_all_data['splits'][s] for s in ref_split]
            self.ref_split = list(chain(*self.ref_split))

        self.ref_data = [self.ref_all_data['sequences'][i] for i in self.ref_split]
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

        ref_sample['dataset_name'] = self.ref_data_path.split('/')[-1].split('.')[0]
        ref_sample['sequence_index'] = ref_seq_idx
        ref_sample['index'] = ref_idx
        ref_sample['centroid'] = np.array([0.,0.,0.])
        ref_sample['radius'] = 1.
        ref_sample['scale'] = 1.
        ref_sample['translate'] = np.array([0.,0.,0.])
        ref_sample['rotation_matrix'] = np.eye(3)

        ref_sample = self.ref_transform(ref_sample)

        return sample, ref_sample
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        for key in ['point_clouds', 'keypoints', 'centroid', 'radius']:
            batch_data[key] = torch.stack([sample[0][key] for sample in batch], dim=0)
            batch_data['ref_'+key] = torch.stack([sample[1][key] for sample in batch], dim=0)

        return batch_data