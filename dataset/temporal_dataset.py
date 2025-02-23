import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain

class TemporalDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = data_path
        self.transform = transform

        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)

        if isinstance(split, str):
            self.split = self.all_data['splits'][split]
        elif isinstance(split, list):
            self.split = [self.all_data['splits'][s] for s in split]
            self.split = list(chain(*self.split))
        
        self.data = [self.all_data['sequences'][i] for i in self.split]
        self.seq_lens = [len(seq['point_clouds']) for seq in self.data]

    def __len__(self):
        return np.sum(self.seq_lens)
    
    def __getitem__(self, idx):
        seq_idx = 0
        while idx >= self.seq_lens[seq_idx]:
            idx -= self.seq_lens[seq_idx]
            seq_idx += 1
        sample = deepcopy(self.data[seq_idx])

        sample['skeleton_type'] = self.data_path.split('/')[-1].split('.')[0].split('_')[0]
        sample['index'] = idx
        sample['centroid'] = np.array([0.,0.,0.])
        sample['radius'] = 1.
        sample['scale'] = 1.
        sample['translate'] = np.array([0.,0.,0.])
        sample['rotation_matrix'] = np.eye(3)

        sample = self.transform(sample)

        return sample
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        for key in ['point_clouds', 'keypoints', 'centroid', 'radius']:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)

        return batch_data