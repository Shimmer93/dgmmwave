import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
from copy import deepcopy

class BiAugDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = data_path
        self.transform1 = transform
        self.transform2 = deepcopy(transform)

        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)

        self.split = self.all_data['splits'][split]
        self.data = [self.all_data['sequences'][i] for i in self.split]
        self.seq_lens = [len(seq['point_clouds']) for seq in self.data]
        # if split.startswith('train'):
        #     random.shuffle(self.data)

    def __len__(self):
        return np.sum(self.seq_lens)
    
    def __getitem__(self, idx):
        seq_idx = 0
        while idx >= self.seq_lens[seq_idx]:
            idx -= self.seq_lens[seq_idx]
            seq_idx += 1
        sample = deepcopy(self.data[seq_idx])
        sample['index'] = idx
        sample['sequence_index'] = seq_idx
        sample['centroid'] = np.array([0.,0.,0.])
        sample['radius'] = 1.
        sample['scale'] = 1.
        sample['translate'] = np.array([0.,0.,0.])
        sample['rotation_matrix'] = np.eye(3)
        # sample = self.data[idx]
        sample2 = deepcopy(sample)
        sample1 = self.transform1(sample)
        sample2 = self.transform2(sample2)
        samples = {}
        for key in sample1.keys():
            samples[key] = torch.cat([sample1[key], sample2[key]], dim=0)
        # sample['point_clouds'] = sample['point_clouds'][..., :-1]
        return samples
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        for key in batch[0].keys():
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)

        return batch_data