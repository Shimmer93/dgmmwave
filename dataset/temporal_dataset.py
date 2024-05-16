import torch
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
from copy import deepcopy

class TemporalDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = data_path
        self.transform = transform

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
        # sample = self.data[idx]
        sample = self.transform(sample)
        return sample