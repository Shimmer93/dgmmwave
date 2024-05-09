import torch
from torch.utils.data import Dataset
import pickle
import random

class TemporalDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = data_path
        self.transform = transform

        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)

        self.split = self.all_data['splits'][split]
        self.data = [self.all_data['sequences'][i] for i in self.split]
        if split.startswith('train'):
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = self.transform(sample)
        return sample