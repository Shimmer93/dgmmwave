import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain

class TemporalDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train', ratio=1, trans=False, both=False):
        self.data_path = data_path
        self.transform = transform
        self.trans = trans
        self.both = both

        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)

        if isinstance(split, str):
            self.split = self.all_data['splits'][split]
        elif isinstance(split, list):
            self.split = [self.all_data['splits'][s] for s in split]
            self.split = list(chain(*self.split))

        self.split = self.split[:int(len(self.split) * ratio)]
        
        self.data = [self.all_data['sequences'][i] for i in self.split]

        if self.trans or self.both:
            self.seq_lens = [len(seq['keypoints'])-1 for seq in self.data]
        else:
            self.seq_lens = [len(seq['keypoints']) for seq in self.data]
        # self.seq_lens = [len(seq['keypoints']) for seq in self.data]

    def __len__(self):
        return np.sum(self.seq_lens)
    
    def __getitem__(self, idx):
        seq_idx = 0
        while idx >= self.seq_lens[seq_idx]:
            idx -= self.seq_lens[seq_idx]
            seq_idx += 1
        sample = deepcopy(self.data[seq_idx])

        # print(len(sample['point_clouds']), len(sample['point_clouds_trans']))
        if self.trans:
            sample['point_clouds'] = sample['point_clouds_trans']
            sample['keypoints'] = sample['keypoints'][:-1]
        elif self.both:
            sample['keypoints'] = sample['keypoints'][:-1]
            if 'point_clouds_trans' not in sample:
                sample['point_clouds_trans'] = deepcopy(sample['point_clouds'])

        sample['dataset_name'] = self.data_path.split('/')[-1].split('.')[0]
        sample['sequence_index'] = seq_idx
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
        keys = ['point_clouds', 'keypoints', 'centroid', 'radius', 'sequence_index']
        # print(type(batch[0]['point_clouds_trans']))
        if 'point_clouds_trans' in batch[0].keys() and isinstance(batch[0]['point_clouds_trans'], torch.Tensor):
            keys.append('point_clouds_trans')
        # print('collate_fn keys: ', keys)
        for key in keys:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)

        return batch_data