import torch
import pickle
import random
import numpy as np
from copy import deepcopy
from itertools import chain

from dataset.temporal_dataset import TemporalDataset

class ReferenceDataset(TemporalDataset):
    def __init__(self, data_path, ref_data_path, transform=None, ref_transform=None, split='train', ref_split='train', ratio=1, ref_ratio=1, trans=False, both=False, ref_trans=False, ref_both=False):
        super().__init__(data_path, transform, split, ratio, trans, both)
        self.ref_data_path = ref_data_path
        self.ref_transform = ref_transform
        self.ref_trans = ref_trans
        self.ref_both = ref_both

        with open(ref_data_path, 'rb') as f:
            self.ref_all_data = pickle.load(f)

        if isinstance(ref_split, str):
            self.ref_split = self.ref_all_data['splits'][ref_split]
        elif isinstance(ref_split, list):
            self.ref_split = [self.ref_all_data['splits'][s] for s in ref_split]
            self.ref_split = list(chain(*self.ref_split))
        # random.shuffle(self.ref_split)
        self.ref_split = self.ref_split[:int(len(self.ref_split) * ref_ratio)]

        self.ref_data = [self.ref_all_data['sequences'][i] for i in self.ref_split]
        if self.ref_trans or ref_both:
            self.ref_seq_lens = [len(seq['keypoints'])-1 for seq in self.ref_data]
        else:
            self.ref_seq_lens = [len(seq['keypoints']) for seq in self.ref_data]
        self.ref_len = np.sum(self.ref_seq_lens)

    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        ref_idx = random.randint(0, self.ref_len - 1)
        ref_seq_idx = 0
        while ref_idx >= self.ref_seq_lens[ref_seq_idx]:
            ref_idx -= self.ref_seq_lens[ref_seq_idx]
            ref_seq_idx += 1
        ref_sample = deepcopy(self.ref_data[ref_seq_idx])

        if self.ref_trans:
            ref_sample['point_clouds'] = ref_sample['point_clouds_trans'] 
            ref_sample['keypoints'] = ref_sample['keypoints'][:-1]
        elif self.ref_both:
            ref_sample['keypoints'] = ref_sample['keypoints'][:-1]

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
        keys = ['point_clouds', 'keypoints', 'centroid', 'radius']
        ref_keys = keys.copy()
        if 'point_clouds_trans' in batch[0][0].keys() and isinstance(batch[0][0]['point_clouds_trans'], torch.Tensor):
            keys.append('point_clouds_trans')
        if 'point_clouds_trans' in batch[0][1].keys() and isinstance(batch[0][1]['point_cloud_trans'], torch.Tensor):
            ref_keys.append('point_cloud_trans')
        for key in keys:
            batch_data[key] = torch.stack([sample[0][key] for sample in batch], dim=0)
        for key in ref_keys:
            batch_data['ref_'+key] = torch.stack([sample[1][key] for sample in batch], dim=0)

        return batch_data