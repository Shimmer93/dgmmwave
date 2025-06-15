import torch
import pickle
import random
import numpy as np
from copy import deepcopy
from itertools import chain

from dataset.temporal_dataset import TemporalDataset

class ReferenceOneToOneDataset(TemporalDataset):
    def __init__(self, data_path, ref_data_path, transform=None, ref_transform=None, split='train', ratio=1):
        super().__init__(data_path, transform, split, ratio, False, True)
        self.ref_data_path = ref_data_path
        self.ref_transform = ref_transform

        with open(ref_data_path, 'rb') as f:
            self.ref_all_data = pickle.load(f)

        if isinstance(split, str):
            self.ref_split = self.ref_all_data['splits'][split]
        elif isinstance(split, list):
            self.ref_split = [self.ref_all_data['splits'][s] for s in split]
            self.ref_split = list(chain(*self.ref_split))

        self.ref_data = [self.ref_all_data['sequences'][i] for i in self.ref_split]
        # self.ref_seq_lens = [len(seq['point_clouds']) for seq in self.ref_data]
        if self.trans or self.both:
            self.ref_seq_lens = [len(seq['keypoints'])-1 for seq in self.ref_data]
        else:
            self.ref_seq_lens = [len(seq['keypoints']) for seq in self.ref_data]
        self.ref_len = np.sum(self.ref_seq_lens)
    
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

        sample['dataset_name'] = self.data_path.split('/')[-1].split('.')[0]
        sample['sequence_index'] = seq_idx
        sample['index'] = idx
        sample['centroid'] = np.array([0.,0.,0.])
        sample['radius'] = 1.
        sample['scale'] = 1.
        sample['translate'] = np.array([0.,0.,0.])
        sample['rotation_matrix'] = np.eye(3)

        ref_idx = idx #random.randint(0, self.ref_len - 1)
        ref_seq_idx = 0
        while ref_idx >= self.ref_seq_lens[ref_seq_idx]:
            ref_idx -= self.ref_seq_lens[ref_seq_idx]
            ref_seq_idx += 1
        ref_sample = deepcopy(self.ref_data[ref_seq_idx])
        sample['point_clouds_trans'] = ref_sample['point_clouds']

        sample = self.transform(sample)
        return sample

        # if self.trans:
        #     ref_sample['point_clouds'] = ref_sample['point_clouds_trans']
        #     ref_sample['keypoints'] = ref_sample['keypoints'][:-1]
        # elif self.both:
        #     ref_sample['keypoints'] = ref_sample['keypoints'][:-1]

        # ref_sample['dataset_name'] = self.ref_data_path.split('/')[-1].split('.')[0]
        # ref_sample['sequence_index'] = ref_seq_idx
        # ref_sample['index'] = ref_idx
        # ref_sample['centroid'] = np.array([0.,0.,0.])
        # ref_sample['radius'] = 1.
        # ref_sample['scale'] = 1.
        # ref_sample['translate'] = np.array([0.,0.,0.])
        # ref_sample['rotation_matrix'] = np.eye(3)

        # ref_sample = self.ref_transform(ref_sample)

        # return sample, ref_sample