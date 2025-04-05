import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain

from dataset.skl_only_dataset import SklOnlyDataset

class SklPredDataset(SklOnlyDataset):
    def __init__(self, data_path, transform=None, split='train'):
        super().__init__(data_path, transform, split)

    def __getitem__(self, idx):
        seq_idx = 0
        while idx >= self.seq_lens[seq_idx]:
            idx -= self.seq_lens[seq_idx]
            seq_idx += 1
        sample = deepcopy(self.data[seq_idx])

        sample['dataset_name'] = self.data_path.split('/')[-1].split('.')[0]
        sample['sequence_index'] = seq_idx
        sample['index'] = idx

        sample_pred = deepcopy(sample)
        sample_pred['keypoints'] = deepcopy(sample['keypoints_pred'])

        sample = self.transform(sample)
        sample_pred = self.transform(sample_pred)

        return sample, sample_pred

    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        keys = ['keypoints', 'sequence_index']
        if 'bone_dirs' in batch[0][0].keys():
            keys.append('bone_dirs')
        if 'bone_motions' in batch[0][0].keys():
            keys.append('bone_motions')
        if 'joint_motions' in batch[0][0].keys():
            keys.append('joint_motions')
        for key in keys:
            batch_data[key] = torch.stack([sample[0][key] for sample in batch], dim=0)
            batch_data['pred_' + key] = torch.stack([sample[1][key] for sample in batch], dim=0)

        return batch_data


        
        