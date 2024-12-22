import torch
from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy
import json
import os

class TiInferenceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.data = []
        self.seq_names = []
        run_dirs = os.listdir(data_dir)
        for run_dir in run_dirs:
            if os.path.isdir(os.path.join(data_dir, run_dir)):
                self.seq_names.append(run_dir)
                fns = os.listdir(os.path.join(data_dir, run_dir))
                fn = sorted(fns)[-1]
                entire_fn = os.path.join(data_dir, run_dir, fn)
                self.data.append(self.read_data_file(entire_fn))

        self.seq_lens = [len(seq['point_clouds']) for seq in self.data]

    def read_data_file(self, data_fn):
        with open(data_fn, 'r') as f:
            data = json.load(f)['data']

        data_out = {}
        data_out['point_clouds'] = []
        for data_i in data:
            if 'pointCloud' not in data_i['frameData']:
                continue
            pc = np.array(data_i['frameData']['pointCloud'])[..., :5]
            if len(pc) == 0:
                continue
            pc[..., 4] = np.clip(pc[..., 4], 0, 1000)
            pc[..., 4] /= 1000
            data_out['point_clouds'].append(pc)

        return data_out

    def __len__(self):
        return np.sum(self.seq_lens)
    
    def __getitem__(self, idx):
        seq_idx = 0
        while idx >= self.seq_lens[seq_idx]:
            idx -= self.seq_lens[seq_idx]
            seq_idx += 1
        sample = deepcopy(self.data[seq_idx])

        sample['name'] = self.seq_names[seq_idx]
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
        for key in ['point_clouds', 'centroid', 'radius']:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        batch_data['index'] = [sample['index'] for sample in batch]
        batch_data['name'] = batch[0]['name']

        return batch_data