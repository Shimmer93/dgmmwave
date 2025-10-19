import torch
from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy
import json
import os
import numpy as np
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors

def filter_point_cloud_outliers(
    point_clouds: List[np.ndarray],
    radius: float = 0.3,
    min_neighbors: int = 1,
    use_statistical_filter: bool = True,
    std_multiplier: float = 2.0
) -> List[np.ndarray]:
    """
    Filter outliers from a sequence of mmWave point clouds.
    
    Args:
        point_clouds: List of point clouds, each as (n_i, 3) numpy array
        radius: Radius for neighbor search (in meters)
        min_neighbors: Minimum number of neighbors required within radius
        use_statistical_filter: Whether to apply additional statistical outlier removal
        std_multiplier: Standard deviation multiplier for statistical filtering
    
    Returns:
        List of filtered point clouds
    """
    filtered_clouds = []
    
    for cloud in point_clouds:
        if cloud.shape[0] == 0:
            filtered_clouds.append(cloud)
            continue
            
        # Step 1: Radius-based outlier removal
        filtered_cloud = _radius_outlier_removal(cloud, radius, min_neighbors)
        
        # Step 2: Statistical outlier removal (optional)
        if use_statistical_filter and filtered_cloud.shape[0] > 0:
            filtered_cloud = _statistical_outlier_removal(filtered_cloud, std_multiplier)
        
        filtered_clouds.append(filtered_cloud)
    
    return filtered_clouds

def _radius_outlier_removal(
    points: np.ndarray, 
    radius: float, 
    min_neighbors: int
) -> np.ndarray:
    """Remove points that don't have enough neighbors within radius."""
    if points.shape[0] <= min_neighbors:
        return points
    
    nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.radius_neighbors(points)
    
    # Count neighbors (excluding the point itself)
    neighbor_counts = np.array([len(idx) - 1 for idx in indices])
    
    # Keep points with sufficient neighbors
    valid_mask = neighbor_counts >= min_neighbors
    return points[valid_mask]

def _statistical_outlier_removal(
    points: np.ndarray, 
    std_multiplier: float = 2.0,
    k_neighbors: int = 10
) -> np.ndarray:
    """Remove points based on statistical analysis of distances to neighbors."""
    if points.shape[0] <= k_neighbors:
        return points
    
    # Use k-nearest neighbors to compute mean distances
    k = min(k_neighbors, points.shape[0] - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Calculate mean distance to k nearest neighbors (excluding self)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # Compute statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # Filter points within statistical threshold
    threshold = global_mean + std_multiplier * global_std
    valid_mask = mean_distances <= threshold
    
    if not np.any(valid_mask):
        # If no points are valid, return the original points
        return points
    return points[valid_mask]

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
                # print(f'in dataset: {run_dir}')
                self.data.append(self.read_data_file(entire_fn, run_dir))

        self.seq_lens = [len(seq['point_clouds']) for seq in self.data]
        # print(self.seq_names, self.seq_lens)

    def read_data_file(self, data_fn, name):
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
            pc[..., 1] -= 0.3

            if 'small' in name:
                mask = pc[..., 1] < 2.5
            else:
                # mask = (pc[..., 0] < -0.5) | (pc[..., 1] < 1.8) | (pc[..., 2] < 1.35)
                mask = (pc[..., 1] < 1.8)
            if np.sum(mask) == 0:
                continue
            pc = pc[mask]

            data_out['point_clouds'].append(pc)
        # data_out['point_clouds'] = filter_point_cloud_outliers(data_out['point_clouds'])

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
        for key in ['point_clouds', 'centroid', 'radius', 'sequence_index']:
            batch_data[key] = torch.stack([sample[key] for sample in batch], dim=0)
        batch_data['index'] = [sample['index'] for sample in batch]
        batch_data['name'] = [sample['name'] for sample in batch]

        return batch_data