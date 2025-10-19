import torch
import pickle
import random
import numpy as np
from copy import deepcopy
from itertools import chain
import json
import os

from dataset.temporal_dataset import TemporalDataset

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from typing import List, Tuple

def filter_point_cloud_outliers(
    point_clouds: List[np.ndarray],
    distance_threshold: float = 0.3,
    min_cluster_size: int = 3,
    static_threshold: float = 0.05,
    min_temporal_frames: int = 3
) -> List[np.ndarray]:
    """
    Filter outliers from a sequence of point clouds for human pose estimation.
    
    Args:
        point_clouds: List of point clouds, each with shape (n_i, 3)
        distance_threshold: Maximum distance for points to be considered neighbors (DBSCAN eps)
        min_cluster_size: Minimum points required to form a cluster (DBSCAN min_samples)
        static_threshold: Maximum movement threshold to consider cluster as static
        min_temporal_frames: Minimum frames a cluster must exist to be considered static
    
    Returns:
        List of filtered point clouds
    """
    if not point_clouds:
        return []
    
    filtered_clouds = []
    
    # Track clusters across frames for static detection
    cluster_history = []
    
    for frame_idx, cloud in enumerate(point_clouds):
        if cloud.shape[0] == 0:
            filtered_clouds.append(cloud)
            cluster_history.append([])
            continue
        
        # Step 1: Filter isolated points using DBSCAN
        if cloud.shape[0] >= min_cluster_size:
            clustering = DBSCAN(eps=distance_threshold, min_samples=min_cluster_size)
            labels = clustering.fit_predict(cloud)
            
            # Keep only points that belong to clusters (label != -1)
            clustered_mask = labels != -1
            clustered_points = cloud[clustered_mask]
            clustered_labels = labels[clustered_mask]
        else:
            # If too few points, keep all
            clustered_points = cloud.copy()
            clustered_labels = np.zeros(len(cloud))
        
        if clustered_points.shape[0] == 0:
            filtered_clouds.append(np.empty((0, 3)))
            cluster_history.append([])
            continue
        
        # Step 2: Analyze cluster movement to filter static clusters
        current_clusters = []
        unique_labels = np.unique(clustered_labels)
        
        for label in unique_labels:
            cluster_mask = clustered_labels == label
            cluster_points = clustered_points[cluster_mask]
            cluster_center = np.mean(cluster_points, axis=0)
            current_clusters.append({
                'center': cluster_center,
                'points': cluster_points,
                'size': len(cluster_points)
            })
        
        cluster_history.append(current_clusters)
        
        # Step 3: Filter static clusters if we have enough history
        if frame_idx >= min_temporal_frames - 1:
            dynamic_points = []
            
            for cluster in current_clusters:
                is_dynamic = True
                
                # Check if this cluster has been static across recent frames
                if len(cluster_history) >= min_temporal_frames:
                    movements = []
                    
                    # Look back through recent frames
                    for hist_idx in range(max(0, len(cluster_history) - min_temporal_frames), 
                                        len(cluster_history) - 1):
                        hist_clusters = cluster_history[hist_idx]
                        
                        if hist_clusters:
                            # Find closest cluster in historical frame
                            distances = [np.linalg.norm(cluster['center'] - hist_cluster['center']) 
                                       for hist_cluster in hist_clusters]
                            
                            if distances:
                                min_distance = min(distances)
                                movements.append(min_distance)
                    
                    # If cluster has been consistently static, mark for removal
                    if movements and np.mean(movements) < static_threshold:
                        is_dynamic = False
                
                if is_dynamic:
                    dynamic_points.append(cluster['points'])
            
            # Combine all dynamic cluster points
            if dynamic_points:
                filtered_points = np.vstack(dynamic_points)
            else:
                filtered_points = np.empty((0, 3))
        else:
            # For early frames, just use clustered points
            filtered_points = clustered_points
        
        filtered_clouds.append(filtered_points)
    
    return filtered_clouds

class ReferenceTIDataset(TemporalDataset):
    def __init__(self, data_path, ref_data_dir, transform=None, ref_transform=None, split='train', ratio=1, trans=False, both=False, ref_trans=False, ref_both=False):
        super().__init__(data_path, transform, split, ratio, trans, both)
        self.ref_data_path = ref_data_dir
        self.ref_transform = ref_transform
        self.ref_trans = ref_trans
        self.ref_both = ref_both

        self.ref_data = []
        self.ref_seq_names = []
        run_dirs = os.listdir(ref_data_dir)
        for run_dir in run_dirs:
            if os.path.isdir(os.path.join(ref_data_dir, run_dir)):
                self.ref_seq_names.append(run_dir)
                fns = os.listdir(os.path.join(ref_data_dir, run_dir))
                fn = sorted(fns)[-1]
                entire_fn = os.path.join(ref_data_dir, run_dir, fn)
                self.ref_data.append(self.read_data_file(entire_fn))

        self.ref_seq_lens = [len(seq['point_clouds']) for seq in self.ref_data]
        self.ref_len = np.sum(self.ref_seq_lens)

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
        data_out['point_clouds'] = filter_point_cloud_outliers(data_out['point_clouds'])

        return data_out
    
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