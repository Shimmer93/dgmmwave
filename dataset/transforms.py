import numpy as np
import torch
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from miniball import get_bounding_ball
# from pytorch3d.loss import chamfer_distance

from misc.skeleton import coco2simplecoco, mmbody2simplecoco, mmfi2simplecoco, itop2simplecoco, mmfi2itop, mmbody2itop, ITOPSkeleton

def log(x):
    print(x)
    with open('log.txt', 'a') as f:
        f.write(str(x) + '\n')

class CalculateBoneDirections():
    def __init__(self, norm=True):
        self.norm = norm

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        bone_dirs = []
        for bone in ITOPSkeleton.bones:
            bone_dir = sample['keypoints'][:, bone[1]] - sample['keypoints'][:, bone[0]]
            if self.norm:
                bone_dir /= np.linalg.norm(bone_dir, axis=-1, keepdims=True)
            bone_dirs.append(bone_dir)
        sample['bone_dirs'] = np.stack(bone_dirs, axis=1)

        return sample

class CalculateBoneMotions():
    def __init__(self):
        pass

    def __call__(self, sample):
        bone_motions = sample['bone_dirs'][1:] - sample['bone_dirs'][:-1]
        # bone_motions /= np.linalg.norm(bone_motions, axis=-1, keepdims=True)
        sample['bone_motions'] = bone_motions
        return sample
    
class CalculateJointMotions():
    def __init__(self, norm=True):
        self.norm = norm

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        joint_motions = sample['keypoints'][1:] - sample['keypoints'][:-1]
        if self.norm:
            joint_motions /= (np.linalg.norm(joint_motions, axis=-1, keepdims=True) + 1e-6)
        sample['joint_motions'] = joint_motions

        return sample
    
class AddNoisyPoints():
    def __init__(self, add_std=0.01, num_added=32, zero_centered=True):
        self.add_std = add_std
        self.num_added = num_added
        self.zero_centered = zero_centered

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            if self.zero_centered:
                noise = np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][i].shape[1]))
            else:
                noise_center = np.random.uniform(-1, 1, sample['point_clouds'][i].shape[1])
                noise = np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][i].shape[1])) + noise_center
            sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i], noise], axis=0)
        
        return sample

class GenerateSegmentationGroundTruth():
    def __init__(self, padding=0.2):
        self.padding = padding

    def __call__(self, sample):
        if 'keypoints' not in sample:
            sample['segmented'] = False
            return sample

        for i in range(len(sample['keypoints'])):
            # mask = np.zeros(sample['point_clouds'][i].shape[0], dtype=np.bool)
            # for bone in ITOPSkeleton.bones:
            #     mask |= self._mask_single_bone(sample['point_clouds'][i][...,:3], sample['keypoints'][i], bone, self.padding)
            # sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i], mask[...,np.newaxis].astype(sample['point_clouds'][i].dtype)], axis=-1)
            # if len(sample['keypoints'][i]) == 0:
            #     sample['point_clouds'][i] = np.zeros((1, 3), dtype=sample['keypoints'][i].dtype)
            
            neighbors = NearestNeighbors(radius=self.padding).fit(sample['keypoints'][i])
            indices = neighbors.radius_neighbors(sample['point_clouds'][i][...,:3], return_distance=False)
            sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i], np.array([len(idx) > 0 for idx in indices], dtype=np.float32)[...,np.newaxis]], axis=-1)

            neighbors = NearestNeighbors(n_neighbors=1).fit(sample['keypoints'][i])
            _, indices = neighbors.kneighbors(sample['point_clouds'][i][...,:3])
            sample['point_clouds'][i][..., -1:] *= (indices + 1).astype(sample['point_clouds'][i].dtype)
            # new_pcs.append(np.concatenate([sample['point_clouds'][i], indices.astype(sample['point_clouds'][i].dtype)], axis=-1))

        sample['segmented'] = True
        return sample

    def _mask_single_bone(self, point_cloud, keypoints, bone, padding=0.1):
        start = keypoints[bone[0]]
        end = keypoints[bone[1]]
        vec = end - start
        length = np.linalg.norm(vec)
        vec /= length
        start -= vec * padding
        end += vec * padding
        mask_cylinder = np.linalg.norm(np.cross(point_cloud - start, point_cloud - end), axis=-1) / length < padding
        mask_start = np.dot(point_cloud - start, vec) > 0
        mask_end = np.dot(point_cloud - end, -vec) > 0
        mask = mask_cylinder & mask_start & mask_end # The mask is a cylinder with padding around the bone
        return mask

class ConvertToMMWavePointCloud():
    def __init__(self, max_dist_threshold=0.1, add_std=0.1, default_num_points=32, num_noisy_points=32):
        self.max_dist_threshold = max_dist_threshold
        self.add_std = add_std
        self.default_num_points = default_num_points
        self.num_noisy_points = num_noisy_points

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        kps0 = sample['keypoints'][:-1]
        kps1 = sample['keypoints'][1:]
        kps_dist = np.linalg.norm(kps0 - kps1, axis=-1)
        # print(kps_dist)
        # dist_threshold = np.random.rand() * self.max_dist_threshold
        dist_threshold = self.max_dist_threshold
        mask = kps_dist > dist_threshold
        # print(mask)

        new_pcs = []
        for i in range(len(sample['keypoints']) - 1):
            pc0 = sample['point_clouds'][i]
            
            num_points = 0
            new_pc = []
            for j in range(sample['keypoints'].shape[1]):
                pc0_j = pc0[sample['point_clouds'][i][..., -1] == j+1]
                if mask[i, j]:
                    new_pc.append(pc0_j)
                    num_points += len(pc0_j)
                    # add_points = sample['keypoints'][i][j][np.newaxis, :].repeat(self.num_noisy_points, axis=0) + \
                    #                 np.random.normal(0, self.add_std, (self.num_noisy_points, sample['point_clouds'][-1].shape[-1]))
                    # new_pc.append(add_points)
                    # num_points += self.num_noisy_points

            if num_points == 0:
                random_idxs = np.random.choice(pc0.shape[0], self.default_num_points)
                new_pc.append(pc0[random_idxs])
            new_pc = np.concatenate(new_pc)
            new_pcs.append(new_pc)

        sample['point_clouds'] = new_pcs
        sample['keypoints'] = sample['keypoints'][:-1]
        return sample

class ConvertToRefinedMMWavePointCloud():
    def __init__(self, max_dist_threshold=0.1, min_dist_threshold=0.05, add_std=0.1, default_num_points=32, num_noisy_points=32):
        self.max_dist_threshold = max_dist_threshold
        self.min_dist_threshold = min_dist_threshold
        self.add_std = add_std
        self.default_num_points = default_num_points
        self.num_noisy_points = num_noisy_points

    def __call__(self, sample):
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = np.stack(sample['keypoints'])

        keypoints = sample['keypoints']
        point_clouds = sample['point_clouds']

        # T, N, _ = point_clouds.shape
        T, J, _ = keypoints.shape
        # flow_thres = self.max_dist_threshold
        # flow_thres = np.random.rand() * self.max_dist_threshold
        # random number between min_dist_threshold and max_dist_threshold
        flow_thres = np.random.uniform(self.min_dist_threshold, self.max_dist_threshold)

        # 1. Calculate keypoint flow (temporal displacement)
        keypoint_flow = keypoints[1:] - keypoints[:-1]  # (T-1, J, 3)

        
        new_pcs = []
        for t in range(T-1):
            pc = point_clouds[t][:, :3]
            kp = keypoints[t][:, :3]  # (J, 3)
            kpf = keypoint_flow[t]
            # print(np.linalg.norm(kpf, axis=-1))
            pc_expanded = pc[:, np.newaxis, :]  # (N, 1, 3)
            kp_expanded = kp[np.newaxis, :, :]  # (1, J, 3)
            kpf_expanded = kpf[np.newaxis, :, :]  # (1, J, 3)

            # Calculate pairwise distances between points and keypoints
            pairwise_distances = np.linalg.norm(pc_expanded - kp_expanded, axis=-1)  # (N, J)

            # 3. Calculate estimated point cloud flow
            # Use inverse distance weighting to estimate flow for each point
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            weights = 1.0 / (pairwise_distances + eps)  # (N, J)
            weights_normalized = weights / (weights.sum(axis=-1, keepdims=True) + eps)  # (N, J)

            # Calculate weighted average of keypoint flows for each point
            weights_expanded = weights_normalized[:, :, np.newaxis]  # (N, J, 1)
            pcf = (weights_expanded * kpf_expanded).sum(axis=1)  # (N, 3)
            pcf = np.linalg.norm(pcf, axis=-1)  # (N,)
            # print(f'Point cloud flow at time {t}: {np.mean(pcf):.4f} ± {np.std(pcf):.4f}, min: {np.min(pcf):.4f}, max: {np.max(pcf):.4f}')

            # 4. Filter points based on flow threshold
            mask = pcf > flow_thres  # (N,)
            # print(np.sum(mask), np.any(mask))
            # print(f'{np.sum(mask)} / {pc.shape[0]} points above flow threshold {flow_thres:.4f} at time {t}')
            if np.any(mask):
                new_pc = pc[mask]
            else:
                # If no points are below the threshold, randomly sample points
                random_idxs = np.random.choice(pc.shape[0], self.default_num_points)
                new_pc = pc[random_idxs]
            new_pcs.append(new_pc)
    
        sample['point_clouds'] = new_pcs
        sample['keypoints'] = sample['keypoints'][:-1]
        return sample

# class ConvertToMMWavePointCloud():
#     def __init__(self, dist_threshold=0.1, default_num_points=32):
#         self.dist_threshold = dist_threshold
#         self.default_num_points = default_num_points

#     def __call__(self, sample):
#         N = len(sample['point_clouds'])
#         J = sample['keypoints'][0].shape[0]

#         max_len = max([pc.shape[0] for pc in sample['point_clouds']])

#         pc0_tensor = torch.zeros((N - 1, J, max_len, 4))
#         pc1_tensor = torch.zeros((N - 1, J, max_len, 4))
#         pc0_lens = torch.zeros((N - 1, J), dtype=torch.long)
#         pc1_lens = torch.zeros((N - 1, J), dtype=torch.long)

#         for i in range(N - 1):
#             pc0 = sample['point_clouds'][i]
#             pc1 = sample['point_clouds'][i+1]
#             seg0 = sample['point_clouds'][i][..., -1:]
#             seg1 = sample['point_clouds'][i+1][..., -1:]
#             for j in range(J):
#                 pc0_j = pc0[seg0[:, 0] == j+1]
#                 pc1_j = pc1[seg1[:, 0] == j+1]
#                 pc0_tensor[i, j, :pc0_j.shape[0]] = torch.from_numpy(pc0_j).float()
#                 pc1_tensor[i, j, :pc1_j.shape[0]] = torch.from_numpy(pc1_j).float()
#                 pc0_lens[i, j] = pc0_j.shape[0]
#                 pc1_lens[i, j] = pc1_j.shape[0]

#         dists = chamfer_distance(pc0_tensor.reshape(-1, max_len, 4)[..., :3], 
#                                  pc1_tensor.reshape(-1, max_len, 4)[..., :3], 
#                                  x_lengths=pc0_lens.reshape(-1), 
#                                  y_lengths=pc1_lens.reshape(-1),
#                                  batch_reduction=None)[0].reshape(N - 1, J)
#         mask = dists > self.dist_threshold

#         new_pcs = []
#         for i in range(N - 1):
#             pc0 = sample['point_clouds'][i]

#             new_pc = []
#             for j in range(J):
#                 pc0_j = pc0_tensor[i, j, :int(pc0_lens[i, j])].numpy()
#                 pc1_j = pc1_tensor[i, j, :int(pc1_lens[i, j])].numpy()
#                 if mask[i, j]:
#                     new_pc.append(pc0_j)

#             if len(new_pc) == 0:
#                 random_idxs = np.random.choice(pc0.shape[0], self.default_num_points)
#                 new_pc.append(pc0[random_idxs])
#             new_pc = np.concatenate(new_pc)
#             new_pcs.append(new_pc)

#         sample['point_clouds'] = new_pcs
#         return sample

class DropPointsAtSegmentedJoints():
    def __init__(self, max_num2drop=3):
        self.max_num2drop = max_num2drop

    def __call__(self, sample):        
        assert 'segmented' in sample
        if not sample['segmented']:
            return sample
        
        num_joints = sample['keypoints'][0].shape[0]
        num2drop = np.random.randint(1, self.max_num2drop)
        idxs2drop = np.random.choice(num_joints, num2drop, replace=False)

        new_pcs = []
        for i in range(len(sample['keypoints'])):
            pc = sample['point_clouds'][i]
            seg = sample['point_clouds'][i][..., -1:]
            mask = np.isin(seg, idxs2drop+1)[:, 0]
            new_pcs.append(pc[~mask, :] if np.any(~mask) else pc)

        sample['point_clouds'] = new_pcs
        return sample
    
class AddPointsAroundJoint():
    def __init__(self, add_std=0.1, max_num2add=1, num_added=32):
        self.add_std = add_std
        self.max_num2add = max_num2add
        self.num_added = num_added

    def __call__(self, sample):
        num_joints = sample['keypoints'][0].shape[0]
        num2add = np.random.randint(1, self.max_num2add)
        idxs2add = np.random.choice(num_joints, num2add, replace=False)

        new_pcs = []
        for i in range(len(sample['keypoints'])):
            pc = sample['point_clouds'][i]
            for idx in idxs2add:
                add_point = sample['keypoints'][i][idx]
                if add_point.shape[-1] < pc.shape[-1]:
                    add_point = np.concatenate([add_point, np.zeros(pc.shape[-1]-add_point.shape[-1])], axis=-1)
                add_points = add_point[np.newaxis, :].repeat(self.num_added, axis=0) + np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][-1].shape[-1]))
                pc = np.concatenate([pc, add_points], axis=0)
            new_pcs.append(pc)

        sample['point_clouds'] = new_pcs
        return sample
    
class GenerateBinaryGroundTruth():
    def __init__(self, radius=0.1):
        self.radius = radius

    def __call__(self, sample):
        new_pcs = []
        for i in range(len(sample['keypoints'])):
            neighbors = NearestNeighbors(radius=self.radius).fit(sample['keypoints'][i])
            indices = neighbors.radius_neighbors(sample['point_clouds'][i][...,:3], return_distance=False)
            new_pcs.append(np.concatenate([sample['point_clouds'][i], np.array([len(idx) > 0 for idx in indices], dtype=np.float32)[...,np.newaxis]], axis=-1))
        sample['point_clouds'] = new_pcs
        return sample

class RemoveOutliers():
    def __init__(self, outlier_type='statistical', num_neighbors=3, std_multiplier=1.0, radius=1.0, min_neighbors=2):
        self.outlier_type = outlier_type
        self.num_neighbors = num_neighbors
        self.std_multiplier = std_multiplier
        self.radius = radius
        self.min_neighbors = min_neighbors
        if outlier_type not in ['statistical', 'radius', 'cluster', 'box']:
            raise ValueError('outlier_type must be "statistical" or "radius" or "cluster" or "box"')

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            if self.outlier_type == 'statistical':
                neighbors = NearestNeighbors(n_neighbors=self.num_neighbors+1).fit(sample['point_clouds'][i][...,:3])
                distances, _ = neighbors.kneighbors(sample['point_clouds'][i][...,:3])
                mean_dist = np.mean(distances[:, 1:], axis=1)
                std_dist = np.std(distances[:, 1:], axis=1)
                dist_threshold = mean_dist + self.std_multiplier * std_dist
                inliers = np.where(distances[:, 1:] < dist_threshold[:, np.newaxis])
            elif self.outlier_type == 'radius':
                neighbors = NearestNeighbors(radius=self.radius).fit(sample['point_clouds'][i][...,:3])
                distances, _ = neighbors.radius_neighbors(sample['point_clouds'][i][...,:3], return_distance=True)
                inliers = np.where([len(d) >= self.min_neighbors for d in distances])
            elif self.outlier_type == 'cluster':
                clusterer = HDBSCAN(min_cluster_size=self.min_neighbors)
                inliers = clusterer.fit_predict(sample['point_clouds'][i][...,:3]) != -1
            elif self.outlier_type == 'box':
                inliers = np.where(np.all(np.abs(sample['point_clouds'][i][...,:2]-np.array([[0,1]])) < self.radius, axis=1))
            else:
                raise ValueError('You should never reach here!')
            if len(inliers[0]) == 0:
                sample['point_clouds'][i] = sample['point_clouds'][i][:1]
                # num_outliers = sample['point_clouds'][i].shape[0] - len(inliers[0])
                # print(f'{num_outliers} outliers removed')
            else:
                sample['point_clouds'][i] = sample['point_clouds'][i][inliers]
        return sample

class Pad():
    def __init__(self, max_len, pad_type='repeat'):
        self.max_len = max_len
        self.pad_type = pad_type
        if pad_type not in ['zero', 'repeat']:
            raise ValueError('pad_type must be "zero" or "repeat"')

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            cur_len = sample['point_clouds'][i].shape[0]
            if cur_len == 0:
                # add random points if the point cloud is empty
                sample['point_clouds'][i] = np.random.normal(0, 1, (self.max_len, sample['point_clouds'][i].shape[1]))
            elif cur_len >= self.max_len:
                indices = np.random.choice(cur_len, self.max_len, replace=False)
                sample['point_clouds'][i] = sample['point_clouds'][i][indices]
            else:
                if self.pad_type == 'zero':
                    sample['point_clouds'][i] = np.pad(sample['point_clouds'][i], ((0, self.max_len - sample['point_clouds'][i].shape[0]), (0, 0)), mode='constant')
                elif self.pad_type == 'repeat':
                    repeat = self.max_len // cur_len
                    residue = self.max_len % cur_len
                    indices = np.random.choice(cur_len, residue, replace=False)
                    sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i] for _ in range(repeat)] + [sample['point_clouds'][i][indices]], axis=0)
                else:
                    raise ValueError('You should never reach here! pad_type must be "zero" or "repeat"')
        sample['point_clouds'] = np.stack(sample['point_clouds'])
        return sample
    
class UniformSample():
    def __init__(self, clip_len, pad_type='both'):
        self.clip_len = clip_len
        if pad_type not in ['both', 'start', 'end']:
            raise ValueError('pad_type must be "both" or "start" or "end"')
        if pad_type == 'both':
            assert clip_len % 2 == 1, 'num_frames must be odd'
            self.pad = (clip_len - 1) // 2
        else:
            self.pad = clip_len - 1
        self.pad_type = pad_type


    def __call__(self, sample):
        # while len(sample['point_clouds']) < self.clip_len:
        #     sample['point_clouds'].append(sample['point_clouds'][-1])
        #     sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)
        if self.pad_type == 'both':
            for _ in range(self.pad):
                if 'point_clouds' in sample:
                    sample['point_clouds'].insert(0, sample['point_clouds'][0])
                    sample['point_clouds'].append(sample['point_clouds'][-1])
                if 'keypoints' in sample:
                    sample['keypoints'] = np.concatenate([sample['keypoints'][0][np.newaxis], sample['keypoints']], axis=0)
                    sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)
                if 'flow' in sample:
                    sample['flow'] = np.concatenate([np.zeros_like(sample['flow'][0][np.newaxis]), sample['flow']], axis=0)
                    sample['flow'] = np.concatenate([sample['flow'], np.zeros_like(sample['flow'][0][np.newaxis])], axis=0)
        elif self.pad_type == 'start':
            for _ in range(self.pad):
                if 'point_clouds' in sample:
                    sample['point_clouds'].insert(0, sample['point_clouds'][0])
                if 'keypoints' in sample:
                    sample['keypoints'] = np.concatenate([sample['keypoints'][0][np.newaxis], sample['keypoints']], axis=0)
                if 'flow' in sample:
                    sample['flow'] = np.concatenate([np.zeros_like(sample['flow'][0][np.newaxis]), sample['flow']], axis=0)
        elif self.pad_type == 'end':
            for _ in range(self.pad):
                if 'point_clouds' in sample:
                    sample['point_clouds'].append(sample['point_clouds'][-1])
                if 'keypoints' in sample:
                    sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)
                if 'flow' in sample:
                    sample['flow'] = np.concatenate([sample['flow'], np.zeros_like(sample['flow'][0][np.newaxis])], axis=0)
        # start_idx = np.random.randint(0, len(sample['point_clouds']) - self.clip_len + 1)
        start_idx = sample['index']
        if 'point_clouds' in sample:
            sample['point_clouds'] = sample['point_clouds'][start_idx:start_idx+self.clip_len]
        if 'keypoints' in sample:
            sample['keypoints'] = sample['keypoints'][start_idx:start_idx+self.clip_len]
        if 'flow' in sample:
            sample['flow'] = sample['flow'][start_idx:start_idx+self.clip_len]
        # print('uniform sample', len(sample['point_clouds']), len(sample['keypoints']))

        return sample
    
class MultiFrameAggregate():
    def __init__(self, num_frames):
        self.num_frames = num_frames
        assert num_frames % 2 == 1, 'num_frames must be odd'
        self.offset = (num_frames - 1) // 2

    def __call__(self, sample):
        total_frames = len(sample['point_clouds'])
        if self.num_frames <= total_frames:
            sample['point_clouds'] = [np.concatenate(sample['point_clouds'][i-self.offset:i+self.offset]) for i in range(self.offset, total_frames-self.offset)]
            # sample['point_clouds'] = [np.concatenate(sample['point_clouds'][np.maximum(0, i-self.offset):np.minimum(i+self.offset+1, total_frames-1)]) for i in range(total_frames)]
            if 'keypoints' in sample:
                sample['keypoints'] = sample['keypoints'][self.offset:-self.offset]
        # print('multi frame aggregate', len(sample['point_clouds']), len(sample['keypoints']))
        return sample

class RandomScale():
    def __init__(self, scale_min=0.9, scale_max=1.1):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample):
        scale = np.random.uniform(self.scale_min, self.scale_max)
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] *= scale
        if 'keypoints' in sample:
            sample['keypoints'] *= scale
        sample['scale'] = scale
        return sample
    
class RandomRotate():
    def __init__(self, angle_min=-np.pi, angle_max=np.pi, deg=False):
        self.angle_min = angle_min
        self.angle_max = angle_max

        if deg:
            angle_min = np.pi * angle_min / 180
            angle_max = np.pi * angle_max / 180

    def __call__(self, sample):
        angle_1 = np.random.uniform(self.angle_min, self.angle_max)
        angle_2 = np.random.uniform(self.angle_min, self.angle_max)
        rot_matrix = np.array([[np.cos(angle_1), -np.sin(angle_1), 0], [np.sin(angle_1), np.cos(angle_1), 0], [0, 0, 1]]) @ np.array([[np.cos(angle_2), 0, np.sin(angle_2)], [0, 1, 0], [-np.sin(angle_2), 0, np.cos(angle_2)]])
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] = sample['point_clouds'][i][...,:3] @ rot_matrix
        if 'keypoints' in sample:
            sample['keypoints'] = sample['keypoints'] @ rot_matrix
        sample['rotation_matrix'] = rot_matrix
        return sample
    
class RandomTranslate():
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, sample):
        translate = np.random.uniform(-self.translate_range, self.translate_range, 3)
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] += translate
        if 'keypoints' in sample:
            sample['keypoints'] += translate
        sample['translate'] = translate
        return sample

class RandomJitter():
    def __init__(self, jitter_std=0.01):
        self.jitter_std = jitter_std

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] += np.random.normal(0, self.jitter_std, sample['point_clouds'][i][...,:3].shape)
        return sample

class RandomJitterKeypoints():
    def __init__(self, jitter_std=0.01):
        self.jitter_std = jitter_std

    def __call__(self, sample):
        for i in range(len(sample['keypoints'])):
            sample['keypoints'][i][...,:3] += np.random.normal(0, self.jitter_std, sample['keypoints'][i][...,:3].shape)
        return sample
    
class RandomDrop():
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            drop_indices = np.random.choice(sample['point_clouds'][i].shape[0], int(sample['point_clouds'][i].shape[0] * self.drop_prob), replace=False)
            sample['point_clouds'][i] = np.delete(sample['point_clouds'][i], drop_indices, axis=0)
        return sample

class GetCentroid():
    def __init__(self, centroid_type='minball'):
        self.centroid_type = centroid_type
        if centroid_type not in ['none', 'zonly', 'mean', 'median', 'minball', 'dataset_median', 'kps']:
            raise ValueError('centroid_type must be "mean" or "minball"')
        
    def __call__(self, sample):
        pc_cat = np.concatenate(sample['point_clouds'], axis=0)
        pc_dedupe = np.unique(pc_cat[...,:3], axis=0)
        if self.centroid_type == 'none':
            centroid = np.zeros(3)
        elif self.centroid_type == 'zonly':
            centroid = np.zeros(3)
            centroid[2] = np.median(pc_dedupe[...,2])
        elif self.centroid_type == 'mean':
            centroid = np.mean(pc_dedupe[...,:3], axis=0)
        elif self.centroid_type == 'median':
            centroid = np.median(pc_dedupe[...,:3], axis=0)
        elif self.centroid_type == 'minball':
            try:
                centroid, _ = get_bounding_ball(pc_dedupe)
            except:
                print('Error in minball')
                centroid = np.mean(pc_dedupe[...,:3], axis=0)
        elif self.centroid_type == 'dataset_median':
            if sample['dataset_name'] == 'mmbody':
                centroid = np.array([0., 0., 3.50313997])
            elif sample['dataset_name'] == 'mri':
                centroid = np.array([0., -0.012695, 2.3711])
            elif sample['dataset_name'] == 'mmfi':
                centroid = np.array([-0.09487205, -0.09743616, 3.04673481])
            elif sample['dataset_name'] == 'mmfi_lidar':
                centroid = np.array([-0.02519154, -0.3084462, 2.99475336])
            elif sample['dataset_name'] == 'itop_side':
                centroid = np.array([-0.0716, -0.25, 2.908])
            elif sample['dataset_name'] == 'itop_top':
                centroid = np.array([-0.05297852, -1.19726562, 0.08111572])
            else:
                raise NotImplementedError
        elif self.centroid_type == 'kps':
            kps_cat = np.concatenate(sample['keypoints'], axis=0)
            centroid = np.array([np.median(kps_cat[:, 0]), np.min(kps_cat[:, 1]), np.median(kps_cat[:, 2])])
        else:
            raise ValueError('You should never reach here! centroid_type must be "mean" or "minball"')
        sample['centroid'] = centroid

        return sample

class Normalize():
    def __init__(self, feat_scale=None):
        self.feat_scale = feat_scale

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] -= sample['centroid'][np.newaxis]
            if self.feat_scale:
                sample['point_clouds'][i][...,3:] /= np.array(self.feat_scale)[np.newaxis][np.newaxis]
                sample['feat_scale'] = self.feat_scale
        if 'keypoints' in sample:
            sample['keypoints'] -= sample['centroid'][np.newaxis][np.newaxis]
        return sample
    
class Flip():
    def __init__(self, left_idxs, right_idxs):
        self.left_idxs = left_idxs
        self.right_idxs = right_idxs

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][..., 0] *= -1 
        indices = np.arange(sample['keypoints'].shape[1], dtype=np.int64)
        left, right = (self.left_idxs, self.right_idxs)
        for l, r in zip(left, right):
            indices[l] = r
            indices[r] = l
        if 'keypoints' in sample:
            sample['keypoints'] = sample['keypoints'][:, indices]
            sample['keypoints'][..., 0] *= -1
        return sample
    
class ToSimpleCOCO():
    def __call__(self, sample):
        if sample['dataset_name'] in ['mmbody', 'lidarhuman26m', 'hmpear']:
            transfer_func = mmbody2simplecoco
        elif sample['dataset_name'] == 'mri':
            transfer_func = coco2simplecoco
        elif sample['dataset_name'] in ['mmfi', 'mmfi_lidar']:
            transfer_func = mmfi2simplecoco
        elif sample['dataset_name'] in ['itop_side', 'itop_top']:
            transfer_func = itop2simplecoco
        else:
            raise ValueError('You should never reach here! dataset_name must be "mmbody", "mri", "mmfi", "itop_side" or "itop_top"')
        
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = [transfer_func(kp) for kp in sample['keypoints']]
        else:
            sample['keypoints'] = transfer_func(sample['keypoints'])
        return sample

class ToITOP():
    def __call__(self, sample):
        if sample['dataset_name'] in ['mmfi', 'mmfi_lidar']:
            transfer_func = mmfi2itop
        elif sample['dataset_name'] in ['mmbody', 'lidarhuman26m', 'hmpear']:
            transfer_func = mmbody2itop
        elif sample['dataset_name'] in ['itop_side', 'itop_top']:
            transfer_func = lambda x: x
        else:
            raise ValueError('You should never reach here! dataset_name must be "mmbody", "mri", "mmfi", "itop_side" or "itop_top"')
        
        if isinstance(sample['keypoints'], list):
            sample['keypoints'] = [transfer_func(kp) for kp in sample['keypoints']]
        else:
            sample['keypoints'] = transfer_func(sample['keypoints'])
        return sample

class ToTensor():
    def __call__(self, sample):
        if 'point_clouds' in sample:
            if isinstance(sample['point_clouds'], list):
                sample['point_clouds'] = [torch.from_numpy(pc).float() for pc in sample['point_clouds']]
            else:
                sample['point_clouds'] = torch.from_numpy(sample['point_clouds']).float()
        if 'keypoints' in sample:
            sample['keypoints'] = torch.from_numpy(sample['keypoints']).float()
        if 'flow' in sample:
            sample['flow'] = torch.from_numpy(sample['flow']).float()
        if 'action' in sample:
            sample['action'] = torch.tensor([sample['action']], dtype=torch.long)
        if 'sequence_index' in sample:
            sample['sequence_index'] = torch.tensor([sample['sequence_index']], dtype=torch.long)
        if 'index' in sample:
            sample['index'] = torch.tensor([sample['index']], dtype=torch.long)
        if 'centroid' in sample:
            sample['centroid'] = torch.from_numpy(sample['centroid']).float()
        if 'radius' in sample:
            sample['radius'] = torch.tensor([sample['radius']]).float()
        if 'scale' in sample:
            sample['scale'] = torch.tensor([sample['scale']]).float()
        if 'translate' in sample:
            sample['translate'] = torch.from_numpy(sample['translate']).float()
        if 'rotation_matrix' in sample:
            sample['rotation_matrix'] = torch.from_numpy(sample['rotation_matrix']).float()
        if 'bone_dirs' in sample:
            sample['bone_dirs'] = torch.from_numpy(sample['bone_dirs']).float()
        if 'bone_motions' in sample:
            sample['bone_motions'] = torch.from_numpy(sample['bone_motions']).float()
        if 'joint_motions' in sample:
            sample['joint_motions'] = torch.from_numpy(sample['joint_motions']).float()
        return sample

class ReduceKeypointLen():
    def __init__(self, only_one=False, keep_type='middle', frame_to_reduce=1, indexs_to_keep=None):
        self.only_one = only_one
        assert keep_type in ['middle', 'start', 'end'], 'keep_type must be "middle", "start" or "end"'
        self.keep_type = keep_type
        self.frame_to_reduce = frame_to_reduce
        self.indexs_to_keep = indexs_to_keep

    def __call__(self, sample):
        if self.only_one:
            num_frames = len(sample['point_clouds'])
            if self.keep_type == 'middle':
                keep_idx = (num_frames - 1) // 2
            elif self.keep_type == 'start':
                keep_idx = 0
            else:
                keep_idx = num_frames - 1
            sample['keypoints'] = sample['keypoints'][keep_idx:keep_idx+1]
        elif self.indexs_to_keep is not None:
            sample['keypoints'] = sample['keypoints'][self.indexs_to_keep]
        else:
            sample['keypoints'] = sample['keypoints'][self.frame_to_reduce:-self.frame_to_reduce]
        return sample

class MultipleKeyAggregate():
    def __init__(self, transforms, ori_key, more_keys):
        self.transforms = transforms
        self.ori_key = ori_key
        self.more_keys = more_keys

    def __call__(self, sample):
        np.random.seed(42)
        for t in self.transforms:
            for another_key in self.more_keys:
                another_sample = deepcopy(sample)
                another_sample[self.ori_key] = another_sample[another_key]
                another_sample = t(another_sample)
                sample[another_key] = another_sample[self.ori_key]
            sample = t(sample)
        np.random.seed(None)
        return sample

class RandomApply():
    def __init__(self, transforms, prob):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            for t in self.transforms:
                sample = t(sample)
        return sample

class ComposeTransform():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

if __name__ == '__main__':
    class hparams:
        multi_frame = True
        random_jitter = True
        flip = True
        normalize = True
        random_scale = True
        random_rotate = True
        reduce_keypoint_len = True
        pad = True
        # parameters
        clip_len = 5
        num_frames = 3
        jitter_std = 0.01
        jitter_prob = 1
        left_idxs = [5,6,7,11,12,13,15,17]
        right_idxs = [2,3,4,8,9,10,14,16]
        flip_prob = 1
        scale_min = 0.8
        scale_max = 1.2
        scale_prob = 1
        angle_min = -0.5
        angle_max = 0.5
        rotate_prob = 1
        only_one = True
        keep_type = 'middle'
        frame_to_reduce = 0
        max_len = 128

    sample = {
        'point_clouds': [np.random.rand(32, 3), np.random.rand(30, 3), np.random.rand(60, 3), np.random.rand(32, 3), np.random.rand(32, 3)],
        'keypoints': [np.random.rand(18, 3), np.random.rand(18, 3), np.random.rand(18, 3), np.random.rand(18, 3), np.random.rand(18, 3)],
        'action': 1
    }
    sample['keypoints'] = np.stack(sample['keypoints'])

    train_transform = TrainTransform(hparams)
    val_transform = ValTransform(hparams)

    sample = train_transform(sample)
    print(sample.keys())
    print(sample['point_clouds'].shape, sample['keypoints'].shape)