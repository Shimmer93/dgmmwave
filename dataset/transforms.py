import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from miniball import get_bounding_ball

from misc.skeleton import coco2simplecoco, mmbody2simplecoco, mmfi2simplecoco, itop2simplecoco, mmfi2itop, mmbody2itop

def log(x):
    print(x)
    with open('log.txt', 'a') as f:
        f.write(str(x) + '\n')

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
    def __call__(self, sample):
        segs = []
        for i in range(len(sample['keypoints'])):
            neighbors = NearestNeighbors(n_neighbors=1).fit(sample['keypoints'][i])
            with open('emm.txt', 'a') as f:
                f.write(f'{sample["point_clouds"][i].shape}, {sample["keypoints"][i].shape}')
            _, indices = neighbors.kneighbors(sample['point_clouds'][i][...,:3])
            segs.append(indices)

        sample['segmentations'] = segs
        return sample

class DropPointsAtSegmentedJoints():
    def __init__(self, max_num2drop=3):
        self.max_num2drop = max_num2drop

    def __call__(self, sample):
        assert 'segmentations' in sample
        num_joints = sample['keypoints'][0].shape[0]
        num2drop = np.random.randint(1, self.max_num2drop)
        idxs2drop = np.random.choice(num_joints, num2drop, replace=False)

        new_pcs = []
        for i in range(len(sample['keypoints'])):
            pc = sample['point_clouds'][i]
            seg = sample['segmentations'][i]
            mask = np.isin(seg, idxs2drop)[:, 0]
            # with open('emm.txt', 'a') as f:
            #     f.write(f'{mask.shape}, {pc.shape}')
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
                # with open('emm.txt', 'a') as f:
                #     x = np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][-1].shape[-1])).shape
                #     f.write(f'{add_point.shape}, {x}')
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
                inliers = np.where(np.all(np.abs(sample['point_clouds'][i][...,:2]) < self.radius, axis=1))
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
            if cur_len >= self.max_len:
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
    def __init__(self, clip_len):
        self.clip_len = clip_len
        assert clip_len % 2 == 1, 'num_frames must be odd'
        self.offset = (clip_len - 1) // 2

    def __call__(self, sample):
        # while len(sample['point_clouds']) < self.clip_len:
        #     sample['point_clouds'].append(sample['point_clouds'][-1])
        #     sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)
        for i in range(self.offset):
            sample['point_clouds'].insert(0, sample['point_clouds'][0])
            sample['point_clouds'].append(sample['point_clouds'][-1])
            if 'keypoints' in sample:
                sample['keypoints'] = np.concatenate([sample['keypoints'][0][np.newaxis], sample['keypoints']], axis=0)
                sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)
        # start_idx = np.random.randint(0, len(sample['point_clouds']) - self.clip_len + 1)
        start_idx = sample['index']
        sample['point_clouds'] = sample['point_clouds'][start_idx:start_idx+self.clip_len]
        if 'keypoints' in sample:
            sample['keypoints'] = sample['keypoints'][start_idx:start_idx+self.clip_len]
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
        if centroid_type not in ['none', 'zonly', 'mean', 'median', 'minball', 'dataset_median']:
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
        if sample['dataset_name'] == 'mmbody':
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
        elif sample['dataset_name'] == 'mmbody':
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
        if isinstance(sample['point_clouds'], list):
            sample['point_clouds'] = [torch.from_numpy(pc).float() for pc in sample['point_clouds']]
        else:
            sample['point_clouds'] = torch.from_numpy(sample['point_clouds']).float()
        if 'keypoints' in sample:
            sample['keypoints'] = torch.from_numpy(sample['keypoints']).float()
        if 'action' in sample:
            sample['action'] = torch.tensor([sample['action']]).float()
        sample['index'] = torch.tensor([sample['index']]).float()
        sample['centroid'] = torch.from_numpy(sample['centroid']).float()
        sample['radius'] = torch.tensor([sample['radius']]).float()
        sample['scale'] = torch.tensor([sample['scale']]).float()
        sample['translate'] = torch.from_numpy(sample['translate']).float()
        sample['rotation_matrix'] = torch.from_numpy(sample['rotation_matrix']).float()
        return sample

class ReduceKeypointLen():
    def __init__(self, only_one=False, keep_type='middle', frame_to_reduce=1):
        self.only_one = only_one
        assert keep_type in ['middle', 'start', 'end'], 'keep_type must be "middle", "start" or "end"'
        self.keep_type = keep_type
        self.frame_to_reduce = frame_to_reduce

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
        else:
            sample['keypoints'] = sample['keypoints'][self.frame_to_reduce:-self.frame_to_reduce]
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