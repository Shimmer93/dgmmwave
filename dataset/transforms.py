import numpy as np
import torch

def log(x):
    print(x)
    with open('log.txt', 'a') as f:
        f.write(str(x) + '\n')

class Pad():
    def __init__(self, max_len, pad_type='zero'):
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
            sample['keypoints'] = np.concatenate([sample['keypoints'][0][np.newaxis], sample['keypoints']], axis=0)
            sample['point_clouds'].append(sample['point_clouds'][-1])
            sample['keypoints'] = np.concatenate([sample['keypoints'], sample['keypoints'][-1][np.newaxis]], axis=0)
        # start_idx = np.random.randint(0, len(sample['point_clouds']) - self.clip_len + 1)
        start_idx = sample['index']
        sample['point_clouds'] = sample['point_clouds'][start_idx:start_idx+self.clip_len]
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
            sample['point_clouds'] = [np.concatenate(sample['point_clouds'][np.maximum(0, i-self.offset):np.minimum(i+self.offset+1, total_frames-1)]) for i in range(total_frames)]
            # sample['keypoints'] = sample['keypoints'][self.offset:-self.offset]
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
        sample['keypoints'] *= scale
        sample['scale'] = scale
        return sample
    
class RandomRotate():
    def __init__(self, angle_min=-np.pi/4, angle_max=np.pi/4):
        self.angle_min = angle_min
        self.angle_max = angle_max

    def __call__(self, sample):
        angle = np.random.uniform(self.angle_min, self.angle_max)
        rot_matrix = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] = sample['point_clouds'][i][...,:3] @ rot_matrix
        sample['keypoints'] = sample['keypoints'] @ rot_matrix
        sample['angle'] = angle
        return sample

class RandomJitter():
    def __init__(self, jitter_std=0.01):
        self.jitter_std = jitter_std

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] += np.random.normal(0, self.jitter_std, sample['point_clouds'][i][...,:3].shape)
        return sample

class GetCentroidRadius():
    def __call__(self, sample):
        pc_cat = np.concatenate(sample['point_clouds'], axis=0)
        centroid = np.mean(pc_cat[...,:3], axis=0)
        radius = np.max(np.linalg.norm(pc_cat[...,:3] - centroid, axis=1))
        sample['centroid'] = centroid
        sample['radius'] = radius
        return sample

class Normalize():
    def __init__(self, feat_scale=None):
        self.feat_scale = feat_scale
    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] -= sample['centroid'][np.newaxis]
            sample['point_clouds'][i][...,:3] /= sample['radius']
            if self.feat_scale:
                sample['point_clouds'][i][...,3:] /= np.array(self.feat_scale)[np.newaxis][np.newaxis]
        # print('normalize', len(sample['point_clouds']), len(sample['keypoints']), sample['centroid'], sample['radius'])
        sample['keypoints'] -= sample['centroid'][np.newaxis][np.newaxis]
        sample['keypoints'] /= sample['radius']
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
        sample['keypoints'] = sample['keypoints'][:, indices]
        sample['keypoints'][..., 0] *= -1
        return sample

class ToTensor():
    def __call__(self, sample):
        if isinstance(sample['point_clouds'], list):
            sample['point_clouds'] = [torch.from_numpy(pc).float() for pc in sample['point_clouds']]
        else:
            sample['point_clouds'] = torch.from_numpy(sample['point_clouds']).float()
        sample['keypoints'] = torch.from_numpy(sample['keypoints']).float()
        # sample['action'] = torch.tensor(sample['action'])
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

class TrainTransform():
    def __init__(self, hparams):
        self.hparams = hparams
        tsfms = []
        tsfms.append(UniformSample(hparams.clip_len))
        if hparams.multi_frame:
            tsfms.append(MultiFrameAggregate(hparams.num_frames))
        if hparams.random_jitter:
            tsfms.append(RandomApply([RandomJitter(hparams.jitter_std)], prob=hparams.jitter_prob))
        if hparams.flip:
            tsfms.append(RandomApply([Flip(hparams.left_idxs, hparams.right_idxs)], prob=hparams.flip_prob))
        tsfms.append(GetCentroidRadius())
        if hparams.normalize:
            tsfms.append(Normalize())
        if hparams.random_scale:
            tsfms.append(RandomApply([RandomScale(hparams.scale_min, hparams.scale_max)], prob=hparams.scale_prob))
        if hparams.random_rotate:
            tsfms.append(RandomApply([RandomRotate(hparams.angle_min, hparams.angle_max)], prob=hparams.rotate_prob))
        if hparams.reduce_keypoint_len:
            tsfms.append(ReduceKeypointLen(hparams.only_one, hparams.keep_type, hparams.frame_to_reduce))
        if hparams.pad:
            tsfms.append(Pad(hparams.max_len))
        tsfms.append(ToTensor())

        self.transforms = tsfms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    
class ValTransform():
    def __init__(self, hparams):
        self.hparams = hparams
        tsfms = []
        tsfms.append(UniformSample(hparams.clip_len))
        if hparams.multi_frame:
            tsfms.append(MultiFrameAggregate(hparams.num_frames))
        tsfms.append(GetCentroidRadius())
        if hparams.normalize:
            tsfms.append(Normalize())
        if hparams.reduce_keypoint_len:
            tsfms.append(ReduceKeypointLen(hparams.only_one, hparams.keep_type, hparams.frame_to_reduce))
        if hparams.pad:
            tsfms.append(Pad(hparams.max_len))
        tsfms.append(ToTensor())

        self.transforms = tsfms

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