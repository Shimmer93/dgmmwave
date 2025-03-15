import numpy as np
import pickle
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import argparse
import h5py
from plyfile import PlyData
import json
from smpl import SMPL
import torch

class Preprocessor():
    def __init__(self, root_dir, out_dir):
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.results = {}
        self.results['splits'] = {}
        self.results['sequences'] = []

    def process(self):
        pass

    def save(self, name):
        with open(os.path.join(self.out_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(self.results, f)

    def _add_to_split(self, split_name, idx):
        if split_name not in self.results['splits']:
            self.results['splits'][split_name] = []
        self.results['splits'][split_name].append(idx)

    def _normalize_intensity(self, feat, max_value):
        feat = np.clip(feat, 0, max_value)
        feat /= max_value
        return feat

class MiliPointPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir):
        super().__init__(root_dir, out_dir)

    def process(self):
        point_fns = [os.path.join(self.root_dir, f'{i}.pkl') for i in range(0, 19)]
        action_labels = np.load(os.path.join(self.root_dir, 'action_label.npy'))

        pcs = []
        kps = []
        for point_fn in point_fns:
            with open(point_fn, 'rb') as f:
                data = pickle.load(f)
            for d in data:
                pcs.append(d['x'])
                kps.append(d['y'])

        assert len(pcs) == len(action_labels), 'Number of point clouds and action labels do not match'
        start_idx = 0
        for i in tqdm(range(len(action_labels)+1)):
            if i == len(action_labels) or (i >= 1 and action_labels[i] != action_labels[i-1]):
                if action_labels[i-1] == -1:
                    continue
                self.results['sequences'].append({
                    'point_clouds': pcs[start_idx:i],
                    'keypoints': np.stack(kps[start_idx:i]),
                    'action': action_labels[i-1]
                })
                start_idx = i

        seq_idxs = np.arange(len(self.results['sequences']))
        np.random.shuffle(seq_idxs)
        num_train = int(len(seq_idxs) * 0.8)
        num_val = int(len(seq_idxs) * 0.1)
        self.results['splits']['train'] = seq_idxs[:num_train]
        self.results['splits']['val'] = seq_idxs[num_train:num_train+num_val]
        self.results['splits']['test'] = seq_idxs[num_train+num_val:]

    def save(self):
        super().save('milipoint')

class MMFiPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir, modality='mmwave'):
        super().__init__(root_dir, out_dir)
        self.action_p1 = ['2', '3', '4', '5', '13', '14', '17', '18', '19', '20', '21', '22', '23', '27']
        assert modality in ['mmwave', 'lidar']
        self.modality = modality

    def process(self):
        dirs = sorted(glob(os.path.join(self.root_dir, 'E*/S*/A*')))

        seq_idxs = np.arange(len(dirs))
        np.random.shuffle(seq_idxs)
        num_train = int(len(seq_idxs) * 0.8)
        num_val = int(len(seq_idxs) * 0.1)
        self.results['splits']['train_rdn_p3'] = sorted(seq_idxs[:num_train])
        self.results['splits']['val_rdn_p3'] = sorted(seq_idxs[num_train:num_train+num_val])
        self.results['splits']['test_rdn_p3'] = sorted(seq_idxs[num_train+num_val:])

        for i, d in tqdm(enumerate(dirs)):
            env = int(d.split('/')[-3][1:])
            subject = int(d.split('/')[-2][1:])
            action = int(d.split('/')[-1][1:])
            pcs = []
            if self.modality == 'mmwave':
                keep_idxs = []
                for bin_fn in sorted(glob(os.path.join(d.replace('MMFi_Dataset', 'filtered_mmwave'), "frame*.bin"))):
                    data_tmp = self._read_bin(bin_fn)
                    data_tmp[:, -1] = self._normalize_intensity(data_tmp[:, -1], 40.0)
                    data_tmp = data_tmp[:, [1, 2, 0, 3, 4]]
                    pcs.append(data_tmp)
                    keep_idx = int(os.path.basename(bin_fn).split('.')[0][5:]) - 1
                    keep_idxs.append(keep_idx)
                kps = np.load(os.path.join(d, 'ground_truth.npy'))[keep_idxs,...]
            else:
                for bin_fn in sorted(glob(os.path.join(d, "lidar", "frame*.bin"))):
                    data_tmp = self._read_bin(bin_fn)
                    data_tmp = data_tmp[:, [1, 2, 0]]
                    data_tmp[..., 0] = -data_tmp[..., 0]
                    pcs.append(data_tmp)
                kps = np.load(os.path.join(d, 'ground_truth.npy'))
                kps[..., 0] = kps[..., 0]
                kps[..., 1] = -kps[..., 1]- 0.2
                kps[..., 2] = kps[..., 2] - 0.1
            new_pcs = []
            for pc, kp in zip(pcs, kps):
                pc = self._filter_pcl(kp, pc, bound=0.2)
                new_pcs.append(pc)
            self.results['sequences'].append({
                'point_clouds': new_pcs,
                'keypoints': kps,
                'action': action
            })

            if i in self.results['splits']['train_rdn_p3']:
                if action in self.action_p1:
                    self._add_to_split('train_rdn_p1', i)
                else:
                    self._add_to_split('train_rdn_p2', i)
            elif i in self.results['splits']['val_rdn_p3']:
                if action in self.action_p1:
                    self._add_to_split('val_rdn_p1', i)
                else:
                    self._add_to_split('val_rdn_p2', i)
            else:
                if action in self.action_p1:
                    self._add_to_split('test_rdn_p1', i)
                else:
                    self._add_to_split('test_rdn_p2', i)

            if subject % 5 == 0:
                self._add_to_split('test_xsub_p3', i)
                if action in self.action_p1:
                    self._add_to_split('test_xsub_p1', i)
                else:
                    self._add_to_split('test_xsub_p2', i)
            elif subject % 5 == 1:
                self._add_to_split('val_xsub_p3', i)
                if action in self.action_p1:
                    self._add_to_split('val_xsub_p1', i)
                else:
                    self._add_to_split('val_xsub_p2', i)
            else:
                self._add_to_split('train_xsub_p3', i)
                if action in self.action_p1:
                    self._add_to_split('train_xsub_p1', i)
                else:
                    self._add_to_split('train_xsub_p2', i)

            if env == 4:
                self._add_to_split('test_xenv_p3', i)
                if action in self.action_p1:
                    self._add_to_split('test_xenv_p1', i)
                else:
                    self._add_to_split('test_xenv_p2', i)
            elif env == 3:
                self._add_to_split('val_xenv_p3', i)
                if action in self.action_p1:
                    self._add_to_split('val_xenv_p1', i)
                else:
                    self._add_to_split('val_xenv_p2', i)
            else:
                self._add_to_split('train_xenv_p3', i)
                if action in self.action_p1:
                    self._add_to_split('train_xenv_p1', i)
                else:
                    self._add_to_split('train_xenv_p2', i)

    def save(self):
        super().save('mmfi_' + self.modality)

    def _read_bin(self, bin_fn):
        with open(bin_fn, 'rb') as f:
            raw_data = f.read()
            data_tmp = np.frombuffer(raw_data, dtype=np.float64)
            if self.modality == 'mmwave':
                data_tmp = data_tmp.copy().reshape(-1, 5)
                data_tmp[:, [3, 4]] = data_tmp[:, [4, 3]]
            else:
                data_tmp = data_tmp.copy().reshape(-1, 3)
        return data_tmp

    def _filter_pcl(self, bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, offset: float = 0):
        """
        Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
        """
        upper_bound = bounding_pcl[:, :3].max(axis=0) + bound
        lower_bound = bounding_pcl[:, :3].min(axis=0) - bound
        lower_bound[2] += offset

        mask_x = (target_pcl[:, 0] >= lower_bound[0]) & (
            target_pcl[:, 0] <= upper_bound[0])
        mask_y = (target_pcl[:, 1] >= lower_bound[1]) & (
            target_pcl[:, 1] <= upper_bound[1])
        mask_z = (target_pcl[:, 2] >= lower_bound[2]) & (
            target_pcl[:, 2] <= upper_bound[2])
        index = mask_x & mask_y & mask_z
        return target_pcl[index]

class MMBodyPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir):
        super().__init__(root_dir, out_dir)

    def process(self):
        train_val_dirs = glob(os.path.join(self.root_dir, 'train/sequence_*'))
        test_dirs = glob(os.path.join(self.root_dir, 'test/*/sequence_*'))

        train_val_seq_idxs = np.arange(len(train_val_dirs))
        np.random.shuffle(train_val_seq_idxs)
        num_train = int(len(train_val_seq_idxs) * 0.9)
        self.results['splits']['train'] = train_val_seq_idxs[:num_train]
        self.results['splits']['val'] = train_val_seq_idxs[num_train:]
        self.results['splits']['test'] = np.arange(len(test_dirs)) + len(train_val_dirs)

        for d in tqdm(train_val_dirs + test_dirs):
            pcs = []
            kps = []
            pc_fns = glob(os.path.join(d, 'radar', '*.npy'))
            bns = sorted([int(os.path.basename(fn).split('.')[0].split('_')[-1]) for fn in pc_fns])
            for bn in bns:
                pc = np.load(os.path.join(d, 'radar', f'frame_{bn}.npy'))
                pc[:,3:] /= np.array([5e-38, 5., 1.])
                pc[:, -1] = self._normalize_intensity(pc[:, -1], 150.0)
                kp = np.load(os.path.join(d, 'mesh', f'frame_{bn}.npz'))['joints'][:22]
                pc = self._filter_pcl(kp, pc)
                if len(pc) == 0:
                    continue
                pcs.append(pc[:, [0, 2, 1, 4, 5]])
                kps.append(kp[:, [0, 2, 1]])
            self.results['sequences'].append({
                'point_clouds': pcs,
                'keypoints': kps,
                'action': -1
            })

    def save(self):
        super().save('mmbody')

    def _filter_pcl(self, bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, offset: float = 0):
        """
        Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
        """
        upper_bound = bounding_pcl[:, :3].max(axis=0) + bound
        lower_bound = bounding_pcl[:, :3].min(axis=0) - bound
        lower_bound[2] += offset

        mask_x = (target_pcl[:, 0] >= lower_bound[0]) & (
            target_pcl[:, 0] <= upper_bound[0])
        mask_y = (target_pcl[:, 1] >= lower_bound[1]) & (
            target_pcl[:, 1] <= upper_bound[1])
        mask_z = (target_pcl[:, 2] >= lower_bound[2]) & (
            target_pcl[:, 2] <= upper_bound[2])
        index = mask_x & mask_y & mask_z
        return target_pcl[index]

class MRIPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir):
        super().__init__(root_dir, out_dir)

    def process(self):
        sub_idxs = np.arange(1, 21)
        shuffled_sub_idxs = sub_idxs.copy()
        np.random.shuffle(shuffled_sub_idxs)
        train_sub_idxs = shuffled_sub_idxs[:14]
        val_sub_idxs = shuffled_sub_idxs[14:16]

        count = 0
        for idx in tqdm(sub_idxs):
            pc_fn = os.path.join(self.root_dir, f'dataset_release/aligned_data/radar/singleframe/subject{idx}.csv')
            label_fn = os.path.join(self.root_dir, f'dataset_release/aligned_data/pose_labels/subject{idx}_all_labels.cpl')
            pc_df = pd.read_csv(pc_fn)
            with open(label_fn, 'rb') as f:
                labels = pickle.load(f)
            splits = list(labels['video_label'].values())[1:13]
            for i, split in enumerate(splits):
                pcs = []
                for j in range(split[0], split[1]):
                    pc = pc_df[pc_df['Camera Frame'] == j][['X', 'Y', 'Z', 'Doppler', 'Intensity']].values
                    pc[:, -1] = self._normalize_intensity(pc[:, -1], 200.0)
                    pc = pc[:, [0, 2, 1, 3, 4]]
                    pc[:, 0] *= -1
                    pcs.append(pc)
                kps = labels['refined_gt_kps'][split[0]:split[1]].transpose(0, 2, 1)
                self.results['sequences'].append({
                    'point_clouds': pcs,
                    'keypoints': kps,
                    'action': i
                })
                if idx in train_sub_idxs:
                    self._add_to_split('train_s2_p1', count)
                elif idx in val_sub_idxs:
                    self._add_to_split('val_s2_p1', count)
                else:
                    self._add_to_split('test_s2_p1', count)
                count += 1

        seq_idxs = np.arange(len(self.results['sequences']))
        np.random.shuffle(seq_idxs)
        num_train = int(len(seq_idxs) * 0.7)
        num_val = int(len(seq_idxs) * 0.1)
        self.results['splits']['train_s1_p1'] = sorted(seq_idxs[:num_train])
        self.results['splits']['val_s1_p1'] = sorted(seq_idxs[num_train:num_train+num_val])
        self.results['splits']['test_s1_p1'] = sorted(seq_idxs[num_train+num_val:])

        for i in tqdm(seq_idxs):
            if i in self.results['splits']['train_s1_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('train_s1_p2', i)
            elif i in self.results['splits']['val_s1_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('val_s1_p2', i)
            elif i in self.results['splits']['test_s1_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('test_s1_p2', i)

            if i in self.results['splits']['train_s2_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('train_s2_p2', i)
            elif i in self.results['splits']['val_s2_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('val_s2_p2', i)
            elif i in self.results['splits']['test_s2_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('test_s2_p2', i)

    def save(self):
        super().save('mri')

class ITOPPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir, view):
        super().__init__(root_dir, out_dir)
        assert view in ['side', 'top'], 'Invalid view'
        self.view = view

    def process(self):
        train_val_data_fn = os.path.join(self.root_dir, f'ITOP_{self.view}_train_point_cloud.h5')
        train_val_labels_fn = os.path.join(self.root_dir, f'ITOP_{self.view}_train_labels.h5')
        test_data_fn = os.path.join(self.root_dir, f'ITOP_{self.view}_test_point_cloud.h5')
        test_labels_fn = os.path.join(self.root_dir, f'ITOP_{self.view}_test_labels.h5')

        train_val_data, train_val_ids = self._read_h5(train_val_data_fn, ['data', 'id'])
        train_val_labels = self._read_h5(train_val_labels_fn, ['is_valid', 'real_world_coordinates', 'segmentation'])

        test_data, test_ids = self._read_h5(test_data_fn, ['data', 'id'])
        test_labels = self._read_h5(test_labels_fn, ['is_valid', 'real_world_coordinates', 'segmentation'])

        train_val_list = self._process_split(train_val_data, train_val_ids, train_val_labels)
        test_list = self._process_split(test_data, test_ids, test_labels)

        train_val_seq_idxs = np.arange(len(train_val_list))
        np.random.shuffle(train_val_seq_idxs)
        num_train = int(len(train_val_seq_idxs) * 0.8)
        self.results['splits']['train'] = train_val_seq_idxs[:num_train]
        self.results['splits']['val'] = train_val_seq_idxs[num_train:]
        self.results['splits']['test'] = np.arange(len(test_list)) + len(train_val_list)

        for d in train_val_list + test_list:
            self.results['sequences'].append(d)

    def save(self):
        super().save('itop_' + self.view)

    def _read_h5(self, fn, keys):
        output = []
        f = h5py.File(fn, 'r')
        for key in keys:
            value = f[key][()]
            output.append(value)
        f.close()
        return tuple(output)
    
    def _segment_human(self, pc, seg):
        pc = pc[np.ravel(seg) != -1]
        return pc
    
    def _process_split(self, data, ids, labels):
        split_list = []
        last_id = None
        for pc, id, valid, kps, seg in tqdm(zip(data, ids, labels[0], labels[1], labels[2])):
            if valid == 0 and (last_id is None or len(split_list[-1]['point_clouds']) > 0):
                split_list.append({
                    'point_clouds': [],
                    'keypoints': [],
                    'action': -1
                })
            else:
                if id.decode().split('_')[0] != last_id and (last_id is None or len(split_list[-1]['point_clouds']) > 0):
                    split_list.append({
                        'point_clouds': [],
                        'keypoints': [],
                        'action': -1
                    })
                pc = self._segment_human(pc, seg)
                if len(pc) == 76800:
                    print('Skipping due to invalid segmentation')
                    continue
                
                if self.view == 'top':
                    pc = pc[..., [0, 2, 1]] * np.array([1, -1, 1])
                    kps = kps[..., [0, 2, 1]] * np.array([1, -1, 1])

                split_list[-1]['point_clouds'].append(pc)
                split_list[-1]['keypoints'].append(kps)
                last_id = id.decode().split('_')[0]

        split_list = [d for d in split_list if len(d['point_clouds']) >= 5]
        return split_list

class LiDARHuman26MPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir):
        super().__init__(root_dir, out_dir)
        self.pc_dir = os.path.join(self.root_dir, 'labels/3d/segment')
        self.kps_dir = os.path.join(self.root_dir, 'labels/3d/pose')

    def process(self):
        with open(os.path.join(self.root_dir, 'train.txt')) as f:
            train_val_ids = f.read().splitlines()
        with open(os.path.join(self.root_dir, 'test.txt')) as f:
            test_ids = f.read().splitlines()

        np.random.shuffle(train_val_ids)
        train_ids = train_val_ids[:int(len(train_val_ids)*0.8)]
        val_ids = train_val_ids[int(len(train_val_ids)*0.8):]

        self._process_split(train_ids, 'train')
        self._process_split(val_ids, 'val')
        self._process_split(test_ids, 'test')

    def _read_point_cloud(self, filename):
        """ read XYZ point cloud from filename PLY file """
        ply_data = PlyData.read(filename)['vertex'].data
        points = np.array([[y, z, x] for x, y, z in ply_data])
        return points
    
    def _read_json(self, filename):
        with open(filename) as f:
            data = json.load(f)
        pose = np.array(data['pose'], dtype=np.float32)
        beta = np.array(data['beta'], dtype=np.float32)
        trans = np.array(data['trans'], dtype=np.float32)
        return pose, beta, trans

    def _process_split(self, ids, split_name):
        for id in tqdm(ids):
            pc_fns = sorted(glob(os.path.join(self.pc_dir, str(id), '*.ply')))

            pcs = []
            poses = []
            betas = []
            transs = []
            for pc_fn in pc_fns:
                kp_fn = pc_fn.replace('segment', 'pose').replace('ply', 'json')
                pc = self._read_point_cloud(pc_fn)
                pose, beta, trans = self._read_json(kp_fn)
                pcs.append(pc)
                poses.append(pose)
                betas.append(beta)
                transs.append(trans)
            poses = np.stack(poses)
            betas = np.stack(betas)
            transs = np.stack(transs)

            smpl = SMPL().cuda()

            batch_size = 8192
            num_batches = len(pcs) // batch_size
            if len(pcs) % batch_size != 0:
                num_batches += 1
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(pcs))
                verts = smpl(torch.from_numpy(poses[start:end]).cuda(), torch.from_numpy(betas[start:end]).cuda())
                kps = smpl.get_full_joints(verts).cpu().numpy()
                kps += transs[start:end, np.newaxis, :]
                if i == 0:
                    all_kps = kps
                else:
                    all_kps = np.concatenate([all_kps, kps], axis=0)
            # all_kps += transs[:, np.newaxis, :]

            self.results['sequences'].append({
                'point_clouds': pcs,
                'keypoints': all_kps[..., [1, 2, 0]],
                'action': -1
            })

            self._add_to_split(split_name, len(self.results['sequences']) - 1)

    def save(self):
        super().save('lidarhuman26m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--dataset', type=str, default='milipoint', help='Dataset name')
    parser.add_argument('--root_dir', type=str, help='Root directory of the dataset')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    np.random.seed(args.seed)
    # Seeds used in original repos:
    # milipoint: 42
    # mmfi: 0
    # mmbody: 35
    # mri: 1234567891

    dataset = args.dataset.lower()
    if dataset == 'milipoint':
        preprocessor = MiliPointPreprocessor(args.root_dir, args.out_dir)
    elif dataset == 'mmfi':
        preprocessor = MMFiPreprocessor(args.root_dir, args.out_dir, 'mmwave')
    elif dataset == 'mmfi_lidar':
        preprocessor = MMFiPreprocessor(args.root_dir, args.out_dir, 'lidar')
    elif dataset == 'mmbody':
        preprocessor = MMBodyPreprocessor(args.root_dir, args.out_dir)
    elif dataset == 'mri':
        preprocessor = MRIPreprocessor(args.root_dir, args.out_dir)
    elif dataset == 'itop_side':
        preprocessor = ITOPPreprocessor(args.root_dir, args.out_dir, 'side')
    elif dataset == 'itop_top':
        preprocessor = ITOPPreprocessor(args.root_dir, args.out_dir, 'top')
    elif dataset == 'lidarhuman26m':
        preprocessor = LiDARHuman26MPreprocessor(args.root_dir, args.out_dir)
    else:
        raise ValueError('Invalid dataset name')
    
    preprocessor.process()
    preprocessor.save()