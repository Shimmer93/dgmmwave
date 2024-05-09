import numpy as np
import pickle
import os
from glob import glob
from tqdm import tqdm
import argparse

def preprocess_milipoint(root_dir, out_dir):
    results = {}
    results['splits'] = {}
    results['sequences'] = []
    point_fns = [os.path.join(root_dir, f'{i}.pkl') for i in range(0, 19)]
    action_labels = np.load(os.path.join(root_dir, 'action_label.npy'))

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
            results['sequences'].append({
                'point_clouds': pcs[start_idx:i],
                'keypoints': np.stack(kps[start_idx:i]),
                'action': action_labels[i-1]
            })
            start_idx = i

    seq_idxs = np.arange(len(results['sequences']))
    np.random.shuffle(seq_idxs)
    num_train = int(len(seq_idxs) * 0.8)
    num_val = int(len(seq_idxs) * 0.1)
    results['splits']['train'] = seq_idxs[:num_train]
    results['splits']['val'] = seq_idxs[num_train:num_train+num_val]
    results['splits']['test'] = seq_idxs[num_train+num_val:]
    with open(os.path.join(out_dir, 'milipoint.pkl'), 'wb') as f:
        pickle.dump(results, f)

def preprocess_mmfi(root_dir, out_dir):

    results = {}
    results['splits'] = {}
    results['sequences'] = []

    action_p1 = ['2', '3', '4', '5', '13', '14', '17', '18', '19', '20', '21', '22', '23', '27']

    def add_to_split(results, split_name, idx):
        if split_name not in results['splits']:
            results['splits'][split_name] = []
        results['splits'][split_name].append(idx)

    dirs = sorted(glob(os.path.join(root_dir, 'all_data/E*/S*/A*')))

    seq_idxs = np.arange(len(results['sequences']))
    np.random.shuffle(seq_idxs)
    num_train = int(len(seq_idxs) * 0.8)
    num_val = int(len(seq_idxs) * 0.1)
    results['splits']['train_rdn_p3'] = sorted(seq_idxs[:num_train])
    results['splits']['val_rdn_p3'] = sorted(seq_idxs[num_train:num_train+num_val])
    results['splits']['test_rdn_p3'] = sorted(seq_idxs[num_train+num_val:])

    for i, d in tqdm(enumerate(dirs)):
        env = int(d.split('/')[-3][1:])
        subject = int(d.split('/')[-2][1:])
        action = int(d.split('/')[-1][1:])
        pcs = []
        for bin_file in sorted(glob(os.path.join(d, "mmwave/frame*.bin"))):
            with open(bin_file, 'rb') as f:
                raw_data = f.read()
                data_tmp = np.frombuffer(raw_data, dtype=np.float64)
                data_tmp = data_tmp.copy().reshape(-1, 5)
                # data_tmp = data_tmp[:, :3]
            pcs.append(data_tmp)
        kps = np.load(os.path.join(d, 'ground_truth.npy'))
        results['sequences'].append({
            'point_clouds': pcs,
            'keypoints': kps,
            'action': action
        })

        if i in results['splits']['train_rdn_p3']:
            if action in action_p1:
                add_to_split(results, 'train_rdn_p1', i)
            else:
                add_to_split(results, 'train_rdn_p2', i)
        elif i in results['splits']['val_rdn_p3']:
            if action in action_p1:
                add_to_split(results, 'val_rdn_p1', i)
            else:
                add_to_split(results, 'val_rdn_p2', i)
        else:
            if action in action_p1:
                add_to_split(results, 'test_rdn_p1', i)
            else:
                add_to_split(results, 'test_rdn_p2', i)

        if subject % 5 == 0:
            add_to_split(results, 'test_xsub_p3', i)
            if action in action_p1:
                add_to_split(results, 'test_xsub_p1', i)
            else:
                add_to_split(results, 'test_xsub_p2', i)
        elif subject % 5 == 1:
            add_to_split(results, 'val_xsub_p3', i)
            if action in action_p1:
                add_to_split(results, 'val_xsub_p1', i)
            else:
                add_to_split(results, 'val_xsub_p2', i)
        else:
            add_to_split(results, 'train_xsub_p3', i)
            if action in action_p1:
                add_to_split(results, 'train_xsub_p1', i)
            else:
                add_to_split(results, 'train_xsub_p2', i)

        if env == 4:
            add_to_split(results, 'test_xenv_p3', i)
            if action in action_p1:
                add_to_split(results, 'test_xenv_p1', i)
            else:
                add_to_split(results, 'test_xenv_p2', i)
        elif env == 3:
            add_to_split(results, 'val_xenv_p3', i)
            if action in action_p1:
                add_to_split(results, 'val_xenv_p1', i)
            else:
                add_to_split(results, 'val_xenv_p2', i)
        else:
            add_to_split(results, 'train_xenv_p3', i)
            if action in action_p1:
                add_to_split(results, 'train_xenv_p1', i)
            else:
                add_to_split(results, 'train_xenv_p2', i)

    with open(os.path.join(out_dir, 'mmfi.pkl'), 'wb') as f:
        pickle.dump(results, f)


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

    if args.dataset == 'milipoint':
        preprocess_milipoint(args.root_dir, args.out_dir)
    elif args.dataset == 'mmfi':
        preprocess_mmfi(args.root_dir, args.out_dir)
    else:
        raise ValueError('Invalid dataset name')