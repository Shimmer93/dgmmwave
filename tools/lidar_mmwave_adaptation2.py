import numpy as np
import pickle
from tqdm import tqdm
import os

import sys
sys.path.append('/home/zpengac/mmwave/dgmmwave')
from dataset.transforms import GenerateSegmentationGroundTruth, ToITOP
import torch
import argparse

BONES = [
    [14, 12], [12, 10], [13, 11], [11, 9], [10, 8], [9, 8], [8, 1], [1, 0], [7, 5], [5, 3], 
    [3, 1], [6, 4], [4, 2], [2, 1]
]

NEIGHBORS = [
    [1],
    [0, 2, 3, 8],
    [1, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    [4],
    [5],
    [1, 9, 10],
    [8, 11],
    [10, 12],
    [9, 13],
    [10, 14],
    [11],
    [12]
]

def calc_similarity_joint(a, b):
    # a: [n, p, 2, 3] n: number of sequences, p: number of bones connecting the joint, 3: direction
    # b: [m, p, 2, 3] m: number of sequences, p: number of bones connecting the joint, 3: direction
    # return: [n, m] similarity between the two joints

    a0 = a[:, :, 0, :] # [n, p, 3]
    b0 = b[:, :, 0, :] # [m, p, 3]

    # a_flow = a[:, :, 1, :] / np. - a[:, :, 0, :] # [n, p, 3]
    # b_flow = b[:, :, 1, :] / np. - b[:, :, 0, :] # [m, p, 3]
    a_normed = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8) # [n, p, 2, 3]
    b_normed = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8) # [m, p, 2, 3]
    

    # cosine similarity
    a0_ = a0.transpose(1, 0, 2) # [p, n, 3]
    b0_ = b0.transpose(1, 0, 2) # [p, m, 3]
    # sim0: # [p, n, m]
    sim0 = np.sum(a0_[:, :, None, :] * b0_[:, None, :, :], axis=-1) # [p, n, m]
    # print(f'sim0 shape: {sim0.shape}, {np.linalg.norm(a0_, axis=-1, keepdims=True).shape}, {np.linalg.norm(b0_, axis=-1, keepdims=True).shape}')
    sim0 = sim0 / (np.linalg.norm(a0_, axis=-1, keepdims=True) * np.linalg.norm(b0_, axis=-1, keepdims=True).transpose(0, 2, 1) + 1e-8) # [p, n, m]
    sim0 = np.mean(sim0, axis=0) # [n, m] average over all bones

    # angle similarity
    angle_a = np.arccos(np.clip(np.sum(a_normed[:, :, 0, :] * a_normed[:, :, 1, :], axis=-1), -1, 1)) # [n, p]
    angle_b = np.arccos(np.clip(np.sum(b_normed[:, :, 0, :] * b_normed[:, :, 1, :], axis=-1), -1, 1)) # [m, p]
    angle_sim = np.exp(-np.abs(angle_a[:, None] - angle_b[None, :]) / (np.pi / 2))[..., 0] # [n, m] values in [0, 1]
    # map to [-1, 1]
    angle_sim = (angle_sim - 0.5) * 2

    # combine the two similarity measures
    sim = (sim0 + angle_sim) / 2 # [n, m] values in [-1, 1]
    return sim

def load_dataset(data_fn, ratio=1.0):
    if os.path.exists(data_fn.replace('.pkl', f'{ratio}_tmp.pkl')):
        with open(data_fn.replace('.pkl', f'{ratio}_tmp.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data

    toitop = ToITOP()
    gengt = GenerateSegmentationGroundTruth(padding=0.25)

    dataset_name = data_fn.split('/')[-1].split('.')[0]
    with open(data_fn, 'rb') as f:
        data = pickle.load(f)

    if ratio < 1.0:
        # all_train_splits = data['splits']['train'].tolist() + data['splits']['val'].tolist()
        all_train_splits = data['splits']['train_rdn_p3'] + data['splits']['val_rdn_p3']
        splits = all_train_splits[:int(len(all_train_splits) * ratio)]
        data['sequences'] = [data['sequences'][i] for i in splits]
        data['splits'] = {
            'train': np.arange(len(data['sequences']))
        }

    for seq in tqdm(data['sequences']):
        seq['dataset_name'] = dataset_name
        seq = toitop(seq)
        seq = gengt(seq)
        if isinstance(seq['keypoints'], torch.Tensor):
            seq['keypoints'] = seq['keypoints'].numpy()


    with open(data_fn.replace('.pkl', f'{ratio}_tmp.pkl'), 'wb') as f:
        pickle.dump(data, f)

    return data

def dump_dataset(data, data_fn):
    with open(data_fn, 'wb') as f:
        pickle.dump(data, f)

def adapt(data_a, data_b, neighbors=NEIGHBORS):
    data_a_new = data_a.copy()
    data_a_new['sequences'] = []

    all_kps_b = []
    all_pcs_b = []
    for seq in data_b['sequences']:
        kps = seq['keypoints']
        kps_ = np.stack([kps[:-1], kps[1:]], axis=2) # [n, 15, 2, 3]
        all_kps_b.append(kps_)
        all_pcs_b.extend(seq['point_clouds'][:-1])
    all_kps_b = np.concatenate(all_kps_b, axis=0) # [n, 15, 3]

    all_max_sims =[list() for _ in range(len(neighbors))]

    for i, seq in enumerate(tqdm(data_a['sequences'])):
        kps_a = seq['keypoints'] # [m, 15, 3]
        pcs_new = [list() for _ in range(kps_a.shape[0]-1)] # list of lists, each list corresponds to a joint

        for j in tqdm(list(range(kps_a.shape[1]))):

            kps_a_neighbors = kps_a[:, neighbors[j], :] # [m, p, 3]
            kps_b_neighbors = all_kps_b[:, neighbors[j], ...] # [n, p, 2, 3]
            kps_a_dirs = kps_a_neighbors - kps_a[:, j, None, :] # [m, p, 3]
            kps_a_dirs = np.stack([kps_a_dirs[:-1], kps_a_dirs[1:]], axis=2) # [m, p, 2, 3]
            kps_b_dirs = kps_b_neighbors - all_kps_b[:, j, None, :] # [n, p, 2, 3]
            
            batch_size = len(kps_b_dirs) // 1
            max_sim = -100
            max_sim_idx = None
            for k in range(0, len(kps_b_dirs), batch_size):
                kps_b_dirs_batch = kps_b_dirs[k:k+batch_size]
                sim_batch = calc_similarity_joint(kps_a_dirs, kps_b_dirs_batch) # [m, n]
                sim_batch = np.clip(sim_batch, -1, 1) # ensure values are in [-1, 1]
                max_sim_batch_idx = np.argmax(sim_batch, axis=1) + k # adjust index for batch
                max_sim_batch = np.max(sim_batch, axis=1)
                if np.max(max_sim_batch) > max_sim:
                    max_sim = np.max(max_sim_batch)
                    max_sim_idx = max_sim_batch_idx
            
            all_max_sims[j].append(max_sim)

            for k in range(kps_a_dirs.shape[0]):
                max_pc = all_pcs_b[max_sim_idx[k]] # [q, 3+1]
                max_kp = all_kps_b[max_sim_idx[k], j, 0, :3] # [3]
                kp = kps_a[k, j, :3] # [3]
                max_pc_j = max_pc[max_pc[:, -1] == j+1]
                pc_new = max_pc_j[:, :3] - max_kp[None, :] + kp[None, :]
                pcs_new[k].append(pc_new) # remove the joint index

        pcs_new = [np.concatenate(pc, axis=0) for pc in pcs_new] # list of arrays, each array corresponds to a point cloud with various number of points

        data_a_new['sequences'].append({
            'point_clouds': seq['point_clouds'],
            'point_clouds_trans': pcs_new,
            'keypoints': seq['keypoints'][:-1]
        })

    all_max_sims = [np.array(sims) for sims in all_max_sims]
    for j in range(len(neighbors)):
        mean_sim = np.mean(all_max_sims[j])
        min_sim = np.min(all_max_sims[j])
        print(f'Joint {j}: mean similarity: {mean_sim:.4f}, min similarity: {min_sim:.4f}')

    return data_a_new

def main(lidar_data_fn, mmwave_data_fn, output_fn, ratio=1.0):
    data_a = load_dataset(lidar_data_fn)
    data_b = load_dataset(mmwave_data_fn, ratio)

    adapted_data = adapt(data_a, data_b)

    dump_dataset(adapted_data, output_fn)
    print(f'Adapted dataset saved to {output_fn}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adapt LiDAR dataset to mmWave dataset')
    parser.add_argument('--lidar_data_fn', type=str, required=True, help='Path to the LiDAR dataset pickle file')
    parser.add_argument('--mmwave_data_fn', type=str, required=True, help='Path to the mmWave dataset pickle file')
    parser.add_argument('--output_fn', type=str, required=True, help='Path to save the adapted dataset pickle file')
    parser.add_argument('--ratio', type=float, default=1.0, help='Ratio of the dataset to use for adaptation')

    args = parser.parse_args()
    
    main(args.lidar_data_fn, args.mmwave_data_fn, args.output_fn, args.ratio)