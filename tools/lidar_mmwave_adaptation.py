import numpy as np
import pickle
from tqdm import tqdm
import os

import sys
sys.path.append('/home/zpengac/pose/dgmmwave')
from dataset.transforms import GenerateSegmentationGroundTruth, ToITOP
import torch
import argparse

JOINT_GROUP_IDX_MAP = {
    'left_arm': [3, 5, 7],
    'right_arm': [2, 4, 6],
    'left_leg': [10, 12, 14],
    'right_leg': [9, 11, 13],
    'torso': [0, 1, 8]
}
JOINT_GROUPS = list(JOINT_GROUP_IDX_MAP.keys())
JOINT_CATEGORIES = ['arm', 'leg', 'torso']

def calc_similarity(a, b):
    # a: [n, 3, 3]
    # b: [m, 3, 3]
    # return: [n, m]

    a0 = a[:, 1, :] - a[:, 0, :]
    a1 = a[:, 2, :] - a[:, 1, :]
    b0 = b[:, 1, :] - b[:, 0, :]
    b1 = b[:, 2, :] - b[:, 1, :]

    sim0 = np.dot(a0, b0.T) / (np.linalg.norm(a0, axis=1)[:, None] * np.linalg.norm(b0, axis=1))
    sim1 = np.dot(a1, b1.T) / (np.linalg.norm(a1, axis=1)[:, None] * np.linalg.norm(b1, axis=1))
    sim = (sim0 + sim1) / 2
    return sim

def calc_similarity_flow(a, b, b_flow):
    sim_dir = calc_similarity(a, b)
    a_ = np.concatenate([a, a[-1:]], axis=0)
    a_flow = a_[1:] - a_[:-1]
    a_flow_amp = np.linalg.norm(a_flow, axis=-1)[:, None, :].repeat(b.shape[0], axis=1).sum(axis=-1)
    b_flow_amp = np.linalg.norm(b_flow, axis=-1)[None, :, :].repeat(a.shape[0], axis=0).sum(axis=-1)

    err_flow = np.abs(a_flow_amp - b_flow_amp) #np.sqrt(np.sum((a_flow_amp - b_flow_amp)**2, axis=-1))
    sim_flow = (1 - err_flow / (np.maximum(a_flow_amp, b_flow_amp) + 1e-6)) ** 0.5
    sim = (sim_dir + sim_flow) / 2
    
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

def get_categorized_results(data, multi_view=False):
    categorized_kps = {category_name: [] for category_name in JOINT_CATEGORIES}
    categorized_pcs = {category_name: [] for category_name in JOINT_CATEGORIES}
    categorized_is_movings = {category_name: [] for category_name in JOINT_CATEGORIES}

    # if ratio < 1.0:
    #     all_train_splits = data['splits']['train'].tolist() + data['splits']['val'].tolist()
    #     # all_train_splits = data['splits']['train_rdn_p3'] + data['splits']['val_rdn_p3']
    #     splits = all_train_splits[:int(len(all_train_splits) * ratio)]
    #     data['sequences'] = [data['sequences'][i] for i in splits]

    for i, seq in tqdm(list(enumerate(data['sequences']))):
        # if i == 2:
        #     break

        if len(seq['point_clouds']) == 0:
            continue

        for j in range(len(seq['point_clouds'])):
            pc = seq['point_clouds'][j]
            kps = seq['keypoints'][j]

            for group_name, group_idxs in JOINT_GROUP_IDX_MAP.items():
                group_pc_ = pc[np.isin(pc[:, -1]-1, group_idxs), :]
                group_pc = group_pc_[..., :3]
                group_kps = kps[group_idxs]
                group_pc -= group_kps[0:1]
                group_kps -= group_kps[0:1]
                if group_name.split('_')[0] == 'right':
                    group_pc[..., 0] *= -1
                    group_kps[..., 0] *= -1

                if len(group_pc) > 5:
                    group_is_moving = np.abs(group_pc_[..., 3:4]).max() > 0.02
                    # assert isinstance(group_is_moving, bool), f'group_is_moving: {group_is_moving}'
                    categorized_kps[group_name.split('_')[-1]].append(group_kps)
                    categorized_pcs[group_name.split('_')[-1]].append(group_pc)
                    categorized_is_movings[group_name.split('_')[-1]].append(group_is_moving)
                    if multi_view:
                        categorized_kps[group_name.split('_')[-1]].append(group_kps[..., [0, 2, 1]])
                        categorized_pcs[group_name.split('_')[-1]].append(group_pc[..., [0, 2, 1]])
                        categorized_is_movings[group_name.split('_')[-1]].append(group_is_moving)
                        categorized_kps[group_name.split('_')[-1]].append(group_kps[..., [1, 2, 0]])
                        categorized_pcs[group_name.split('_')[-1]].append(group_pc[..., [1, 2, 0]])
                        categorized_is_movings[group_name.split('_')[-1]].append(group_is_moving)
                        categorized_kps[group_name.split('_')[-1]].append(group_kps[..., [1, 0, 2]])
                        categorized_pcs[group_name.split('_')[-1]].append(group_pc[..., [1, 0, 2]])
                        categorized_is_movings[group_name.split('_')[-1]].append(group_is_moving)
                        categorized_kps[group_name.split('_')[-1]].append(group_kps[..., [2, 1, 0]])
                        categorized_pcs[group_name.split('_')[-1]].append(group_pc[..., [2, 1, 0]])
                        categorized_is_movings[group_name.split('_')[-1]].append(group_is_moving)
                        categorized_kps[group_name.split('_')[-1]].append(group_kps[..., [2, 0, 1]])
                        categorized_pcs[group_name.split('_')[-1]].append(group_pc[..., [2, 0, 1]])
                        categorized_is_movings[group_name.split('_')[-1]].append(group_is_moving)

    categorized_kps = {category_name: np.stack(categorized_kps[category_name]).astype(np.float32) for category_name in JOINT_CATEGORIES}
    categorized_is_movings = {category_name: np.array(categorized_is_movings[category_name], dtype=bool) for category_name in JOINT_CATEGORIES}
    
    return categorized_kps, categorized_pcs, categorized_is_movings
    
def get_categorized_results_flow(data):
    categorized_kps = {category_name: [] for category_name in JOINT_CATEGORIES}
    categorized_pcs = {category_name: [] for category_name in JOINT_CATEGORIES}
    categorized_flows = {category_name: [] for category_name in JOINT_CATEGORIES}

    for i, seq in tqdm(list(enumerate(data['sequences']))):
        # if i == 2:
        #     break

        if len(seq['point_clouds']) == 0:
            continue
        last_kps = seq['keypoints'][0]

        for j in range(len(seq['point_clouds'])):
            pc = seq['point_clouds'][j]
            kps = seq['keypoints'][j]

            for group_name, group_idxs in JOINT_GROUP_IDX_MAP.items():
                group_pc = pc[np.isin(pc[:, -1]-1, group_idxs), :3]
                group_kps = kps[group_idxs]
                group_pc -= group_kps[0:1]
                group_kps -= group_kps[0:1]
                if group_name.split('_')[0] == 'right':
                    group_pc[..., 0] *= -1
                    group_kps[..., 0] *= -1

                if j > 0:
                    categorized_kps[group_name.split('_')[-1]].append(group_kps)
                    categorized_pcs[group_name.split('_')[-1]].append(group_pc)
                    categorized_flows[group_name.split('_')[-1]].append(group_kps - last_kps[group_idxs])

            if j > 0:
                last_kps = seq['keypoints'][j]

    categorized_kps = {category_name: np.stack(categorized_kps[category_name]).astype(np.float32) for category_name in JOINT_CATEGORIES}
    categorized_flows = {category_name: np.stack(categorized_flows[category_name]).astype(np.float32) for category_name in JOINT_CATEGORIES}      
    print(categorized_kps['arm'].shape, categorized_flows['arm'].shape)
    return categorized_kps, categorized_pcs, categorized_flows   

def dataset_a2b(data_a, all_categorized_kps_b, all_categorized_pcs_b):
    data_a_new = data_a.copy()
    data_a_new['sequences'] = []
    grouped_best_sims = {group_name: [] for group_name in JOINT_GROUPS}

    for i, seq in tqdm(enumerate(data_a['sequences'])):
        # if i == 2:
        #     break

        new_grouped_pcs = {group_name: [] for group_name in JOINT_GROUPS}
        seq_len = len(seq['point_clouds'])
        seq_kps = seq['keypoints']

        for group_name, group_idxs in JOINT_GROUP_IDX_MAP.items():
            seq_grouped_kps_a = seq_kps[:, group_idxs]
            seq_grouped_kps_a_ = seq_grouped_kps_a - seq_grouped_kps_a[:, 0:1]
            category_name = group_name.split('_')[-1]
            seq_grouped_sims = calc_similarity(seq_grouped_kps_a_, all_categorized_kps_b[category_name])

            for j in range(seq_len):
                grouped_kps_a = seq_grouped_kps_a[j]
                grouped_sims_ = seq_grouped_sims[j]
                if len(grouped_sims_) > 0:
                    best_idx = np.argmax(grouped_sims_)
                    best_sim = grouped_sims_[best_idx]
                    grouped_best_sims[group_name].append(best_sim)
                    all_categorized_pcs_b_ = all_categorized_pcs_b[category_name]
                    best_pc = all_categorized_pcs_b_[best_idx]
                    if group_name.split('_')[0] == 'right':
                        best_pc[..., 0] *= -1
                    grouped_best_pc = best_pc + grouped_kps_a[None, 0]
                    # print(grouped_best_pc.shape)
                    new_grouped_pcs[group_name].append(grouped_best_pc)
                else:
                    new_grouped_pcs[group_name].append(np.zeros((0, 3)))
        
        new_pcs = [np.concatenate([grouped_pc[i] for grouped_pc in new_grouped_pcs.values()], axis=0) for i in range(seq_len)]
        data_a_new['sequences'].append({
            'point_clouds': seq['point_clouds'],
            'point_clouds_trans': new_pcs,
            'keypoints': seq_kps
        })

    for group_name in JOINT_GROUPS:
        grouped_best_sims[group_name] = np.array(grouped_best_sims[group_name])
        mean_grouped_best_sim = np.mean(grouped_best_sims[group_name], axis=0)
        min_grouped_best_sim = np.min(grouped_best_sims[group_name], axis=0)
        print(f'{group_name} best sim: mean: {mean_grouped_best_sim}, min: {min_grouped_best_sim}')
    
    return data_a_new


def dataset_a2b_ismoving(data_a, all_categorized_kps_b, all_categorized_pcs_b, all_categorized_is_movings_b):
    data_a_new = data_a.copy()
    data_a_new['sequences'] = []
    grouped_best_sims = {group_name: [] for group_name in JOINT_GROUPS}

    for i, seq in tqdm(enumerate(data_a['sequences'])):
        # if i == 2:
        #     break

        new_grouped_pcs = {group_name: [] for group_name in JOINT_GROUPS}
        seq_len = len(seq['point_clouds'])
        seq_kps = seq['keypoints']

        for group_name, group_idxs in JOINT_GROUP_IDX_MAP.items():
            seq_grouped_kps_a = seq_kps[:, group_idxs]
            seq_grouped_kps_a_ = seq_grouped_kps_a - seq_grouped_kps_a[:, 0:1]
            seq_grouped_last_kps_a = np.concatenate([seq_grouped_kps_a[0:1], seq_grouped_kps_a[:-1]], axis=0)
            seq_grouped_flow_a = seq_grouped_kps_a - seq_grouped_last_kps_a
            seq_grouped_is_moving_a = np.abs(np.linalg.norm(seq_grouped_flow_a, axis=-1)).max(axis=1) > 0.02
            category_name = group_name.split('_')[-1]
            seq_grouped_sims = calc_similarity(seq_grouped_kps_a_, all_categorized_kps_b[category_name])

            for j in range(seq_len):
                grouped_kps_a = seq_grouped_kps_a[j]
                grouped_sims = seq_grouped_sims[j]
                grouped_is_moving_a = seq_grouped_is_moving_a[j]
                # print(grouped_is_moving_a)
                # print(all_categorized_is_movings_b[category_name].shape, all_categorized_is_movings_b[category_name].dtype, grouped_sims.shape, all_categorized_pcs_b[category_name].shape)
                grouped_sims_ = grouped_sims[all_categorized_is_movings_b[category_name] == grouped_is_moving_a]
                if len(grouped_sims_) > 0:
                    best_idx = np.argmax(grouped_sims_)
                    best_sim = grouped_sims_[best_idx]
                    grouped_best_sims[group_name].append(best_sim)
                    all_categorized_pcs_b_ = [all_categorized_pcs_b[category_name][i] for i in range(len(all_categorized_pcs_b[category_name])) if all_categorized_is_movings_b[category_name][i] == grouped_is_moving_a]
                    best_pc = all_categorized_pcs_b_[best_idx]
                    if group_name.split('_')[0] == 'right':
                        best_pc[..., 0] *= -1
                    grouped_best_pc = best_pc + grouped_kps_a[None, 0]
                    # print(grouped_best_pc.shape)
                    new_grouped_pcs[group_name].append(grouped_best_pc)
                else:
                    new_grouped_pcs[group_name].append(np.zeros((0, 3)))
        
        new_pcs = [np.concatenate([grouped_pc[i] for grouped_pc in new_grouped_pcs.values()], axis=0) for i in range(seq_len)]
        data_a_new['sequences'].append({
            'point_clouds': seq['point_clouds'],
            'point_clouds_trans': new_pcs,
            'keypoints': seq_kps
        })

    for group_name in JOINT_GROUPS:
        grouped_best_sims[group_name] = np.array(grouped_best_sims[group_name])
        mean_grouped_best_sim = np.mean(grouped_best_sims[group_name], axis=0)
        min_grouped_best_sim = np.min(grouped_best_sims[group_name], axis=0)
        print(f'{group_name} best sim: mean: {mean_grouped_best_sim}, min: {min_grouped_best_sim}')
    
    return data_a_new

def dataset_a2b_flow(data_a, all_categorized_kps_b, all_categorized_pcs_b, all_categorized_flows_b):
    data_a_new = data_a.copy()
    data_a_new['sequences'] = []
    grouped_best_sims = {group_name: [] for group_name in JOINT_GROUPS}

    for i, seq in tqdm(enumerate(data_a['sequences'])):
        # if i == 2:
        #     break

        new_grouped_pcs = {group_name: [] for group_name in JOINT_GROUPS}
        seq_len = len(seq['point_clouds'])
        seq_kps = seq['keypoints']

        for group_name, group_idxs in JOINT_GROUP_IDX_MAP.items():
            seq_grouped_kps_a = seq_kps[:, group_idxs]
            seq_grouped_kps_a_ = seq_grouped_kps_a - seq_grouped_kps_a[:, 0:1]
            category_name = group_name.split('_')[-1]
            seq_grouped_sims = calc_similarity_flow(seq_grouped_kps_a_, all_categorized_kps_b[category_name], all_categorized_flows_b[category_name])

            for j in range(seq_len):
                grouped_kps_a = seq_grouped_kps_a[j]
                grouped_sims = seq_grouped_sims[j]
                best_idx = np.argmax(grouped_sims)
                best_sim = grouped_sims[best_idx]
                grouped_best_sims[group_name].append(best_sim)
                best_pc = all_categorized_pcs_b[category_name][best_idx]
                if group_name.split('_')[0] == 'right':
                    best_pc[..., 0] *= -1
                grouped_best_pc = best_pc + grouped_kps_a[None, 0]
                new_grouped_pcs[group_name].append(grouped_best_pc)
        
        new_pcs = [np.concatenate([grouped_pc[i] for grouped_pc in new_grouped_pcs.values()], axis=0) for i in range(seq_len)]
        data_a_new['sequences'].append({
            'point_clouds': new_pcs,
            'keypoints': seq_kps
        })

    for group_name in JOINT_GROUPS:
        grouped_best_sims[group_name] = np.array(grouped_best_sims[group_name])
        mean_grouped_best_sim = np.mean(grouped_best_sims[group_name], axis=0)
        min_grouped_best_sim = np.min(grouped_best_sims[group_name], axis=0)
        print(f'{group_name} best sim: mean: {mean_grouped_best_sim}, min: {min_grouped_best_sim}')
    
    return data_a_new


def main(lidar_data_fn, mmwave_data_fn, output_lidar_data_fn, output_mmwave_data_fn):
    lidar_data = load_dataset(lidar_data_fn)
    mmwave_data = load_dataset(mmwave_data_fn, ratio=0.1)
    lidar_categorized_kps, lidar_categorized_pcs, _ = get_categorized_results(lidar_data)
    mmwave_categorized_kps, mmwave_categorized_pcs, mmwave_categorized_is_movings = get_categorized_results(mmwave_data, multi_view=True)
    
    lidar_data_new = dataset_a2b_ismoving(lidar_data, mmwave_categorized_kps, mmwave_categorized_pcs, mmwave_categorized_is_movings)
    dump_dataset(lidar_data_new, output_lidar_data_fn)
    mmwave_data_new = dataset_a2b(mmwave_data, lidar_categorized_kps, lidar_categorized_pcs)
    dump_dataset(mmwave_data_new, output_mmwave_data_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lidar to mmwave adaptation')
    parser.add_argument('--lidar_data_fn', type=str, required=True, help='Path to the lidar data file')
    parser.add_argument('--mmwave_data_fn', type=str, required=True, help='Path to the mmwave data file')
    parser.add_argument('--output_lidar_data_fn', type=str, required=True, help='Path to save the adapted lidar data file')
    parser.add_argument('--output_mmwave_data_fn', type=str, required=True, help='Path to save the adapted mmwave data file')

    args = parser.parse_args()
    # main(args.lidar_data_fn, args.mmwave_data_fn, args.output_lidar_data_fn)
    main(args.lidar_data_fn, args.mmwave_data_fn, args.output_lidar_data_fn, args.output_mmwave_data_fn)