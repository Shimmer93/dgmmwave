import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from tqdm import tqdm
import argparse
import os
from functools import partial

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from misc.skeleton import ITOPSkeleton, SimpleCOCOSkeleton, JOINT_COLOR_MAP

def load_results(path):
    with open(path, 'rb') as f:
        results = pickle.load(f)

    # pcs = results['input'][..., [0, 2, 1]]
    # kps_gt = results['gt'][..., [0, 2, 1]]
    # kps_gt = kps_gt - kps_gt[:, 8:9, :]
    # kps_pred = results['pred'][..., [0, 2, 1]]
    # kps_pred = kps_pred - kps_pred[:, 8:9, :]
    # seq_idxs = results['seq_idx']

    pcs = []
    kps_gt = []
    kps_pred = []
    seq_idxs = []
    for i, seq in enumerate(results):
        pcs.append(seq['flow'][..., [0, 2, 1]])
        kps_gt.append(seq['keypoints'][..., [0, 2, 1]])
        kps_gt[-1] = kps_gt[-1] - kps_gt[-1][:, 8:9, :]
        kps_pred.append(seq['keypoints_pred'][..., [0, 2, 1]])
        kps_pred[-1] = kps_pred[-1] - kps_pred[-1][:, 8:9, :]
        seq_idxs.append(np.ones(seq['keypoints'].shape[0]) * i)
    pcs = np.concatenate(pcs, axis=0)
    kps_gt = np.concatenate(kps_gt, axis=0)
    kps_pred = np.concatenate(kps_pred, axis=0)
    seq_idxs = np.concatenate(seq_idxs, axis=0)

    return pcs, kps_gt, kps_pred, seq_idxs

def get_edges(skeleton_type):
    if skeleton_type == 'simplecoco':
        edges = SimpleCOCOSkeleton.bones
    elif skeleton_type == 'itop':
        edges = ITOPSkeleton.bones
    else:
        raise ValueError(f'Unknown skeleton type: {skeleton_type}')
    
    return edges

def preprocess_clip(data_dict, seq_idxs, start, end):
    new_data_dict = {}
    new_pcs = None
    all_kps = []
    for key in data_dict:
        data = data_dict[key]
        if new_pcs is None:
            new_pcs = data['pcs'][start:end]
        all_kps.append(data['kps'][start:end])
        
    all_kps = np.concatenate(all_kps, axis=0)
    all_ps = all_kps.reshape(-1, 3)
    center = all_ps.mean(axis=0)
    floor = all_ps.min(axis=0)[2]
    disp = center
    disp[2] = floor
    disp = disp.reshape(1, 1, 3)

    all_ps -= disp[0]
    min_x, max_x = np.min(all_ps[:, 0]), np.max(all_ps[:, 0])
    min_y, max_y = np.min(all_ps[:, 1]), np.max(all_ps[:, 1])
    min_z, max_z = np.min(all_ps[:, 2]), np.max(all_ps[:, 2])

    for key in data_dict:
        data = data_dict[key]
        new_data_dict[key] = {
            'pcs': new_pcs - disp,
            'kps': data['kps'][start:end] - disp
        }

    new_seq_idxs = seq_idxs[start:end]

    return new_data_dict, new_seq_idxs, (min_x, max_x, min_y, max_y, min_z, max_z)

def process_frame(index, fig, axes_list, data_dict, seq_idxs, edges, bounds):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    fig.suptitle(f'Sequence: {seq_idxs[index]}', fontsize=30)

    for i, key in enumerate(data_dict):
        data = data_dict[key]
        pc = data['pcs'][index]
        kps = data['kps'][index]
        ax_3d, ax_xy, ax_xz, ax_yz = axes_list[i]

        ax_3d.clear()
        # ax.grid(False)
        ax_3d.set_xticks([])
        ax_3d.set_yticks([])
        ax_3d.set_zticks([])
        ax_3d.set_box_aspect([range_x, range_y, range_z])
        ax_3d.set_xlim(min_x - range_x * 0.1, max_x + range_x * 0.1)
        ax_3d.set_ylim(min_y - range_y * 0.1, max_y + range_y * 0.1)
        ax_3d.set_zlim(min_z - range_z * 0.1, max_z + range_z * 0.1)

        ax_3d.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=10, c='slategrey')
        for j, kp in enumerate(kps):
            ax_3d.plot(kp[0], kp[1], kp[2], marker='o', markersize=10, c=JOINT_COLOR_MAP[j])
        for edge in edges:
            ax_3d.plot(kps[edge, 0], kps[edge, 1], kps[edge, 2], c='grey', linewidth=3)

        ax_3d.set_title(f'{key}_3D', fontsize=20)

        for ax, view, name in zip([ax_xy, ax_xz, ax_yz], [(0, 1), (0, 2), (1, 2)], ['XY', 'XZ', 'YZ']):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            min_list = [min_x, min_y, min_z]
            max_list = [max_x, max_y, max_z]
            range_list = [range_x, range_y, range_z]
            ax.set_aspect('equal')
            ax.set_xlim(min_list[view[0]] - range_list[view[0]] * 0.1, max_list[view[0]] + range_list[view[0]] * 0.1)
            ax.set_ylim(min_list[view[1]] - range_list[view[1]] * 0.1, max_list[view[1]] + range_list[view[1]] * 0.1)

            ax.scatter(pc[:, view[0]], pc[:, view[1]], s=10, c='slategrey')
            for j, kp in enumerate(kps):
                ax.plot(kp[view[0]], kp[view[1]], marker='o', markersize=10, c=JOINT_COLOR_MAP[j])
            for edge in edges:
                ax.plot(kps[edge, view[0]], kps[edge, view[1]], c='grey', linewidth=3)

            ax.set_title(f'{key}_{name}', fontsize=20)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    labels = ['GT'] + args.labels
    edges = get_edges(args.skeleton_type)

    data_dict = {}
    for i, results_path in enumerate(args.results):
        pcs, kps_gt, kps_pred, seq_idxs = load_results(results_path)
        if 'GT' not in data_dict:
            data_dict['GT'] = {
                'pcs': pcs,
                'kps': kps_gt
            }
        data_dict[labels[i+1]] = {
            'pcs': pcs,
            'kps': kps_pred
        }

    if args.clips is not None:
        if args.clips[-1] == -1:
            args.clips[-1] = len(data_dict['GT']['pcs'])
        clips = [(args.clips[i], args.clips[i + 1]) for i in range(0, len(args.clips), 2)]
    else:
        clips = []
        for i in args.seqs:
            start = -1
            end = -1
            for j, seq_idx in enumerate(seq_idxs):
                if seq_idx == i and start == -1:
                    start = j
                if j == len(seq_idxs) - 1 and start != -1:
                    end = j + 1
                if seq_idx != i and start != -1:
                    end = j
                    break
            if start != -1 and end != -1:
                clips.append((start, end))

    for clip in tqdm(clips, desc='Clip'):
        fig = plt.figure(figsize=(len(labels) * args.width, 4 * args.height))
        axes_list = []
        for i in range(len(labels)):
            axes = []
            ax_3d = fig.add_subplot(4, len(labels), i+1, projection='3d')
            ax_xy = fig.add_subplot(4, len(labels), i+1 + len(labels))
            ax_xz = fig.add_subplot(4, len(labels), i+1 + 2 * len(labels))
            ax_yz = fig.add_subplot(4, len(labels), i+1 + 3 * len(labels))
            axes.append(ax_3d)
            axes.append(ax_xy)
            axes.append(ax_xz)
            axes.append(ax_yz)
            axes_list.append(axes)
        fig.tight_layout()

        num_frames = clip[1] - clip[0]
        clipped_data_dict, clipped_seq_idxs, bounds = preprocess_clip(data_dict, seq_idxs, clip[0], clip[1])
        output_path = os.path.join(args.output_dir, f'vis_{clip[0]}_{clip[1]}.mp4')
        
        ani_func = partial(process_frame, fig=fig, axes_list=axes_list, data_dict=clipped_data_dict, seq_idxs=clipped_seq_idxs, edges=edges, bounds=bounds)
        ani = FuncAnimation(fig, ani_func, frames=num_frames, interval=1000/args.fps)
        ani.save(output_path, writer='ffmpeg')

        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', type=str, required=True)
    parser.add_argument('--labels', nargs='+', type=str, required=True)
    parser.add_argument('--clips', nargs='+', type=int)
    parser.add_argument('--seqs', nargs='+', type=int)
    parser.add_argument('--skeleton_type', type=str, default='simplecoco')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--height', type=int, default=6)
    parser.add_argument('--output_dir', type=str, default='vis')
    args = parser.parse_args()
    main(args)