import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
from tqdm import tqdm
import argparse
import os
from functools import partial

def load_results(path):
    with open(path, 'rb') as f:
        results = pickle.load(f)

    pcs = results['input'][:, 2][..., [0, 2, 1]]
    kps_gt = results['gt'][:, 0][..., [0, 2, 1]]
    kps_pred = results['pred'][:, 0][..., [0, 2, 1]]
    return pcs, kps_gt, kps_pred

def get_edges(skeleton_type):
    if skeleton_type == 'simplecoco':
        edges = [
            [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [1, 7],
            [2, 8], [7, 9], [8, 10], [9, 11], [10, 12]
        ]
    elif skeleton_type == 'itop':
        edges = [
            [14, 12], [12, 10], [13, 11], [11, 9], [10, 8], [9, 8], [8, 1], [1, 0], [7, 5], [5, 3], 
            [3, 1], [6, 4], [4, 2], [2, 1]
        ]
    else:
        raise ValueError(f'Unknown skeleton type: {skeleton_type}')
    
    return edges

def preprocess_clip(data_dict, start, end):
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

    return new_data_dict, (min_x, max_x, min_y, max_y, min_z, max_z)

def process_frame(index, axes, data_dict, edges, bounds):
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z

    for i, key in enumerate(data_dict):
        data = data_dict[key]
        pc = data['pcs'][index]
        kps = data['kps'][index]
        ax = axes[i]

        ax.clear()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect([range_x, range_y, range_z])
        ax.set_xlim(min_x - range_x * 0.1, max_x + range_x * 0.1)
        ax.set_ylim(min_y - range_y * 0.1, max_y + range_y * 0.1)
        ax.set_zlim(min_z - range_z * 0.1, max_z + range_z * 0.1)

        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=10, c='slategrey')
        ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], s=50, c='crimson')
        for edge in edges:
            ax.plot(kps[edge, 0], kps[edge, 1], kps[edge, 2], c='royalblue', linewidth=3)

        ax.set_title(key, fontsize=20)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    labels = ['GT'] + args.labels
    edges = get_edges(args.skeleton_type)

    data_dict = {}
    for i, results_path in enumerate(args.results):
        pcs, kps_gt, kps_pred = load_results(results_path)
        if 'GT' not in data_dict:
            data_dict['GT'] = {
                'pcs': pcs,
                'kps': kps_gt
            }
        data_dict[labels[i+1]] = {
            'pcs': pcs,
            'kps': kps_pred
        }

    if args.clips[-1] == -1:
        args.clips[-1] = len(data_dict['GT']['pcs'])
    clips = [(args.clips[i], args.clips[i + 1]) for i in range(0, len(args.clips), 2)]

    for clip in tqdm(clips, desc='Clip'):
        fig = plt.figure(figsize=(len(labels) * args.width, args.height))
        axes = []
        for i in range(len(labels)):
            ax = fig.add_subplot(1, len(labels), i+1, projection='3d')
            axes.append(ax)
        fig.tight_layout()

        num_frames = clip[1] - clip[0]
        clipped_data_dict, bounds = preprocess_clip(data_dict, clip[0], clip[1])
        output_path = os.path.join(args.output_dir, f'vis_{clip[0]}_{clip[1]}.mp4')
        
        ani_func = partial(process_frame, axes=axes, data_dict=clipped_data_dict, edges=edges, bounds=bounds)
        ani = FuncAnimation(fig, ani_func, frames=num_frames, interval=1000/args.fps)
        ani.save(output_path, writer='ffmpeg')

        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', type=str, required=True)
    parser.add_argument('--labels', nargs='+', type=str, required=True)
    parser.add_argument('--clips', nargs='+', type=int, required=True)
    parser.add_argument('--skeleton_type', type=str, default='simplecoco')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--height', type=int, default=6)
    parser.add_argument('--output_dir', type=str, default='vis')
    args = parser.parse_args()
    main(args)