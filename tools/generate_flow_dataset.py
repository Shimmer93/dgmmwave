import pickle
import argparse
import numpy as np

def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def generate_split(data):
    split_seqs = []
    split_idxs = []
    last_seq_idx = -1
    for i in range(len(data['pred'])):
        seq_idx = int(data['seq_idx'][i].item())
        kps = data['gt'][i]
        kps_pred = data['pred'][i]
        flow = data['flow'][i]
        if seq_idx != last_seq_idx:
            last_seq_idx = seq_idx
            split_seqs.append({'keypoints': [], 'keypoints_pred': [], 'flow': []})
            split_idxs.append(seq_idx)
        split_seqs[-1]['keypoints'].append(kps)
        split_seqs[-1]['keypoints_pred'].append(kps_pred)
        split_seqs[-1]['flow'].append(flow)

    flow_std_list = []
    for i in range(len(split_seqs)):
        split_seqs[i]['keypoints'] = np.array(split_seqs[i]['keypoints'])[:, 0, ...]
        split_seqs[i]['keypoints_pred'] = np.array(split_seqs[i]['keypoints_pred'])[:, 0, ...]
        flow = np.array(split_seqs[i]['flow'])
        N, T, J, D = flow.shape

        flow = flow[:, -1, ...]

        # flow_ = np.zeros((N + T - 1, J, D))
        # # flow_mag_table = np.zeros((N + T - 1, N, J))
        # for j in range(T):
        #     flow_[j:j+N] += flow[:, j]
        #     flow_[-1 * j] *= (T / (j + 1))

        # for k in range(N):
        #     flow_mag_table[k:k+T, k] = np.linalg.norm(flow[k], axis=-1)
        # flow_stds = np.zeros((N, J))
        # for k in range(N):
        #     for l in range(J):
        #         flow_mags = flow_mag_table[k, :, l]
        #         flow_mags = flow_mags[flow_mags > 0]
        #         print(flow_mags)
        #         flow_std = np.std(flow_mags)
        #         flow_stds[k, l] = flow_std
        # flow_std_list.append(flow_stds)

        # flow = flow_[T-1:]
        # split_seqs[i]['flow'] = flow / T
        split_seqs[i]['flow'] = flow

    # write_pkl('flow_std.pkl', flow_std_list)

    return split_seqs, split_idxs

def main(train_data, val_data, test_data):
    train_split_seqs, train_split_idxs = generate_split(train_data)
    val_split_seqs, val_split_idxs = generate_split(val_data)
    test_split_seqs, test_split_idxs = generate_split(test_data)
    seqs = train_split_seqs + val_split_seqs + test_split_seqs

    splits = {
        'train': train_split_idxs,
        'val': val_split_idxs,
        'test': test_split_idxs
    }

    data = {
        'splits': splits,
        'sequences': seqs
    }

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate flow dataset')
    parser.add_argument('train_data', type=str, help='Path to train data pkl file')
    parser.add_argument('val_data', type=str, help='Path to validation data pkl file')
    parser.add_argument('test_data', type=str, help='Path to test data pkl file')
    parser.add_argument('output_path', type=str, help='Output path for the generated dataset')

    args = parser.parse_args()

    train_data = read_pkl(args.train_data)
    val_data = read_pkl(args.val_data)
    test_data = read_pkl(args.test_data)

    data = main(train_data, val_data, test_data)

    write_pkl(args.output_path, data)
    print(f"Dataset generated and saved to {args.output_path}")