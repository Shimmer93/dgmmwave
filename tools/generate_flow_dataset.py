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

def process_data(data):
    processed_data = []
    for seq in data:
        del seq['input']
        processed_data.append(seq)
    return processed_data

def main(train_data, val_data, test_data):
    train_data = process_data(train_data)
    val_data = process_data(val_data)
    test_data = process_data(test_data)

    seqs = train_data + val_data + test_data
    
    idxs = np.arange(len(seqs))
    train_split_idxs = idxs[:len(train_data)]
    val_split_idxs = idxs[len(train_data):len(train_data) + len(val_data)]
    test_split_idxs = idxs[len(train_data) + len(val_data):]

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