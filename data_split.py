from pathlib import Path
import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def load_labels(label_file):
    label_dict = dict()
    f = open(label_file, 'r')
    lines = f.readlines()[1:]
    f.close()
    
    subject_list = []
    label_list = []

    for line in lines:
        subject_id, label = line.strip().split(',')
        subject_list.append(subject_id)
        label_list.append(label)
    return subject_list, label_list

def stratified_split_random(target_dir, X, y, cv=5, random_state=42):
    df = pd.DataFrame()
    df['subject_id'] = X
    df['label'] = y
    df = df.sort_values(by='subject_id', ascending=1)
    X_data = np.asarray(df['subject_id'])
    y_label = np.asarray(df['label'])
    sss = StratifiedShuffleSplit(n_splits=cv, test_size=0.2, random_state=random_state)
    fold_idx = 0
    for train_index, test_index in sss.split(X_data, y_label):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_label[train_index], y_label[test_index]

        train_file_path = Path(target_dir, f'train_{fold_idx:02}.txt')
        f_train = open(str(train_file_path), 'w')
        f_train.write('subject_id,label\n')
        for i, infant_id in enumerate(X_train):
            label = y_train[i]
            f_train.write(f'{infant_id},{label}\n')
        f_train.close()

        test_file_path = Path(target_dir, f'test_{fold_idx:02}.txt')
        f_test = open(test_file_path, 'w')
        f_test.write('subject_id,label\n')
        for i, infant_id in enumerate(X_test):
            label = y_test[i]
            f_test.write(f'{infant_id},{label}\n')
        f_test.close()
        fold_idx += 1

def load_data(data_root, label_dict):
    data_paths = Path(data_root).rglob('pose_sequence.npy')

    infant_list = []
    labels = []

    for data_path in data_paths:
        if not data_path.exists():
            continue
        data_dir = data_path.resolve().parents[0]
        dir_name = str(data_dir.stem)
        infant_id = dir_name[:13]
        infant_list.append(dir_name)
        if infant_id in label_dict.keys():
            labels.append(label_dict[infant_id])
        else:
            labels.append(0)
    return infant_list, labels

def main():
    parser = argparse.ArgumentParser(description="Parse the parameters for the datasets split.")
    parser.add_argument('--cv', type=int, default=10, help='Number of splits.')
    parser.add_argument('--label-file', type=str, help='Path to the label file.')
    parser.add_argument('--out-dir', type=str, help='Path to the output directory.')
    parser.add_argument('--random-state', type=int, default=0, help='Random state.')
    args = parser.parse_args()
    cv = args.cv
    random_state = args.random_state
    out_dir = args.out_dir
    label_file = args.label_file
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    subject_list, label_list = load_labels(label_file)

    # infant_list, labels = load_data(data_root, label_dict)
    stratified_split_random(out_dir, subject_list, label_list, cv=cv, random_state=random_state)

if __name__ == '__main__':
    main()
