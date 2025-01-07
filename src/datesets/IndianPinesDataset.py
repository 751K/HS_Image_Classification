import os

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class IndianPinesDataset(Dataset):
    def __init__(self, data, labels, dim=3):
        self.dim = dim
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_data(data_path, gt_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件未找到：{data_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"地面真实标签文件未找到：{gt_path}")

    data = loadmat(data_path)['indian_pines_corrected']
    labels = loadmat(gt_path)['indian_pines_gt']

    # 数据归一化
    data = (data - np.mean(data, axis=(0, 1))) / np.std(data, axis=(0, 1))

    return data, labels


def create_patches(data, labels, patch_size=5):
    rows, cols, bands = data.shape
    pad_width = patch_size // 2

    # 对数据和标签进行填充
    padded_data = np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')
    padded_labels = np.pad(labels, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=0)

    patches = []
    patch_labels = []

    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:  # 只选择非背景像素
                patch = padded_data[i:i + patch_size, j:j + patch_size, :]
                label_patch = padded_labels[i:i + patch_size, j:j + patch_size]
                patches.append(patch.transpose(2, 0, 1))  # 转换为 (200, 5, 5)
                patch_labels.append(label_patch)

    return np.array(patches), np.array(patch_labels)


def prepare_data(data, labels, test_size=0.6, val_size=0.1, random_state=42, dim=1):
    if dim == 1:
        mask = labels != 0
        samples = data[mask]
        targets = labels[mask] - 1

        X_train, X_temp, y_train, y_temp = train_test_split(samples, targets, test_size=test_size + val_size,
                                                            random_state=random_state, stratify=targets)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size),
                                                        random_state=random_state, stratify=y_temp)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    elif dim == 2:
        # 创建5x5的图像块
        patches, patch_labels = create_patches(data, labels, patch_size=5)

        # 分割数据集
        X_train, X_temp, y_train, y_temp = train_test_split(patches, patch_labels, test_size=test_size + val_size,
                                                            random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size),
                                                        random_state=random_state)

    elif dim == 3:
        patches, patch_labels = create_patches(data, labels, patch_size=5)
        patches = torch.from_numpy(patches).unsqueeze(1)

        # 分割数据集
        X_train, X_temp, y_train, y_temp = train_test_split(patches, patch_labels, test_size=test_size + val_size,
                                                            random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size),
                                                        random_state=random_state)

    else:
        raise ValueError("dim must be 1, 2, or 3")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers, dim=1):
    train_dataset = IndianPinesDataset(X_train, y_train, dim)
    val_dataset = IndianPinesDataset(X_val, y_val, dim)
    test_dataset = IndianPinesDataset(X_test, y_test, dim)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for inputs, labels in train_loader:
        print(f"Inputs shape: {inputs.shape}")
        print(f"Labels shape: {labels.shape}")
        break  # 只打印第一个batch

    return train_loader, val_loader, test_loader
