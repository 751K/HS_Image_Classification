import os

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split


class IndianPinesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # (samples, bands, 1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

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


def prepare_data(data, labels, test_size=0.1, val_size=0.1, random_state=42, dim=1):
    global X_train, y_train, X_val, y_val, X_test, y_test
    rows, cols, bands = data.shape

    # 创建掩码来选择非背景像素
    mask = labels != 0

    if dim == 1:
        # 对于1D情况，我们保持原来的处理方式
        samples = data[mask]
        targets = labels[mask] - 1  # 将类别标签从1-based改为0-based

        # 分割数据集
        X_train, X_temp, y_train, y_temp = train_test_split(samples, targets, test_size=test_size + val_size,
                                                            random_state=random_state, stratify=targets)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size),
                                                        random_state=random_state, stratify=y_temp)

        # 调整数据形状为 [samples, channels, sequence_length]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    elif dim == 2:
        # 对于2D情况，我们需要保持空间结构
        # 创建一个包含非背景像素索引的列表
        indices = np.argwhere(mask)

        # 分割索引
        train_indices, temp_indices = train_test_split(indices, test_size=test_size + val_size,
                                                       random_state=random_state)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size / (test_size + val_size),
                                                     random_state=random_state)

        # 创建训练、验证和测试掩码
        train_mask = np.zeros_like(mask, dtype=bool)
        val_mask = np.zeros_like(mask, dtype=bool)
        test_mask = np.zeros_like(mask, dtype=bool)

        train_mask[train_indices[:, 0], train_indices[:, 1]] = True
        val_mask[val_indices[:, 0], val_indices[:, 1]] = True
        test_mask[test_indices[:, 0], test_indices[:, 1]] = True

        # 准备数据集
        X_train = data[train_mask]
        y_train = labels[train_mask] - 1
        X_val = data[val_mask]
        y_val = labels[val_mask] - 1
        X_test = data[test_mask]
        y_test = labels[test_mask] - 1

        # 调整数据形状为 [samples, channels, height, width]
        # 在这里，我们保持每个样本的空间结构为1x1
        X_train = X_train.reshape(X_train.shape[0], bands, 1, 1)
        X_val = X_val.reshape(X_val.shape[0], bands, 1, 1)
        X_test = X_test.reshape(X_test.shape[0], bands, 1, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test


def reshape_to_1d(X):
    return X.reshape(X.shape[0], X.shape[1], 1)


def reshape_to_2d(X):
    return X.reshape(X.shape[0], 1, X.shape[1], 1)


def remove_background(X, y):
    """
    去除背景类（标签0）并调整标签
    """
    mask = y > 0
    X = X[mask]
    y = y[mask] - 1  # 类别从0开始
    return X, y


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers):
    train_dataset = IndianPinesDataset(X_train, y_train)
    val_dataset = IndianPinesDataset(X_val, y_val)
    test_dataset = IndianPinesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
