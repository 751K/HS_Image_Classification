# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.util import view_as_windows
from sklearn.model_selection import StratifiedShuffleSplit


class HSIDataset(Dataset):
    def __init__(self, input_data, input_labels, dim=1):
        self.dim = dim
        self.data = torch.as_tensor(input_data, dtype=torch.float32).contiguous()
        self.labels = torch.tensor(input_labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[-1] if self.dim == 1 else len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_patches(input_data, input_labels, patch_size=5):
    rows, cols, bands = input_data.shape
    pad_width = patch_size // 2

    # 对数据进行填充
    padded_data = np.pad(input_data, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')
    padded_labels = np.pad(input_labels, pad_width, mode='constant', constant_values=0)

    # 使用 view_as_windows 生成 patches
    patches = view_as_windows(padded_data, (patch_size, patch_size, bands), step=1)
    patches = patches.reshape(rows * cols, patch_size, patch_size, bands)

    # 确保 patch_labels 的形状与 patches 的第一个维度一致
    patch_labels = padded_labels[pad_width:pad_width + rows, pad_width:pad_width + cols].flatten()

    # 找到有效的（非零）标签索引
    valid_indices = np.where(patch_labels != 0)[0]

    # 使用有效索引选择 patches 和 labels
    patches = patches[valid_indices]
    patch_labels = patch_labels[valid_indices]

    # 转换 patches 的形状为 (num_patches, bands, patch_size, patch_size)
    patches = patches.transpose(0, 3, 1, 2)

    return patches, patch_labels


def reshape_data_1D(data, labels):
    rows, cols, bands, = data.shape
    y = labels.flatten()
    valid_pixels = y != 0
    X = data.reshape(-1, bands)
    X = X[valid_pixels]
    y = y[valid_pixels]
    return X, y


def prepare_data(data, labels, test_size=0.9, random_state=42, dim=1, patch_size=5):
    if dim not in [1, 2, 3]:
        raise ValueError("Dim must be 1, 2, or 3")

    if dim == 1:
        X, y = reshape_data_1D(data, labels)

    else:
        # patch: (num_patches, bands, patch_size, patch_size)
        patches, patch_labels = create_patches(data, labels, patch_size)
        if dim == 2:
            X = patches
        else:
            X = patches.reshape(patches.shape[0], 1, patches.shape[1], patch_size, patch_size)
        y = patch_labels

    if test_size == 1:
        return X, y, None, None, None, None
    else:
        val_size = 0.5
        train_data, train_label, test_data, test_label, val_data, val_label = (
            spilt_three(X, y, test_size=test_size, val_size=val_size, random_state=random_state))
        return train_data, train_label, test_data, test_label, val_data, val_label


def split_two(data, labels, test_size=0.95, random_state=42):
    train_data, test_data, train_labels, test_labels = None, None, None, None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_indices, test_indices in sss.split(data, labels):
        train_data, test_data = data[train_indices], data[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

    return train_data, train_labels, test_data, test_labels


def spilt_three(data, labels, test_size=0.95, val_size=0.5, random_state=42):
    train_data, test_data, train_labels, test_labels, val_data, val_labels = None, None, None, None, None, None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_indices, test_indices in sss.split(data, labels):
        train_data, test_data = data[train_indices], data[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    for train_indices, val_indices in sss_val.split(train_data, train_labels):
        train_data, val_data = train_data[train_indices], train_data[val_indices]
        train_labels, val_labels = train_labels[train_indices], train_labels[val_indices]

    return train_data, train_labels, test_data, test_labels, val_data, val_labels


def create_data_loaders(X_train, y_train, X_test, y_test, batch_size, num_workers, dim=1, logger=None, X_val=None,
                        y_val=None):
    train_dataset = HSIDataset(X_train, y_train, dim)
    test_dataset = HSIDataset(X_test, y_test, dim)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    try:
        inputs, input_labels = next(iter(train_dataloader))
        if logger is not None:
            logger.info(f"Inputs shape: {inputs.shape}")
            logger.info(f"Labels shape: {input_labels.shape}")
    except Exception as e:
        logger.error(f"An error occurred while checking the data loaders: {e}")

    if X_val is not None and y_val is not None:
        val_dataset = HSIDataset(X_val, y_val, dim)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, test_dataloader, val_dataloader
    else:
        return train_dataloader, test_dataloader


# 使用示例
if __name__ == "__main__":
    # 假设您已经加载了数据和标签
    data = np.random.rand(145, 145, 200)  # 示例数据
    labels = np.random.randint(0, 10, (145, 145))  # 示例标签

    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(data, labels, dim=3)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=32, num_workers=4, dim=3
    )
