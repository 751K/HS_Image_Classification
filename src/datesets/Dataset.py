# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


class HSIDataset(Dataset):
    def __init__(self, data, labels, dim=1):
        self.dim = dim
        self.data = torch.as_tensor(data, dtype=torch.float32).contiguous()
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[-1] if self.dim == 1 else len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_patches(data, labels, patch_size=5):
    rows, cols, bands = data.shape
    pad_width = patch_size // 2

    # 对数据进行填充
    padded_data = np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')
    padded_labels = np.pad(labels, pad_width, mode='constant', constant_values=0)

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


def create_spectral_samples(data, labels, sequence_length=25):
    bands, rows, cols = data.shape
    flattened_data = data.reshape(bands, -1)  # shape: (bands, rows*cols)
    flattened_labels = labels.flatten()

    # 只保留非背景像素
    valid_pixels = flattened_labels != 0
    samples = flattened_data[:, valid_pixels]
    sample_labels = flattened_labels[valid_pixels] - 1

    # 如果像素数不能被 sequence_length 整除，截断到最近的倍数
    num_samples = (samples.shape[1] // sequence_length) * sequence_length
    samples = samples[:, :num_samples]
    sample_labels = sample_labels[:num_samples]

    # 重塑为 (bands, sequence_length, num_sequences)
    samples = samples.reshape(bands, sequence_length, -1)

    # 转换标签以匹配序列
    sample_labels = sample_labels.reshape(-1, sequence_length)[:, 0]

    return samples, sample_labels


def create_spectral_patches(data, labels, patch_size, stride):
    bands, rows, cols = data.shape

    # 计算可以创建的patch数量
    num_patches = (bands - patch_size) // stride + 1

    # 初始化patch数组
    patches = np.zeros((rows * cols * num_patches, patch_size))
    patch_labels = np.zeros(rows * cols * num_patches, dtype=int)

    index = 0
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:  # 只处理非背景像素
                for k in range(0, bands - patch_size + 1, stride):
                    patches[index] = data[k:k + patch_size, i, j]
                    patch_labels[index] = labels[i, j] - 1  # 将类别标签从1-based改为0-based
                    index += 1

    # 裁剪数组到实际使用的大小
    patches = patches[:index]
    patch_labels = patch_labels[:index]

    return patches, patch_labels


def prepare_data(data, labels, test_size=0.65, val_size=0.05, random_state=42, dim=1, patch_size=5):
    if dim not in [1, 2, 3]:
        raise ValueError("Dim must be 1, 2, or 3")

    if dim == 1:
        rows, cols, bands, = data.shape

        y = labels.flatten()
        valid_pixels = y != 0

        # 重塑数据为 (pixels, bands)
        X = data.reshape(-1, bands)

        # 使用相同的索引选择有效像素
        X = X[valid_pixels]
        y = y[valid_pixels]

    else:  # dim == 2 or dim == 3
        # patch: (num_patches, bands, patch_size, patch_size)
        patches, patch_labels = create_patches(data, labels, patch_size)
        if dim == 2:
            X = patches.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
        else:  # dim == 3
            X = patches.reshape(patches.shape[0], 1, patches.shape[1], patch_size, patch_size)
        y = patch_labels

    X_train, y_train, X_val, y_val, X_test, y_test = optimized_split(X, y, test_size=0.65, val_size=0.05,
                                                                     random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


def optimized_split(X, y, test_size=0.65, val_size=0.05, random_state=42):
    # 计算训练集大小
    train_size = 1 - test_size - val_size

    # 使用 StratifiedShuffleSplit 一次性分割数据
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_size, random_state=random_state)

    for train_index, temp_index in sss.split(X, y):
        X_train, X_temp = X[train_index], X[temp_index]
        y_train, y_temp = y[train_index], y[temp_index]

    # 计算验证集在剩余数据中的比例
    val_ratio = val_size / (test_size + val_size)

    # 再次使用 StratifiedShuffleSplit 分割验证集和测试集
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio, random_state=random_state)

    for val_index, test_index in sss_val.split(X_temp, y_temp):
        X_val, X_test = X_temp[val_index], X_temp[test_index]
        y_val, y_test = y_temp[val_index], y_temp[test_index]

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers, dim=1, logger=None):
    train_dataset = HSIDataset(X_train, y_train, dim)
    val_dataset = HSIDataset(X_val, y_val, dim)
    test_dataset = HSIDataset(X_test, y_test, dim)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for inputs, labels in train_loader:
        break  # 只打印第一个batch

    try:
        inputs, labels = next(iter(train_loader))
        logger.info(f"Inputs shape: {inputs.shape}")
        logger.info(f"Labels shape: {labels.shape}")
    except Exception as e:
        logger.error(f"An error occurred while checking the data loaders: {e}")
    return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
    # 假设您已经加载了数据和标签
    data = np.random.rand(145, 145, 200)  # 示例数据
    labels = np.random.randint(0, 10, (145, 145))  # 示例标签

    sequence_length = 25  # 设置 sequence_length
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(data, labels, dim=1, sequence_length=sequence_length)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32, num_workers=4, dim=1
    )
