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
        self.labels = torch.as_tensor(input_labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[-1] if self.dim == 1 else len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_patches(input_data: np.ndarray, input_labels: np.ndarray, patch_size: int = 5, batch_size: int = 1024):
    """通过分批处理减少内存峰值使用"""
    rows, cols, bands = input_data.shape
    pad_width = patch_size // 2

    padded_data = np.pad(input_data, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='reflect')

    # 找出所有有效像素位置
    valid_positions = []
    valid_labels = []

    for i in range(rows):
        for j in range(cols):
            if input_labels[i, j] != 0:
                valid_positions.append((i, j))
                valid_labels.append(input_labels[i, j] - 1)

    num_patches = len(valid_positions)
    all_patches = []

    # 分批处理
    for start_idx in range(0, num_patches, batch_size):
        end_idx = min(start_idx + batch_size, num_patches)
        batch_patches = np.zeros((end_idx - start_idx, bands, patch_size, patch_size), dtype=input_data.dtype)

        for i, (r, c) in enumerate(valid_positions[start_idx:end_idx]):
            r_pad, c_pad = r + pad_width, c + pad_width
            patch = padded_data[r_pad - pad_width:r_pad + pad_width + 1, c_pad - pad_width:c_pad + pad_width + 1, :]
            batch_patches[i] = patch.transpose(2, 0, 1)

        all_patches.append(batch_patches)

    # 合并所有批次
    final_patches = np.vstack(all_patches)
    return final_patches, np.array(valid_labels)


def reshape_data_1D(data: np.ndarray, labels: np.ndarray):
    """
    将数据和标签重塑为一维形式，并移除背景像素。

    Args:
        data (np.ndarray): 输入数据数组，形状为 (rows, cols, bands)。
        labels (np.ndarray): 输入标签数组，形状为 (rows, cols)。

    Returns:
        tuple: 包含以下两个元素的元组：
            - X (np.ndarray): 重塑后的数据数组，形状为 (num_valid_pixels, bands)。
            - y (np.ndarray): 重塑后的标签数组，形状为 (num_valid_pixels,)，背景像素已移除，标签值已减 1。
    """
    rows, cols, bands, = data.shape
    y = labels.flatten()
    valid_pixels = y != 0
    X = data.reshape(-1, bands)
    X = X[valid_pixels]
    y = y[valid_pixels]
    y = y - 1
    return X, y


def prepare_data(data: np.ndarray, labels: np.ndarray, test_size: float = None, random_state: int = 42, dim: int = 1,
                 patch_size: int = 5, logger=None):
    """
    准备数据，根据指定的维度进行处理，并划分为训练集、测试集和验证集。

    Args:
        data (np.ndarray): 输入数据数组，形状为 (rows, cols, bands)。
        labels (np.ndarray): 输入标签数组，形状为 (rows, cols)。
        test_size (float, optional): 测试集比例。默认为 0.9。
        random_state (int, optional): 随机种子，用于重复实验。默认为 42。
        dim (int, optional): 数据维度。1 表示一维数据，2 表示二维patch数据, 3表示三维patch数据。默认为 1。
        patch_size (int, optional): patch的大小，当dim>1时有效。默认为 5。

    Returns:
        tuple: 包含训练集、测试集和验证集数据的元组，格式为 (X_train, y_train, X_test, y_test, X_val, y_val)。
               如果 test_size 为 1，则只返回处理后的数据和标签 (processed_data, processed_labels)。
    """
    if dim not in [1, 2, 3]:
        raise ValueError("Dim must be 1, 2, or 3")

    # 根据维度处理数据
    if dim == 1:
        processed_data, processed_labels = reshape_data_1D(data, labels)
    else:
        patches, patch_labels = create_patches(data, labels, patch_size)
        if dim == 2:
            processed_data = patches
        else:  # dim == 3
            processed_data = patches.reshape(patches.shape[0], 1, patches.shape[1], patch_size, patch_size)
        processed_labels = patch_labels

    if logger:
        logger.info(f"数据处理完成，数据形状: {processed_data.shape}, 标签形状: {processed_labels.shape}")

    if test_size == 1:
        return processed_data, processed_labels
    else:
        train_data, train_labels, test_data, test_labels, val_data, val_labels = (
            spilt_three(processed_data, processed_labels,
                        test_size=test_size,
                        random_state=random_state))
        return train_data, train_labels, test_data, test_labels, val_data, val_labels


def split_two(data: np.ndarray, labels: np.ndarray, test_size: float = 0.95, random_state: int = 42):
    """
        将数据划分为训练集和测试集。

        Args:
            data (np.ndarray): 输入数据数组。
            labels (np.ndarray): 输入标签数组。
            test_size (float, optional): 测试集比例。默认为 0.95。
            random_state (int, optional): 随机种子，用于重复实验。默认为 42。

        Returns:
            tuple: 包含训练集数据、训练集标签、测试集数据和测试集标签的元组，
                   格式为 (train_data, train_labels, test_data, test_labels)。
     """
    train_data, test_data, train_labels, test_labels = None, None, None, None

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_indices, test_indices in sss.split(data, labels):
        train_data, test_data = data[train_indices], data[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

    return train_data, train_labels, test_data, test_labels


def spilt_three(data: np.ndarray, labels: np.ndarray, test_size: float = 0.95, val_size: float = 0.5,
                random_state: int = 42):
    """
           将数据划分为训练集,测试集与验证集。

           Args:
               data (np.ndarray): 输入数据数组。
               labels (np.ndarray): 输入标签数组。
               test_size (float, optional): 测试集比例。默认为 0.95。
               val_size(float, optional): 除去测试集后的验证集比例。默认为 0.5。
               random_state (int, optional): 随机种子，用于重复实验。默认为 42。

           Returns:
               tuple: 包含训练集数据、训练集标签、测试集数据，测试集标签，验证集数据，验证集标签的元组，
                      格式为 (train_data, train_labels, test_data, test_labels)。
    """
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


def create_one_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 32, num_workers: int = 4, dim: int = 1):
    """
    创建单个数据集的DataLoader。

    Args:
        X (np.ndarray): 输入数据。
        y (np.ndarray): 输入标签。
        batch_size (int, optional): 批量大小。默认为 32。
        num_workers (int, optional): 用于数据加载的worker数量。默认为 4。
        dim (int, optional): 数据维度。默认为 1。

    Returns:
        DataLoader: 用于加载数据的DataLoader。
    """
    dataset = HSIDataset(X, y, dim)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader


def create_two_loader(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                      batch_size: int = 32, num_workers: int = 4, dim: int = 1,
                      logger=None):
    """
        创建训练集和测试集的数据加载器。

        Args:
            X_train (np.ndarray): 训练数据。
            y_train (np.ndarray): 训练标签。
            X_test (np.ndarray): 测试数据。
            y_test (np.ndarray): 测试标签。
            batch_size (int, optional): 批量大小。默认为 32。
            num_workers (int, optional): 用于数据加载的worker数量。默认为 4。
            dim (int, optional): 数据维度。默认为 1。
            logger (logging.Logger, optional): 日志记录器。默认为 None。

        Returns:
            tuple: 包含训练数据加载器和测试数据加载器的元组。
                   格式为 (train_dataloader, test_dataloader)。
    """
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
        import traceback
        error_msg = f"程序执行过程中发生错误:\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(error_msg)

    return train_dataloader, test_dataloader


def create_three_loader(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 32, num_workers: int = 4, dim: int = 1,
                        logger=None):
    """
    创建训练集、测试集和验证集的数据加载器。

    Args:
        X_train (np.ndarray): 训练数据。
        y_train (np.ndarray): 训练标签。
        X_test (np.ndarray): 测试数据。
        y_test (np.ndarray): 测试标签。
        X_val (np.ndarray): 验证数据。
        y_val (np.ndarray): 验证标签。
        batch_size (int, optional): 批量大小。默认为 32。
        num_workers (int, optional): 用于数据加载的worker数量。默认为 4。
        dim (int, optional): 数据维度。默认为 1。
        logger (logging.Logger, optional): 日志记录器。默认为 None。

    Returns:
        tuple: 包含训练数据加载器、测试数据加载器和验证数据加载器的元组。
               格式为 (train_dataloader, test_dataloader, val_dataloader)。
    """
    train_dataset = HSIDataset(X_train, y_train, dim)
    test_dataset = HSIDataset(X_test, y_test, dim)
    val_dataset = HSIDataset(X_val, y_val, dim)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    try:
        inputs, input_labels = next(iter(train_dataloader))
        if logger is not None:
            logger.info(f"Inputs shape: {inputs.shape}")
            logger.info(f"Labels shape: {input_labels.shape}")
    except Exception as e:
        import traceback
        error_msg = f"程序执行过程中发生错误:\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(error_msg)

    return train_dataloader, test_dataloader, val_dataloader


# 使用示例
if __name__ == "__main__":
    # 假设您已经加载了数据和标签
    data = np.random.rand(145, 145, 200)  # 示例数据
    labels = np.random.randint(0, 10, (145, 145))  # 示例标签

    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(data, labels, dim=3)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    train_loader = create_one_loader(X_train, y_train, batch_size=32, num_workers=4, dim=3)
    for inputs, labels in train_loader:
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
    train_loader, test_loader = create_two_loader(X_train, y_train, X_test, y_test, batch_size=32, num_workers=4, dim=3)
    for inputs, labels in train_loader:
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
    train_loader, test_loader, val_loader = create_three_loader(X_train, y_train, X_test, y_test, X_val, y_val,
                                                                batch_size=32, num_workers=4, dim=3)
    for inputs, labels in train_loader:
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
