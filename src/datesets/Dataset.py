import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class HSIDataset(Dataset):
    def __init__(self, data, labels, dim=1):
        self.dim = dim
        self.data = torch.as_tensor(data, dtype=torch.float32).contiguous()
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.data.shape[-1] if self.dim == 1 else len(self.data)

    def __getitem__(self, idx):
        if self.dim == 1:
            return self.data[:, :, idx], self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]


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


def prepare_data(data, labels, test_size=0.65, val_size=0.05, random_state=42, dim=1, patch_size=5, sequence_length=25):
    if dim == 1:
        data = np.transpose(data, (2, 0, 1))  # 转置为 (bands, rows, cols)
        samples, sample_labels = create_spectral_samples(data, labels, sequence_length)

        # 分割数据
        num_sequences = samples.shape[2]
        indices = np.arange(num_sequences)
        train_indices, temp_indices = train_test_split(indices, test_size=test_size + val_size,
                                                       random_state=random_state, stratify=sample_labels)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size / (test_size + val_size),
                                                     random_state=random_state, stratify=sample_labels[temp_indices])

        X_train = samples[:, :, train_indices]
        X_val = samples[:, :, val_indices]
        X_test = samples[:, :, test_indices]
        y_train = sample_labels[train_indices]
        y_val = sample_labels[val_indices]
        y_test = sample_labels[test_indices]

    elif dim == 2 or dim == 3:
        # 保持原有的 dim=2 和 dim=3 的处理逻辑
        patches, patch_labels = create_patches(data, labels, patch_size)

        if dim == 2:
            patches = patches.reshape(patches.shape[0], patches.shape[1], patch_size, patch_size)
        else:  # dim == 3
            patches = patches.reshape(patches.shape[0], 1, patches.shape[1], patch_size, patch_size)

        X_train, X_temp, y_train, y_temp = train_test_split(patches, patch_labels, test_size=test_size + val_size,
                                                            random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size),
                                                        random_state=random_state)

    else:
        raise ValueError("Dim must be 1, 2, or 3")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers, dim=1):
    train_dataset = HSIDataset(X_train, y_train, dim)
    val_dataset = HSIDataset(X_val, y_val, dim)
    test_dataset = HSIDataset(X_test, y_test, dim)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for inputs, labels in train_loader:
        print(f"Inputs shape: {inputs.shape}")
        print(f"Labels shape: {labels.shape}")
        break  # 只打印第一个batch

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
