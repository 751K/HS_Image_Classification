import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.kernel_approximation import Nystroem

from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


def spectral_kernel_pca_reduction(data, n_components=20, kernel='rbf', gamma=None, random_state=None, n_samples=10000):
    """
    对高光谱图像的光谱维度进行优化的核PCA降维

    Args:
        data : numpy array, 形状为 (height, width, n_bands)
        n_components : int, 可选 (默认为 20)
        kernel : string, 可选 (默认为 'rbf')
        gamma : float, 可选 (默认为 None)
        random_state : int 或 RandomState 实例, 可选 (默认为 None)
        n_samples : int, 可选 (默认为 10000)
            用于近似的样本数

    Returns:
        reduced_data : numpy array, 形状为 (height, width, n_components)
        metrics : dict
            包含评估指标的字典
    """
    height, width, n_bands = data.shape
    data_2d = data.reshape(-1, n_bands).astype(np.float32)

    # 标准化数据
    scaler = StandardScaler()
    data_2d_scaled = scaler.fit_transform(data_2d)

    # 使用 Nystroem 方法进行特征近似
    feature_map_nystroem = Nystroem(kernel=kernel, n_components=n_components, random_state=random_state, n_jobs=-1)
    data_2d_approximated = feature_map_nystroem.fit_transform(data_2d_scaled)

    # 使用 PCA 进行最终降维
    pca = KernelPCA(n_components=n_components, kernel='linear', random_state=random_state, n_jobs=-1)
    reduced_data_2d = pca.fit_transform(data_2d_approximated)

    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    # 计算近似的解释方差比
    explained_variance_ratio = pca.lambdas_ / np.sum(pca.lambdas_)
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    metrics = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_explained_variance_ratio': cumulative_explained_variance_ratio
    }

    return reduced_data, metrics


# 使用示例
if __name__ == "__main__":
    from src.datesets.datasets_load import load_dataset
    import matplotlib.pyplot as plt

    # 加载数据集
    data, labels, dataset_info = load_dataset('Pavia')

    # 应用优化的核 PCA 光谱降维
    n_components = 20
    reduced_data, metrics = spectral_kernel_pca_reduction(data, n_components=n_components, kernel='rbf')

    print("原始数据形状:", data.shape)
    print("降维后的数据形状:", reduced_data.shape)
    print(f"累积解释方差比: {metrics['cumulative_explained_variance_ratio'][-1]:.4f}")

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data[50, 50, :])
    plt.title("Original Spectrum")
    plt.xlabel("Band")
    plt.ylabel("Intensity")

    plt.subplot(1, 2, 2)
    plt.plot(reduced_data[50, 50, :])
    plt.title(f"Reduced Spectrum ({n_components} components)")
    plt.xlabel("Kernel PCA Component")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

    # 可视化累积解释方差比
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), metrics['cumulative_explained_variance_ratio'])
    plt.title("Cumulative Explained Variance Ratio")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.grid(True)
    plt.show()
