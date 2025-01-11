import numpy as np
from umap import UMAP
from src.datesets.datasets_load import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt


def spectral_umap_reduction(data, n_components=10, n_neighbors=15, min_dist=0.1, random_seed=42):
    """
    对高光谱图像进行UMAP降维

    :param data: 原始数据，形状为 (height, width, n_bands)
    :param n_components: 降维后的维度
    :param n_neighbors: UMAP参数，临近点数量
    :param min_dist: UMAP参数，最小距离
    :param random_seed: 随机种子
    :return: 降维后的数据，形状为 (height, width, n_components)
    """
    height, width, n_bands = data.shape
    data_2d = data.reshape(-1, n_bands)

    print(f"开始 UMAP 降维，数据形状: {data_2d.shape}")
    start_time = time.time()

    reducer = UMAP(n_components=n_components,
                   n_neighbors=n_neighbors,
                   min_dist=min_dist,
                   random_state=random_seed)

    reduced_data_2d = reducer.fit_transform(data_2d)

    end_time = time.time()
    print(f"UMAP 降维完成，耗时: {end_time - start_time:.2f} 秒")

    reduced_data = reduced_data_2d.reshape(height, width, n_components)
    return reduced_data


def compute_evaluation_metrics(original_data, reduced_data, n_neighbors=5):
    """
    计算降维结果的评估指标

    :param original_data: 原始数据，形状为 (n_samples, n_features)
    :param reduced_data: 降维后的数据，形状为 (n_samples, n_components)
    :param n_neighbors: 用于计算可信度的邻居数
    :return: 字典，包含相关系数和可信度
    """
    print("开始计算评估指标...")
    start_time = time.time()

    # 计算相关系数
    correlation, _ = pearsonr(pairwise_distances(original_data).flatten(),
                              pairwise_distances(reduced_data).flatten())

    # 计算可信度
    trustworthiness = compute_trustworthiness(original_data, reduced_data, n_neighbors)

    end_time = time.time()
    print(f"评估指标计算完成，耗时: {end_time - start_time:.2f} 秒")

    return {
        "correlation": correlation,
        "trustworthiness": trustworthiness
    }


def compute_trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean'):
    n_samples = X.shape[0]
    dist_X = pairwise_distances(X, metric=metric)
    dist_X_embedded = pairwise_distances(X_embedded, metric=metric)

    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)

    n_neighbors = min(n_neighbors, n_samples - 1)
    t = 0
    for i in range(n_samples):
        for j in ind_X_embedded[i, 1:n_neighbors + 1]:
            rank = np.where(ind_X[i] == j)[0][0]
            t += rank - n_neighbors if rank > n_neighbors else 0

    t = 1 - t * (2 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)))
    return t


def plot_spectra(original_data, reduced_data, save_path='umap_comparison.png'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(original_data[50, 50, :])
    plt.title("Original Spectrum")
    plt.xlabel("Band")
    plt.ylabel("Intensity")

    plt.subplot(1, 2, 2)
    plt.plot(reduced_data[50, 50, :])
    plt.title("Reduced Spectrum")
    plt.xlabel("UMAP Component")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"图像已保存为 {save_path}")
    plt.show()


if __name__ == "__main__":
    print("开始加载数据...")
    start_time = time.time()
    data, labels, dataset_info = load_dataset('Pavia')
    end_time = time.time()
    print(f"数据加载完成，耗时: {end_time - start_time:.2f} 秒")

    # UMAP降维
    n_components = 30
    reduced_data = spectral_umap_reduction(data, n_components=n_components)

    # 计算评估指标
    original_2d = data.reshape(-1, data.shape[-1])
    reduced_2d = reduced_data.reshape(-1, n_components)
    metrics = compute_evaluation_metrics(original_2d, reduced_2d)

    print("原始数据形状:", data.shape)
    print("降维后的数据形状:", reduced_data.shape)
    print(f"相关系数: {metrics['correlation']:.4f}")
    print(f"可信度: {metrics['trustworthiness']:.4f}")

    # 绘制并保存图像
    plot_spectra(data, reduced_data)