import cupy as cp
import numpy as np
from cuml import DBSCAN, PCA
from cuml.manifold import UMAP
from cuml.metrics import trustworthiness
from src.datesets.datasets_load import load_dataset
import time
import matplotlib.pyplot as plt


def spectral_umap_reduction_gpu(data: object, n_components: object = 10, n_neighbors: object = 15,
                                min_dist: object = 0.1, random_seed: object = 42) -> object:
    """
    使用 GPU 加速的 UMAP 对高光谱图像进行降维

    :param data: 原始数据，形状为 (height, width, n_bands)
    :param n_components: 降维后的维度
    :param n_neighbors: UMAP参数，临近点数量
    :param min_dist: UMAP参数，最小距离
    :param random_seed: 随机种子
    :return: 降维后的数据，形状为 (height, width, n_components)
    """
    height, width, n_bands = data.shape
    data_2d = data.reshape(-1, n_bands)

    print(f"开始 GPU UMAP 降维，数据形状: {data_2d.shape}")
    start_time = time.time()

    # 将数据转移到 GPU
    data_gpu = cp.asarray(data_2d)

    # 初始化 GPU UMAP
    # noinspection PyCallingNonCallable
    reducer = UMAP(n_components=n_components,
                   n_neighbors=n_neighbors,
                   min_dist=min_dist,
                   random_state=random_seed)

    # 执行降维
    reduced_data_gpu = reducer.fit_transform(data_gpu)

    # 将结果转回 CPU
    reduced_data_2d = cp.asnumpy(reduced_data_gpu)

    end_time = time.time()
    print(f"GPU UMAP 降维完成，耗时: {end_time - start_time:.2f} 秒")

    reduced_data = reduced_data_2d.reshape(height, width, n_components)
    return reduced_data


def cluster_and_visualize(reduced_data, eps=0.5, min_samples=5):
    """
    对降维后的数据进行聚类并可视化

    :param reduced_data: 降维后的数据
    :param eps: DBSCAN 的 eps 参数
    :param min_samples: DBSCAN 的 min_samples 参数
    """
    print("开始聚类和可视化...")
    start_time = time.time()

    # 使用 DBSCAN 进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(reduced_data)

    # 使用 PCA 进行降维到 2D 用于可视化
    pca = PCA(n_components=2)
    reduced_2d = pca.fit_transform(reduced_data)

    # 将数据转回 CPU 进行绘图
    reduced_2d_cpu = cp.asnumpy(reduced_2d)
    clusters_cpu = cp.asnumpy(clusters)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_2d_cpu[:, 0], reduced_2d_cpu[:, 1], c=clusters_cpu, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('UMAP + DBSCAN Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    end_time = time.time()
    print(f"聚类和可视化完成，耗时: {end_time - start_time:.2f} 秒")

    # 保存图像
    plt.savefig('umap_dbscan_clustering.png')
    print("图像已保存为 umap_dbscan_clustering.png")
    plt.close()

    # 打印聚类统计信息
    n_clusters = len(np.unique(clusters_cpu))
    n_noise = np.sum(clusters_cpu == -1)
    print(f"聚类数量: {n_clusters}")
    print(f"噪声点数量: {n_noise}")


def compute_evaluation_metrics_gpu(original_data, reduced_data, n_neighbors=5):
    """
    使用 GPU 计算降维结果的评估指标

    :param original_data: 原始数据，形状为 (n_samples, n_features)
    :param reduced_data: 降维后的数据，形状为 (n_samples, n_components)
    :param n_neighbors: 用于计算可信度的邻居数
    :return: 字典，包含可信度
    """
    print("开始计算评估指标...")
    start_time = time.time()

    # 将数据转移到 GPU
    original_gpu = cp.asarray(original_data)
    reduced_gpu = cp.asarray(reduced_data)

    # 计算可信度
    trust = trustworthiness(original_gpu, reduced_gpu, n_neighbors=n_neighbors)

    end_time = time.time()
    print(f"评估指标计算完成，耗时: {end_time - start_time:.2f} 秒")

    return {
        "trustworthiness": trust
    }


if __name__ == "__main__":
    print("开始加载数据...")
    start_time = time.time()
    data, labels, dataset_info = load_dataset('Pavia')
    end_time = time.time()
    print(f"数据加载完成，耗时: {end_time - start_time:.2f} 秒")

    # GPU UMAP降维
    n_components = 80

    reduced_data = spectral_umap_reduction_gpu(data, n_components=n_components, n_neighbors=200, min_dist=0.1)

    # 计算评估指标
    original_2d = data.reshape(-1, data.shape[-1])
    reduced_2d = reduced_data.reshape(-1, n_components)

    print("原始数据形状:", data.shape)
    print("降维后的数据形状:", reduced_data.shape)
    cluster_and_visualize(reduced_2d)
