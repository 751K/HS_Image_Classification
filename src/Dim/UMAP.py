import numpy as np
import os
import time
import matplotlib.pyplot as plt
from umap import UMAP
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from joblib import parallel_backend, Parallel, delayed
from tqdm import tqdm


def optimized_umap_reduction(data, n_components=20, n_neighbors=15, min_dist=0.1, random_state=None,
                             max_samples=10000, batch_size=5000, use_cache=True, cache_dir='./cache',
                             n_jobs=-1, metric='euclidean', verbose=1):
    """
    多核并行优化的UMAP降维算法

    参数:
        data: numpy array, 形状为 (height, width, n_bands)
            高光谱图像数据
        n_components: int, 目标维度 (默认 20)
        n_neighbors: int, UMAP临近点数量
        min_dist: float, UMAP最小距离参数
        random_state: int, 随机种子
        max_samples: int, 用于拟合模型的最大样本数
        batch_size: int, 批处理大小
        use_cache: bool, 是否使用缓存
        cache_dir: str, 缓存目录
        n_jobs: int, 使用的CPU核数，-1表示全部
        metric: str, 距离度量方法
        verbose: int, 显示进度信息的级别 (0-2)

    返回:
        reduced_data: numpy array, 形状为 (height, width, n_components)
            降维后的数据
        metrics: dict, 包含评估指标的字典
    """
    # 记录开始时间
    start_time = time.time()

    height, width, n_bands = data.shape
    total_pixels = height * width
    data_2d = data.reshape(-1, n_bands).astype(np.float32)  # 降低内存占用

    # 缓存处理
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        import hashlib
        cache_id = f"{data.shape}_{n_components}_{n_neighbors}_{min_dist}_{random_state}_{n_jobs}_{metric}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"umap_{cache_hash}.npz")

        if os.path.exists(cache_file):
            if verbose > 0:
                print("从缓存加载UMAP结果...")
            cached = np.load(cache_file, allow_pickle=True)
            return cached['reduced_data'], cached['metrics'].item()

    if verbose > 0:
        print(f"开始UMAP降维 (多核并行版本, n_jobs={n_jobs})...")

    # 随机采样训练数据
    if total_pixels > max_samples:
        np.random.seed(random_state if random_state is not None else 42)
        sample_indices = np.random.choice(total_pixels, max_samples, replace=False)
        train_data = data_2d[sample_indices]
    else:
        train_data = data_2d
        sample_indices = np.arange(total_pixels)

    if verbose > 0:
        print(f"使用 {train_data.shape[0]} 个样本训练UMAP模型...")

    # 标准化数据以提高性能
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    # 使用多线程配置创建和训练UMAP模型
    umap_time_start = time.time()
    with parallel_backend('threading', n_jobs=n_jobs):
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            low_memory=False,  # 使用更多内存以提高速度
            n_jobs=n_jobs
        )
        train_embedding = reducer.fit_transform(train_data_scaled)
    umap_time = time.time() - umap_time_start

    # 创建结果数组
    reduced_data_2d = np.zeros((total_pixels, n_components), dtype=np.float32)
    reduced_data_2d[sample_indices] = train_embedding

    # 处理剩余数据
    if total_pixels > max_samples:
        remaining_indices = np.setdiff1d(np.arange(total_pixels), sample_indices)

        if verbose > 0:
            print(f"处理剩余 {len(remaining_indices)} 个像素...")

        # 使用最近邻插值而不是直接转换（UMAP不支持直接transform）
        # 这是一个近似方法，可以显著提高处理大型数据集的性能
        k = min(30, len(sample_indices))
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=n_jobs)
        nn.fit(train_data_scaled)

        # 分批处理剩余数据
        remaining_batches = [remaining_indices[i:i + batch_size] for i in range(0, len(remaining_indices), batch_size)]

        if verbose > 0:
            remaining_batches = tqdm(remaining_batches, desc="UMAP批处理")

        for batch_indices in remaining_batches:
            batch_data = data_2d[batch_indices]
            batch_scaled = scaler.transform(batch_data)

            # 找到k个最近邻
            distances, neighbors = nn.kneighbors(batch_scaled)

            # 基于距离的权重
            weights = 1.0 / np.maximum(distances, 1e-10)
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # 向量化计算加权平均
            for i, idx in enumerate(batch_indices):
                reduced_data_2d[idx] = np.sum(train_embedding[neighbors[i]] * weights[i][:, np.newaxis], axis=0)

    # 重塑为原始图像形状
    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    # 计算评估指标
    metrics_time_start = time.time()

    # 只使用采样数据计算指标以提高效率
    sample_original = train_data_scaled
    sample_reduced = train_embedding

    # 计算相关系数 (使用采样以提高速度)
    max_corr_samples = min(5000, len(sample_original))
    if len(sample_original) > max_corr_samples:
        corr_idx = np.random.choice(len(sample_original), max_corr_samples, replace=False)
        corr_orig = sample_original[corr_idx]
        corr_red = sample_reduced[corr_idx]
    else:
        corr_orig = sample_original
        corr_red = sample_reduced

    with parallel_backend('threading', n_jobs=n_jobs):
        dist_orig = pairwise_distances(corr_orig, metric=metric, n_jobs=n_jobs)
        dist_red = pairwise_distances(corr_red, metric=metric, n_jobs=n_jobs)

    correlation, _ = pearsonr(dist_orig.flatten(), dist_red.flatten())

    # 计算可信度 (trustworthiness)
    n_trust_samples = min(3000, len(sample_original))
    if len(sample_original) > n_trust_samples:
        trust_idx = np.random.choice(len(sample_original), n_trust_samples, replace=False)
        trust_orig = sample_original[trust_idx]
        trust_red = sample_reduced[trust_idx]
    else:
        trust_orig = sample_original
        trust_red = sample_reduced

    trustworthiness = compute_trustworthiness(trust_orig, trust_red,
                                              n_neighbors=min(10, len(trust_red) - 1),
                                              metric=metric, n_jobs=n_jobs)

    metrics_time = time.time() - metrics_time_start
    processing_time = time.time() - start_time

    # 收集指标
    metrics = {
        'correlation': correlation,
        'trustworthiness': trustworthiness,
        'processing_time': processing_time,
        'umap_time': umap_time,
        'metrics_time': metrics_time,
        'n_samples_used': len(train_data)
    }

    # 缓存结果
    if use_cache:
        np.savez(cache_file, reduced_data=reduced_data, metrics=metrics)

    if verbose > 0:
        print(f"UMAP降维完成，处理时间: {processing_time:.2f}秒")
        print(f"相关系数: {correlation:.4f}, 可信度: {trustworthiness:.4f}")

    return reduced_data, metrics


def compute_trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean', n_jobs=1):
    """优化的可信度计算函数"""
    with parallel_backend('threading', n_jobs=n_jobs):
        dist_X = pairwise_distances(X, metric=metric, n_jobs=n_jobs)
        dist_X_embedded = pairwise_distances(X_embedded, metric=metric, n_jobs=n_jobs)

    n_samples = X.shape[0]
    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = np.argsort(dist_X_embedded, axis=1)

    n_neighbors = min(n_neighbors, n_samples - 1)
    t = 0

    # 向量化计算以提高速度
    for i in range(n_samples):
        ranks = np.where(np.in1d(ind_X[i], ind_X_embedded[i, 1:n_neighbors + 1]))[0]
        t += np.sum(np.maximum(0, ranks - n_neighbors))

    t = 1 - t * (2 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)))
    return t


def analyze_umap_results(data, reduced_data, metrics, pixel_pos=(50, 50), output_dir=None):
    """
    分析UMAP降维结果并可视化

    参数:
        data: 原始高光谱数据
        reduced_data: UMAP降维后的数据
        metrics: 包含评估指标的字典
        pixel_pos: 用于显示光谱的像素位置 (行, 列)
        output_dir: 输出目录，如果指定则保存图像
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. 可视化原始数据和降维后数据的光谱
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data[pixel_pos[0], pixel_pos[1], :])
    plt.title(f"原始光谱 ({data.shape[2]} 波段)")
    plt.xlabel("波段")
    plt.ylabel("强度")

    plt.subplot(1, 2, 2)
    plt.plot(reduced_data[pixel_pos[0], pixel_pos[1], :])
    plt.title(f"降维后特征 ({reduced_data.shape[2]} 维)")
    plt.xlabel("UMAP组件")
    plt.ylabel("值")

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'umap_spectrum_comparison.png'))
    plt.tight_layout()
    plt.show()

    # 2. 打印性能指标
    print(f"UMAP降维性能指标:")
    print(f"相关系数: {metrics['correlation']:.4f}")
    print(f"可信度: {metrics['trustworthiness']:.4f}")
    print(f"总处理时间: {metrics['processing_time']:.2f}秒")
    print(f"UMAP计算时间: {metrics['umap_time']:.2f}秒")
    print(f"使用样本数量: {metrics['n_samples_used']}")


# 使用示例
if __name__ == "__main__":
    from src.datesets.datasets_load import load_dataset

    # 加载数据集
    data, labels, dataset_info = load_dataset('Pavia')

    # 应用优化的UMAP降维
    n_components = 32
    reduced_data, metrics = optimized_umap_reduction(
        data,
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        max_samples=8000,
        batch_size=2000,
        use_cache=True
    )

    print("原始数据形状:", data.shape)
    print("降维后的数据形状:", reduced_data.shape)

    # 分析结果
    analyze_umap_results(data, reduced_data, metrics)
