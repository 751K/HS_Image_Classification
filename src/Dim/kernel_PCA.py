import numpy as np
import os
import time
from sklearn.decomposition import KernelPCA
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def optimized_kernel_pca_reduction(data, n_components=20, kernel='rbf', gamma=None,
                                   random_state=None, max_samples=10000, batch_size=5000,
                                   use_cache=True, cache_dir='./cache'):
    """
    优化高光谱图像的核PCA降维性能

    Args:
        data: numpy array, 形状为 (height, width, n_bands)
        n_components: int, 目标维度 (默认 20)
        kernel: string, 核函数 (默认 'rbf')
        gamma: float, 核参数 (默认 None)
        random_state: int, 随机种子
        max_samples: int, 用于拟合模型的最大样本数
        batch_size: int, 批处理大小
        use_cache: bool, 是否使用缓存
        cache_dir: str, 缓存目录

    Returns:
        reduced_data: numpy array, 形状为 (height, width, n_components)
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
        cache_id = f"{data.shape}_{n_components}_{kernel}_{gamma}_{random_state}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"kpca_{cache_hash}.npz")

        if os.path.exists(cache_file):
            cached = np.load(cache_file, allow_pickle=True)
            return cached['reduced_data'], cached['metrics'].item()

    # 随机采样训练数据
    if total_pixels > max_samples:
        np.random.seed(random_state if random_state is not None else 42)
        sample_indices = np.random.choice(total_pixels, max_samples, replace=False)
        train_data = data_2d[sample_indices]
    else:
        train_data = data_2d

    # 标准化数据
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    # 使用Nystroem特征近似
    feature_map = Nystroem(
        kernel=kernel,
        n_components=min(n_components * 2, train_data.shape[0]),
        gamma=gamma,
        random_state=random_state,
        n_jobs=-1
    )
    train_data_approx = feature_map.fit_transform(train_data_scaled)

    # 使用KernelPCA进行降维
    pca = KernelPCA(n_components=n_components, kernel='linear',
                    random_state=random_state, n_jobs=-1)
    _ = pca.fit(train_data_approx)  # 只对训练数据拟合

    # 分批处理全部数据
    reduced_data_2d = np.zeros((total_pixels, n_components), dtype=np.float32)

    n_batches = (total_pixels - 1) // batch_size + 1
    for i in range(0, total_pixels, batch_size):
        batch_end = min(i + batch_size, total_pixels)

        # 处理当前批次
        batch = data_2d[i:batch_end]
        batch_scaled = scaler.transform(batch)
        batch_approx = feature_map.transform(batch_scaled)
        reduced_batch = pca.transform(batch_approx)

        reduced_data_2d[i:batch_end] = reduced_batch

    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    # 计算指标
    # 计算指标
    explained_variance_ratio = pca.eigenvalues_ / np.sum(pca.eigenvalues_)
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    processing_time = time.time() - start_time

    metrics = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_explained_variance_ratio': cumulative_explained_variance_ratio,
        'processing_time': processing_time
    }

    # 缓存结果
    if use_cache:
        np.savez(cache_file, reduced_data=reduced_data, metrics=metrics)

    return reduced_data, metrics


def analyze_variance_ratio(metrics, n_components=None, threshold=0.95):
    """
    分析核PCA降维的累积解释方差比

    Args:
        metrics: 包含'cumulative_explained_variance_ratio'的指标字典
        n_components: 显示的最大组件数，None表示全部
        threshold: 目标解释方差比阈值，默认0.95
    """
    cum_var_ratio = metrics['cumulative_explained_variance_ratio']

    if n_components is None:
        n_components = len(cum_var_ratio)
    else:
        n_components = min(n_components, len(cum_var_ratio))

    # 找出达到阈值所需的组件数
    components_needed = np.argmax(cum_var_ratio >= threshold) + 1

    # 绘制累积解释方差比曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), cum_var_ratio[:n_components], marker='o', markersize=4)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold * 100}% Threshold')
    plt.axvline(x=components_needed, color='g', linestyle='--',
                label=f'Components Needed: {components_needed} for {threshold * 100}%')

    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.legend()

    # 打印关键数据点
    print(f"前10个组件的累积解释方差比: {cum_var_ratio[9]:.4f}")
    print(f"达到{threshold * 100}%解释方差需要的组件数: {components_needed}")
    print(f"最终累积解释方差比(全部{len(cum_var_ratio)}个组件): {cum_var_ratio[-1]:.4f}")

    plt.tight_layout()
    plt.show()

    # 绘制每个主成分的单独解释方差比
    variance_ratio = metrics['explained_variance_ratio'][:n_components]
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components + 1), variance_ratio)
    plt.title('Individual Variance Ratio per Component')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    from src.datesets.datasets_load import load_dataset

    # 加载数据集
    data, labels, dataset_info = load_dataset('Wuhan')

    # 应用优化的核PCA光谱降维
    n_components = 32
    reduced_data, metrics = optimized_kernel_pca_reduction(
        data,
        n_components=n_components,
        kernel='rbf',
        max_samples=8000,  # 减少训练样本数
        batch_size=5000,  # 调整批处理大小
        use_cache=True  # 启用缓存加速重复计算
    )

    print("原始数据形状:", data.shape)
    print("降维后的数据形状:", reduced_data.shape)
    print(f"累积解释方差比: {metrics['cumulative_explained_variance_ratio'][-1]:.4f}")
    print(f"总处理时间: {metrics['processing_time']:.2f}秒")

    analyze_variance_ratio(metrics, threshold=0.95)
