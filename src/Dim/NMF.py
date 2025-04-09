import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from joblib import parallel_backend, Parallel, delayed
from tqdm import tqdm


def optimized_nmf_reduction(data, n_components=32, max_iter=200, tol=1e-4, random_state=None,
                            max_samples=10000, batch_size=5000, use_cache=True, cache_dir='./cache',
                            n_jobs=-1, verbose=1):
    """
    优化的非负矩阵分解 (NMF) 算法实现，支持多核并行、批处理和缓存

    参数:
        data: numpy array, 形状为 (height, width, n_bands)
            高光谱图像数据
        n_components: int, 目标维度 (默认 32)
        max_iter: int, 最大迭代次数
        tol: float, 收敛容差
        random_state: int, 随机种子
        max_samples: int, 用于拟合模型的最大样本数
        batch_size: int, 批处理大小
        use_cache: bool, 是否使用缓存
        cache_dir: str, 缓存目录
        n_jobs: int, 使用的CPU核数，-1表示全部
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

    # 确保数据非负
    data_2d = np.maximum(data_2d, 0)

    # 缓存处理
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        import hashlib
        cache_id = f"{data.shape}_{n_components}_{max_iter}_{tol}_{random_state}_{n_jobs}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"nmf_{cache_hash}.npz")

        if os.path.exists(cache_file):
            if verbose > 0:
                print("从缓存加载NMF结果...")
            cached = np.load(cache_file, allow_pickle=True)
            return cached['reduced_data'], cached['metrics'].item()

    if verbose > 0:
        print(f"开始NMF降维 (多核并行版本, n_jobs={n_jobs})...")

    # 随机采样训练数据
    if total_pixels > max_samples:
        np.random.seed(random_state if random_state is not None else 42)
        sample_indices = np.random.choice(total_pixels, max_samples, replace=False)
        train_data = data_2d[sample_indices]
    else:
        train_data = data_2d
        sample_indices = np.arange(total_pixels)

    if verbose > 0:
        print(f"使用 {train_data.shape[0]} 个样本训练NMF模型...")

    # 使用并行配置初始化NMF模型
    with parallel_backend('threading', n_jobs=n_jobs):
        nmf_model = NMF(
            n_components=n_components,
            init='random',
            solver='cd',  # 坐标下降法通常更快
            beta_loss='frobenius',
            max_iter=max_iter,
            tol=tol,
            random_state=random_state
        )

        # 拟合模型
        reduced_train = nmf_model.fit_transform(train_data)
        components = nmf_model.components_

        # 收集收敛信息
        reconstruction_err = nmf_model.reconstruction_err_
        n_iter = nmf_model.n_iter_

    # 创建结果数组
    reduced_data_2d = np.zeros((total_pixels, n_components), dtype=np.float32)
    reduced_data_2d[sample_indices] = reduced_train

    # 处理剩余数据（分批变换）
    if total_pixels > max_samples:
        remaining_indices = np.setdiff1d(np.arange(total_pixels), sample_indices)

        if verbose > 0:
            print(f"处理剩余 {len(remaining_indices)} 个像素...")

        # 分批处理剩余数据
        batches = np.array_split(remaining_indices,
                                 max(1, len(remaining_indices) // batch_size))

        if verbose > 0:
            batches = tqdm(batches, desc="NMF批处理")

        for batch_indices in batches:
            batch_data = data_2d[batch_indices]
            batch_reduced = nmf_model.transform(batch_data)
            reduced_data_2d[batch_indices] = batch_reduced

    # 重塑为原始图像形状
    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    # 计算重建误差
    if verbose > 0 and len(sample_indices) <= 10000:
        sample_recon = np.dot(reduced_data_2d[sample_indices], components)
        recon_err = np.mean((train_data - sample_recon) ** 2)
    else:
        recon_err = reconstruction_err

    processing_time = time.time() - start_time

    # 收集指标
    metrics = {
        'reconstruction_error': recon_err,
        'n_iterations': n_iter,
        'processing_time': processing_time,
        'components': components
    }

    # 缓存结果
    if use_cache:
        np.savez(cache_file, reduced_data=reduced_data, metrics=metrics)

    if verbose > 0:
        print(f"NMF降维完成，处理时间: {processing_time:.2f}秒")
        print(f"迭代次数: {n_iter}, 重建误差: {recon_err:.6f}")

    return reduced_data, metrics


def analyze_nmf_results(data, reduced_data, metrics, n_components_to_show=5, pixel_pos=(50, 50), output_dir=None):
    """
    分析NMF降维结果并可视化

    参数:
        data: 原始高光谱数据
        reduced_data: NMF降维后的数据
        metrics: 包含NMF评估指标的字典
        n_components_to_show: 要显示的基矩阵组件数量
        pixel_pos: 用于显示光谱的像素位置 (行, 列)
        output_dir: 输出目录，如果指定则保存图像
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 获取基矩阵
    components = metrics['components']

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
    plt.xlabel("NMF组件")
    plt.ylabel("权重")

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'nmf_spectrum_comparison.png'))
    plt.tight_layout()
    plt.show()

    # 2. 可视化前n个基矩阵组件
    plt.figure(figsize=(12, 6))
    n_show = min(n_components_to_show, components.shape[0])
    for i in range(n_show):
        plt.plot(components[i], label=f'组件 {i + 1}')

    plt.title(f"NMF基矩阵前{n_show}个组件")
    plt.xlabel("原始波段")
    plt.ylabel("权重")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'nmf_components.png'))
    plt.tight_layout()
    plt.show()

    # 3. 打印性能指标
    print(f"NMF降维性能指标:")
    print(f"重建误差: {metrics['reconstruction_error']:.6f}")
    print(f"迭代次数: {metrics['n_iterations']}")
    print(f"处理时间: {metrics['processing_time']:.2f}秒")


if __name__ == "__main__":
    from src.datesets.datasets_load import load_dataset

    # 加载数据集
    data, labels, dataset_info = load_dataset('Indian')

    # 应用优化的NMF降维
    n_components = 32
    reduced_data, metrics = optimized_nmf_reduction(
        data,
        n_components=n_components,
        max_iter=200,
        max_samples=8000,
        batch_size=2000,
        use_cache=True
    )

    print("原始数据形状:", data.shape)
    print("降维后的数据形状:", reduced_data.shape)

    # 分析结果
    analyze_nmf_results(data, reduced_data, metrics)
