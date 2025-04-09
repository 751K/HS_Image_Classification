from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state
from joblib import parallel_backend
import numpy as np
import os
import time
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def stress(D, d):
    """
    计算应力值（Stress）

    Args:
        D : numpy array
            原始高维空间中的距离矩阵
        d : numpy array
            降维后低维空间中的距离矩阵

    Returns:
        float : 应力值
    """
    return np.sqrt(np.sum((D - d) ** 2) / np.sum(D ** 2))


def trustworthiness(D, d, n_neighbors):
    """
    计算可信度（Trustworthiness）

    Args:
        D : numpy array
            原始高维空间中的距离矩阵
        d : numpy array
            降维后低维空间中的距离矩阵
        n_neighbors : int
            考虑的邻居数量

    Returns:
        float : 可信度值
    """
    n = D.shape[0]
    t = np.zeros(n)
    for i in range(n):
        rank_D = np.argsort(D[i])
        rank_d = np.argsort(d[i])
        for j in range(n_neighbors):
            if rank_d[j + 1] not in rank_D[:n_neighbors + 1]:
                t[i] += (rank_D[rank_d[j + 1]] - n_neighbors)
    return 1 - t.sum() * 2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))


def continuity(D, d, n_neighbors):
    """
    计算连续性（Continuity）

    Args:
        D : numpy array
            原始高维空间中的距离矩阵
        d : numpy array
            降维后低维空间中的距离矩阵
        n_neighbors : int
            考虑的邻居数量

    Returns:
        float : 连续性值
    """
    n = D.shape[0]
    c = np.zeros(n)
    for i in range(n):
        rank_D = np.argsort(D[i])
        rank_d = np.argsort(d[i])
        for j in range(n_neighbors):
            if rank_D[j + 1] not in rank_d[:n_neighbors + 1]:
                c[i] += (rank_d[rank_D[j + 1]] - n_neighbors)
    return 1 - c.sum() * 2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))


# 向量化计算梯度的函数
def compute_gradient_batch(X, D):
    dist = euclidean_distances(X)
    np.fill_diagonal(dist, 1)  # 避免除零错误

    # 向量化计算梯度
    grad = np.zeros_like(X)
    n = X.shape[0]

    # 预计算所有点对差异
    diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]

    # 计算权重因子 (dist_ij - D_ij) / dist_ij
    weights = (dist - D) / (dist + np.finfo(float).eps)
    np.fill_diagonal(weights, 0)  # 对角线设为0

    # 权重乘以差异的加权累加
    for i in range(n):
        grad[i] = np.sum(weights[i, :, np.newaxis] * diffs[i, :, :], axis=0)

    return grad, dist


def optimized_mds_reduction(data, n_components=20, max_iter=300, eps=1e-3, random_state=None,
                            max_samples=10000, batch_size=5000, use_cache=True, cache_dir='./cache',
                            n_jobs=-1, verbose=1):
    """
    多核并行优化版MDS降维算法

    Args:
        data: numpy array, 形状为 (height, width, n_bands)
        n_components: int, 目标维度 (默认 20)
        max_iter: int, 最大迭代次数
        eps: float, 收敛阈值
        random_state: int, 随机种子
        max_samples: int, 用于拟合模型的最大样本数
        batch_size: int, 批处理大小
        use_cache: bool, 是否使用缓存
        cache_dir: str, 缓存目录
        n_jobs: int, 使用的CPU核数，-1表示全部 (默认 -1)
        verbose: int, 显示进度信息的级别 (0-2)

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
        cache_id = f"{data.shape}_{n_components}_{max_iter}_{eps}_{random_state}_{n_jobs}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"mds_parallel_{cache_hash}.npz")

        if os.path.exists(cache_file):
            if verbose > 0:
                print("从缓存加载MDS结果...")
            cached = np.load(cache_file, allow_pickle=True)
            return cached['reduced_data'], cached['metrics'].item()

    if verbose > 0:
        print(f"开始MDS降维 (多核并行版本, n_jobs={n_jobs})...")

    # 随机采样训练数据
    if total_pixels > max_samples:
        np.random.seed(random_state if random_state is not None else 42)
        sample_indices = np.random.choice(total_pixels, max_samples, replace=False)
        train_data = data_2d[sample_indices]
    else:
        train_data = data_2d
        sample_indices = np.arange(total_pixels)

    if verbose > 0:
        print(f"使用 {train_data.shape[0]} 个样本训练MDS模型...")

    # 标准化数据
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    # 计算训练数据的距离矩阵 (使用并行计算)
    with parallel_backend('threading', n_jobs=n_jobs):
        D_train = euclidean_distances(train_data_scaled)

    # 使用MDS算法对训练数据进行降维
    random_state_obj = check_random_state(random_state)
    X_train = random_state_obj.rand(train_data.shape[0], n_components)

    # 迭代优化训练数据的低维表示
    stress_values = []

    iterator = range(max_iter)
    if verbose > 0:
        iterator = tqdm(iterator, desc="MDS迭代")

    for iteration in iterator:
        # 计算新的梯度和距离
        grad, dist = compute_gradient_batch(X_train, D_train)

        # 计算应力值
        stress_value = stress(D_train, dist)
        stress_values.append(stress_value)

        # 更新坐标
        step_size = 0.2 / (1 + iteration * 0.3)  # 学习率衰减
        X_train -= step_size * grad

        # 收敛检查
        if stress_value < eps:
            break

    # 对全部数据进行处理
    reduced_data_2d = np.zeros((total_pixels, n_components), dtype=np.float32)
    reduced_data_2d[sample_indices] = X_train

    # 处理剩余数据
    if total_pixels > max_samples:
        remaining_indices = np.setdiff1d(np.arange(total_pixels), sample_indices)

        if verbose > 0:
            print(f"处理剩余 {len(remaining_indices)} 个点...")

        # 使用NearestNeighbors加速k近邻搜索
        k = min(5, len(sample_indices))
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=n_jobs)
        nn.fit(train_data_scaled)

        # 分批处理剩余数据
        for i in range(0, len(remaining_indices), batch_size):
            batch_indices = remaining_indices[i:i + batch_size]
            batch_data = data_2d[batch_indices]
            batch_scaled = scaler.transform(batch_data)

            # 使用NearestNeighbors快速找到k个最近邻
            distances, neighbors = nn.kneighbors(batch_scaled)

            # 计算基于距离的权重
            weights = 1.0 / np.maximum(distances, 1e-10)
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # 向量化计算加权平均 - 大幅提高速度
            X_batch = np.zeros((len(batch_indices), n_components), dtype=np.float32)
            for b in range(len(batch_indices)):
                X_batch[b] = np.sum(X_train[neighbors[b]] * weights[b][:, np.newaxis], axis=0)

            reduced_data_2d[batch_indices] = X_batch

    # 重塑为原始图像形状
    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    # 计算评估指标
    final_dist = euclidean_distances(X_train, n_jobs=n_jobs)
    final_stress = stress(D_train, final_dist)
    trust = trustworthiness(D_train, final_dist, n_neighbors=min(10, X_train.shape[0] - 1))
    cont = continuity(D_train, final_dist, n_neighbors=min(10, X_train.shape[0] - 1))

    processing_time = time.time() - start_time

    metrics = {
        'final_stress': final_stress,
        'stress_values': stress_values,
        'trustworthiness': trust,
        'continuity': cont,
        'processing_time': processing_time,
        'n_iterations': len(stress_values)
    }

    # 缓存结果
    if use_cache:
        np.savez(cache_file, reduced_data=reduced_data, metrics=metrics)

    if verbose > 0:
        print(f"MDS降维完成，处理时间: {processing_time:.2f}秒")

    return reduced_data, metrics


def analyze_mds_results(metrics, output_dir=None):
    """
    分析MDS降维结果并可视化

    Args:
        metrics: 包含MDS评估指标的字典
        output_dir: 输出目录，如果指定则保存图像
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 绘制应力值变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics['stress_values']) + 1), metrics['stress_values'], marker='o', markersize=4)
    plt.title('MDS应力值收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('应力值')
    plt.grid(True)

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'mds_stress_convergence.png'))
    else:
        plt.show()

    # 打印评估指标
    print(f"最终应力值: {metrics['final_stress']:.4f}")
    print(f"可信度(Trustworthiness): {metrics['trustworthiness']:.4f}")
    print(f"连续性(Continuity): {metrics['continuity']:.4f}")
    print(f"总处理时间: {metrics['processing_time']:.2f}秒")
    print(f"迭代次数: {metrics['n_iterations']}")


def torch_stress(D, d):
    """使用PyTorch计算应力值"""
    return torch.sqrt(torch.sum((D - d) ** 2) / torch.sum(D ** 2)).item()


def torch_pdist(X):
    """计算成对欧氏距离矩阵"""
    X_norm = (X ** 2).sum(1).view(-1, 1)
    dist = X_norm + X_norm.view(1, -1) - 2.0 * torch.mm(X, X.t())
    # 修正数值不稳定性
    dist = torch.clamp(dist, 0.0)
    return torch.sqrt(dist)


def compute_gradient_torch(X, D):
    """使用PyTorch计算梯度，完全向量化"""
    device = X.device
    n = X.size(0)

    # 计算低维空间距离矩阵
    dist = torch_pdist(X)
    # 避免除零
    dist = torch.clamp(dist, min=1e-7)

    # 计算差异矩阵
    diff = dist - D

    # 计算梯度(向量化实现)
    grad = torch.zeros_like(X)
    for i in range(n):
        for j in range(n):
            if i != j:
                grad[i] += diff[i, j] / dist[i, j] * (X[i] - X[j])

    return grad, dist


def torch_mds_reduction(data, n_components=20, max_iter=300, eps=1e-3, random_state=None,
                        max_samples=10000, batch_size=5000, use_cache=True, cache_dir='./cache',
                        use_gpu=True, verbose=1):
    # TODO:fix
    """
    使用PyTorch加速的MDS降维算法

    Args:
        data: numpy array, 形状为 (height, width, n_bands)
        n_components: int, 目标维度 (默认 20)
        max_iter: int, 最大迭代次数
        eps: float, 收敛阈值
        random_state: int, 随机种子
        max_samples: int, 用于拟合模型的最大样本数
        batch_size: int, 批处理大小
        use_cache: bool, 是否使用缓存
        cache_dir: str, 缓存目录
        use_gpu: bool, 是否使用GPU (如果可用)
        verbose: int, 显示进度信息的级别 (0-2)

    Returns:
        reduced_data: numpy array, 形状为 (height, width, n_components)
        metrics: dict, 包含评估指标的字典
    """
    # 检测设备
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose > 0:
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose > 0 and use_gpu:
            print("GPU不可用，使用CPU")

    # 记录开始时间
    start_time = time.time()

    height, width, n_bands = data.shape
    total_pixels = height * width
    data_2d = data.reshape(-1, n_bands).astype(np.float32)

    # 缓存处理
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        import hashlib
        cache_id = f"{data.shape}_{n_components}_{max_iter}_{eps}_{random_state}_torch_{'gpu' if use_gpu and torch.cuda.is_available() else 'cpu'}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"mds_torch_{cache_hash}.npz")

        if os.path.exists(cache_file):
            if verbose > 0:
                print("从缓存加载MDS结果...")
            cached = np.load(cache_file, allow_pickle=True)
            return cached['reduced_data'], cached['metrics'].item()

    if verbose > 0:
        print(f"开始MDS降维 (PyTorch{'GPU' if device.type == 'cuda' else 'CPU'}版本)...")

    # 随机采样训练数据
    if total_pixels > max_samples:
        np.random.seed(random_state if random_state is not None else 42)
        sample_indices = np.random.choice(total_pixels, max_samples, replace=False)
        train_data = data_2d[sample_indices]
    else:
        train_data = data_2d
        sample_indices = np.arange(total_pixels)

    if verbose > 0:
        print(f"使用 {train_data.shape[0]} 个样本训练MDS模型...")

    # 标准化数据
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    # 转换为PyTorch张量并移至设备
    train_data_torch = torch.tensor(train_data_scaled, dtype=torch.float32, device=device)

    # 计算高维空间距离矩阵
    D_train = torch_pdist(train_data_torch)

    # 初始化低维嵌入
    torch.manual_seed(random_state if random_state is not None else 42)
    X_train = torch.rand((train_data.shape[0], n_components), dtype=torch.float32, device=device)

    # 迭代优化
    stress_values = []

    iterator = range(max_iter)
    if verbose > 0:
        iterator = tqdm(iterator, desc="MDS迭代")

    for iteration in iterator:
        # 计算梯度和距离
        grad, dist = compute_gradient_torch(X_train, D_train)

        # 计算应力值
        stress_value = torch_stress(D_train, dist)
        stress_values.append(stress_value)

        # 更新坐标
        step_size = 0.2 / (1 + iteration * 0.3)  # 学习率衰减
        X_train -= step_size * grad

        # 收敛检查
        if stress_value < eps:
            break

    # 将结果转回CPU和NumPy格式
    X_train_np = X_train.cpu().numpy()

    # 对全部数据进行处理
    reduced_data_2d = np.zeros((total_pixels, n_components), dtype=np.float32)
    reduced_data_2d[sample_indices] = X_train_np

    # 处理剩余数据
    if total_pixels > max_samples:
        remaining_indices = np.setdiff1d(np.arange(total_pixels), sample_indices)

        if verbose > 0:
            print(f"处理剩余 {len(remaining_indices)} 个点...")

        # 使用最近邻插值
        k = min(5, len(sample_indices))
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
        nn.fit(train_data_scaled)

        # 分批处理剩余数据
        for i in range(0, len(remaining_indices), batch_size):
            batch_indices = remaining_indices[i:i + batch_size]
            batch_data = data_2d[batch_indices]
            batch_scaled = scaler.transform(batch_data)

            # 找到k个最近邻
            distances, neighbors = nn.kneighbors(batch_scaled)

            # 基于距离的权重
            weights = 1.0 / np.maximum(distances, 1e-10)
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # 向量化计算加权平均
            X_batch = np.zeros((len(batch_indices), n_components), dtype=np.float32)
            for b in range(len(batch_indices)):
                X_batch[b] = np.sum(X_train_np[neighbors[b]] * weights[b][:, np.newaxis], axis=0)

            reduced_data_2d[batch_indices] = X_batch

    # 重塑为原始图像形状
    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    # 计算评估指标 (在CPU上)
    D_train_np = D_train.cpu().numpy()
    final_dist = torch_pdist(torch.tensor(X_train_np)).cpu().numpy()

    # 使用原始函数计算评估指标
    final_stress = stress(D_train_np, final_dist)
    trust = trustworthiness(D_train_np, final_dist, n_neighbors=min(10, X_train_np.shape[0] - 1))
    cont = continuity(D_train_np, final_dist, n_neighbors=min(10, X_train_np.shape[0] - 1))

    processing_time = time.time() - start_time

    metrics = {
        'final_stress': final_stress,
        'stress_values': stress_values,
        'trustworthiness': trust,
        'continuity': cont,
        'processing_time': processing_time,
        'n_iterations': len(stress_values),
        'device': device.type
    }

    # 缓存结果
    if use_cache:
        np.savez(cache_file, reduced_data=reduced_data, metrics=metrics)

    if verbose > 0:
        print(f"MDS降维完成，处理时间: {processing_time:.2f}秒")

    return reduced_data, metrics


# 使用示例
if __name__ == "__main__":
    from src.datesets.datasets_load import load_dataset

    # 加载数据集
    data, labels, dataset_info = load_dataset('Pavia')

    # 应用优化的MDS降维
    n_components = 32
    reduced_data, metrics = optimized_mds_reduction(
        data,
        n_components=n_components,
        max_iter=100,
        max_samples=5000,
        batch_size=2000,
        use_cache=True
    )
    # reduced_data, metrics = torch_mds_reduction(
    #     data,
    #     n_components=n_components,
    #     max_iter=100,
    #     max_samples=5000,
    #     batch_size=2000,
    #     use_cache=True
    # )

    print("原始数据形状:", data.shape)
    print("降维后的数据形状:", reduced_data.shape)

    # 分析结果
    analyze_mds_results(metrics)

    # 可视化原始数据和降维后数据的光谱
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data[50, 50, :])
    plt.title(f"原始光谱 ({data.shape[2]} 波段)")
    plt.xlabel("波段")
    plt.ylabel("强度")

    plt.subplot(1, 2, 2)
    plt.plot(reduced_data[50, 50, :])
    plt.title(f"降维后光谱 ({n_components} 维)")
    plt.xlabel("MDS组件")
    plt.ylabel("值")

    plt.tight_layout()
    plt.show()
