import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state


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


def spectral_mds_reduction(data, n_components=20, max_iter=300, eps=1e-3, random_state=None):
    """
    对高光谱图像的光谱维度进行MDS降维

    Args:
        data : numpy array, 形状为 (height, width, n_bands)
            高光谱图像数据
        n_components : int, 可选 (默认为 20)
            降维后的光谱波段数
        max_iter : int, 可选 (默认为 300)
            最大迭代次数
        eps : float, 可选 (默认为 1e-3)
            收敛阈值
        random_state : int 或 RandomState 实例, 可选 (默认为 None)
            随机数生成器的种子

    Returns:
        reduced_data : numpy array, 形状为 (height, width, n_components)
            光谱维度降维后的数据
        metrics : dict
            包含各种评估指标的字典，包括：
            - 'final_stress': float, 最终应力值
            - 'stress_values': list, 每次迭代的应力值
            - 'trustworthiness': float, 可信度
            - 'continuity': float, 连续性
    """
    height, width, n_bands = data.shape
    data_2d = data.reshape(-1, n_bands)

    D = euclidean_distances(data_2d)
    random_state = check_random_state(random_state)
    X = random_state.rand(data_2d.shape[0], n_components)

    stress_values = []
    for _ in range(max_iter):
        dist = euclidean_distances(X)
        stress_value = stress(D, dist)
        stress_values.append(stress_value)

        grad = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                grad[i] += (dist[i, j] - D[i, j]) / (dist[i, j] + 1e-7) * (X[i] - X[j])
                grad[j] -= (dist[i, j] - D[i, j]) / (dist[i, j] + 1e-7) * (X[i] - X[j])

        X -= grad / X.shape[0]

        if stress_value < eps:
            break

    reduced_data = X.reshape(height, width, n_components)

    final_dist = euclidean_distances(X)
    final_stress = stress(D, final_dist)
    trust = trustworthiness(D, final_dist, n_neighbors=10)
    cont = continuity(D, final_dist, n_neighbors=10)

    metrics = {
        'final_stress': final_stress,
        'stress_values': stress_values,
        'trustworthiness': trust,
        'continuity': cont
    }

    return reduced_data, metrics


# 使用示例
if __name__ == "__main__":
    # 生成一个模拟的高光谱图像数据
    np.random.seed(42)
    hyperspectral_data = np.random.rand(10, 10, 200)  # 100x100 像素，200 个波段

    # 应用光谱MDS降维
    data = spectral_mds_reduction(hyperspectral_data, n_components=64)

    print("原始数据形状:", hyperspectral_data.shape)
    (reduced_data, metric) = data
    print("降维后的数据形状:", reduced_data.shape)

    # 可视化原始数据和降维后数据的一个像素的光谱
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hyperspectral_data[5, 5, :])
    plt.title("Original Spectrum (200 bands)")
    plt.xlabel("Band")
    plt.ylabel("Intensity")

    plt.subplot(1, 2, 2)
    plt.plot(reduced_data[5, 5, :])
    plt.title("Reduced Spectrum (20 components)")
    plt.xlabel("MDS Component")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()
