import numpy as np


def pca(X, n_components=None):
    """
    执行主成分分析 (PCA)

    参数:
    X : numpy array, 形状为 (n_samples, n_features)
        要降维的数据
    n_components : int, 可选 (默认为 None)
        要保留的主成分数量。如果为 None，则保留所有主成分

    返回:
    X_pca : numpy array, 形状为 (n_samples, n_components)
        降维后的数据
    explained_variance_ratio : numpy array
        每个主成分解释的方差比例
    """

    # 1. 数据中心化
    X_centered = X - np.mean(X, axis=0)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_centered, rowvar=False)

    # 3. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. 对特征值和特征向量进行排序（降序）
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. 选择主成分的数量
    if n_components is None:
        n_components = X.shape[1]

    # 6. 选择前 n_components 个特征向量
    selected_eigenvectors = eigenvectors[:, :n_components]

    # 7. 投影数据
    X_pca = np.dot(X_centered, selected_eigenvectors)

    # 8. 计算解释方差比
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[:n_components] / total_variance

    return X_pca, explained_variance_ratio, selected_eigenvectors


def spectral_pca_reduction(data, n_components=None):
    """
    对高光谱图像的光谱维度进行 PCA 降维

    参数:
    data : numpy array, 形状为 (height, width, n_bands)
        高光谱图像数据
    n_components : int, 可选 (默认为 None)
        要保留的主成分数量。如果为 None，则保留所有主成分

    返回:
    reduced_data : numpy array, 形状为 (height, width, n_components)
        光谱维度降维后的数据
    explained_variance_ratio : numpy array
        每个主成分解释的方差比例
    """
    # 获取原始数据的形状
    height, width, n_bands = data.shape

    # 重塑数据为 2D 数组，每一行代表一个像素的光谱
    data_2d = data.reshape(-1, n_bands)

    # 应用 PCA
    data_pca, explained_var_ratio, eigenvectors = pca(data_2d, n_components=n_components)

    # 将降维后的数据重塑回原始的空间维度
    reduced_data = data_pca.reshape(height, width, -1)

    return reduced_data, explained_var_ratio, eigenvectors


# 使用示例
if __name__ == "__main__":
    # 生成一个模拟的高光谱图像数据
    np.random.seed(0)
    hyperspectral_data = np.random.rand(145, 145, 200)  # 100x100 像素，200 个波段

    # 执行光谱 PCA，保留 20 个主成分
    reduced_data, explained_var_ratio, eigenvectors = spectral_pca_reduction(hyperspectral_data, n_components=20)

    print("原始数据形状:", hyperspectral_data.shape)
    print("PCA 后的数据形状:", reduced_data.shape)
    print("解释方差比:", explained_var_ratio)
    print("总解释方差:", np.sum(explained_var_ratio))

    # 可视化原始数据和降维后数据的一个像素的光谱
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hyperspectral_data[50, 50, :])
    plt.title("Original Spectrum (200 bands)")
    plt.xlabel("Band")
    plt.ylabel("Intensity")

    plt.subplot(1, 2, 2)
    plt.plot(reduced_data[50, 50, :])
    plt.title("Reduced Spectrum (20 principal components)")
    plt.xlabel("Principal Component")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

    # 可视化前几个主成分的特征向量
    plt.figure(figsize=(12, 5))
    for i in range(5):  # 显示前5个主成分
        plt.plot(eigenvectors[:, i], label=f'PC {i + 1}')
    plt.title("First 5 Principal Components")
    plt.xlabel("Original Band")
    plt.ylabel("Loading")
    plt.legend()
    plt.tight_layout()
    plt.show()
