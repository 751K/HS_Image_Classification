from sklearn.decomposition import KernelPCA
import numpy as np


def spectral_kernel_pca_reduction(data, n_components=20, kernel='rbf', gamma=None, random_state=None):
    """
    对高光谱图像的光谱维度进行核PCA降维

    Args:
        data : numpy array, 形状为 (height, width, n_bands)
            高光谱图像数据
        n_components : int, 可选 (默认为 20)
            降维后的光谱波段数
        kernel : string, 可选 (默认为 'rbf')
            核函数类型。可选 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'
        gamma : float, 可选 (默认为 None)
            核系数。如果为None，则使用1/n_features
        random_state : int 或 RandomState 实例, 可选 (默认为 None)
            随机数生成器的种子

    Returns:
        reduced_data : numpy array, 形状为 (height, width, n_components)
            光谱维度降维后的数据
        metrics : dict
            包含评估指标的字典
            - 'explained_variance_ratio': 解释方差比
            - 'cumulative_explained_variance_ratio': 累积解释方差比
    """
    height, width, n_bands = data.shape
    data_2d = data.reshape(-1, n_bands)

    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, random_state=random_state,
                     fit_inverse_transform=True)
    reduced_data_2d = kpca.fit_transform(data_2d)

    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    # 计算解释方差比
    X_reconstructed = kpca.inverse_transform(reduced_data_2d)
    total_variance = np.var(data_2d, axis=0).sum()
    explained_variance = np.var(X_reconstructed, axis=0).sum()
    explained_variance_ratio = explained_variance / total_variance

    # 计算累积解释方差比
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

    metrics = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_explained_variance_ratio': cumulative_explained_variance_ratio
    }

    return reduced_data, metrics
