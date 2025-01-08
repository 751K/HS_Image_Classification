import numpy as np
from umap import UMAP
from src.datesets.datasets_load import load_dataset
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def spectral_umap_reduction(data, n_components=10, n_neighbors=15, min_dist=0.1, random_seed=42):
    """
    对高光谱图像的光谱维度进行 UMAP 降维

    Args:
        data : numpy array, 形状为 (height, width, n_bands)
        n_components : int, 可选 (默认为 10)
            降维后的光谱波段数
        n_neighbors : int, 可选 (默认为 15)
            用于构建高维图的邻居数
        min_dist : float, 可选 (默认为 0.1)
            控制嵌入点在低维空间中的最小距离
        random_seed : int, 可选 (默认为 42)
            随机种子

    Return:
        tuple: (reduced_data, reconstruction_error, correlation)
            reduced_data : numpy array, 形状为 (height, width, n_components)
                光谱维度降维后的数据
            reconstruction_error : float
                重构误差
            correlation : float
                原始数据和重构数据之间的相关系数
    """
    # 获取原始数据的形状
    height, width, n_bands = data.shape

    # 重塑数据为 2D 数组，每一行代表一个像素的光谱
    data_2d = data.reshape(-1, n_bands)

    # 初始化 UMAP
    reducer = UMAP(n_components=n_components,
                   n_neighbors=n_neighbors,
                   min_dist=min_dist,
                   random_state=random_seed)

    # 应用 UMAP 降维
    reduced_data_2d = reducer.fit_transform(data_2d)

    # 计算重构误差
    reconstructed_data_2d = reducer.inverse_transform(reduced_data_2d)
    reconstruction_error = mean_squared_error(data_2d, reconstructed_data_2d)

    # 计算相关系数
    correlation, _ = pearsonr(data_2d.flatten(), reconstructed_data_2d.flatten())

    # 将降维后的数据重塑回原始的空间维度
    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    return reduced_data, (reconstruction_error, correlation)


# 使用示例
if __name__ == "__main__":
    # 生成一个模拟的高光谱图像数据
    np.random.seed(42)
    data, labels, dataset_info = load_dataset('Pavia')

    # 应用光谱降维
    reduced_data, x = spectral_umap_reduction(data, n_components=80)
    reconstruction_error, correlation = x

    print("原始数据形状:", data.shape)
    print("降维后的数据形状:", reduced_data.shape)
    print(f"重构误差: {reconstruction_error:.4f}")
    print(f"相关系数: {correlation:.4f}")

    # 可视化原始数据和降维后数据的一个像素的光谱
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data[50, 50, :])
    plt.title("Original Spectrum")
    plt.xlabel("Band")
    plt.ylabel("Intensity")

    plt.subplot(1, 2, 2)
    plt.plot(reduced_data[50, 50, :])
    plt.title("Reduced Spectrum")
    plt.xlabel("UMAP Component")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()
