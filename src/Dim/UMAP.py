import numpy as np
from umap import UMAP


def spectral_umap_reduction(data, n_components=10, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    对高光谱图像的光谱维度进行 UMAP 降维

    参数:
    data : numpy array, 形状为 (height, width, n_bands)
        高光谱图像数据
    n_components : int, 可选 (默认为 10)
        降维后的光谱波段数
    n_neighbors : int, 可选 (默认为 15)
        用于构建高维图的邻居数
    min_dist : float, 可选 (默认为 0.1)
        控制嵌入点在低维空间中的最小距离
    random_state : int, 可选 (默认为 42)
        随机种子，用于结果的复现

    返回:
    reduced_data : numpy array, 形状为 (height, width, n_components)
        光谱维度降维后的数据
    """
    # 获取原始数据的形状
    height, width, n_bands = data.shape

    # 重塑数据为 2D 数组，每一行代表一个像素的光谱
    data_2d = data.reshape(-1, n_bands)

    # 初始化 UMAP
    reducer = UMAP(n_components=n_components,
                   n_neighbors=n_neighbors,
                   min_dist=min_dist,
                   random_state=random_state)

    # 应用 UMAP 降维
    reduced_data_2d = reducer.fit_transform(data_2d)

    # 将降维后的数据重塑回原始的空间维度
    reduced_data = reduced_data_2d.reshape(height, width, n_components)

    return reduced_data


# 使用示例
if __name__ == "__main__":
    # 生成一个模拟的高光谱图像数据
    np.random.seed(0)
    hyperspectral_data = np.random.rand(145, 145, 200)  # 100x100 像素，200 个波段

    # 应用光谱降维
    reduced_data = spectral_umap_reduction(hyperspectral_data, n_components=20)

    print("原始数据形状:", hyperspectral_data.shape)
    print("降维后的数据形状:", reduced_data.shape)

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
    plt.title("Reduced Spectrum (20 bands)")
    plt.xlabel("UMAP Component")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()
