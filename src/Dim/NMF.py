import numpy as np
from sklearn.decomposition import NMF
from src.datesets.datasets_load import load_dataset


def nmf_reduction(X, n_components=None):
    """
    执行非负矩阵分解 (NMF)

    参数:
    X : numpy array, 形状为 (n_samples, n_features)
        要降维的数据
    n_components : int, 可选 (默认为 None)
        要保留的成分数量。如果为 None，则默认为 min(n_samples, n_features)

    返回:
    X_nmf : numpy array, 形状为 (n_samples, n_components)
        降维后的数据
    components : numpy array
        NMF 模型的基矩阵
    """

    # 初始化 NMF 模型
    nmf_model = NMF(n_components=n_components, init='random', random_state=42)
    X_nmf = nmf_model.fit_transform(X)
    components = nmf_model.components_

    return X_nmf, components


def spectral_nmf_reduction(data, n_components=None):
    """
    对高光谱图像的光谱维度进行 NMF 降维

    参数:
    data : numpy array, 形状为 (height, width, n_bands)
        高光谱图像数据
    n_components : int, 可选 (默认为 None)
        要保留的成分数量

    返回:
    reduced_data : numpy array, 形状为 (height, width, n_components)
        光谱维度降维后的数据
    components : numpy array
        NMF 模型的基矩阵
    """
    # 获取原始数据的形状
    height, width, n_bands = data.shape

    # 重塑数据为 2D 数组，每一行代表一个像素的光谱
    data_2d = data.reshape(-1, n_bands)

    # 应用 NMF
    data_nmf, components = nmf_reduction(data_2d, n_components=n_components)

    # 将降维后的数据重塑回原始的空间维度
    reduced_data = data_nmf.reshape(height, width, -1)

    return reduced_data, components


# 使用示例
if __name__ == "__main__":
    # 生成一个模拟的高光谱图像数据
    np.random.seed(42)
    data, labels, dataset_info = load_dataset('Pavia')

    reduced_data, components = spectral_nmf_reduction(data, n_components=80)

    print("原始数据形状:", data.shape)
    print("NMF 后的数据形状:", reduced_data.shape)

    # 可视化原始数据和降维后数据的一个像素的光谱
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data[50, 50, :])
    plt.title("Original Spectrum (200 bands)")
    plt.xlabel("Band")
    plt.ylabel("Intensity")

    plt.subplot(1, 2, 2)
    plt.plot(reduced_data[50, 50, :])
    plt.title("Reduced Spectrum (80 components)")
    plt.xlabel("Component")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

    # 可视化前几个成分的基矩阵
    plt.figure(figsize=(12, 5))
    for i in range(5):  # 显示前5个成分
        plt.plot(components[i, :], label=f'Component {i + 1}')
    plt.title("First 5 Components")
    plt.xlabel("Original Band")
    plt.ylabel("Loading")
    plt.legend()
    plt.tight_layout()
    plt.show()
