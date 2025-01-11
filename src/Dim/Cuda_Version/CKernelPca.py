import cupy as cp
import numpy as np
from cuml.decomposition import PCA
import time


class KernelPCA:
    def __init__(self, n_components, kernel='rbf', gamma=None):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.alphas = None
        self.K = None

    def fit_transform(self, X):
        start_time = time.time()

        # 将数据转移到 GPU
        X_gpu = cp.asarray(X)

        # 计算核矩阵
        if self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]
            self.K = self._rbf_kernel(X_gpu, X_gpu, self.gamma)
        elif self.kernel == 'linear':
            self.K = cp.dot(X_gpu, X_gpu.T)
        else:
            raise ValueError("Unsupported kernel type.")

        # 中心化核矩阵
        N = X_gpu.shape[0]
        one_n = cp.ones((N, N)) / N
        self.K = self.K - cp.dot(one_n, self.K) - cp.dot(self.K, one_n) + cp.dot(cp.dot(one_n, self.K), one_n)

        # 使用 cuML 的 PCA 进行特征值分解
        pca = PCA(n_components=self.n_components)
        self.alphas = pca.fit_transform(self.K)

        end_time = time.time()
        print(f"Kernel PCA fit_transform 完成，耗时: {end_time - start_time:.2f} 秒")

        return self.alphas

    def transform(self, X):
        X_gpu = cp.asarray(X)
        if self.kernel == 'rbf':
            K = self._rbf_kernel(X_gpu, cp.asarray(self.X_fit), self.gamma)
        elif self.kernel == 'linear':
            K = cp.dot(X_gpu, self.X_fit.T)
        else:
            raise ValueError("Unsupported kernel type.")

        return cp.dot(K, self.alphas)

    @staticmethod
    def _rbf_kernel(X, Y, gamma):
        X_norm = cp.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = cp.sum(Y ** 2, axis=1).reshape(1, -1)
        K = cp.exp(-gamma * (X_norm + Y_norm - 2 * cp.dot(X, Y.T)))
        return K


# 使用示例
if __name__ == "__main__":
    from src.datesets.datasets_load import load_dataset

    print("开始加载数据...")
    start_time = time.time()
    data, labels, dataset_info = load_dataset('Pavia')
    end_time = time.time()
    print(f"数据加载完成，耗时: {end_time - start_time:.2f} 秒")

    # 重塑数据
    height, width, n_bands = data.shape
    data_2d = data.reshape(-1, n_bands)

    # 应用 Kernel PCA
    n_components = 30
    kpca = KernelPCA(n_components=n_components, kernel='rbf')
    reduced_data = kpca.fit_transform(data_2d)

    print("原始数据形状:", data_2d.shape)
    print("降维后的数据形状:", reduced_data.shape)

    # 将结果转回 CPU 并重塑为原始图像形状
    reduced_data_cpu = cp.asnumpy(reduced_data)
    reduced_data_reshaped = reduced_data_cpu.reshape(height, width, n_components)

    print("重塑后的数据形状:", reduced_data_reshaped.shape)
