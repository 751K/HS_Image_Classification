import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from Dim.PCA import spectral_pca_reduction
from Dim.kernel_PCA import optimized_kernel_pca_reduction
from datesets.datasets_load import load_dataset

font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
plt.rcParams['font.family'] = ['Arial Unicode MS']


def plot_spectral_curve_by_class(data, labels, label_names):
    """
    按类别提取并绘制每个类别的光谱响应曲线（直接平均）

    """

    # 获取图像的高和宽
    H, W, C = data.shape

    # 准备绘图
    plt.figure(figsize=(10, 6))

    # 对每个类别的像素进行处理
    for class_id, class_name in enumerate(label_names):
        # 获取当前类别的所有像素坐标
        class_pixels = np.where(labels == class_id)

        if len(class_pixels[0]) == 0:
            continue  # 如果当前类别没有像素，跳过

        # 提取当前类别的光谱响应数据
        class_spectral_values = data[class_pixels[0], class_pixels[1], :]

        # 计算每个类别的光谱响应的均值
        mean_spectrum = np.mean(class_spectral_values, axis=0)

        # 绘制当前类别的光谱响应曲线
        plt.plot(np.arange(C), mean_spectrum, label=class_name)

    # 设置图形标签和标题
    plt.xlabel('光谱波段')
    plt.ylabel('光谱响应值')
    # plt.title(f'{dataset_name} 数据集的类别光谱响应曲线')
    plt.legend(loc='upper left')

    # 保存和显示图形
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_光谱响应曲线.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    dataset_name = 'Wuhan'  # 'Pavia', 'Botswana', 'Wuhan', 'Indian', 'Salinas'
    data, labels, label_names = load_dataset(dataset_name)
    # reduce_data, _ = spectral_pca_reduction(data, n_components=80)
    # reduce_data, _ = optimized_kernel_pca_reduction(data, n_components=80)
    plot_spectral_curve_by_class(data, labels, label_names)
