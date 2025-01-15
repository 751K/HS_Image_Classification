import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.datesets.Dataset import create_patches, create_spectral_samples
from matplotlib.patches import Patch


def visualize_classification(model, data, labels, device, config, class_names, save_path=None):
    """
    对原始数据进行逐像素分类并生成可视化图。

    Args:
        model (torch.nn.Module): 训练好的模型
        data (np.ndarray): 原始高光谱数据，形状为 (height, width, channels)
        labels (np.ndarray): 原始标签数据，形状为 (height, width)
        device (torch.device): 用于计算的设备（CPU或GPU）
        config (Config): 配置对象
        class_names (list): 类别名称列表

    Returns:
        np.ndarray: 分类结果图，形状为 (height, width)
    """
    model.eval()
    height, width, channels = data.shape

    if model.dim == 1:
        data_transposed = np.transpose(data, (2, 0, 1))  # 转置为 (bands, rows, cols)
        samples, _ = create_spectral_samples(data_transposed, labels, config.sequence_length)
        X = samples.transpose(2, 0, 1)  # 调整为 (num_samples, bands, sequence_length)
    else:  # dim == 2 or dim == 3
        patches, _ = create_patches(data, labels, config.patch_size)
        if model.dim == 2:
            X = patches.reshape(patches.shape[0], patches.shape[1], config.patch_size, config.patch_size)
        else:  # dim == 3
            X = patches.reshape(patches.shape[0], 1, patches.shape[1], config.patch_size, config.patch_size)

    # 逐批次进行预测
    batch_size = config.batch_size
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="Classifying pixels"):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # 重塑预测结果
    classification_map = np.zeros((height, width), dtype=int)
    index = 0
    for i in range(height):
        for j in range(width):
            if labels[i, j] != 0:  # 只对非背景像素进行预测
                classification_map[i, j] = predictions[index]
                index += 1

    # 创建颜色映射
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = plt.cm.jet(np.linspace(0, 1, num_classes))

    # 创建分类结果的彩色图
    rgb_image = colors[classification_map]

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 原始标签
    ax1.imshow(labels, cmap='jet')
    ax1.set_title('Original Labels')
    ax1.axis('off')

    # 分类结果
    ax2.imshow(classification_map, cmap='jet')
    ax2.set_title('Classification Result')
    ax2.axis('off')

    # 创建图例
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=class_names[i])
                       for i in range(len(class_names)) if i in unique_labels]

    # 在图像之间添加图例
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        # 获取项目根目录并保存图像
        root_dir = get_project_root()
        pic_dir = os.path.join(root_dir, "Pic")
        os.makedirs(pic_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        filename = f"classification_visualization_{timestamp}.png"
        save_path = os.path.join(pic_dir, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 计算准确率
    mask = labels != 0  # 假设 0 是背景类
    accuracy = np.mean(classification_map[mask] == labels[mask])
    print(f"Classification accuracy: {accuracy:.4f}")


def get_project_root():
    """返回项目根目录的路径"""
    # 获取当前文件（vis.py）的路径
    current_path = os.path.abspath(__file__)

    # 获取 src 目录的父目录，即项目根目录
    return os.path.dirname(os.path.dirname(os.path.dirname(current_path)))


if __name__ == "__main__":
    print(get_project_root())
