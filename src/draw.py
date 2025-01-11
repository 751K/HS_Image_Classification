import os
from datetime import datetime

from src.Dim.PCA import spectral_pca_reduction
from src.config import Config
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.datesets.Dataset import create_patches
from src.datesets.datasets_load import load_dataset
from src.model_init import create_model


def get_project_root():
    """返回项目根目录的路径"""
    # 获取当前文件（draw.py）的路径
    current_path = os.path.abspath(__file__)

    # 获取 src 目录的父目录，即项目根目录
    return os.path.dirname(os.path.dirname(current_path))


def visualize_classification(model, data, labels, device, config, class_names):
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

    # 创建patches
    patches, _ = create_patches(data, labels, patch_size=config.patch_size)

    # 逐批次进行预测
    batch_size = config.batch_size
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Classifying pixels"):
            batch = torch.FloatTensor(patches[i:i + batch_size]).to(device)
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

    # 获取项目根目录并保存图像
    root_dir = get_project_root()
    pic_dir = os.path.join(root_dir, "Pic")
    os.makedirs(pic_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filename = f"classification_visualization_{timestamp}.png"
    save_path = os.path.join(pic_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 计算准确率
    mask = labels != 0  # 假设 0 是背景类
    accuracy = np.mean(classification_map[mask] == labels[mask])
    print(f"Classification accuracy: {accuracy:.4f}")

    return classification_map


def load_checkpoint(checkpoint_path, model, device):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return


if __name__ == "__main__":
    # 加载数据集
    data, labels, dataset_info = load_dataset('Pavia')
    config = Config()
    # 创建模型
    data, _ = spectral_pca_reduction(data, n_components=config.n_components)
    model = create_model(config.model_name, input_channels=data.shape[-1], num_classes=len(np.unique(labels)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 加载最佳模型
    checkpoint_path = config.resume_checkpoint

    load_checkpoint(checkpoint_path, model, device)

    class_names = ['Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted Metal Sheets', 'Bare Soil',
                   'Bitumen', 'Self-Blocking Bricks', 'Shadow']

    classification_map = visualize_classification(model, data, labels, device, config, class_names)