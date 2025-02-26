import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Train_and_Eval.eval import evaluate_model
from datesets.Dataset import prepare_data, HSIDataset
from matplotlib.patches import Patch


def visualize_classification(model, data, labels, device, config, class_names, logger=None, save_path=None):
    """
    对原始数据进行逐像素分类并生成可视化图。

    Args:
        model (torch.nn.Module): 训练好的模型
        data (np.ndarray): 原始高光谱数据，形状为 (height, width, channels)
        labels (np.ndarray): 原始标签数据，形状为 (height, width)
        device (torch.device): 用于计算的设备（CPU或GPU）
        config (Config): 配置对象
        class_names (list): 类别名称列表
        logger (logging.Logger): 日志记录器
        save_path (str, optional): 保存可视化图的路径。如果为 None，则使用默认路径保存

    """
    model.eval()

    height, width, channels = data.shape
    # data:Original data shape: (512, 217, 80)
    X, y, _, _ = prepare_data(data, labels, test_size=1, dim=model.dim, patch_size=config.patch_size)
    all_dataset = HSIDataset(X, y, model.dim)
    all_dataloader = DataLoader(all_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    all_preds = []

    if logger is not None:
        logger.info("开始全图推理")
    else:
        print("开始全图推理")

    with torch.no_grad():
        for inputs, batch_labels in all_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)

    # 重塑预测结果到原始图像尺寸
    classification_map = np.zeros((height, width), dtype=int)
    index = 0
    for i in range(height):
        for j in range(width):
            if labels[i, j] != 0:
                classification_map[i, j] = all_preds[index]
                index += 1

    # 创建颜色映射
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = plt.cm.jet(np.linspace(0, 1, num_classes))

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

    if logger is not None:
        logger.info(f"Classification accuracy: {accuracy:.4f}")
    else:
        print(f"Classification accuracy: {accuracy:.4f}")


def get_project_root():
    """返回项目根目录的路径"""
    # 获取当前文件（vis.py）的路径
    current_path = os.path.abspath(__file__)

    # 获取 src 目录的父目录，即项目根目录
    return os.path.dirname(os.path.dirname(os.path.dirname(current_path)))


if __name__ == "__main__":
    print(get_project_root())
