import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Train_and_Eval.eval import evaluate_model
from datesets.Dataset import prepare_data, HSIDataset, create_one_loader
from matplotlib.patches import Patch


def visualize_classification(model, data, label, device, config, logger=None):
    # TODO:fix the bug of the function
    """
    对原始数据进行逐像素分类并生成可视化图。

    Args:
        model (torch.nn.Module): 训练好的模型
        data (np.ndarray): 原始高光谱数据，形状为 (height, width, channels)
        label (np.ndarray): 原始标签数据，形状为 (height, width)
        device (torch.device): 用于计算的设备（CPU或GPU）
        config (Config): 配置对象
        logger (logging.Logger): 日志记录器

    """
    model.eval()
    height, width, channels = data.shape

    X, y = prepare_data(data, label, test_size=1, dim=model.dim, patch_size=config.patch_size)
    dataloader = create_one_loader(X, y, batch_size=128, num_workers=0, dim=model.dim)

    all_preds = []
    all_labels = []

    if logger is not None:
        logger.info("开始全图推理")
    else:
        print("开始全图推理")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_accuracies = []

    unique_classes = np.unique(all_labels)
    logger.info("\n各类别准确率:")
    for class_id in unique_classes:
        class_mask = all_labels == class_id
        class_accuracy = accuracy_score(all_labels[class_mask], all_preds[class_mask])
        class_accuracies.append(class_accuracy)
        logger.info("类别 %d: 准确率 = %.4f", class_id, class_accuracy)

    oa = accuracy_score(all_labels, all_preds)
    aa = np.mean(class_accuracies)
    kappa = cohen_kappa_score(all_labels, all_preds)

    # 输出总体结果
    logger.info(" 总体准确率(OA): %.4f", oa)
    logger.info("平均准确率 (AA): %.4f", aa)
    logger.info("Kappa 系数: %.4f", kappa)

    logger.info('开始绘图')
    classification_map = np.zeros((height, width), dtype=int)
    index = 0

    for i in range(height):
        for j in range(width):
            if label[i, j] != 0:
                classification_map[i, j] = all_preds[index] + 1
                index += 1

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 原始标签
    ax1.imshow(label, cmap='jet')
    ax1.set_title('Original Labels')
    ax1.axis('off')

    # 分类结果
    ax2.imshow(classification_map, cmap='jet')
    ax2.set_title('Classification Result')
    ax2.axis('off')

    plt.tight_layout()

    root_dir = get_project_root()
    pic_dir = os.path.join(root_dir, "Pic")
    os.makedirs(pic_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filename = f"classification_visualization_{timestamp}.png"
    save_path = os.path.join(pic_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def get_project_root():
    """返回项目根目录的路径"""
    # 获取当前文件（vis.py）的路径
    current_path = os.path.abspath(__file__)

    # 获取 src 目录的父目录，即项目根目录
    return os.path.dirname(os.path.dirname(os.path.dirname(current_path)))


if __name__ == "__main__":
    print(get_project_root())
