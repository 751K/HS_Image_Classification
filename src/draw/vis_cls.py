import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm

from datesets.Dataset import prepare_data, create_one_loader
from src.utils.paths import get_project_root, ensure_dir, sanitize_filename


def visualize_classification(model, data, label, device, config, logger=None):
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
    mask = label != 0  # 创建非背景像素的掩码

    X, y = prepare_data(data, label, test_size=1, dim=model.dim, patch_size=config.patch_size)
    dataloader = create_one_loader(X, y, batch_size=128, num_workers=0, dim=model.dim)

    all_preds = []
    all_labels = []

    if logger is not None:
        logger.info("开始全图推理")
    else:
        print("开始全图推理")

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算各类别准确率
    class_accuracies = []
    unique_classes = np.unique(all_labels)

    if logger is not None:
        logger.info("\n各类别准确率:")
        for class_id in unique_classes:
            class_mask = all_labels == class_id
            class_accuracy = accuracy_score(all_labels[class_mask], all_preds[class_mask])
            class_accuracies.append(class_accuracy)
            logger.info("类别 %d: 准确率 = %.4f", class_id, class_accuracy)

        oa = accuracy_score(all_labels, all_preds)
        aa = np.mean(class_accuracies)
        kappa = cohen_kappa_score(all_labels, all_preds)

        logger.info("总体准确率(OA): %.4f", oa)
        logger.info("平均准确率 (AA): %.4f", aa)
        logger.info("Kappa 系数: %.4f", kappa)
        logger.info('开始绘图')

    # 创建分类映射图 - 只使用掩码索引方式填充预测结果
    classification_map = np.zeros((height, width), dtype=int)
    classification_map[mask] = all_preds + 1  # 添加1以匹配原始标签类别

    # 获取所有类别以便颜色映射
    all_classes = np.unique(np.concatenate([label.flatten(), classification_map.flatten()]))
    max_class = max(np.max(label), np.max(classification_map))

    # 创建一个从jet复制的自定义colormap
    cmap = plt.cm.jet.copy()
    # 创建一个新的颜色列表，其中第一个颜色是黑色
    colors = cmap(np.linspace(0, 1, max_class + 1))
    colors[0] = [0, 0, 0, 1]  # 将索引0的颜色设为黑色
    custom_cmap = plt.matplotlib.colors.ListedColormap(colors)

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 原始标签
    im1 = ax1.imshow(label, cmap=custom_cmap, vmin=0, vmax=max_class)
    ax1.set_title('Original Labels')
    ax1.axis('off')

    # 分类结果
    im2 = ax2.imshow(classification_map, cmap=custom_cmap, vmin=0, vmax=max_class)
    ax2.set_title('Classification Result')
    ax2.axis('off')

    plt.tight_layout()

    # 优化保存逻辑，确保跨平台兼容
    root_dir = get_project_root()
    pic_dir = os.path.join(root_dir, "Pic")
    ensure_dir(pic_dir)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    base_filename = f"classification_visualization_{timestamp}.png"
    # 清理文件名中的非法字符
    safe_filename = sanitize_filename(base_filename)
    save_path = os.path.join(pic_dir, safe_filename)

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"分类可视化结果已保存到: {save_path}")
        else:
            print(f"分类可视化结果已保存到: {save_path}")
    except Exception as e:
        error_msg = f"保存可视化图片失败: {str(e)}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)

    plt.show()
    plt.close()


if __name__ == "__main__":
    print(get_project_root())
