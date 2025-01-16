import numpy as np
import torch
from sklearn.metrics import accuracy_score


def evaluate_model(model, data_loader, criterion, device, logger, class_result=False):
    """
    评估模型性能。

    Args:
        model (torch.nn.Module): 要评估的神经网络模型。
        data_loader (torch.utils.data.DataLoader): 包含评估数据的 DataLoader。
        criterion (torch.nn.Module): 损失函数。
        device (torch.device): 用于计算的设备（CPU 或 GPU）。
        logger (logging.Logger): 用于记录输出的日志对象。
        class_result (bool, optional): 是否计算并输出每个类别的准确率。默认为 False。

    Returns:
        tuple: 包含以下元素的元组：
            - avg_loss (float): 平均损失。
            - accuracy (float): 总体准确率。
            - all_preds (numpy.ndarray): 所有预测标签的数组。
            - all_labels (numpy.ndarray): 所有真实标签的数组。
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 转换为 numpy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算总体准确率
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / len(all_labels)

    # 计算每个类别的准确率
    class_accuracies = {}
    if class_result:
        unique_classes = np.unique(all_labels)
        logger.info("\n各类别准确率:")
        for class_id in unique_classes:
            class_mask = all_labels == class_id
            class_accuracy = accuracy_score(all_labels[class_mask], all_preds[class_mask])
            class_accuracies[class_id] = class_accuracy
            logger.info("类别 %d: 准确率 = %.4f", class_id, class_accuracy)

    # 输出总体结果
    logger.info("平均损失: %.4f, 总体准确率: %.4f", avg_loss, accuracy)

    return avg_loss, accuracy, all_preds, all_labels
