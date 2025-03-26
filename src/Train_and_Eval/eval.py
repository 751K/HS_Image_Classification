import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score


def evaluate_model(model, data_loader, criterion, device, logger, class_result=False) \
        -> [float, tuple[float, float, float], np.ndarray, np.ndarray]:
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
            - aa (float): 平均准确率 (Average Accuracy)。
            - kappa (float): Kappa 系数。
            - all_preds (numpy.ndarray): 所有预测标签的数组。
            - all_labels (numpy.ndarray): 所有真实标签的数组。
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # class_result=True时显示进度条
        if class_result:
            from tqdm import tqdm
            data_iter = tqdm(data_loader, desc="评估中", leave=True)
        else:
            data_iter = data_loader

        for i, (inputs, labels) in enumerate(data_iter):
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
    overall_accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / len(all_labels)

    # 计算每个类别的准确率
    class_accuracies = []
    unique_classes = np.unique(all_labels)

    for class_id in unique_classes:
        class_mask = all_labels == class_id
        class_accuracy = accuracy_score(all_labels[class_mask], all_preds[class_mask])
        class_accuracies.append(class_accuracy)
        if class_result:
            logger.info("类别 %d: 准确率 = %.4f", class_id, class_accuracy)

    class_accuracies = np.array(class_accuracies)
    aa = np.mean(class_accuracies)

    # 计算 Kappa 系数
    kappa = cohen_kappa_score(all_labels, all_preds)

    # 输出总体结果
    logger.info("平均损失: %.4f, 总体准确率(OA): %.4f", avg_loss, overall_accuracy)
    logger.info("平均准确率 (AA): %.4f, Kappa 系数: %.4f", aa, kappa)

    return avg_loss, (overall_accuracy, aa, kappa), all_preds, all_labels
