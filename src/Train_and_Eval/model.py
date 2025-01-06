import json
import logging
import random

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def set_seed(seed):
    """
    设置随机种子以确保结果可复现。
    Args:
        seed (int): 用于随机数生成器的种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, logger):
    """
    训练模型。

    Args:
        model (torch.nn.Module): 要训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        val_loader (torch.utils.data.DataLoader): 验证数据加载器。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        num_epochs (int): 训练的轮数。
        device (torch.device): 设备（CPU或GPU）。
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard的记录器。
        logger (logging.Logger): 日志记录器。
    """

    best_val_accuracy = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:  # 每100个批次打印一次
                logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.4f',
                            epoch + 1, num_epochs, i + 1, len(train_loader), loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # 验证
        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion, device, logger)
        # 记录训练和验证指标
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        logger.info('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f',
                    epoch + 1, num_epochs, epoch_loss, epoch_acc, val_loss, val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            logger.info('New best model saved with validation accuracy: %.4f', best_val_accuracy)

    return best_model


def evaluate_model(model, data_loader, criterion, device, logger):
    """
    评估模型在给定数据加载器上的表现。

    Args:
        model (torch.nn.Module): 要评估的模型。
        data_loader (torch.utils.data.DataLoader): 用于评估的数据加载器。
        criterion (torch.nn.Module): 损失函数。
        device (torch.device): 运行评估的设备（CPU或GPU）。
        logger (logging.Logger): 用于记录信息的日志记录器。

    Returns:
        tuple: 包含平均损失、准确率、所有预测和所有标签的元组。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    avg_loss = running_loss / len(data_loader.dataset)
    logger.info("平均损失: %.4f, 准确率: %.4f", avg_loss, accuracy)

    return avg_loss, accuracy, all_preds, all_labels


def save_model(state_dict, path):
    """
    保存模型的状态字典。

    Args:
        state_dict (dict): 模型的状态字典。
        path (str): 保存路径。
    """
    torch.save(state_dict, path)
    print(f"Model state_dict saved to {path}")


def save_test_results(all_preds, all_labels, accuracy, classification_report, path):
    """
    保存测试结果。

    Args:
        all_preds (list): 所有预测标签。
        all_labels (list): 所有真实标签。
        accuracy (float): 测试准确率。
        classification_report (str): 分类报告。
        path (str): 保存路径。
    """

    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

    results = {
        "predictions": all_preds,
        "true_labels": all_labels,
        "accuracy": float(accuracy),
        "classification_report": classification_report
    }

    with open(path, 'w') as f:
        json.dump(results, f, indent=4, default=convert)

    print(f"Test results saved to {path}")


def plot_and_save_confusion_matrix(labels, preds, num_classes, save_path):
    """
    计算并保存混淆矩阵。

    Args:
        labels (list or np.ndarray): 真实标签。
        preds (list or np.ndarray): 预测标签。
        num_classes (int): 类别数量。
        save_path (str): 保存图像的路径。
    """
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)

    # 可视化混淆矩阵
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # 保存图像
    plt.savefig(save_path)

    # 显示图像
    plt.show()