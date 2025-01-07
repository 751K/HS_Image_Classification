# main.py
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Train_and_Eval.learing_rate import WarmupCosineSchedule
from Train_and_Eval.log import setup_logger
from Train_and_Eval.model import save_model, save_test_results, set_seed, plot_and_save_confusion_matrix
from Train_and_Eval.model import train_model, evaluate_model
from config import Config
from datesets.IndianPinesDataset import load_data, prepare_data, create_data_loaders
from model_init import create_model


def main():
    config = Config()
    set_seed(config.seed)

    # 创建保存模型和结果的目录
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = os.path.join("..", "results", f"{config.model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 设置日志记录器
    logger = setup_logger(save_dir)

    logger.info(f"使用模型: {config.model_name}")
    logger.info(
        f"配置参数：epochs={config.num_epochs}, batch_size={config.batch_size}, num_workers={config.num_workers}")

    # 加载和准备数据
    data, labels = load_data(config.data_path, config.gt_path)
    num_classes = len(np.unique(labels))
    input_channels = data.shape[-1]

    # 创建模型
    model = create_model(config.model_name, input_channels, num_classes)
    logger.info(f"模型创建完成：{config.model_name}")

    # 准备数据
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(data, labels, dim=model.dim)  # 假设所有模型都是3D的
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, config.batch_size, config.num_workers, dim=model.dim,
    )

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    total_steps = config.num_epochs * len(train_loader)
    scheduler = WarmupCosineSchedule(optimizer, config.warmup_steps, total_steps)

    # 设置TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    # 训练模型
    logger.info("开始训练模型...")
    best_model_state_dict = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                        config.num_epochs,
                                        device, writer, logger)

    # 保存最佳模型
    model_save_path = os.path.join(save_dir, "best_model.pth")
    save_model(best_model_state_dict, model_save_path)

    # 评估模型
    model.load_state_dict(best_model_state_dict)
    avg_loss, accuracy, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device, logger)

    # 保存结果和生成可视化
    results_save_path = os.path.join(save_dir, "test_results.json")
    save_test_results(all_preds, all_labels, accuracy, avg_loss, results_save_path)

    confusion_matrix_save_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(all_labels, all_preds, num_classes, confusion_matrix_save_path)

    writer.close()
    logger.info("程序执行完毕")


if __name__ == '__main__':
    main()
