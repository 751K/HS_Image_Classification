import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

from src.CNNBase.LeeEtAl3D import LeeEtAl3D
from src.Train_and_Eval.learing_rate import WarmupCosineSchedule
from src.Train_and_Eval.log import setup_logger
from src.Train_and_Eval.model import init_weights
from src.Train_and_Eval.model import save_model, save_test_results
from src.Train_and_Eval.model import train_model, evaluate_model, set_seed, plot_and_save_confusion_matrix
from src.datesets.IndianPinesDataset import load_data, prepare_data, create_data_loaders


def main():
    num_epochs = 100
    batch_size = 32
    num_workers = 4
    warmup_steps = 10
    set_seed(42)

    # 创建保存模型和结果的目录
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    save_dir = os.path.join("..", "results", f"ResNet2D_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # 设置日志记录器
    logger = setup_logger(save_dir)

    logger.info("配置参数：epochs=%d, batch_size=%d, num_workers=%d", num_epochs, batch_size, num_workers)

    # 数据文件路径
    data_path = '../datasets/Indian/Indian_pines_corrected.mat'
    gt_path = '../datasets/Indian/Indian_pines_gt.mat'

    logger.info("数据路径：data_path=%s, gt_path=%s", data_path, gt_path)

    # data: (145, 145, 200), labels: (145, 145)
    data, labels = load_data(data_path, gt_path)
    num_classes = len(np.unique(labels))
    input_channels = data.shape[-1]
    logger.info("数据加载完成：data shape=%s, labels shape=%s, num_classes=%d",
                data.shape, labels.shape, num_classes)

    # 创建模型

    # model = ResNet2D(input_channels=input_channels, num_classes=num_classes)
    # model = ResNet1D(input_channels=input_channels, num_classes=num_classes)
    model = LeeEtAl3D(in_channels=input_channels, n_classes=num_classes)

    model.apply(init_weights)
    model_name = model.__class__.__name__

    logger.info("模型创建完成：%s", model_name)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(data, labels, dim=model.dim)

    logger.info("数据预处理完成：train=%s, val=%s, test=%s",
                X_train.shape, X_val.shape, X_test.shape)

    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size, num_workers, dim=model.dim

    )

    logger.info("数据加载器创建完成")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logger.info("使用设备：%s", device)

    # 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    total_steps = num_epochs * len(train_loader)  # 总步数
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)

    logger.info("优化器和学习率调度器设置完成：warmup_steps=%d, total_steps=%d", warmup_steps, total_steps)

    # 设置TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    logger.info("TensorBoard设置完成")

    # 训练模型
    logger.info("开始训练模型...")
    best_model_state_dict = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs,
                                        device, writer, logger)
    logger.info("模型训练完成")

    # 保存最佳模型
    model_save_path = os.path.join(save_dir, "best_model.pth")
    save_model(best_model_state_dict, model_save_path)
    logger.info("最佳模型保存在：%s", model_save_path)

    # 加载最佳模型并评估
    model.load_state_dict(best_model_state_dict)
    logger.info("加载最佳模型进行评估...")
    avg_loss, accuracy, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device, logger)

    # 生成分类报告
    report = classification_report(all_labels, all_preds)

    # 保存测试结果
    results_save_path = os.path.join(save_dir, "test_results.json")

    all_preds = all_preds.tolist() if isinstance(all_preds, np.ndarray) else all_preds
    all_labels = all_labels.tolist() if isinstance(all_labels, np.ndarray) else all_labels

    save_test_results(all_preds, all_labels, accuracy, report, results_save_path)
    logger.info("测试结果保存在：%s", results_save_path)

    logger.info("测试准确率: %.4f", accuracy)
    logger.info("分类报告:\n%s", report)

    writer.close()
    logger.info("TensorBoard写入器已关闭")

    confusion_matrix_save_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(all_labels, all_preds, num_classes, confusion_matrix_save_path)
    logger.info("混淆矩阵已保存在：%s", confusion_matrix_save_path)

    logger.info("程序执行完毕")


if __name__ == '__main__':
    main()
