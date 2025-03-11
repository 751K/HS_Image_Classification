# Description: Optuna 超参数优化

import os
import optuna
import torch.nn as nn
import numpy as np
import torch.optim as optim
from optuna import Trial
from torch.utils.tensorboard import SummaryWriter

from src.MambaBase import AllinMamba
from src.Train_and_Eval.device import get_device
from src.Train_and_Eval.log import setup_logger
from src.Train_and_Eval.model import set_seed
from src.config import Config
from src.datesets.datasets_load import load_dataset
from src.Dim.api import apply_dimension_reduction
from src.datesets.Dataset import prepare_data, HSIDataset
from src.Train_and_Eval.learing_rate import WarmupCosineSchedule
from src.Train_and_Eval.train import train_model
from src.Train_and_Eval.eval import evaluate_model
from torch.utils.data import DataLoader


def create_data_loaders_for_optimization(X_train, y_train, X_val, y_val, batch_size, num_workers, dim=1, logger=None):
    """
    创建用于优化的数据加载器，只包括训练集和验证集。

    Args:
        X_train (np.array): 训练数据
        y_train (np.array): 训练标签
        X_val (np.array): 验证数据
        y_val (np.array): 验证标签
        batch_size (int): 批量大小
        num_workers (int): 数据加载的工作线程数
        dim (int): 数据维度 (1, 2, or 3)
        logger (logging.Logger): 日志记录器

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    # 数据检查
    if logger:
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Training labels shape: {y_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Validation labels shape: {y_val.shape}")

    # 检查数据是否包含 NaN 或 inf
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        error_msg = "Training data contains NaN or inf values"
        logger.error(error_msg) if logger else print(error_msg)
        raise ValueError(error_msg)

    if np.isnan(X_val).any() or np.isinf(X_val).any():
        error_msg = "Validation data contains NaN or inf values"
        logger.error(error_msg) if logger else print(error_msg)
        raise ValueError(error_msg)

    # 创建数据集
    try:
        train_dataset = HSIDataset(X_train, y_train, dim)
        val_dataset = HSIDataset(X_val, y_val, dim)
    except Exception as e:
        error_msg = f"Error creating datasets: {str(e)}"
        logger.error(error_msg) if logger else print(error_msg)
        raise

    # 创建数据加载器
    try:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    except Exception as e:
        error_msg = f"Error creating dataloaders: {str(e)}"
        logger.error(error_msg) if logger else print(error_msg)
        raise

    # 检查数据加载器
    if logger:
        try:
            train_inputs, train_labels = next(iter(train_dataloader))
            logger.info(f"Training batch shape: {train_inputs.shape}")
            logger.info(f"Training labels shape: {train_labels.shape}")

            val_inputs, val_labels = next(iter(val_dataloader))
            logger.info(f"Validation batch shape: {val_inputs.shape}")
            logger.info(f"Validation labels shape: {val_labels.shape}")
        except Exception as e:
            import traceback
            error_msg = f"检查dataloader时发生错误:\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
            print(error_msg)

    return train_dataloader, val_dataloader


def objective(trial: Trial, config, logger, data, labels, num_classes, input_channels):
    """
    Optuna超参数优化目标函数。

    参数:
        trial (optuna.trial.Trial): Optuna试验对象。
        config (Config): 包含模型和训练设置的配置对象。
        logger (Logger): 用于记录信息的日志对象。
        data (numpy.ndarray): 训练数据。
        labels (numpy.ndarray): 训练标签。
        num_classes (int): 输出类别的数量。
        input_channels (int): 模型输入通道数。

    返回:
        float: 当前试验配置下的验证集准确率。
    """

    # 超参数搜索空间
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)  # 学习率范围
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])  # 批大小选择
    patch_size = trial.suggest_int('patch_size', 3, 11, step=2)  # 补丁大小
    depth = trial.suggest_int('depth', 1, 2)  # 网络深度
    feature_dim = trial.suggest_categorical('feature_dim', [16, 32, 64, 128])  # 特征维度
    mlp_dim = trial.suggest_categorical('mlp_dim', [16, 32, 64, 128])  # MLP维度
    dropout = trial.suggest_float('dropout', 0.1, 0.4)  # Dropout率
    d_state = trial.suggest_categorical('d_state', [16, 32, 48, 64, 80])  # Mamba状态维度
    expand = trial.suggest_categorical('expand', [4, 8])  # 扩展因子

    set_seed(config.seed)

    device = get_device()

    # 使用Optuna配置的超参数创建模型
    model = AllinMamba(input_channels=input_channels, num_classes=num_classes, patch_size=patch_size,
                       depth=depth, feature_dim=feature_dim, mlp_dim=mlp_dim, dropout=dropout,
                       d_state=d_state, expand=expand, mode=2)
    model.to(device)

    # 数据准备
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(data, labels, dim=model.dim, patch_size=patch_size)
    logger.info(
        f"准备的数据集形状: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 创建训练和验证数据加载器
    train_loader, val_loader = create_data_loaders_for_optimization(
        X_train, y_train, X_val, y_val, batch_size, config.num_workers,
        dim=model.dim, logger=logger
    )

    # 设置训练过程的损失函数、优化器、学习率调度器等
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = config.num_epochs * len(train_loader)
    scheduler = WarmupCosineSchedule(optimizer, config.warmup_steps, total_steps)

    # 创建 TensorBoard 记录器
    writer = SummaryWriter(log_dir=os.path.join(config.sa65ve_dir, f'optuna_trial_{trial.number}'))

    # 训练模型
    best_model_state_dict = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                        config.num_epochs, device, writer, logger, start_epoch=0, config=config)

    # 评估模型
    model.load_state_dict(best_model_state_dict)
    _, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion, device, logger, class_result=False)
    oa, aa, kappa = val_accuracy
    writer.close()

    # 记录该试验的最佳验证精度
    trial.set_user_attr('best_val_accuracy', oa)
    logger.info(f"试验 {trial.number} 完成，验证准确率: {oa:.4f}")

    # 返回当前试验的验证准确率，供 Optuna 用来优化
    return oa


def main():
    config = Config()
    if config.model_name is None:
        print("未选择模型，程序退出。")
        return

    # 设置日志记录器
    logger = setup_logger(config.save_dir)
    logger.info("开始 Optuna 超参数优化")

    # 加载数据（只加载一次）
    data, labels, _ = load_dataset(config.datasets, logger)

    # 应用降维（如果需要）
    if config.perform_dim_reduction:
        data = apply_dimension_reduction(data, config, logger)

    # 获取类别数和输入通道数
    num_classes = len(np.unique(labels))
    input_channels = data.shape[-1]

    logger.info("数据加载和预处理完成")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, logger, data, labels, num_classes, input_channels),
                   n_trials=config.optuna_trials)

    logger.info("Optuna 优化完成")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    logger.info("优化后的最佳参数：")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    logger.info("程序执行完毕")


if __name__ == '__main__':
    main()
