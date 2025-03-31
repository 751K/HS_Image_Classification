# Description: Optuna 超参数优化

import os
import optuna
import torch.nn as nn
import numpy as np
import torch.optim as optim
from optuna import Trial
from torch.utils.tensorboard import SummaryWriter

from model_init import init_weights
from src.MambaBase import AllinMamba
from src.utils.device import get_device
from src.utils.log import setup_logger
from src.Train_and_Eval.model import set_seed
from src.config import Config
from src.datesets.datasets_load import load_dataset
from src.Dim.api import apply_dimension_reduction
from src.datesets.Dataset import prepare_data, HSIDataset
from src.Train_and_Eval.learing_rate import WarmupCosineSchedule
from src.Train_and_Eval.train import train_model
from src.Train_and_Eval.eval import evaluate_model
from torch.utils.data import DataLoader
from src.utils.paths import get_optuna_dir, get_plot_path, ensure_dir


# TensorBoard日志保存
def create_tensorboard_writer(config, trial_number):
    log_dir = get_optuna_dir(config.save_dir, trial_number)
    return SummaryWriter(log_dir=log_dir)


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
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      drop_last=True)  # 丢弃最后不完整批次
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    drop_last=True)  # 丢弃最后不完整批次
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
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])  # 批大小选择

    mlp_dim = trial.suggest_categorical('mlp_dim', [16, 32, 64, 128])  # MLP维度
    dropout = trial.suggest_float('dropout', 0.1, 0.5)  # Dropout率
    d_state = trial.suggest_categorical('d_state', [16, 32, 48, 64, 80, 96])  # Mamba状态维度
    expand = trial.suggest_categorical('expand', [8, 16, 32, 64])  # 扩展因子
    head_dim = trial.suggest_categorical('head_dim', [4, 8, 16, 32])  # 头维度
    d_conv = trial.suggest_categorical('d_conv', [2, 4, 8, 16])  # 卷积维度

    # 学习率调度器超参数
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)  # 学习率范围
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.01, 0.2)  # 预热步数占总步数比例
    cycles = trial.suggest_float('cycles', 0.3, 1.0)  # 余弦周期
    min_lr_ratio = trial.suggest_float('min_lr_ratio', 0.0, 0.2)  # 最小学习率与初始学习率的比例
    # 优化器衰减系数
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)  # 权重衰减范围

    learning_rate = config.learning_rate
    warmup_ratio = config.warmup_ratio
    cycles = config.cycles
    min_lr_ratio = config.min_lr_ratio
    weight_decay = config.weight_decay
    feature_dim = config.feature_dim

    set_seed(config.seed)

    device = get_device()

    # 使用Optuna配置的超参数创建模型
    model = AllinMamba(input_channels=input_channels, num_classes=num_classes, patch_size=config.patch_size,
                       depth=1, feature_dim=feature_dim, mlp_dim=mlp_dim, dropout=dropout,
                       d_state=d_state, expand=expand, mode=2, head_dim=head_dim, d_conv=d_conv)
    model.apply(init_weights)
    model.to(device)

    # 数据准备
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(data, labels, test_size=config.test_size,
                                                                  dim=model.dim, patch_size=config.patch_size)
    logger.info(
        f"准备的数据集形状: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 创建训练和验证数据加载器
    train_loader, val_loader = create_data_loaders_for_optimization(
        X_train, y_train, X_val, y_val, batch_size, config.num_workers,
        dim=model.dim, logger=logger
    )

    # 设置训练过程的损失函数、优化器、学习率调度器等
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 基于优化参数创建调度器
    total_steps = config.num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)  # 将比例转换为步数
    min_lr = min_lr_ratio * learning_rate  # 计算最小学习率

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        t_total=total_steps,
        cycles=cycles,
        min_lr=min_lr
    )

    logger.info(f"优化器设置: lr={learning_rate:.2e}, weight_decay={weight_decay:.2e},")
    logger.info(f"学习率调度器设置: warmup_steps={warmup_steps}/{total_steps} ({warmup_ratio:.2f}), "
                f"cycles={cycles}, min_lr={min_lr:.2e} ({min_lr_ratio:.2f}×LR)")
    logger.info(f'批大小: {batch_size}, 特征维度: {feature_dim}, MLP维度: {mlp_dim}, ')
    logger.info(f'Dropout: {dropout:.2f}, 状态维度: {d_state}, 扩展因子: {expand}')

    # 创建 TensorBoard 记录器
    writer = create_tensorboard_writer(config, trial.number)

    # 训练模型
    best_model_state_dict = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                        config.num_epochs, device, writer, logger, start_epoch=0,
                                        config=config, save_checkpoint=False)

    # 评估模型
    model.load_state_dict(best_model_state_dict)
    _, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion, device, logger, class_result=False)
    oa, aa, kappa = val_accuracy
    writer.close()

    # 记录该试验的最佳验证精度
    trial.set_user_attr('best_val_accuracy', oa)
    logger.info(f"试验 {trial.number} 完成，验证准确率: {oa:.4f}")
    logger.info(f'平均准确率: {aa:.4f}, Kappa系数: {kappa:.4f}')

    return kappa


def plot_lr_schedule(config, best_params, save_path=None):
    """
    可视化最佳学习率调度曲线

    Args:
        config: 配置对象
        best_params: 最佳参数字典
        save_path: 保存路径，如果为None则显示图表
    """
    import matplotlib.pyplot as plt
    import torch.optim as optim

    # 创建虚拟模型和优化器
    dummy_model = nn.Linear(10, 2)
    optimizer = optim.Adam(dummy_model.parameters(), lr=best_params['learning_rate'])

    # 推断总步数和预热步数
    # 假设一个批次大小来估计总步数
    estimated_steps_per_epoch = 100  # 这是一个估计值
    total_steps = config.num_epochs * estimated_steps_per_epoch
    warmup_steps = int(best_params['warmup_ratio'] * total_steps)
    min_lr = best_params['min_lr_ratio'] * best_params['learning_rate']

    # 创建调度器
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        t_total=total_steps,
        cycles=best_params['cycles'],
        min_lr=min_lr
    )

    # 记录学习率
    lrs = []
    for i in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    # 标记预热阶段
    plt.axvline(x=warmup_steps, color='r', linestyle='--')
    plt.text(warmup_steps, max(lrs) / 2, f'Warmup End: {warmup_steps} steps',
             rotation=90, verticalalignment='center')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def save_lr_schedule_plot(config, best_params):
    try:
        save_path = get_plot_path(config.save_dir)
        ensure_dir(os.path.dirname(save_path))
        plot_lr_schedule(config, best_params, save_path=save_path)
        return save_path
    except Exception as e:
        return None, str(e)


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

    try:
        best_params = trial.params.copy()
        saved_path = save_lr_schedule_plot(config, best_params)
        if saved_path:
            logger.info(f"学习率调度曲线已保存至: {saved_path}")
        else:
            logger.error(f"保存学习率调度曲线失败")
    except Exception as e:
        logger.error(f"绘制学习率调度曲线时发生错误: {str(e)}")

    logger.info("程序执行完毕")


if __name__ == '__main__':
    main()
