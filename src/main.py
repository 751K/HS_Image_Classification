# main.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from Train_and_Eval.device import get_device
from Train_and_Eval.learing_rate import WarmupCosineSchedule
from Train_and_Eval.log import setup_logger
from Train_and_Eval.model import save_model, save_test_results, set_seed
from config import Config
from datesets.Dataset import prepare_data, create_data_loaders
from model_init import create_model
from Dim.api import apply_dimension_reduction
from datesets.datasets_load import load_dataset
from src.Train_and_Eval.eval import evaluate_model
from src.Train_and_Eval.train import train_model
from src.vis import visualize_classification
from src.draw.matrix import plot_and_save_confusion_matrix


def main():
    try:
        config = Config()
        if config.model_name is None:
            print("未选择模型，程序退出。")
            return

        set_seed(config.seed)

        # 设置日志记录器
        logger = setup_logger(config.save_dir)
        logger.info("程序开始执行")

        logger.info(f"使用模型: {config.model_name}")
        logger.info(
            f"配置参数：epochs={config.num_epochs}, batch_size={config.batch_size}, num_workers={config.num_workers}")

        # 加载和准备数据
        data, labels, dataset_info = load_dataset(config.datasets, logger)

        logger.info(f"数据加载完成：{config.datasets}")

        data = apply_dimension_reduction(data, config, logger)
        logger.info(f"使用{config.dim_reduction}降维完成")

        num_classes = len(np.unique(labels))
        input_channels = data.shape[-1]

        # 创建模型
        model = create_model(config.model_name, input_channels, num_classes, config.patch_size)

        device = get_device()
        logger.info('使用设备: {}'.format(device))
        model.to(device)

        logger.info(f"模型创建完成：{config.model_name}")

        # 准备数据
        X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(data, labels, test_size=config.test_size,
                                                                      dim=model.dim, patch_size=config.patch_size,
                                                                      random_state=config.seed)
        logger.info(f"训练集尺寸: {X_train.shape}")
        logger.info(f"测试集尺寸: {X_test.shape}")
        logger.info(f'验证集尺寸: {X_val.shape}')
        logger.info("数据预处理完成")
        train_loader, test_loader, val_loader = create_data_loaders(
            X_train, y_train, X_test, y_test, config.batch_size, config.num_workers,
            dim=model.dim, logger=logger, X_val=X_val, y_val=y_val
        )
        logger.info("Dataloader创建完成")

        # 设置训练参数
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)learning_rateΩ

        total_steps = config.num_epochs * len(train_loader)
        scheduler = WarmupCosineSchedule(optimizer, config.warmup_steps, total_steps)
        logger.info("训练参数设置完成")

        # 设置TensorBoard
        writer = SummaryWriter(log_dir=os.path.join(config.save_dir, 'tensorboard'))

        # 检查是否有断点
        start_epoch = 0
        if config.resume_checkpoint:
            checkpoint = torch.load(config.resume_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"从断点 {config.resume_checkpoint} 恢复训练，从 epoch {start_epoch} 开始")

        # 训练模型
        logger.info("开始训练模型...")
        best_model_state_dict = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                            config.num_epochs, device, writer, logger, start_epoch, config)
        logger.info("模型训练完成")
        # 保存最佳模型
        model_save_path = os.path.join(config.save_dir, "best_model.pth")
        save_model(best_model_state_dict, model_save_path, logger)
        logger.info(f"最佳模型已保存到: {model_save_path}")

        # 评估模型
        if config.test_mode is not True:
            model.load_state_dict(best_model_state_dict)
        logger.info('测试集评估中...')
        avg_loss, accuracy, all_preds, all_labels = evaluate_model(model, test_loader, criterion,
                                                                   device, logger, class_result=True)

        # 保存结果和生成可视化
        results_save_path = os.path.join(config.save_dir, "test_results.json")
        save_test_results(all_preds, all_labels, accuracy, avg_loss, results_save_path, logger)

        confusion_matrix_save_path = os.path.join(config.save_dir, "confusion_matrix.png")
        plot_and_save_confusion_matrix(all_labels, all_preds, num_classes, confusion_matrix_save_path)
        visualize_save_path = os.path.join(config.save_dir, "visualization.png")
        visualize_classification(model, data, labels, device, config, dataset_info, logger,
                                 save_path=visualize_save_path)
        writer.close()
        logger.info("程序执行完毕")

    except SystemExit:
        print("程序已按用户请求退出。")
    except Exception as e:
        import traceback
        error_msg = f"程序执行过程中发生错误:\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(error_msg)
        sys.exit(1)


if __name__ == '__main__':
    main()
