import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Train_and_Eval.model import set_seed
from src.Train_and_Eval.learing_rate import WarmupCosineSchedule
from src.utils.log import setup_logger, log_training_details
from src.utils.paths import create_comparison_dir, get_file_path
from config import Config
from src.datesets.Dataset import prepare_data, create_three_loader
from src.model_init import create_model
from src.Dim.api import apply_dimension_reduction
from src.datesets.datasets_load import load_dataset
from src.Train_and_Eval.eval import evaluate_model
from src.Train_and_Eval.train import train_model
from utils.cache import clear_cache


def run_experiment(dataset_name, model_name):
    """针对单个数据集运行实验"""
    try:
        # 创建配置
        config = Config()
        config.test_mode = False
        config.update_dataset_config(dataset_name)
        config.model_name = model_name

        # 设置保存目录
        config.save_dir = create_comparison_dir(dataset_name, model_name)

        # 设置日志记录器
        logger = setup_logger(config.save_dir)
        logger.info(f"开始对数据集 {dataset_name} 进行实验")

        logger.info(f"使用设备: {config.device}")
        set_seed(config.seed)

        log_training_details(logger, config)

        # 加载和准备数据
        data, labels, dataset_info = load_dataset(config.datasets, logger)
        data = apply_dimension_reduction(data, config, logger)

        # 创建模型
        model = create_model(config.model_name, config)

        # 准备数据
        X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(data, labels, test_size=config.test_size,
                                                                      dim=model.dim, patch_size=config.patch_size,
                                                                      random_state=config.seed)

        # 确保模型的类别数匹配实际标签数量
        num_classes = config.num_classes

        logger.info(f"训练集标签范围: {np.min(y_train)}-{np.max(y_train)}")
        logger.info(f"测试集标签范围: {np.min(y_test)}-{np.max(y_test)}")
        logger.info(f"模型类别数: {num_classes}")

        train_loader, test_loader, val_loader = create_three_loader(
            X_train, y_train, X_test, y_test, X_val, y_val, config.batch_size,
            config.num_workers, dim=model.dim, logger=logger
        )

        # 设置训练参数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # 设置学习率调度器
        total_steps = config.num_epochs * len(train_loader)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(config.warmup_ratio * total_steps),
            t_total=total_steps,
            cycles=config.cycles,
            min_lr=config.min_lr_ratio * config.learning_rate
        )

        # 设置TensorBoard
        tensorboard_dir = os.path.join(config.save_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)

        # 训练模型
        logger.info(f"开始训练模型（数据集: {dataset_name}）...")
        try:
            best_model_state_dict = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                                config.num_epochs, config.device, writer, logger, 0, config)

            # 保存最佳模型
            model_save_path = get_file_path(config.save_dir, "best_model.pth")
            torch.save(best_model_state_dict, model_save_path)

            # 评估模型
            model.load_state_dict(best_model_state_dict)
            logger.info(f'测试集评估中（数据集: {dataset_name}）...')
            avg_loss, accuracy, all_preds, all_labels = evaluate_model(model, test_loader, criterion,
                                                                       config.device, logger, class_result=True)
        except RuntimeError as e:
            if 'out of memory' in str(e) or 'CUDA' in str(e):
                logger.error(f"CUDA内存不足，尝试使用CPU: {e}")
                torch.cuda.empty_cache()
                config.device = torch.device('cpu')
                model = model.to(config.device)

                best_model_state_dict = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                                    config.num_epochs, config.device, writer, logger, 0, config)

                model_save_path = get_file_path(config.save_dir, "best_model.pth")
                torch.save(best_model_state_dict, model_save_path)

                model.load_state_dict(best_model_state_dict)
                avg_loss, accuracy, all_preds, all_labels = evaluate_model(model, test_loader, criterion,
                                                                           config.device, logger, class_result=True)
            else:
                raise

        # 关闭TensorBoard writer
        writer.close()

        # 保存结果
        results = {
            'dataset': dataset_name,
            'model': model_name,
            'oa': float(accuracy[0]),
            'aa': float(accuracy[1]),
            'kappa': float(accuracy[2]),
            'avg_loss': float(avg_loss),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        results_save_path = get_file_path(config.save_dir, "test_results.json")
        with open(results_save_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(
            f"数据集 {dataset_name} 实验完成，OA: {accuracy[0]:.4f}， AA: {accuracy[1]:.4f}, Kappa: {accuracy[2]:.4f}")

        # 清理内存
        logger.info("清理内存...")
        del model, optimizer, scheduler, criterion, train_loader, test_loader, val_loader
        del data, labels, X_train, y_train, X_test, y_test, X_val, y_val
        clear_cache(logger)
        logger.info("内存清理完成")

        return results

    except Exception as e:
        import traceback
        error_msg = f"数据集 {dataset_name} 实验失败: {str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(error_msg)

        # 异常情况下也清理内存
        print("清理内存...")
        clear_cache(None)
        print("内存清理完成")

        return {
            'dataset': dataset_name,
            'model': model_name,
            'oa': None,
            'aa': None,
            'kappa': None,
            'avg_loss': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def main():
    # 设置要测试的数据集和模型
    datasets = ['Indian', 'Pavia', 'Salinas', 'Botswana']
    model_name = 'AllinMamba'  # 可根据需要更改模型

    # 保存所有结果
    all_results = []

    # 对每个数据集运行实验
    for dataset in datasets:
        print(f"\n{'=' * 20} 开始 {dataset} 数据集实验 {'=' * 20}")

        result = run_experiment(dataset, model_name)
        if result:
            all_results.append(result)

        # 每个数据集处理完后额外清理一次内存
        print(f"完成 {dataset} 数据集，执行额外内存清理...")
        clear_cache(None)

    # 创建比较结果目录
    compare_dir = create_comparison_dir()

    # 保存汇总结果
    df_results = pd.DataFrame(all_results)
    csv_path = get_file_path(compare_dir, "comparison_results.csv")
    df_results.to_csv(csv_path, index=False)

    # 保存汇总JSON
    json_path = get_file_path(compare_dir, "comparison_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n所有数据集实验完成，结果已保存到 {compare_dir}")

    # 打印比较表格
    if not df_results.empty:
        print("\n===== 数据集比较结果 =====")
        columns_to_show = [col for col in ['dataset', 'oa', 'aa', 'kappa'] if col in df_results.columns]
        print(df_results[columns_to_show])


if __name__ == "__main__":
    main()
