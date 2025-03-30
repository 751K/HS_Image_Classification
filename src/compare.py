# compare_datasets.py
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from src.Train_and_Eval.learing_rate import WarmupCosineSchedule
from src.utils.log import setup_logger, log_training_details
from src.Train_and_Eval.model import set_seed
from config import Config
from src.datesets.Dataset import prepare_data, create_three_loader
from src.model_init import create_model
from src.Dim.api import apply_dimension_reduction
from src.datesets.datasets_load import load_dataset
from src.Train_and_Eval.eval import evaluate_model
from src.Train_and_Eval.train import train_model
from src.utils.paths import (get_project_root, ensure_dir, sanitize_filename,
                             ROOT_DIR, create_experiment_dir, create_comparison_dir, get_file_path)


def run_experiment(dataset_name, model_name):
    """
    针对单个数据集运行实验
    """
    try:
        # 创建配置
        config = Config()
        config.test_mode = False
        config.datasets = dataset_name
        config.model_name = model_name

        # 设置保存目录
        config.save_dir = create_comparison_dir(dataset_name, model_name)

        # 设置随机种子
        set_seed(config.seed)

        # 设置日志记录器
        logger = setup_logger(config.save_dir)
        logger.info(f"开始对数据集 {dataset_name} 进行实验")
        log_training_details(logger, config)

        # 获取设备
        device = config.device

        # 加载和准备数据
        data, labels, dataset_info = load_dataset(config.datasets, logger)
        data = apply_dimension_reduction(data, config, logger)

        # 创建模型
        model = create_model(config.model_name, config)

        # 准备数据
        X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(data, labels, test_size=config.test_size,
                                                                      dim=model.dim, patch_size=config.patch_size,
                                                                      random_state=config.seed)

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
        ensure_dir(tensorboard_dir)
        writer = SummaryWriter(log_dir=tensorboard_dir)

        # 训练模型
        logger.info(f"开始训练模型（数据集: {dataset_name}）...")
        best_model_state_dict = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                            config.num_epochs, device, writer, logger, 0, config)

        # 保存最佳模型
        model_save_path = get_file_path(config.save_dir, "best_model.pth")
        torch.save(best_model_state_dict, model_save_path)

        # 评估模型
        model.load_state_dict(best_model_state_dict)
        logger.info(f'测试集评估中（数据集: {dataset_name}）...')
        avg_loss, accuracy, all_preds, all_labels = evaluate_model(model, test_loader, criterion,
                                                                   device, logger, class_result=True)

        # 关闭 TensorBoard writer
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
            json.dump(results, f, indent=4)  # type: ignore

        logger.info(
            f"数据集 {dataset_name} 实验完成，OA: {accuracy[0]:.4f}， AA: {accuracy[1]:.4f}, Kappa: {accuracy[2]:.4f}")
        return results

    except Exception as e:
        import traceback
        error_msg = f"数据集 {dataset_name} 实验失败: {str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(error_msg)
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
    datasets = ['Indian', 'Pavia', 'Salinas', 'KSC', 'Botswana']
    model_name = 'AllinMamba'  # 可以根据需要更改模型

    # 保存所有结果
    all_results = []

    # 对每个数据集运行实验
    for dataset in datasets:
        result = run_experiment(dataset, model_name)
        if result:
            all_results.append(result)

    # 创建比较结果目录
    compare_dir = create_comparison_dir()

    # 保存汇总结果
    df_results = pd.DataFrame(all_results)
    csv_path = get_file_path(compare_dir, "comparison_results.csv")
    df_results.to_csv(csv_path, index=False)

    # 保存汇总JSON
    json_path = get_file_path(compare_dir, "comparison_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)  # type: ignore

    print(f"所有数据集实验完成，结果已保存到 {compare_dir}")

    # 打印比较表格
    if not df_results.empty:
        print("\n===== 数据集比较结果 =====")
        columns_to_show = [col for col in ['dataset', 'oa', 'aa', 'kappa'] if col in df_results.columns]
        print(df_results[columns_to_show])


if __name__ == "__main__":
    main()
