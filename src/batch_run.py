# batch_run.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

from config import Config
from src.utils.log import setup_logger
from src.Train_and_Eval.model import set_seed
from src.datesets.datasets_load import load_dataset
from src.Dim.api import apply_dimension_reduction
from model_init import AVAILABLE_MODELS, create_model
from src.datesets.Dataset import prepare_data, create_three_loader
from src.Train_and_Eval.train import train_model
from src.Train_and_Eval.eval import evaluate_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.Train_and_Eval.learing_rate import WarmupCosineSchedule


def batch_run(dataset_name, models_to_run=None, result_dir=None):
    """批量执行多个模型并保存比较结果"""
    # 创建结果目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if result_dir is None:
        result_dir = f"../results/batch_comparison_{dataset_name}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # 设置日志
    logger = setup_logger(result_dir)
    logger.info(f"开始在 {dataset_name} 数据集上批量执行模型")

    # 确定要运行的模型
    if models_to_run is None:
        models_to_run = list(AVAILABLE_MODELS.keys())

    logger.info(f"将运行以下模型: {', '.join(models_to_run)}")

    # 准备对比结果存储
    comparison_results = {
        "models": [],
        "accuracy": [],
        "aa": [],
        "kappa": [],
        "training_time": [],
        "parameters": []
    }

    # 加载数据集（只加载一次）
    logger.info(f"加载数据集: {dataset_name}")
    config = Config()
    config.datasets = dataset_name
    config.test_mode = False  # 确保是训练模式

    set_seed(config.seed)

    # 加载数据
    data, labels, dataset_info = load_dataset(config.datasets, logger)
    data = apply_dimension_reduction(data, config, logger)

    # 设备
    device = config.device
    logger.info(f'使用设备: {device}')

    # 对每个模型执行训练和评估
    for model_name in tqdm(models_to_run, desc="执行模型"):
        try:
            logger.info(f"开始执行模型: {model_name}")

            # 更新配置
            config.model_name = model_name
            config.save_dir = os.path.join(result_dir, f"{model_name}")
            os.makedirs(config.save_dir, exist_ok=True)

            # 创建模型
            model = create_model(model_name, config, logger)
            logger.info(f'test_size: {config.test_size}')
            logger.info(f'patch_size: {config.patch_size}')

            # 准备数据
            X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(
                data, labels, test_size=config.test_size,
                dim=model.dim, patch_size=config.patch_size,
                random_state=config.seed, logger=logger
            )

            train_loader, test_loader, val_loader = create_three_loader(
                X_train, y_train, X_test, y_test, X_val, y_val,
                config.batch_size, config.num_workers, dim=model.dim, logger=logger
            )

            # 设置训练参数
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

            total_steps = config.num_epochs * len(train_loader)
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=int(config.warmup_ratio * total_steps),
                t_total=total_steps,
                cycles=config.cycles,
                min_lr=config.min_lr_ratio * config.learning_rate
            )

            # 设置TensorBoard
            writer = SummaryWriter(log_dir=os.path.join(config.save_dir, 'tensorboard'))

            # 训练计时
            start_time = datetime.now()

            # 训练模型 - 使用原始train_model函数
            best_model_state_dict = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                config.num_epochs, device, writer, logger, 0, config
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # 保存最佳模型
            model_save_path = os.path.join(config.save_dir, "best_model.pth")
            torch.save(best_model_state_dict, model_save_path)

            # 评估模型
            model.load_state_dict(best_model_state_dict)
            avg_loss, accuracy, all_preds, all_labels = evaluate_model(
                model, test_loader, criterion, device, logger, class_result=True
            )

            oa, aa, kappa = accuracy

            # 保存结果
            model_result = {
                "model_name": model_name,
                "oa": float(oa),
                "aa": float(aa),
                "kappa": float(kappa),
                "loss": float(avg_loss),
                "training_time": training_time,
                "parameters": sum(p.numel() for p in model.parameters())
            }

            # 保存单个模型结果
            with open(os.path.join(config.save_dir, "results.json"), "w") as f:
                json.dump(model_result, f, indent=4)

            # 添加到比较结果中
            comparison_results["models"].append(model_name)
            comparison_results["accuracy"].append(float(oa))
            comparison_results["aa"].append(float(aa))
            comparison_results["kappa"].append(float(kappa))
            comparison_results["training_time"].append(training_time)
            comparison_results["parameters"].append(sum(p.numel() for p in model.parameters()))

            logger.info(f"模型 {model_name} 执行完成，OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}")

        except Exception as e:
            logger.error(f"模型 {model_name} 执行失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # 保存总体比较结果
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(result_dir, "model_comparison.csv"), index=False)

    # 可视化比较结果
    plt.figure(figsize=(12, 8))
    plt.bar(comparison_results["models"], comparison_results["accuracy"])
    plt.title(f"模型OA比较 - {dataset_name}数据集")
    plt.xlabel("模型")
    plt.ylabel("总体准确率(OA)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "accuracy_comparison.png"))

    # 参数量vs准确率
    plt.figure(figsize=(10, 8))
    plt.scatter(comparison_results["parameters"], comparison_results["accuracy"])
    for i, model_name in enumerate(comparison_results["models"]):
        plt.annotate(model_name, (comparison_results["parameters"][i], comparison_results["accuracy"][i]))
    plt.title(f"参数量 vs 准确率 - {dataset_name}数据集")
    plt.xlabel("参数量")
    plt.ylabel("准确率")
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "params_vs_accuracy.png"))

    logger.info(f"批量执行完成，结果已保存至 {result_dir}")
    return comparison_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量执行模型比较")
    parser.add_argument("--dataset", type=str, default="Indian", help="数据集名称")
    parser.add_argument("--models", type=str, nargs="+", help="要运行的模型列表，不指定则运行所有模型")
    args = parser.parse_args()

    batch_run(args.dataset, args.models)
