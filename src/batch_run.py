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
import time

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
from utils.cache import clear_cache


def measure_inference_time(model, test_loader, device):
    """测量模型推理时间"""
    model.eval()
    total_samples = 0
    total_time = 0

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            # 计时推理过程
            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize() if 'cuda' in str(device) else None
            end_time = time.time()

            total_time += (end_time - start_time)

    # 计算平均每个样本的推理时间（毫秒）
    avg_inference_time = (total_time / total_samples) * 1000
    return avg_inference_time


def batch_run(dataset_name, models_to_run=None, result_dir=None):
    """批量执行多个模型并保存比较结果"""
    # 创建结果目录
    from src.utils.paths import create_batch_result_dir

    # 修改结果目录创建逻辑
    if result_dir is None:
        result_dir = create_batch_result_dir(dataset_name=dataset_name)
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
        "parameters": [],
        "inference_time": []  # 添加推理时间字段
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
                config.num_epochs, device, writer, logger, 0, config,save_checkpoint=False
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # 保存最佳模型
            model_save_path = os.path.join(config.save_dir, "best_model.pth")
            torch.save(best_model_state_dict, model_save_path)

            # 加载最佳模型
            model.load_state_dict(best_model_state_dict)

            # 测量推理时间
            logger.info(f"测量模型 {model_name} 的推理时间...")
            inference_time = measure_inference_time(model, test_loader, device)
            logger.info(f"模型 {model_name} 的平均推理时间: {inference_time:.2f} 毫秒/样本")

            # 评估模型
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
                "parameters": sum(p.numel() for p in model.parameters()),
                "inference_time": float(inference_time)  # 添加推理时间
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
            comparison_results["inference_time"].append(float(inference_time))  # 添加推理时间

            logger.info(f"模型 {model_name} 执行完成，OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}")
            del model, train_loader, test_loader, val_loader, optimizer, scheduler, best_model_state_dict, writer
            clear_cache(logger)
        except Exception as e:
            logger.error(f"模型 {model_name} 执行失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            clear_cache(logger)

    # 保存总体比较结果
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(result_dir, f"model_comparison_{dataset_name}.csv"), index=False)

    # 分组柱状图同时展示OA、AA和Kappa
    x = np.arange(len(comparison_results["models"]))  # 模型位置
    width = 0.25  # 每组柱的宽度

    plt.figure(figsize=(14, 8))
    # 绘制三组柱状图
    plt.bar(x - width, comparison_results["accuracy"], width, label='OA')
    plt.bar(x, comparison_results["aa"], width, label='AA')
    plt.bar(x + width, comparison_results["kappa"], width, label='Kappa')

    plt.title(f"Model Performance Comparison - {dataset_name} Dataset")
    plt.xlabel("Models")
    plt.ylabel("Accuracy Metrics")
    plt.xticks(x, comparison_results["models"], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "accuracy_comparison.png"))

    # 参数量vs准确率
    plt.figure(figsize=(10, 8))
    plt.scatter(comparison_results["parameters"], comparison_results["kappa"])
    for i, model_name in enumerate(comparison_results["models"]):
        plt.annotate(model_name, (comparison_results["parameters"][i], comparison_results["kappa"][i]))
    plt.title(f"Model Parameters vs Kappa - {dataset_name} Dataset")
    plt.xlabel("Parameters")
    plt.ylabel("Kappa")
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "params_vs_accuracy.png"))

    # 运行时间vs准确率
    plt.figure(figsize=(10, 8))
    plt.scatter(comparison_results["training_time"], comparison_results["kappa"])
    for i, model_name in enumerate(comparison_results["models"]):
        plt.annotate(model_name, (comparison_results["training_time"][i], comparison_results["kappa"][i]))
    plt.title(f"Training Time vs Kappa - {dataset_name}Dataset")
    plt.xlabel("Training Time(s)")
    plt.ylabel("Kappa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "time_vs_accuracy.png"))

    # 添加推理时间vs准确率图
    plt.figure(figsize=(10, 8))
    plt.scatter(comparison_results["inference_time"], comparison_results["kappa"])
    for i, model_name in enumerate(comparison_results["models"]):
        plt.annotate(model_name, (comparison_results["inference_time"][i], comparison_results["kappa"][i]))
    plt.title(f"Inference Time vs Kappa - {dataset_name} Dataset")
    plt.xlabel("Inference Time(ms/sample)")
    plt.ylabel("Kappa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "inference_time_vs_accuracy.png"))

    logger.info(f"批量执行完成，结果已保存至 {result_dir}")
    clear_cache(logger)
    return comparison_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量执行模型比较")
    parser.add_argument("--dataset", type=str, default="Wuhan", help="数据集名称")
    parser.add_argument("--models", type=str, nargs="+", help="要运行的模型列表，不指定则运行所有模型")
    args = parser.parse_args()

    batch_run(args.dataset, args.models)
    clear_cache(None)
