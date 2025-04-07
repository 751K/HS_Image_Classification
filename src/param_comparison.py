# param_comparison.py

import os
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse
import time
from copy import deepcopy
from tqdm import tqdm

from config import Config
from src.utils.log import setup_logger
from src.Train_and_Eval.model import set_seed
from src.datesets.datasets_load import load_dataset
from src.Dim.api import apply_dimension_reduction
from model_init import create_model
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


def compare_parameters_values(model_name, dataset_name, parameters_name, values, result_dir=None):
    """比较模型在不同参数下的性能"""
    # 创建结果目录
    from src.utils.paths import create_param_comparison_dir

    # 修改结果目录创建逻辑
    if result_dir is None:
        result_dir = create_param_comparison_dir(model_name, parameters_name, dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    # 设置日志
    logger = setup_logger(result_dir)
    logger.info(f"开始在{dataset_name}数据集上比较{model_name}模型的不同{parameters_name}参数")
    logger.info(f"将测试以下{parameters_name}值: {values}")

    # 准备比较结果存储
    comparison_results = {
        f"{parameters_name}": [],
        "oa": [],
        "aa": [],
        "kappa": [],
        "training_time": [],
        "parameters": [],
        "inference_time": []
    }

    # 加载数据集（只加载一次）
    logger.info(f"加载数据集: {dataset_name}")
    config = Config()
    config.datasets = dataset_name
    config.test_mode = False
    config.model_name = model_name

    set_seed(config.seed)

    # 加载数据
    data, labels, dataset_info = load_dataset(config.datasets, logger)
    data = apply_dimension_reduction(data, config, logger)

    # 获取不带背景的类别数
    num_classes = config.num_classes
    logger.info(f"数据集类别数: {num_classes}")

    # 设备
    device = config.device
    logger.info(f'使用设备: {device}')

    # 准备数据（只准备一次，所有模型使用相同数据分割）
    logger.info("准备数据...")

    # 创建基础模型以获取维度信息
    base_model = create_model(config.model_name, config, logger)

    logger.info(config.test_size)

    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(
        data, labels, test_size=config.test_size,
        dim=base_model.dim, patch_size=config.patch_size,
        random_state=config.seed, logger=logger
    )
    train_loader, test_loader, val_loader = create_three_loader(
        X_train, y_train, X_test, y_test, X_val, y_val,
        config.batch_size, config.num_workers, dim=base_model.dim, logger=logger
    )

    # 对每个parameter值执行训练和评估
    for value in tqdm(values, desc=f"测试不同{parameters_name}值"):
        try:
            value_str = str(value).replace('.', '_')
            logger.info(f"开始测试{parameters_name}={value}")

            # 更新配置
            config_copy = deepcopy(config)
            config_copy.save_dir = os.path.join(result_dir, f"{parameters_name}_{value_str}")
            os.makedirs(config_copy.save_dir, exist_ok=True)
            setattr(config_copy, parameters_name, value)

            if parameters_name == "patch_size" or parameters_name == "test_size" or parameters_name == "random_state":
                X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(
                    data, labels, test_size=config_copy.test_size,
                    dim=base_model.dim, patch_size=config_copy.patch_size,
                    random_state=config_copy.seed, logger=logger
                )
                train_loader, test_loader, val_loader = create_three_loader(
                    X_train, y_train, X_test, y_test, X_val, y_val,
                    config_copy.batch_size, config_copy.num_workers, dim=base_model.dim, logger=logger
                )
            elif parameters_name == "batch_size":
                train_loader, test_loader, val_loader = create_three_loader(
                    X_train, y_train, X_test, y_test, X_val, y_val,
                    config_copy.batch_size, config_copy.num_workers, dim=base_model.dim, logger=logger
                )

            # 创建模型 - 带有特定的参数
            model = create_model(model_name, config_copy, logger)

            # 设置训练参数
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=config_copy.learning_rate,
                                    weight_decay=config_copy.weight_decay)

            total_steps = config_copy.num_epochs * len(train_loader)
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=int(config_copy.warmup_ratio * total_steps),
                t_total=total_steps,
                cycles=config_copy.cycles,
                min_lr=config_copy.min_lr_ratio * config_copy.learning_rate
            )

            # 设置TensorBoard
            writer = SummaryWriter(log_dir=os.path.join(config_copy.save_dir, 'tensorboard'))

            # 训练计时
            start_time = datetime.now()

            # 训练模型
            best_model_state_dict = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                config_copy.num_epochs, device, writer, logger, 0, config_copy, save_checkpoint=False
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # 保存最佳模型
            model_save_path = os.path.join(config_copy.save_dir, "best_model.pth")
            torch.save(best_model_state_dict, model_save_path)

            # 加载最佳模型
            model.load_state_dict(best_model_state_dict)

            # 测量推理时间
            logger.info(f"测量模型{parameters_name}={value}的推理时间...")
            inference_time = measure_inference_time(model, test_loader, device)
            logger.info(f"模型{parameters_name}={value}的平均推理时间: {inference_time:.2f} 毫秒/样本")

            # 评估模型
            avg_loss, accuracy, all_preds, all_labels = evaluate_model(
                model, test_loader, criterion, device, logger, class_result=True
            )

            oa, aa, kappa = accuracy

            # 保存结果
            model_result = {
                f"{parameters_name}": float(value),
                "oa": float(oa),
                "aa": float(aa),
                "kappa": float(kappa),
                "loss": float(avg_loss),
                "training_time": training_time,
                "parameters": sum(p.numel() for p in model.parameters()),
                "inference_time": float(inference_time)
            }

            # 保存单个模型结果
            with open(os.path.join(config_copy.save_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(model_result, f, indent=4)

            # 添加到比较结果中
            comparison_results[f"{parameters_name}"].append(float(value))
            comparison_results["oa"].append(float(oa))
            comparison_results["aa"].append(float(aa))
            comparison_results["kappa"].append(float(kappa))
            comparison_results["training_time"].append(training_time)
            comparison_results["parameters"].append(sum(p.numel() for p in model.parameters()))
            comparison_results["inference_time"].append(float(inference_time))

            logger.info(f"{parameters_name}={value}测试完成，OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}")
            del model, optimizer, scheduler, best_model_state_dict
            clear_cache(logger)

        except Exception as e:
            logger.error(f"{parameters_name}={value}测试失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            clear_cache(logger)

    # 保存总体比较结果
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(result_dir, f"{parameters_name}_comparison.csv"), index=False)

    # 绘制values vs kappa
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_results[f"{parameters_name}"], comparison_results["kappa"], marker='o', linestyle='-')
    plt.title(f"Influence of {model_name} {parameters_name} Parameter on Kappa - {dataset_name}Dataset")
    plt.xlabel(f"{parameters_name}")
    plt.ylabel("Kappa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{parameters_name}_vs_kappa.png"))

    # 绘制values vs 参数量
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_results[f"{parameters_name}"], comparison_results["parameters"], marker='o', linestyle='-')
    plt.title(f"Influence of {model_name} {parameters_name} Parameter on Model Parameters - {dataset_name}Dataset")
    plt.xlabel(f"{parameters_name}")
    plt.ylabel("Parameters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{parameters_name}_vs_params.png"))

    # 绘制values vs 推理时间
    plt.figure(figsize=(10, 6))
    plt.plot(comparison_results[f"{parameters_name}"], comparison_results["inference_time"], marker='o', linestyle='-')
    plt.title(f"Influence of {model_name} {parameters_name} Parameter on Inference Time - {dataset_name}Dataset")
    plt.xlabel(f"{parameters_name}")
    plt.ylabel("Inference TIme(ms/sample)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{parameters_name}_vs_inference_time.png"))

    # 双Y轴图：values vs (kappa和推理时间)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color1 = 'tab:blue'
    ax1.set_xlabel(f'{parameters_name}')
    ax1.set_ylabel('Kappa', color=color1)
    ax1.plot(comparison_results[f"{parameters_name}"], comparison_results["kappa"], marker='o', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Inference TIme(ms/sample)', color=color2)
    ax2.plot(comparison_results[f"{parameters_name}"], comparison_results["inference_time"], marker='s', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(
        f"Influence of {model_name} {parameters_name} Parameter on Inference Time and Kappa - {dataset_name}Dataset")
    fig.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{parameters_name}_vs_kappa_and_time.png"))

    logger.info(f"比较完成，结果已保存至 {result_dir}")
    clear_cache(logger)
    return comparison_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较模型在不同参数下的性能")
    parser.add_argument("--model", type=str, default="AllinMamba", help="模型名称")
    parser.add_argument("--dataset", type=str, default="Wuhan", help="数据集名称")
    parser.add_argument("--parameters_name", type=str, default="test_size", )
    parser.add_argument("--value", type=float,
                        default=[0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
                        nargs='+',
                        help="参数值列表")
    # parser.add_argument("--value", type=int,
    #                     default=[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    #                     nargs='+',
    #                     help="参数值列表")

    args = parser.parse_args()

    compare_parameters_values(args.model, args.dataset, args.parameters_name, args.value)
    clear_cache(None)
