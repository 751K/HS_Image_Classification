# src/draw/feature.py

import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime
import argparse
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
import pandas as pd

from src.utils.paths import ROOT_DIR, ensure_dir
from src.utils.device import get_device
from src.utils.log import setup_logger
from src.datesets.datasets_load import load_dataset
from src.datesets.Dataset import prepare_data, create_three_loader
from model_init import create_model
from src.Dim.api import apply_dimension_reduction
from utils.attention_vis import visualize_mamba_attention

# 创建特征可视化专用目录
VIZ_DIR = os.path.join(ROOT_DIR, "feature_visualization")
ensure_dir(VIZ_DIR)


class FeatureHook:
    """捕获模型中间层特征的钩子"""

    def __init__(self):
        self.features = {}
        self.handles = []

    def hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach().cpu()

        return hook

    def register_hooks(self, model, layers_dict):
        """注册多个钩子到模型的指定层"""
        for name, layer in layers_dict.items():
            handle = layer.register_forward_hook(self.hook_fn(name))
            self.handles.append(handle)

    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []


def get_projection_layers(model):
    """识别模型中的投影层"""
    projection_layers = {}

    # 遍历模型的命名模块
    for name, module in model.named_modules():
        # 识别投影层 (in_proj, out_proj)
        if isinstance(module, torch.nn.Linear) and ('in_proj' in name or 'out_proj' in name):
            projection_layers[name] = module

    return projection_layers


def extract_features(model, dataloader, device, layers_dict, n_samples=100):
    """提取指定层的特征"""
    feature_hook = FeatureHook()
    feature_hook.register_hooks(model, layers_dict)

    model.eval()
    features = defaultdict(list)
    labels = []
    count = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="提取特征"):
            if count >= n_samples:
                break

            batch_size = inputs.size(0)
            if count + batch_size > n_samples:
                inputs = inputs[:n_samples - count]
                targets = targets[:n_samples - count]

            inputs = inputs.to(device)
            _ = model(inputs)

            # 收集特征
            for name, feature in feature_hook.features.items():
                # 对每个样本单独处理
                for i in range(feature.size(0)):
                    if count + i < n_samples:
                        # 获取第i个样本的特征
                        feat = feature[i]

                        # 根据特征维度适当处理
                        if feat.dim() == 0:  # 标量
                            feat_avg = feat.unsqueeze(0)
                        elif feat.dim() == 1:  # 向量
                            feat_avg = feat
                        else:  # 多维张量
                            # 将除第一维外的所有维度平均化
                            feat_avg = feat.view(feat.size(0), -1).mean(dim=1)

                        features[name].append(feat_avg.numpy())

            # 收集标签
            labels.extend(targets.cpu().numpy())
            count += inputs.size(0)

    feature_hook.remove_hooks()

    # 将列表转换为numpy数组
    for name in features:
        features[name] = np.stack(features[name])
        print(f"层 '{name}' 特征形状: {features[name].shape}")

    return features, np.array(labels)


def visualize_features(features, labels, output_dir, method='tsne', class_names=None):
    """使用降维方法可视化特征"""
    os.makedirs(output_dir, exist_ok=True)

    for layer_name, layer_features in features.items():
        # 检查特征维度
        n_features = layer_features.shape[1]

        # 跳过维度不足的特征
        if n_features < 2:
            print(f"跳过层 '{layer_name}'，特征维度({n_features})不足以进行{method.upper()}降维")
            continue

        # 跳过太大的特征集
        if layer_features.shape[1] > 10000:
            continue

        plt.figure(figsize=(10, 8))

        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=min(2, n_features))
        else:
            raise ValueError(f"Unsupported visualization method: {method}")

        # 应用降维
        reduced_features = reducer.fit_transform(layer_features)

        # 确定唯一标签
        unique_labels = np.unique(labels)

        # 绘制散点图
        for label in unique_labels:
            idx = labels == label
            plt.scatter(
                reduced_features[idx, 0],
                reduced_features[idx, 1] if reduced_features.shape[1] > 1 else np.zeros(sum(idx)),
                label=class_names[label] if class_names else f"Class {label}",
                alpha=0.7
            )

        plt.title(f"Feature Distribution: {layer_name}")
        plt.xlabel(f"{method.upper()} Dimension 1")
        plt.ylabel(f"{method.upper()} Dimension 2")
        plt.legend()
        plt.tight_layout()

        # 保存图像
        clean_name = layer_name.replace('.', '_')
        plt.savefig(os.path.join(output_dir, f"{clean_name}_{method}.png"), dpi=300)
        plt.close()


def calculate_feature_statistics(features):
    """计算特征的统计信息"""
    stats = {}

    for name, feature in features.items():
        stats[name] = {
            'mean': np.mean(feature, axis=0),
            'std': np.std(feature, axis=0),
            'min': np.min(feature, axis=0),
            'max': np.max(feature, axis=0),
            'sparsity': np.mean(feature == 0),
            'activation_ratio': np.mean(feature > 0),
            'avg_magnitude': np.mean(np.abs(feature)),
            'shape': feature.shape
        }

    return stats


def compare_projection_dimensions(model_name, dataset_name, dimensions, n_samples=100):
    """比较不同投影维度对特征表达的影响"""
    from config import Config

    # 设置基本配置
    config = Config()
    config.datasets = dataset_name
    config.model_name = model_name

    # 创建结果目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.join(VIZ_DIR, f"{model_name}_projection_dim_analysis_{timestamp}")
    ensure_dir(output_dir)

    # 设置日志
    logger = setup_logger(output_dir)
    logger.info(f"开始分析{model_name}在{dataset_name}数据集上不同投影维度的特征表达")

    # 加载数据集
    data, labels, dataset_info = load_dataset(config.datasets, logger)
    data = apply_dimension_reduction(data, config, logger)

    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")

    # 创建基础模型
    base_model = create_model(model_name, config, logger)

    # 准备数据
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(
        data, labels, test_size=config.test_size,
        dim=base_model.dim, patch_size=config.patch_size,
        random_state=config.seed, logger=logger
    )

    _, test_loader, _ = create_three_loader(
        X_train, y_train, X_test, y_test, X_val, y_val,
        config.batch_size, config.num_workers, dim=base_model.dim, logger=logger
    )

    # 获取类别名称
    class_names = dataset_info

    feature_stats = {}
    all_features = {}  # 存储所有维度的特征，用于后续分析

    # 为每个投影维度创建模型并提取特征
    for dim in dimensions:
        logger.info(f"分析投影维度: {dim}")

        # 设置模型参数
        if model_name == 'AllinMamba':
            config.feature_dim = dim  # 修改特征维度参数
        elif model_name == 'Mamba2':
            config.d_model = dim  # 修改模型维度参数

            # 创建模型
            model = create_model(model_name, config, logger)
            model = model.to(device)

            # 加载该维度对应的预训练权重
            checkpoint_path = os.path.join(ROOT_DIR, "results", f"{dataset_name}_{model_name}_{dim}", "best_model.pth")
            if os.path.exists(checkpoint_path):
                logger.info(f"加载训练好的权重: {checkpoint_path}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    logger.info("权重加载成功")
                except Exception as e:
                    logger.error(f"加载权重失败: {e}")
                    logger.warning("使���未训练的模型进行分析，结果可能不具备实际意义")
            else:
                # 如果找不到预训练权重，可以尝试查找其他可能的权重文件
                alternative_paths = [
                    os.path.join(ROOT_DIR, "results", f"{dataset_name}_{model_name}_*", "best_model.pth"),
                    os.path.join(ROOT_DIR, "checkpoints", f"{model_name}_{dataset_name}_{dim}.pth")
                ]

                weights_loaded = False
                for pattern in alternative_paths:
                    import glob
                    for path in glob.glob(pattern):
                        try:
                            logger.info(f"尝试加载替代权重: {path}")
                            checkpoint = torch.load(path, map_location=device)
                            if 'model_state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['model_state_dict'])
                            else:
                                model.load_state_dict(checkpoint)
                            logger.info("替代权重加载成功")
                            weights_loaded = True
                            break
                        except Exception:
                            continue

                    if weights_loaded:
                        break

                if not weights_loaded:
                    logger.warning(f"未找到任何可用的预训练权重")
                    logger.warning("使用未训练的模型进行分析，结果可能不具备实际意义")

        # 识别投影层
        projection_layers = get_projection_layers(model)
        logger.info(f"找到{len(projection_layers)}个投影层: {list(projection_layers.keys())}")

        # 提取特征
        logger.info(f"正在提取维度{dim}的特征...")
        features, feature_labels = extract_features(model, test_loader, device, projection_layers, n_samples)

        # 存储特征用于后续分析
        all_features[dim] = features

        # 计算特征统计信息
        stats = calculate_feature_statistics(features)
        feature_stats[dim] = stats

        # 创建该维度的输出目录
        dim_output_dir = os.path.join(output_dir, f"dim_{dim}")
        ensure_dir(dim_output_dir)

        # 可视化特征
        logger.info(f"正在可视化维度{dim}的特征...")
        visualize_features(features, feature_labels, dim_output_dir, 'tsne', class_names)
        visualize_features(features, feature_labels, dim_output_dir, 'pca', class_names)

        # 保存每个层的激活分布
        for layer_name, layer_features in features.items():
            plt.figure(figsize=(10, 6))
            sns.histplot(layer_features.flatten(), bins=50, kde=True)
            plt.title(f"Feature Distribution: {layer_name} (dim={dim})")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(dim_output_dir, f"{layer_name.replace('.', '_')}_hist.png"), dpi=300)
            plt.close()

    # 创建CSV导出目录
    csv_dir = os.path.join(output_dir, "csv_data")
    ensure_dir(csv_dir)

    # 统计数据CSV导出
    stats_data = {
        'dimension': [],
        'layer_name': [],
        'avg_magnitude': [],
        'sparsity': [],
        'activation_ratio': [],
        'num_params': []
    }

    # 提取每个维度下每个层的统计数据
    for dim in dimensions:
        for layer_name, layer_stats in feature_stats[dim].items():
            stats_data['dimension'].append(dim)
            stats_data['layer_name'].append(layer_name)
            stats_data['avg_magnitude'].append(layer_stats['avg_magnitude'])
            stats_data['sparsity'].append(layer_stats['sparsity'])
            stats_data['activation_ratio'].append(layer_stats['activation_ratio'])

            # 获取该层参数数量
            num_params = 0
            for name, module in model.named_modules():
                if name == layer_name and hasattr(module, 'weight'):
                    num_params = module.weight.numel()
                    if hasattr(module, 'bias') and module.bias is not None:
                        num_params += module.bias.numel()
            stats_data['num_params'].append(num_params)

    # 保存统计数据到CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(csv_dir, 'projection_stats.csv'), index=False)
    logger.info(f"统计数据已保存到: {os.path.join(csv_dir, 'projection_stats.csv')}")

    # 为每个层创建单独的CSV文件，记录维度变化趋势
    for layer in list(feature_stats[dimensions[0]].keys()):
        layer_data = {
            'dimension': dimensions,
            'avg_magnitude': [feature_stats[dim][layer]['avg_magnitude'] for dim in dimensions],
            'sparsity': [feature_stats[dim][layer]['sparsity'] for dim in dimensions],
            'activation_ratio': [feature_stats[dim][layer]['activation_ratio'] for dim in dimensions]
        }
        layer_df = pd.DataFrame(layer_data)
        clean_layer_name = layer.replace('.', '_')
        layer_df.to_csv(os.path.join(csv_dir, f"{clean_layer_name}_trends.csv"), index=False)
        logger.info(f"层{layer}的趋势数据已保存到CSV")

    # 保存原始特征数据样本
    for dim in dimensions:
        dim_dir = os.path.join(csv_dir, f"dim_{dim}")
        ensure_dir(dim_dir)
        for layer_name, features_array in all_features[dim].items():
            # 限制只保存100个样本，��多100个维度
            max_cols = min(100, features_array.shape[1])
            sample_features = features_array[:100, :max_cols]
            feature_df = pd.DataFrame(sample_features)
            clean_name = layer_name.replace('.', '_')
            feature_df.to_csv(os.path.join(dim_dir, f"{clean_name}_features.csv"), index=False)
        logger.info(f"维度{dim}的特征样本已保存到CSV")

    # 比较不同维度的特征统计信息
    for layer in list(feature_stats[dimensions[0]].keys()):
        # 提取该层在不同维度下的统计信息
        magnitude_by_dim = [feature_stats[dim][layer]['avg_magnitude'] for dim in dimensions]
        sparsity_by_dim = [feature_stats[dim][layer]['sparsity'] for dim in dimensions]
        activation_by_dim = [feature_stats[dim][layer]['activation_ratio'] for dim in dimensions]

        # 绘制平均幅度随维度的变化
        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, magnitude_by_dim, marker='o')
        plt.title(f"Average Feature Magnitude vs Projection Dimension: {layer}")
        plt.xlabel("Projection Dimension")
        plt.ylabel("Average Magnitude")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{layer.replace('.', '_')}_magnitude.png"), dpi=300)
        plt.close()

        # 绘制稀疏度随维度的变化
        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, sparsity_by_dim, marker='o')
        plt.title(f"Feature Sparsity vs Projection Dimension: {layer}")
        plt.xlabel("Projection Dimension")
        plt.ylabel("Sparsity (% of zeros)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{layer.replace('.', '_')}_sparsity.png"), dpi=300)
        plt.close()

        # 绘制激活率随维度的变化
        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, activation_by_dim, marker='o')
        plt.title(f"Positive Activation Ratio vs Projection Dimension: {layer}")
        plt.xlabel("Projection Dimension")
        plt.ylabel("% of Positive Activations")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{layer.replace('.', '_')}_activation.png"), dpi=300)
        plt.close()

    logger.info(f"分析完成，结果已保存到: {output_dir}")
    return output_dir


def analyze_attention_weights(model_name, dataset_name, n_samples=16):
    """分析模型注意力权重"""
    from config import Config

    # 设置配置
    config = Config()
    config.datasets = dataset_name
    config.model_name = model_name

    # 创建结果目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.join(VIZ_DIR, f"{model_name}_attention_analysis_{timestamp}")
    ensure_dir(output_dir)

    # 设置日志
    logger = setup_logger(output_dir)
    logger.info(f"开始分析{model_name}在{dataset_name}数据集上的注意力权重")

    # 加载数据集
    data, labels, class_names = load_dataset(config.datasets, logger)
    data = apply_dimension_reduction(data, config, logger)

    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")

    # 创建模型
    model = create_model(model_name, config, logger)
    model = model.to(device)

    # 加载训练好的权重

    checkpoint_path = os.path.join(ROOT_DIR, "results", "Indian_AllinMamba_0401_1740", "best_model.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"加载训练好的权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device,pickle_module=pickle)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning(f"未找到训练好的权重文件: {checkpoint_path}")
        logger.warning("使用未训练的模型进行分析，结果可能不具备实际意义")

    # 准备数据
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(
        data, labels, test_size=config.test_size,
        dim=model.dim, patch_size=config.patch_size,
        random_state=config.seed, logger=logger
    )

    _, test_loader, _ = create_three_loader(
        X_train, y_train, X_test, y_test, X_val, y_val,
        256, config.num_workers, dim=model.dim, logger=logger
    )

    # 可视化注意力权重
    logger.info("开始可视化注意力权重...")
    visualize_mamba_attention(model, test_loader, device, output_dir, class_names, n_samples)

    logger.info(f"注意力权重可视化完成，结果保存在: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型特征与注意力权重分析")
    parser.add_argument("--model", type=str, default="AllinMamba", help="模型名称")
    parser.add_argument("--dataset", type=str, default="Indian", help="数据集名称")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[64, 128, 256, 384, 512],
                        help="要分析的投影维度")
    parser.add_argument("--samples", type=int, default=100, help="用于特征提取的样本数")
    parser.add_argument("--analysis", type=str, default="attention",
                        choices=["feature", "attention"], help="分析类型")

    args = parser.parse_args()

    if args.analysis == "feature":
        compare_projection_dimensions(args.model, args.dataset, args.dimensions, args.samples)
    else:
        analyze_attention_weights(args.model, args.dataset, args.samples)
