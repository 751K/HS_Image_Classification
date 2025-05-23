import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from tqdm import tqdm
import torch
from glob import glob
import importlib

from src.Dim.PCA import spectral_pca_reduction
from src.config import Config
from src.datesets.Dataset import create_patches, reshape_data_1D
from src.datesets.datasets_load import load_dataset


def get_model_class(model_name):
    # 模型名称到模块路径的映射
    model_map = {
        'ResNet1D': 'src.CNNBase.ResNet1D',
        'ResNet2D': 'src.CNNBase.ResNet2D',
        'ResNet3D': 'src.CNNBase.ResNet3D',
        'SwimTransformer': 'src.TransformerBase.SwimTransformer',
        'VisionTransformer': 'src.TransformerBase.VisionTransformer',
        'SSMamba': 'src.MambaBase.SSMamba',
        'MambaHSI': 'src.MambaBase.MambaHSI',
        'LeeEtAl3D': 'src.CNNBase.LeeEtAl3D',
        'MSAFMamba': 'src.MambaBase.MSAFMamba',
        'SSFTT': 'src.TransformerBase.SSFTT',
        'GCN2D': 'src.CNNBase.GCN2D'
    }

    # 根据模型名称动态导入相应的类
    for key, module_path in model_map.items():
        if key in model_name:
            module_name, class_name = module_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

    raise ValueError(f"未知的模型类型: {model_name}")


def visualize_all_models(all_data, all_labels, vis_device, vis_config, label_name):
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
    model_dirs = glob(os.path.join(results_dir, f"{vis_config.datasets}_*"))

    n_models = len(model_dirs)
    n_cols = 3
    n_rows = ((n_models - 1) + 2) // 2  # +2 为第一行的两个大图

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))

    # 确保axes始终是2D的
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    unique_labels = np.unique(all_labels)
    num_classes = len(unique_labels)
    colors = plt.cm.jet(np.linspace(0, 1, num_classes))

    # 绘制放大的Ground Truth
    ax_gt = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=2)
    ax_gt.imshow(all_labels, cmap='jet')
    ax_gt.set_title('Ground Truth', fontsize=16)
    ax_gt.axis('off')

    # 处理每个模型
    msafmamba_result = None
    other_models = []

    for model_dir in model_dirs:
        model_path = os.path.join(model_dir, "best_model.pth")
        state_dict = torch.load(model_path, map_location=vis_device)

        model_name = os.path.basename(model_dir).split('_')[1]
        ModelClass = get_model_class(model_name)

        model = ModelClass(input_channels=all_data.shape[-1], num_classes=len(label_name))
        model.load_state_dict(state_dict)
        model.to(vis_device)
        model.eval()

        print(f"Processing model: {os.path.basename(model_dir)}")
        classification_map = inference(model, all_data, all_labels, vis_device, vis_config)

        if 'MSAFMamba' in model_name:
            msafmamba_result = (model_name, classification_map)
        else:
            other_models.append((model_name, classification_map))

    # 绘制放大的MSAFMamba结果
    if msafmamba_result:
        ax_msaf = plt.subplot2grid((n_rows, n_cols), (0, 2), colspan=1)
        ax_msaf.imshow(msafmamba_result[1], cmap='jet')
        ax_msaf.set_title(f'Model: {msafmamba_result[0]}', fontsize=16)
        ax_msaf.axis('off')

        mask = all_labels != 0
        accuracy = np.mean(msafmamba_result[1][mask] == all_labels[mask])
        ax_msaf.text(0.5, -0.1, f"Accuracy: {accuracy:.4f}", ha='center', va='center',
                     transform=ax_msaf.transAxes, fontsize=12)

    # 绘制其他模型结果
    for idx, (model_name, classification_map) in enumerate(other_models):
        row = (idx + 2) // n_cols
        col = (idx + 2) % n_cols

        axes[row, col].imshow(classification_map, cmap='jet')
        axes[row, col].set_title(f'Model: {model_name}')
        axes[row, col].axis('off')

        mask = all_labels != 0
        accuracy = np.mean(classification_map[mask] == all_labels[mask])
        axes[row, col].text(0.5, -0.1, f"Accuracy: {accuracy:.4f}", ha='center', va='center',
                            transform=axes[row, col].transAxes)

    # 移除空白子图
    for row in range(1, n_rows):
        for col in range(n_cols):
            if row * n_cols + col >= n_models + 1:
                fig.delaxes(axes[row, col])

    # 创建图例
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=label_name[i])
                       for i in range(len(label_name)) if i in unique_labels]

    # 在子图下方添加图例
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 为图例腾出空间

    save_dir = os.path.join(results_dir, "comparison")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_comparison_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def inference(model, inference_data, gt_labels, inference_device, inference_config):
    """
    使用给定的模型对数据进行推理。

    Args:
        model (torch.nn.Module): 训练好的模型
        inference_data (np.ndarray): 原始高光谱数据，形状为 (height, width, channels)
        gt_labels (np.ndarray): 原始标签数据，形状为 (height, width)
        inference_device (torch.device): 用于计算的设备（CPU或GPU）
        inference_config (Config): 配置对象

    Returns:
        np.ndarray: 分类结果图，形状为 (height, width)
    """
    height, width, channels = inference_data.shape

    if model.dim == 1:
        X, _ = reshape_data_1D(inference_data, gt_labels)
    else:
        patches, patch_labels = create_patches(inference_data, gt_labels, inference_config.patch_size)
        if model.dim == 2:
            X = patches.reshape(patches.shape[0], patches.shape[1], inference_config.patch_size,
                                inference_config.patch_size)
        else:  # dim == 3
            X = patches.reshape(patches.shape[0], 1, patches.shape[1], inference_config.patch_size,
                                inference_config.patch_size)

    # 逐批次进行预测
    batch_size = inference_config.batch_size
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="分类像素"):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(inference_device)
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # 重塑预测结果
    classification_map = np.zeros((height, width), dtype=int)
    index = 0
    for i in range(height):
        for j in range(width):
            if gt_labels[i, j] != 0:  # 只对非背景像素进行预测
                classification_map[i, j] = predictions[index]
                index += 1

    return classification_map


if __name__ == "__main__":
    datasets = 'Salinas'
    data, labels, label_names = load_dataset(datasets)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    config.datasets = datasets
    reduced_data, _ = spectral_pca_reduction(data, n_components=config.n_components)
    visualize_all_models(reduced_data, labels, device, config, label_names)
