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
from src.datesets.Dataset import create_spectral_samples, create_patches
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
        'LeeEtAl3D': 'src.CNNBase.LeeEtAl3D'
    }

    # 根据模型名称动态导入相应的类
    for key, module_path in model_map.items():
        if key in model_name:
            module_name, class_name = module_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

    raise ValueError(f"未知的模型类型: {model_name}")


def visualize_all_models(data, labels, device, config, label_names):
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
    model_dirs = glob(os.path.join(results_dir, f"{config.datasets}_*"))

    n_models = len(model_dirs)
    n_cols = 3  # 地面实况 + 每行2个模型
    n_rows = (n_models + 1) // 2  # +1 为地面实况，然后除以2并向上取整

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))

    # 确保axes始终是2D的
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # 绘制地面实况
    axes[0, 0].imshow(labels, cmap='jet')
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    colors = plt.cm.jet(np.linspace(0, 1, num_classes))

    # 为模型推理添加进度条
    for idx, model_dir in enumerate(tqdm(model_dirs, desc="处理模型")):
        row = idx // 2
        col = (idx % 2) + 1  # +1 因为地面实况在第一列

        model_path = os.path.join(model_dir, "best_model.pth")
        state_dict = torch.load(model_path, map_location=device)

        model_name = os.path.basename(model_dir).split('_')[1]
        ModelClass = get_model_class(model_name)

        model = ModelClass(input_channels=data.shape[-1], num_classes=len(label_names))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        classification_map = inference(model, data, labels, device, config)

        axes[row, col].imshow(classification_map, cmap='jet')
        axes[row, col].set_title(f'Model: {os.path.basename(model_dir)}')
        axes[row, col].axis('off')

        mask = labels != 0
        accuracy = np.mean(classification_map[mask] == labels[mask])
        axes[row, col].text(0.5, -0.1, f"Accuracy: {accuracy:.4f}", ha='center', va='center',
                            transform=axes[row, col].transAxes)

    # 移除空白子图
    for row in range(n_rows):
        for col in range(n_cols):
            if col == 0 and row > 0:  # 移除第一列多余的图（地面实况之后）
                fig.delaxes(axes[row, col])
            elif row == n_rows - 1 and col > (n_models % 2):  # 移除最后一行多余的图
                fig.delaxes(axes[row, col])

    # 创建图例
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=label_names[i])
                       for i in range(len(label_names)) if i in unique_labels]

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


def inference(model, data, labels, device, config):
    """
    使用给定的模型对数据进行推理。

    Args:
        model (torch.nn.Module): 训练好的模型
        data (np.ndarray): 原始高光谱数据，形状为 (height, width, channels)
        labels (np.ndarray): 原始标签数据，形状为 (height, width)
        device (torch.device): 用于计算的设备（CPU或GPU）
        config (Config): 配置对象

    Returns:
        np.ndarray: 分类结果图，形状为 (height, width)
    """
    height, width, channels = data.shape

    if model.dim == 1:
        data_transposed = np.transpose(data, (2, 0, 1))  # 转置为 (bands, rows, cols)
        samples, _ = create_spectral_samples(data_transposed, labels, config.sequence_length)
        X = samples.transpose(2, 0, 1)  # 调整为 (num_samples, bands, sequence_length)
    else:  # dim == 2 or dim == 3
        patches, _ = create_patches(data, labels, config.patch_size)
        if model.dim == 2:
            X = patches.reshape(patches.shape[0], patches.shape[1], config.patch_size, config.patch_size)
        else:  # dim == 3
            X = patches.reshape(patches.shape[0], 1, patches.shape[1], config.patch_size, config.patch_size)

    # 逐批次进行预测
    batch_size = config.batch_size
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="分类像素"):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # 重塑预测结果
    classification_map = np.zeros((height, width), dtype=int)
    index = 0
    for i in range(height):
        for j in range(width):
            if labels[i, j] != 0:  # 只对非背景像素进行预测
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
