import os

from src.Dim.PCA import spectral_pca_reduction
from src.config import Config

import numpy as np
import torch

from src.datesets.datasets_load import load_dataset
from src.draw.vis_cls import visualize_classification
from src.model_init import create_model




if __name__ == "__main__":
    # 加载数据集
    data, labels, dataset_info = load_dataset('Pavia')
    config = Config()
    # 创建模型
    data, _ = spectral_pca_reduction(data, n_components=config.n_components)
    model = create_model(config.model_name, input_channels=data.shape[-1], num_classes=len(np.unique(labels)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 加载最佳模型
    checkpoint_path = config.resume_checkpoint

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

    classification_map = visualize_classification(model, data, labels, device, config, dataset_info)
