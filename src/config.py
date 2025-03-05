# config.py
import inspect
import os
import sys
from datetime import datetime

import torch
from sympy import false

from src.model_init import AVAILABLE_MODELS


class Config:
    def __init__(self):
        self.test_mode = false

        # 获取调用此类的文件名
        caller_frame = inspect.stack()[1]
        caller_filename = os.path.basename(caller_frame.filename)
        # torch.backends.cudnn.benchmark = True

        if caller_filename == 'main.py' and self.test_mode is not True:
            self.model_name = self.select_model()
            self.num_epochs = 20
        else:
            self.model_name = 'AllinMamba'
            self.num_epochs = 10
            # torch.autograd.set_detect_anomaly = True

        self.batch_size = 128
        self.num_workers = 0
        self.warmup_steps = 10
        self.learning_rate = 0.0003

        self.test_size = 0.95

        self.seed = 42
        self.datasets = 'Salinas'  # 可选:'Indian', 'Pavia', 'Salinas', 'KSC', 'Botswana'
        self.patch_size = 7

        # self.resume_checkpoint = '../results/Salinas_AllinMamba_0226_1559/checkpoint_epoch_100.pth'
        self.resume_checkpoint = None

        # 降维相关配置
        self.perform_dim_reduction = True
        self.dim_reduction = 'PCA'  # 可选: 'PCA', 'KernelPCA', 'MDS', 'UMAP'，‘NMF’
        self.n_components = 64  # 降维后的组件数
        self.pca_whiten = False
        self.kpca_kernel = 'rbf'
        self.kpca_gamma = None
        self.mds_metric = True
        self.mds_n_init = 4
        self.umap_n_neighbors = 15
        self.umap_min_dist = 0.1

        self.stop_train = True
        self.patience = 10  # 早停的耐心值
        self.min_delta = 0.005  # 被视为改进的最小变化量
        if caller_filename == 'main.py':
            self.save_dir = os.path.join("..", "results",
                                         f"{self.datasets}_{self.model_name}_{datetime.now().strftime('%m%d_%H%M')}")
        else:
            self.save_dir = os.path.join("...", "test_results",
                                         f"{self.datasets}_{self.model_name}_{datetime.now().strftime('%m%d_%H%M')}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.label_smoothing = 0.08

        self.optuna_trials = 100  # Optuna 试验次数

    @staticmethod
    def select_model():
        print("Available models:")
        print("0. Exit")
        for i, model_name in enumerate(AVAILABLE_MODELS.keys(), 1):
            print(f"{i}. {model_name}")

        while True:
            try:
                choice = int(input("Enter the number of the model you want to use: "))
                if choice == 0:
                    print("You chose to exit the program. Goodbye!")
                    sys.exit(0)
                elif 1 <= choice <= len(AVAILABLE_MODELS):
                    return list(AVAILABLE_MODELS.keys())[choice - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
