# config.py
import sys
import torch
from src.model_init import AVAILABLE_MODELS


class Config:
    def __init__(self):
        self.test_mode = True
        if self.test_mode:
            self.model_name = 'SSMamba'
            self.num_epochs = 2
            torch.autograd.set_detect_anomaly = True
        else:
            self.model_name = self.select_model()
            self.num_epochs = 100
            torch.backends.cudnn.benchmark = True

        self.batch_size = 64
        self.num_workers = 0
        self.warmup_steps = 10
        self.learning_rate = 0.001
        self.seed = 42
        self.datasets = 'Pavia'  # 可选:'Indian', 'Pavia', 'Salinas'
        self.patch_size = 3
        self.resume_checkpoint = '../results/SSMamba_0111_221046/checkpoint_epoch_100.pth'
        # self.resume_checkpoint = None

        # 降维相关配置
        self.perform_dim_reduction = True
        self.dim_reduction = 'PCA'  # 可选: 'PCA', 'KernelPCA', 'MDS', 'UMAP'，‘NMF’
        self.n_components = 80  # 降维后的组件数
        self.pca_whiten = False
        self.kpca_kernel = 'rbf'
        self.kpca_gamma = None
        self.mds_metric = True
        self.mds_n_init = 4
        self.umap_n_neighbors = 15
        self.umap_min_dist = 0.1

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
