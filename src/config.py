# config.py
import torch
import inspect
import os
import sys
from datetime import datetime

from Train_and_Eval.device import get_device
from src.model_init import AVAILABLE_MODELS


class Config:
    def __init__(self):
        self.test_mode = True

        # 获取调用此类的文件名
        caller_frame = inspect.stack()[1]
        caller_filename = os.path.basename(caller_frame.filename)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.num_epochs = 80
        self.vis_enable = False  # 是否可视化

        self.device = get_device()

        if caller_filename == 'main.py' and self.test_mode is not True:
            self.model_name = self.select_model()

        else:
            self.model_name = 'AllinMamba'
            self.num_epochs = 80
            # torch.autograd.set_detect_anomaly(True)

        self.batch_size = 32
        self.num_workers = 0

        # 学习率超参数配置
        self.learning_rate = 0.0006716351350049811
        self.warmup_ratio = 0.19163467965038367
        self.cycles = 0.31
        self.min_lr_ratio = 0.19

        # 优化器超参数配置
        self.weight_decay = 1e-04
        self.beta1 = 0.88
        self.beta2 = 0.98
        self.eps = 2.5e-07

        self.seed = 3407
        self.datasets = 'Salinas'  # 可选:'Indian', 'Pavia', 'Salinas', 'KSC', 'Botswana'

        if self.datasets == 'Indian':
            self.test_size = 0.9
        elif self.datasets == 'KSC':
            self.test_size = 0.75
        elif self.datasets == 'Botswana':
            self.test_size = 0.65
        else:
            self.test_size = 0.95

        self.patch_size = 9

        self.resume_checkpoint = None
        # self.resume_checkpoint = '../results/Salinas_AllinMamba_0309_1804/checkpoint_epoch_40.pth'

        # 降维相关配置
        self.perform_dim_reduction = True
        self.dim_reduction = 'PCA'  # 可选: 'PCA', 'KernelPCA', 'MDS', 'UMAP'，‘NMF’, 'Concat'
        self.n_components = 80  # 降维后的组件数
        self.pca_whiten = False
        self.kpca_kernel = 'rbf'
        self.kpca_gamma = None
        self.mds_metric = True
        self.mds_n_init = 4
        self.umap_n_neighbors = 15
        self.umap_min_dist = 0.1

        self.stop_train = True
        self.patience = 20  # 早停的耐心值
        self.min_delta = 0.0005  # 被视为改进的最小变化量

        if caller_filename == 'main.py':
            self.save_dir = os.path.join("..", "results",
                                         f"{self.datasets}_{self.model_name}_{datetime.now().strftime('%m%d_%H%M')}")
        else:
            self.save_dir = os.path.join("...", "test_results",
                                         f"{self.datasets}_{self.model_name}_{datetime.now().strftime('%m%d_%H%M')}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.optuna_trials = 40  # Optuna 试验次数

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
