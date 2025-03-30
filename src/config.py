# config.py
import torch
import inspect
import os
import sys
from datetime import datetime

from src.utils.device import get_device
from src.model_init import AVAILABLE_MODELS
from src.utils.paths import create_experiment_dir


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

        self.datasets = 'KSC'  # 可选:'Indian', 'Pavia', 'Salinas', 'KSC', 'Botswana'

        self.num_workers = 0

        self.seed = 3407

        self.feature_dim = 128
        self.expand = 32
        self.depth = 1
        self.dropout = 0.25

        if self.datasets == 'Indian':
            self.batch_size = 32
            self.test_size = 0.85
            self.num_classes = 16
            self.mlp_dim = 64
            self.expand = 64
            self.d_state = 48
            self.dropout = 0.1824152805122036
            self.learning_rate = 0.0008420354656015855
            self.warmup_ratio = 0.06176653864344813
            self.cycles = 0.5988120910182956
            self.min_lr_ratio = 0.0468285619606516
            self.weight_decay = 2.767587061278965e-05
        elif self.datasets == 'Pavia':
            self.batch_size = 64
            self.test_size = 0.95
            self.num_classes = 9
            self.mlp_dim = 64
            self.dropout = 0.41433006712651566
            self.d_state = 32
            self.cycles = 0.3092527265390655
            self.expand = 16
            self.learning_rate = 0.0002236164386445011
            self.warmup_ratio = 0.04089041139272309
            self.min_lr_ratio = 0.082177205991796
            self.weight_decay = 1.4964546437063115e-05
        elif self.datasets == 'Salinas':
            self.batch_size = 32
            self.mlp_dim = 32
            self.dropout = 0.129859960550872
            self.d_state = 64
            self.cycles = 0.50
            self.learning_rate = 0.0005144888086719854
            self.warmup_ratio = 0.010269880544951123
            self.min_lr_ratio = 0.1855273798968476
            self.weight_decay = 0.00015654784836898332
            self.test_size = 0.95
            self.num_classes = 16
        elif self.datasets == 'KSC':
            self.batch_size = 32
            self.mlp_dim = 64
            self.dropout = 0.47419792880823625
            self.d_state = 16
            self.learning_rate = 0.0007942357117505358
            self.warmup_ratio = 0.018587729415862373
            self.cycles = 0.5031109547446925
            self.min_lr_ratio = 0.09210873093398604
            self.weight_decay = 0.00018084984175502887
            self.test_size = 0.75
            self.num_classes = 13
        elif self.datasets == 'Botswana':
            self.batch_size = 16
            self.mlp_dim = 16
            self.dropout = 0.2654449469982165
            self.d_state = 64
            self.learning_rate = 0.00041645493840305884
            self.warmup_ratio = 0.010097737876680557
            self.cycles = 0.5004982345432989
            self.min_lr_ratio = 0.0828985932694622
            self.weight_decay = 0.0004960078104701533
            self.test_size = 0.65
            self.num_classes = 14
        else:
            raise ValueError(f"Unsupported dataset: {self.datasets}. Supported datasets are: 'Indian', 'Pavia', "
                             f"'Salinas', 'KSC', 'Botswana'")

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

        # 将原来的路径生成代码替换为：
        self.save_dir = create_experiment_dir(
            self.datasets,
            self.model_name,
            is_main=(caller_filename == 'main.py' and not self.test_mode)
        )

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
