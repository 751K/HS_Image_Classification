# config.py
import torch
import inspect
import os
import sys

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

        self.multi_gpu = False
        self.multi_gpu_flag = False

        if self.device == torch.device('cuda') and torch.cuda.device_count() > 1:
            self.multi_gpu_flag = True

        if caller_filename == 'main.py' and self.test_mode is not True:
            self.model_name = self.select_model()

        else:
            self.model_name = 'AllinMamba'
            self.num_epochs = 80
            # torch.autograd.set_detect_anomaly(True)

        self.datasets = 'Wuhan'  # 可选:'Indian', 'Pavia', 'Salinas', 'KSC', 'Botswana', 'Wuhan'

        self.num_workers = 0

        self.seed = 3407

        self.feature_dim = 128
        self.expand = 8
        self.depth = 1
        self.dropout = 0.25
        self.d_conv = 16

        self.chunk_size = 16

        if self.datasets == 'Indian':
            self.batch_size = 64
            self.test_size = 0.8
            self.num_classes = 16
            self.mlp_dim = 32
            self.expand = 32
            self.d_state = 56
            self.d_conv = 5
            self.head_dim = 4
            self.dropout = 0.4519834690833574
            self.learning_rate = 0.0002684994708259589
            self.warmup_ratio = 0.04006485849074962
            self.cycles = 0.6310212386924017
            self.min_lr_ratio = 0.12814831606095006
            self.weight_decay = 0.00014499372875857687
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
            self.mlp_dim = 32
            self.dropout = 0.29942626972917785
            self.d_state = 80
            self.expand = 64
            self.learning_rate = 0.000317246216733878
            self.warmup_ratio = 0.03542440054169354
            self.cycles = 0.8386501740444608
            self.min_lr_ratio = 0.015853889118795032
            self.weight_decay = 0.0006216143401161145
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
            self.test_size = 0.7
            self.num_classes = 14
        elif self.datasets == 'Wuhan':
            self.batch_size = 64
            self.mlp_dim = 32
            self.dropout = 0.3365819719387231
            self.d_state = 96
            self.cycles = 0.9085242721331501
            self.head_dim = 8
            self.d_conv = 4
            self.expand = 16
            self.learning_rate = 0.0007742320544396837
            self.warmup_ratio = 0.057967770097392214
            self.min_lr_ratio = 0.19095951952778517
            self.weight_decay = 4.546505725523656e-05
            self.test_size = 0.95
            self.num_classes = 22
        else:
            raise ValueError(f"Unsupported dataset: {self.datasets}. Supported datasets are: 'Indian', 'Pavia', "
                             f"'Salinas', 'KSC', 'Botswana'， 'Wuhan'.")

        if self.multi_gpu_flag:
            self.batch_size = self.batch_size * torch.cuda.device_count()
            self.learning_rate = self.learning_rate * torch.cuda.device_count()

        self.patch_size = 9

        self.resume_checkpoint = None
        # self.resume_checkpoint = '../results/Salinas_AllinMamba_0309_1804/checkpoint_epoch_40.pth'

        # 降维相关配置
        self.perform_dim_reduction = True
        self.dim_reduction = 'KernelPCA'  # 可选: 'PCA', 'KernelPCA', 'MDS', 'UMAP'，‘NMF’, 'Concat'
        self.n_components = 32  # 降维后的组件数

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
