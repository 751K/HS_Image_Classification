import time
from enum import Enum, auto
from typing import Optional, Tuple, Union, Literal

import torch
from torch import nn
from einops import rearrange

from Train_and_Eval.device import get_device


class DataPrepMethod(Enum):
    """数据预处理方法枚举"""
    STANDARD = auto()  # 标准化处理
    MINMAX = auto()  # Min-Max 归一化
    RCS = auto()  # RCS (Range-Channel-Space) 归一化
    NONE = auto()  # 不进行预处理


class FeatureExtractionMethod(Enum):
    """特征提取方法枚举"""
    BASELINE = auto()  # 使用 conv3d_sep + conv2d_sep
    RESIDUAL = auto()  # 添加残差连接
    NO_CONV3D = auto()  # 移除 3D 卷积
    NO_CONV2D_SEP = auto()  # 移除分离式 2D 卷积


class WeightMethod(Enum):
    """权重计算方法枚举"""
    MEAN = auto()  # 均等权重
    COS = auto()  # 基于高斯函数的权重
    TRIANGULAR = auto()  # 三角形分布权重
    NEW = auto()  # 自定义权重计算


class FusionMethod(Enum):
    """特征融合方法枚举"""
    BASELINE = auto()  # 使用 sigmoid 门控
    COSINE = auto()  # 基于余弦相似度的门控
    NONE = auto()  # 无融合，直接使用空间特征


class Canary_Model(nn.Module):
    """
    AllinMamba: 融合空间和光谱信息的 Mamba 模型架构

    该模型使用螺旋扫描和状态空间模型处理遥感图像分类任务，
    包含空间路径和光谱路径两个分支，最终通过融合门控机制组合特征。
    """

    def __init__(
            self,
            input_channels: int,
            num_classes: int,
            patch_size: int,
            feature_dim: int = 128,
            depth: int = 1,
            mlp_dim: int = 64,
            dropout: float = 0.37,
            d_state: int = 16,
            expand: int = 8,
            mode: int = 2,
            root_mamba: bool = False,
            data_prep_method: DataPrepMethod = DataPrepMethod.STANDARD,
            feature_extraction_method: FeatureExtractionMethod = FeatureExtractionMethod.BASELINE,
            weight_method: WeightMethod = WeightMethod.NEW,
            fusion_method: FusionMethod = FusionMethod.BASELINE
    ):
        """
        初始化 AllinMamba 模型

        Args:
            input_channels: 输入通道数
            num_classes: 分类类别数
            patch_size: 图像块大小
            feature_dim: 特征维度
            depth: Mamba块重复深度
            mlp_dim: 分类器中MLP的隐藏维度
            dropout: Dropout比率
            d_state: Mamba状态维度
            expand: 扩展因子
            mode: Mamba版本模式(1或2)
            root_mamba: 是否使用官方mamba_ssm包
            data_prep_method: 数据预处理方法
            feature_extraction_method: 特征提取方法
            weight_method: 权重计算方法
            fusion_method: 特征融合方法
        """
        super().__init__()

        # 基本参数设置
        self.dim = 2
        self.feature_dim = feature_dim
        self.depth = depth
        self.patch_size = patch_size
        self.dropout = dropout
        self.scan_length = patch_size // 2 * 4 + 4
        self.input_channels = input_channels
        self.chunk_size = 4

        # 配置方法选择
        self.data_prep_method = data_prep_method
        self.feature_extraction_method = feature_extraction_method
        self.weight_method = weight_method
        self.fusion_method = fusion_method

        # 卷积网络组件
        self._init_conv_layers()

        # 线性层和融合组件
        self.linear = nn.Linear(input_channels, feature_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 2),
            nn.Softmax(dim=1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
            nn.BatchNorm1d(num_classes)
        )

        # 初始化 Mamba 层
        self._init_mamba_layers(mode, d_state, expand, root_mamba)

        # 如果需要feature_extraction_no_conv2d_sep方法，初始化conv_map
        if self.feature_extraction_method == FeatureExtractionMethod.NO_CONV2D_SEP:
            self.conv_map = nn.Conv2d(input_channels * 8, feature_dim, kernel_size=1)

    def _init_conv_layers(self):
        """初始化卷积层组件"""
        # 3D卷积分离模块
        self.conv3d_sep = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(7, 3, 3), padding=(3, 1, 1), groups=1),
            nn.BatchNorm3d(1),
            nn.SiLU(),
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=1),
            nn.BatchNorm3d(8),
            nn.SiLU()
        )

        # 2D深度可分离卷积模块
        self.conv2d_sep = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels * 8,
                out_channels=self.input_channels * 8,
                kernel_size=3,
                padding=1,
                groups=self.input_channels * 8
            ),
            nn.BatchNorm2d(self.input_channels * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.input_channels * 8, out_channels=self.feature_dim, kernel_size=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU()
        )

        # 维度调整1×1卷积
        self.conv2d_dim = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.feature_dim, kernel_size=1),
            nn.ReLU()
        )

    def _init_mamba_layers(self, mode, d_state, expand, root_mamba):
        """初始化Mamba层"""
        # 导入适当的Mamba实现
        if root_mamba:
            try:
                from mamba_ssm import Mamba2, Mamba as Mamba1
            except ImportError:
                from src.MambaBase.Mamba1 import Mamba1
                from src.MambaBase.Mamba2 import Mamba2
        else:
            from src.MambaBase.Mamba1 import Mamba1
            from src.MambaBase.Mamba2 import Mamba2

        # 根据模式选择Mamba版本
        if mode == 1:
            self.MambaLayer1 = Mamba1(
                d_model=self.feature_dim,
                d_state=d_state,
                expand=expand
            )
            self.MambaLayer2 = Mamba1(
                d_model=self.scan_length,
                d_state=d_state,
                expand=expand
            )
            self.MambaLayer3 = Mamba1(
                d_model=self.input_channels,
                d_state=d_state,
                expand=expand
            )
        else:
            # Mamba2 版本
            self.MambaLayer1 = Mamba2(
                d_model=self.feature_dim,
                d_state=d_state,
                headdim=16,
                expand=expand,
                chunk_size=self.chunk_size
            )
            self.MambaLayer2 = Mamba2(
                d_model=self.scan_length,
                d_state=d_state,
                headdim=16,
                expand=expand,
                chunk_size=self.chunk_size
            )
            self.MambaLayer3 = Mamba2(
                d_model=self.input_channels,
                d_state=d_state,
                headdim=16,
                expand=expand,
                chunk_size=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模型前向传播

        Args:
            x: 输入张量，形状为 (B, C, H, W)

        Returns:
            分类输出张量，形状为 (B, num_classes)
        """
        # 1. 数据预处理
        x = self._prepare_data(x)

        # 2. 双路径特征提取
        x1 = self.spatial_route(x)  # 空间路径
        x2 = self.spectral_route(x)  # 光谱路径

        # 3. 特征融合
        x1 = self._fusion_way(x1, x2)

        # 4. 门控机制融合
        combined_features = torch.cat([x1, x2], dim=1)
        weights = self.fusion_gate(combined_features)
        x = weights[:, 0:1] * x1 + weights[:, 1:2] * x2

        # 5. 分类
        x = self.classifier(x)
        return x

    def spatial_route(self, x: torch.Tensor) -> torch.Tensor:
        """
        空间信息处理路径

        Args:
            x: 输入张量，形状为 (B, C, H, W)

        Returns:
            特征张量，形状为 (B, feature_dim)
        """
        # 1. 特征提取
        x = self._feature_extraction(x)

        # 2. 螺旋扫描
        x = self.smallScan(x)

        # 3. Mamba序列建模
        x = self.MambaBlock(x)

        # 4. 取前4个位置的特征平均作为最终表示
        x = x[:, 0:4].mean(dim=1)
        return x

    def spectral_route(self, x: torch.Tensor) -> torch.Tensor:
        """
        光谱信息处理路径

        Args:
            x: 输入张量，形状为 (B, C, H, W)

        Returns:
            特征张量，形状为 (B, feature_dim)
        """
        # 提取中心像素的光谱信息
        center_pixel = x[:, :, self.patch_size // 2, self.patch_size // 2]
        center_pixel = center_pixel.unsqueeze(1)  # (B, 1, C)

        # 使用Mamba处理光谱序列
        for _ in range(self.depth):
            center_pixel = self.MambaLayer3(center_pixel)

        # 投影到特征空间
        center_pixel = self.linear(center_pixel.squeeze(1))
        return center_pixel

    def MambaBlock(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mamba序列建模块

        Args:
            x: 输入张量，形状为 (B, scan_length, hidden_dim)

        Returns:
            处理后的张量，形状为 (B, scan_length, hidden_dim)
        """
        for _ in range(self.depth):
            # 沿特征维度处理
            x = self.MambaLayer1(x)
            # 转置后沿序列长度维度处理
            x = x.transpose(1, 2)
            x = self.MambaLayer2(x)
            x = x.transpose(1, 2)
        return x

    def smallScan(self, x: torch.Tensor) -> torch.Tensor:
        """
        螺旋扫描实现，从中心点开始逐层扫描图像，提取特征

        Args:
            x: 输入特征图，形状为 (B, hidden_dim, H, W)

        Returns:
            螺旋扫描后的序列特征，形状为 (B, scan_length, hidden_dim)
        """
        center = self.patch_size // 2
        x_scan = torch.zeros((x.shape[0], x.shape[1], self.scan_length), device=x.device)

        # 初始化中心点及周围位置
        for i in range(4):
            x_scan[:, :, i] = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        x_scan[:, :, 0] = x[:, :, center, center]  # 中心点

        # 逐层螺旋扫描
        for i in range(center):
            weights = self._get_weights(2 * i + 3, device=x.device)

            # 顶行扫描
            x_slice = x[:, :, center - i - 1, center - i - 1:center + i + 2]
            x_scan[:, :, i * 4 + 4] = (x_slice * weights.view(1, 1, -1)).sum(dim=-1) / weights.sum()

            # 右列扫描
            x_slice = x[:, :, center - i - 1:center + i + 2, center + i + 1]
            x_scan[:, :, i * 4 + 5] = (x_slice * weights.view(1, 1, -1)).sum(dim=-1) / weights.sum()

            # 底行扫描
            x_slice = x[:, :, center + i + 1, center - i - 1:center + i + 2]
            x_scan[:, :, i * 4 + 6] = (x_slice * weights.view(1, 1, -1)).sum(dim=-1) / weights.sum()

            # 左列扫描
            x_slice = x[:, :, center - i - 1:center + i + 2, center - i - 1]
            x_scan[:, :, i * 4 + 7] = (x_slice * weights.view(1, 1, -1)).sum(dim=-1) / weights.sum()

        return x_scan.transpose(1, 2)  # (B, scan_length, hidden_dim)

    def _prepare_data(self, x: torch.Tensor) -> torch.Tensor:
        """数据预处理分发器"""
        if self.data_prep_method == DataPrepMethod.STANDARD:
            return self._prepare_data_standard(x)
        elif self.data_prep_method == DataPrepMethod.MINMAX:
            return self._prepare_data_minmax(x)
        elif self.data_prep_method == DataPrepMethod.RCS:
            return self._prepare_data_RCS(x)
        else:  # DataPrepMethod.NONE
            return x

    def _feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取分发器"""
        if self.feature_extraction_method == FeatureExtractionMethod.BASELINE:
            return self._feature_extraction_baseline(x)
        elif self.feature_extraction_method == FeatureExtractionMethod.NO_CONV3D:
            return self._feature_extraction_no_conv3d(x)
        elif self.feature_extraction_method == FeatureExtractionMethod.NO_CONV2D_SEP:
            return self._feature_extraction_no_conv2d_sep(x)
        else:  # FeatureExtractionMethod.RESIDUAL
            return self._feature_extraction_residual(x)

    def _get_weights(self, length: int, device='cuda') -> torch.Tensor:
        """权重计算分发器"""
        if self.weight_method == WeightMethod.MEAN:
            return self._get_weights_mean(length, device)
        elif self.weight_method == WeightMethod.COS:
            return self._get_weights_cos(length, device)
        elif self.weight_method == WeightMethod.TRIANGULAR:
            return self._get_weights_triangular(length, device)
        else:  # WeightMethod.NEW
            return self._get_weights_new(length, device)

    def _fusion_way(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """融合方式分发器"""
        if self.fusion_method == FusionMethod.BASELINE:
            return self._fusion_way_baseline(x1, x2)
        elif self.fusion_method == FusionMethod.COSINE:
            return self._fusion_way_cosine(x1, x2)
        else:  # FusionMethod.NONE
            return x1

    # 以下为各种预处理方法的实现
    @staticmethod
    def _prepare_data_RCS(x: torch.Tensor) -> torch.Tensor:
        """RCS 数据预处理"""
        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)
        mean = x_reshaped.mean(dim=0, keepdim=True)
        std = x_reshaped.std(dim=0, keepdim=True) + 1e-6
        x_normalized = (x_reshaped - mean) / std
        x_out = x_normalized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_out = x_out * torch.rsqrt(x_out.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-5)
        return nn.SiLU()(x_out)

    @staticmethod
    def _prepare_data_standard(x: torch.Tensor) -> torch.Tensor:
        """标准化数据预处理"""
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        return (x - mean) / std

    @staticmethod
    def _prepare_data_minmax(x: torch.Tensor) -> torch.Tensor:
        """Min-Max 归一化数据预处理"""
        min_val = x.amin(dim=(0, 2, 3), keepdim=True)
        max_val = x.amax(dim=(0, 2, 3), keepdim=True)
        return (x - min_val) / (max_val - min_val + 1e-6)

    # 以下为各种特征提取方法的实现
    def _feature_extraction_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """基准特征提取方法"""
        x_out = self.conv3d_sep(x.unsqueeze(1))
        x_out = rearrange(x_out, 'b t c h w -> b (t c) h w')
        x_out = self.conv2d_sep(x_out)
        return x_out

    def _feature_extraction_residual(self, x: torch.Tensor) -> torch.Tensor:
        """带残差连接的特征提取方法"""
        residual_x = self.conv2d_dim(x)
        x_out = self.conv3d_sep(x.unsqueeze(1))
        x_out = rearrange(x_out, 'b t c h w -> b (t c) h w')
        x_out = self.conv2d_sep(x_out)
        return x_out + residual_x

    def _feature_extraction_no_conv3d(self, x: torch.Tensor) -> torch.Tensor:
        """无3D卷积的特征提取方法"""
        residual_x = self.conv2d_dim(x)
        x_expanded = x.repeat(1, 8, 1, 1)  # 通道扩展
        x_out = self.conv2d_sep(x_expanded)
        return x_out + residual_x

    def _feature_extraction_no_conv2d_sep(self, x: torch.Tensor) -> torch.Tensor:
        """无分离式2D卷积的特征提取方法"""
        residual_x = self.conv2d_dim(x)
        x_out = self.conv3d_sep(x.unsqueeze(1))
        x_out = rearrange(x_out, 'b t c h w -> b (t c) h w')
        x_out = self.conv_map(x_out)  # 使用预定义的映射卷积
        return x_out + residual_x

    # 以下为各种权重计算方法的实现
    @staticmethod
    def _get_weights_mean(length: int, device='cuda') -> torch.Tensor:
        """均等权重计算"""
        return torch.ones(length, device=device) / length

    @staticmethod
    def _get_weights_cos(length: int, device='cuda') -> torch.Tensor:
        """基于高斯函数的权重计算"""
        center = length // 2
        weights = torch.exp(-0.5 * ((torch.arange(length) - center) / (center / 2)) ** 2).to(device)
        return weights

    @staticmethod
    def _get_weights_triangular(length: int, device='cuda') -> torch.Tensor:
        """三角形分布权重计算"""
        if length == 1:
            return torch.ones(1, device=device)
        center = (length - 1) / 2.0
        indices = torch.arange(length, device=device, dtype=torch.float32)
        weights = 1 - torch.abs(indices - center) / center
        return weights / weights.sum()

    @staticmethod
    def _get_weights_new(length: int, device='cuda', sigma=1.0) -> torch.Tensor:
        """自定义权重计算方法"""
        center = length // 2
        indices = torch.arange(length, device=device)
        squared_distances = (indices - center) ** 2 + center ** 2
        decay_factor = -0.5 / (sigma ** 2)
        weights = torch.exp(decay_factor * squared_distances)
        return weights / weights.sum()

    # 以下为各种融合方法的实现
    @staticmethod
    def _fusion_way_baseline(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """基准融合方法"""
        gate = torch.sigmoid(x2)
        return gate * x1

    @staticmethod
    def _fusion_way_cosine(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """基于余弦相似度的融合方法"""
        cos_sim = nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8)
        gate = (cos_sim + 1) / 2
        return gate.unsqueeze(1) * x1


if __name__ == "__main__":
    # 模型参数
    config = {
        "input_channels": 80,
        "num_classes": 10,
        "patch_size": 9,
        "batch_size": 4,
        "depth": 1,
        "data_prep_method": DataPrepMethod.STANDARD,
        "feature_extraction_method": FeatureExtractionMethod.BASELINE,
        "weight_method": WeightMethod.NEW,
        "fusion_method": FusionMethod.BASELINE
    }

    # 初始化模型
    model = Canary_Model(
        input_channels=config["input_channels"],
        num_classes=config["num_classes"],
        patch_size=config["patch_size"],
        depth=config["depth"],
        data_prep_method=config["data_prep_method"],
        feature_extraction_method=config["feature_extraction_method"],
        weight_method=config["weight_method"],
        fusion_method=config["fusion_method"]
    )

    # 设备选择和模型迁移
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)

    # 创建测试数据
    x = torch.randn(
        config["batch_size"],
        config["input_channels"],
        config["patch_size"],
        config["patch_size"]
    ).to(device)

    # 测量推理时间
    start_time = time.time()
    with torch.no_grad():
        out = model(x)
    inference_time = time.time() - start_time

    # 输出结果
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Output shape: {out.shape}")