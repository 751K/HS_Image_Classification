import torch
from torch import nn
from einops import rearrange
from mamba_ssm import Mamba2
from mamba_ssm import Mamba


class AllinMamba(nn.Module):
    def __init__(self, input_channels, num_classes, patch_size, hidden_dim=128, depth=1, mlp_dim=128, dropout=0.,
                 emb_dropout=0., d_state=4, d_conv=4, expand=8, mode=2):
        super().__init__()

        self.dim = 3
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.patch_size = patch_size
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.scan_length = patch_size // 2 * 4 + 2

        self.conv3d_sep = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(7, 3, 3), padding=(3, 1, 1), groups=1),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        self.conv2d_sep = nn.Sequential(
            nn.Conv2d(in_channels=input_channels * 8, out_channels=input_channels * 8, kernel_size=3, padding=1,
                      groups=input_channels * 8),
            nn.BatchNorm2d(input_channels * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels * 8, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
            nn.BatchNorm1d(num_classes)
        )

        if mode == 1:
            self.MambaLayer1 = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.MambaLayer2 = Mamba(
                d_model=self.scan_length,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        else:
            self.MambaLayer1 = Mamba2(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=32
            )
            self.MambaLayer2 = Mamba2(
                d_model=self.scan_length,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand * 16,
                headdim=16
            )

    @staticmethod
    def prepare_data(x):
        """
        Args:
            x: 原始输入数据 (B, 1, C, H, W)
        Returns:
            标准化的输入数据 (B, 1, C, H, W)
        """
        B, _, C, H, W = x.shape

        x_reshaped = x.squeeze(1).permute(0, 2, 3, 1).reshape(-1, C)

        mean = x_reshaped.mean(dim=0, keepdim=True)
        std = x_reshaped.std(dim=0, keepdim=True) + 1e-6
        x_normalized = (x_reshaped - mean) / std

        x_out = x_normalized.reshape(B, H, W, C).permute(0, 3, 1, 2).unsqueeze(1)
        return x_out

    def Feature_extraction(self, x):
        """
        Args:
            x: 输入张量, shape=(B, 1, C, H, W)
        Returns:
            特征张量, shape=(B, hidden_dim, H, W)
        """
        residual_x = x.squeeze(1)
        x = self.conv3d_sep(x)
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.conv2d_sep(x)
        conv = nn.Conv2d(residual_x.shape[1], x.shape[1], 1).to(x.device)
        residual_x = conv(residual_x.squeeze(1))
        return x + residual_x

    # @staticmethod
    # def get_weights(length, device='cuda'):
    #     center = length // 2
    #     weights = torch.exp(-0.5 * ((torch.arange(length) - center) / (center / 2)) ** 2).to(device)
    #     return weights

    @staticmethod
    def get_weights(length, device='cuda'):
        weights = torch.ones(length, device=device) / length
        return weights

    def smallScan(self, x):
        """
        Args:
            x: 输入张量, shape=(B, hidden_dim, H, W)
        Returns:
            螺旋扫描后的特征, shape=(B, scan_length, hidden_dim)
        """
        center = self.patch_size // 2
        x_scan = torch.zeros((x.shape[0], x.shape[1], self.scan_length), device=x.device)
        x_scan[:, :, 1] = x[:, :, center, center]
        x_scan[:, :, 0] = torch.zeros((x.shape[0], x.shape[1]), device=x.device)

        for i in range(center):
            weights = self.get_weights(2 * i + 3, device=x.device)

            # 顶行
            x_slice = x[:, :, center - i - 1, center - i - 1:center + i + 2]
            x_scan[:, :, i * 4 + 2] = (x_slice * weights.view(1, 1, -1)).sum(dim=-1) / weights.sum()

            # 右列
            x_slice = x[:, :, center - i - 1:center + i + 2, center + i + 1]
            x_scan[:, :, i * 4 + 3] = (x_slice * weights.view(1, 1, -1)).sum(dim=-1) / weights.sum()

            # 底行
            x_slice = x[:, :, center + i + 1, center - i - 1:center + i + 2]
            x_scan[:, :, i * 4 + 4] = (x_slice * weights.view(1, 1, -1)).sum(dim=-1) / weights.sum()

            # 左列
            x_slice = x[:, :, center - i - 1:center + i + 2, center - i - 1]
            x_scan[:, :, i * 4 + 5] = (x_slice * weights.view(1, 1, -1)).sum(dim=-1) / weights.sum()

        return x_scan.transpose(1, 2)

    def MambaBlock(self, x):
        """
        Args:
            x: 输入张量, shape=(B, scan_length, hidden_dim)
        Returns:
            经过MambaBlock处理后的张量, shape=(B, scan_length, hidden_dim)
        """
        for _ in range(self.depth):
            x = self.MambaLayer1(x)
            x = x.transpose(1, 2)
            x = self.MambaLayer2(x)
            x = x.transpose(1, 2)
        return x

    def forward(self, x):
        x = self.prepare_data(x)
        x = self.Feature_extraction(x)
        x = self.smallScan(x)
        x = self.MambaBlock(x)
        x = self.classifier(x[:, 0])
        return x


if __name__ == "__main__":
    input_channels = 20
    num_classes = 10
    patch_size = 9
    batch_size = 4
    depth = 2

    model = AllinMamba(input_channels=input_channels, num_classes=num_classes, patch_size=patch_size, depth=depth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    x = torch.randn(batch_size, 1, input_channels, patch_size, patch_size).to(device)
    out = model(x)
    print(out.shape)  # [4, 10]
