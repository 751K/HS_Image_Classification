import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.utils.device import get_device


class SpectralGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SpectralGCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gcn(x, edge_index)
        x = self.bn(x)
        return F.relu(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.conv(x)
        att = torch.sigmoid(att)
        return x * att


class GCN2D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden_channels: int = 64, num_gcn_layers: int = 3,
                 patch_size=7):
        super(GCN2D, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dim = 2
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.spatial_attention = SpatialAttention(hidden_channels)

        self.gcn_layers = nn.ModuleList([
            SpectralGCNLayer(hidden_channels, hidden_channels)
            for _ in range(num_gcn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.size()

        x = self.feature_extraction(x)
        x = self.spatial_attention(x)

        edge_index = self._create_grid_graph(height, width).to(x.device)
        x = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))

        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)

        x = x.view(batch_size, height, width, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x

    @staticmethod
    def _create_grid_graph(height: int, width: int) -> torch.Tensor:
        edge_index = []

        for i in range(height):
            for j in range(width):
                node = i * width + j
                if i > 0:
                    edge_index.append([node, (i - 1) * width + j])
                if i < height - 1:
                    edge_index.append([node, (i + 1) * width + j])
                if j > 0:
                    edge_index.append([node, i * width + (j - 1)])
                if j < width - 1:
                    edge_index.append([node, i * width + (j + 1)])

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


# 测试代码
if __name__ == '__main__':
    patch_size = [3, 5, 7, 9]
    device = get_device()
    model = GCN2D(input_channels=200, num_classes=16).to(device)

    for size in patch_size:
        print(f"Patch size: {size}")
        input_data = torch.randn(16, 200, size, size).to(device)
        output = model(input_data)
        print(f"Input shape: {input_data.shape}")
        print(f"Output shape: {output.shape}")
