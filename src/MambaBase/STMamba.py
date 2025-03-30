# This code is from the repository: https://github.com/AlanLowell/STMamba This is the official implementation of the
# paper: R. Ming, N. Chen, J. Peng, W. Sun and Z. Ye, "Semantic Tokenization-Based Mamba for Hyperspectral Image
# Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,
# with some modifications made here.

import torch
from einops import rearrange
from torch import nn
import torch.nn.init as init

from src.utils.device import get_device


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):  # 64 ,8
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SToken(nn.Module):  # 97.4 oa:  97.802 #  --> L
    def __init__(self, dim):
        super(SToken, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_normal_(self.bias)

    def forward(self, x):
        # 在序列维度上进行SE注意力计算
        squeeze_seq = torch.mean(x, dim=1).unsqueeze(1)  # 在band维度上进行全局平均池化  [64, 1 ,16]

        squeeze_seq = self.fc1(squeeze_seq)

        out = squeeze_seq + self.bias.expand(x.shape[0], -1, -1)
        return out * x


class S6_noC(nn.Module):
    def __init__(self, seq_len, d_model):
        super(S6_noC, self).__init__()

        self.state_dim = d_model

        self.LN_B = nn.Linear(d_model, d_model)
        self.LN_C = nn.Linear(d_model, d_model)
        self.LN_delta = nn.Linear(d_model, d_model)

        self.delta = nn.Parameter(torch.zeros(1, seq_len, d_model))  # [1, L, D]
        nn.init.xavier_normal_(self.delta)

        self.A = nn.Parameter(torch.zeros(seq_len, d_model))  # [L, D]
        nn.init.xavier_normal_(self.A)
        self.ST = SToken(d_model)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数

        Args:
            x: 输入张量，形状为 [B, L, D]
        """
        batch_size, seq_len, feat_dim = x.shape

        # 门控机制
        z = self.Sigmoid(x)

        # 线性变换
        B_0 = self.LN_B(x)  # [B, L, D]
        C_ = self.LN_C(x)  # [B, L, D]

        # 特征增强与调制
        T_ = self.ST(x)  # [B, L, D] - 通过SToken增强特征
        delta = self.Sigmoid(self.LN_delta(x) + self.delta)  # [B, L, D]

        # 系数矩阵调制 - 使用广播代替einsum
        A_ = delta * self.A.unsqueeze(0)  # [B, L, D] = [B, L, D] * [1, L, D]
        B_ = delta * B_0  # [B, L, D]

        # 预分配输出张量
        x_out = torch.zeros_like(x, device=x.device)

        # 初始状态
        s = torch.zeros(batch_size, feat_dim, device=x.device)  # [B, D]

        # 序列处理循环
        for t in range(seq_len):
            # 状态更新 - 使用元素级乘法代替einsum
            s = A_[:, t] * s + B_[:, t] * x[:, t]  # [B, D]

            # 输出预测
            x_out[:, t] = C_[:, t] * s + T_[:, t]  # [B, D]

        # 应用门控机制
        x_out = x_out * z

        return x_out


class Scan(nn.Module):
    def __init__(self):  # 64, 1, 8 ,8
        super().__init__()

    @staticmethod
    def forward(x):  # [64,21,5,5]
        cen = x.shape[2] // 2  # 5
        x = rearrange(x, 'b c h w -> b h w c')  # [64,5,5,21]
        x_out = torch.zeros(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]).to(x.device)  # [64,25,21]
        x_out[:, 0, :] = x[:, cen, cen, :]

        assignments_map = {
            0: [
                (slice(1, 3), (cen - 1, slice(cen, cen + 2))),
                (slice(3, 5), (slice(cen, cen + 2), cen + 1)),
                (slice(5, 7), (cen + 1, slice(cen - 1, cen + 1))),
                (slice(7, 9), (slice(cen - 1, cen + 1), cen - 1)),
            ],
            1: [
                (slice(9, 13), (cen - 2, slice(cen - 1, cen + 3))),
                (slice(13, 17), (slice(cen - 1, cen + 3), cen + 2)),
                (slice(17, 21), (cen + 2, slice(cen - 2, cen + 2))),
                (slice(21, 25), (slice(cen - 2, cen + 2), cen - 2)),
            ],
            2: [
                (slice(25, 31), (cen - 3, slice(cen - 2, cen + 4))),
                (slice(31, 37), (slice(cen - 2, cen + 4), cen + 3)),
                (slice(37, 43), (cen + 3, slice(cen - 3, cen + 3))),
                (slice(43, 49), (slice(cen - 3, cen + 3), cen - 3)),
            ],
            3: [
                (slice(49, 57), (cen - 4, slice(cen - 3, cen + 5))),
                (slice(57, 65), (slice(cen - 3, cen + 5), cen + 4)),
                (slice(65, 73), (cen + 4, slice(cen - 4, cen + 4))),
                (slice(73, 81), (slice(cen - 4, cen + 4), cen - 4)),
            ],
            4: [
                (slice(81, 91), (cen - 5, slice(cen - 4, cen + 6))),
                (slice(91, 101), (slice(cen - 4, cen + 6), cen + 5)),
                (slice(101, 111), (cen + 5, slice(cen - 5, cen + 5))),
                (slice(111, 121), (slice(cen - 5, cen + 5), cen - 5)),
            ],
        }

        for i in range(cen):
            for dest, (row_idx, col_idx) in assignments_map.get(i, []):
                x_out[:, dest, :] = x[:, row_idx, col_idx, :]

        return x_out


class STMambaBlock(nn.Module):
    def __init__(self, seq_len, dim, depth, mlp_dim, dropout):  # 64, 1, 8 ,8
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, S6_noC(seq_len, dim))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):  # x:[64,5,64]
        for S6_noC, mlp in self.layers:
            x = S6_noC(x)  # go to attention
            x = nn.AdaptiveAvgPool1d(1)(x.transpose(1, 2), ).squeeze()  # [B D]
            x = mlp(x)  # go to MLP_Block
        return x


class STMamba(nn.Module):
    def __init__(self, input_channels=1, num_classes=21, depth=1, patch_size=9, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(STMamba, self).__init__()
        self.dim = 3
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * input_channels, out_channels=num_classes, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )

        self.pos_embedding = torch.nn.init.uniform_(nn.Parameter(torch.empty(1, (patch_size ** 2 + 1), num_classes)))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_classes))
        self.dropout = nn.Dropout(emb_dropout)
        self.scan = Scan()
        self.STMambaBlock = STMambaBlock(patch_size ** 2 + 1, num_classes, depth, mlp_dim, dropout)

    def forward(self, input_data):  # x:[64, 1, 30, 9, 9]

        input_data = self.conv3d_features(input_data)  # ->x:[64,8,28,7,7 ]

        input_data = rearrange(input_data, 'b c h w y -> b (c h) w y')  # 8个通道合一，增强光谱空间特征 -> [64,8*28,7,7]
        input_data = self.conv2d_features(input_data)  # ->[64,21,5,5]

        input_data = self.scan(input_data)  # ->[64,p2,21]
        cls_tokens = self.cls_token.expand(input_data.shape[0], -1, -1)  # ->[64,1,21]

        input_data = torch.cat((cls_tokens, input_data), dim=1)
        input_data += self.pos_embedding
        input_data = self.dropout(input_data)

        input_data = self.STMambaBlock(input_data)

        if input_data.shape[0] == 16 or input_data.shape[0] == 21:
            input_data = input_data.unsqueeze(0)  # 最后一个批次只有一个数据

        return input_data


if __name__ == "__main__":
    torch.manual_seed(42)

    # 设置模型参数
    in_chans = 30  # 输入通道数
    embed_dim = 128  # 嵌入维度
    num_classes = 21  # 类别数量
    patch = 9  # patch大小
    depth = 1  # STMambaBlock的深度
    mlp_dim = 8  # MLP隐藏层维度
    dropout = 0.1  # Dropout比率

    # 创建模型
    model = STMamba(input_channels=in_chans, patch_size=patch, num_classes=num_classes, depth=depth, mlp_dim=mlp_dim,
                    dropout=dropout)

    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)

    # 测试不同输入尺寸
    test_sizes = [(9, 9)]

    for size in test_sizes:
        H, W = size
        x = torch.randn(4, 1, in_chans, H, W).to(device)  # 假设输入是[batch_size, in_channels, seq_len, height, width]

        try:
            with torch.no_grad():
                output = model(x)

            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")

        except Exception as e:
            import traceback
            error_msg = f"程序执行过程中发生错误:\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
            print(error_msg)

    # 计算并打印模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
