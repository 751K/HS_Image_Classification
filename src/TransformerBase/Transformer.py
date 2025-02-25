import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, input_channels, num_classes, d_model=128, nhead=8, num_layers=3, dropout=0.1, patch_size=5):
        super(Transformer, self).__init__()

        self.model_type = 'Transformer'
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.sequence_length = 25  # 固定序列长度为25
        self.dim = 1
        # 初始的线性层，将输入转换为 d_model 维度
        self.input_projection = nn.Linear(input_channels, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model * self.sequence_length, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # 输入形状: (batch_size, input_channels, sequence_length)
        # 需要转置为 (sequence_length, batch_size, input_channels)
        x = x.permute(2, 0, 1)

        # 投影到 d_model 维度
        x = self.input_projection(x)

        # 添加位置编码
        x = self.pos_encoder(x)

        # 通过 Transformer 编码器
        output = self.transformer_encoder(x)

        # 重塑输出并应用分类器
        output = output.transpose(0, 1).contiguous().view(output.size(1), -1)
        output = self.classifier(output)

        return output


# 使用示例
if __name__ == "__main__":
    # 参数设置
    input_channels = 80  # 根据您的数据调整
    num_classes = 10  # 根据您的分类任务调整

    # 创建模型
    model = Transformer(input_channels, num_classes)

    # 创建一个随机输入进行测试
    batch_size = 32
    x = torch.randn(batch_size, input_channels, 25)  # 序列长度固定为25

    # 前向传播
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
