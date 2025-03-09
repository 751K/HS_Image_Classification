import torch
from torch import nn
from torch.nn import init

from Train_and_Eval.device import get_device


class HybridSN(nn.Module):
    def __init__(self, input_channels, num_classes, patch_size=19):
        super().__init__()
        self.dim = 2
        self.patch_size = patch_size
        self.in_chs = input_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True))

        self.x1_shape = self.get_shape_after_3dconv()
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.x1_shape[1] * self.x1_shape[2], out_channels=64, kernel_size=(3, 3),
                      padding=(1, 1)),
            nn.ReLU(inplace=True))
        self.x2_shape = self.get_shape_after_2dconv()

        self.dense1 = nn.Sequential(
            nn.Linear(self.x2_shape, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4))

        self.dense3 = nn.Sequential(
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def forward(self, X):
        X = X.unsqueeze(1)
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.conv4(x)

        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out

    def get_shape_after_2dconv(self):
        x = torch.zeros((1, self.x1_shape[1] * self.x1_shape[2], self.x1_shape[3], self.x1_shape[4]))
        with torch.no_grad():
            x = self.conv4(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def get_shape_after_3dconv(self):
        x = torch.zeros((1, 1, self.in_chs, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape

    def _initialize_weights(self):
        # 对卷积层进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


if __name__ == "__main__":
    torch.manual_seed(42)

    # 设置模型参数
    in_chans = 64
    num_classes = 10
    batch_size = 4

    # 设置设备
    device = get_device()
    print(f"Using device: {device}")

    # 测试不同输入尺寸
    test_sizes = [(7, 7), (14, 14), (28, 28)]

    for size in test_sizes:
        x = torch.randn(batch_size, in_chans, size[0], size[1]).to(device)
        model = HybridSN(input_channels=in_chans, num_classes=num_classes, patch_size=size[0])
        model = model.to(device)
        try:
            with torch.no_grad():
                output = model(x)

            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")

            # 计算并打印模型参数总数
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params}")

        except Exception as e:
            import traceback
            error_msg = f"input size {size}执行过程中发生错误:\n{str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"
            print(error_msg)

