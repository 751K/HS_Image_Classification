import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LeeEtAl(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        self.name = 'LeeEtAl'

        self.inception = self._create_inception_module(in_channels)
        self.residual_blocks = self._create_residual_blocks()
        self.classifier = self._create_classifier(n_classes)

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)
        self.dropout = nn.Dropout(p=0.4)

        self.apply(self._weight_init)

    def forward(self, x):
        x = self._inception_forward(x)
        x = self._residual_blocks_forward(x)
        x = self._classifier_forward(x)
        return x

    @staticmethod
    def _create_inception_module(in_channels):
        return nn.ModuleDict({
            'conv_3x3': nn.Conv3d(1, 128, (3, 3, in_channels), stride=(1, 1, 2), padding=(1, 1, 0)),
            'conv_1x1': nn.Conv3d(1, 128, (1, 1, in_channels), stride=(1, 1, 1), padding=0)
        })

    def _create_residual_blocks(self):
        return nn.ModuleList([
            self._create_residual_block(256, 128),
            self._create_residual_block(128, 128)
        ])

    @staticmethod
    def _create_residual_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (1, 1))
        )

    @staticmethod
    def _create_classifier(n_classes):
        return nn.Sequential(
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, n_classes, (9, 9))
        )

    def _inception_forward(self, x):
        x_3x3 = self.inception['conv_3x3'](x)
        x_1x1 = self.inception['conv_1x1'](x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        x = torch.squeeze(x)
        return F.relu(self.lrn1(x))

    def _residual_blocks_forward(self, x):
        x = self.residual_blocks[0](x)
        x = F.relu(self.lrn2(x))
        for block in self.residual_blocks[1:]:
            x = F.relu(x + block(x))
        return x

    def _classifier_forward(self, x):
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i < len(self.classifier) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x.squeeze(2).squeeze(2)

    @staticmethod
    def _weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)
