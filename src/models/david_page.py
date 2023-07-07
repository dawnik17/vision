"""
CIFAR 10
INPUT - [3, 32, 32]
"""
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(BasicBlock, self).__init__()
        self.cblock = nn.Sequential(
            *[
                self._get_base_layer(
                    in_channel if i == 0 else out_channel, out_channel, dropout
                )
                for i in range(2)
            ]
        )

    def _get_base_layer(self, in_channel, out_channel, dropout):
        return nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.cblock(x) + x


class DavidPageNet(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], dropout=0.01):
        super(DavidPageNet, self).__init__()
        self.block0 = self._get_base_layer(3, channels[0], pool=False)
        self.block1 = nn.Sequential(
            *[
                self._get_base_layer(channels[0], channels[1]),
                BasicBlock(channels[1], channels[1], dropout),
            ]
        )

        self.block2 = self._get_base_layer(channels[1], channels[2])
        self.block3 = nn.Sequential(
            *[
                self._get_base_layer(channels[2], channels[3]),
                BasicBlock(channels[3], channels[3], dropout),
            ]
        )

        self.logit = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def _get_base_layer(self, in_channel, out_channel, pool=True):
        return nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                stride=1,
                padding=1,
                kernel_size=3,
                bias=False,
                padding_mode="replicate",
            ),
            nn.MaxPool2d(2) if pool else nn.Identity(),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block0(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return self.logit(x)
