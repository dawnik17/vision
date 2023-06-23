import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, inchannel, dropout, number_of_layers=2):
        super(BasicBlock, self).__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=inchannel,
                        out_channels=inchannel,
                        kernel_size=3,
                        stride=1,
                        bias=False,
                        padding=1,
                    ),
                    nn.BatchNorm2d(inchannel),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(self, x):
        identity = x.clone()

        for block in self.blocks:
            x = block(x)
            identity = torch.add(identity, x)

        return identity


class InputBlock(nn.Module):
    def __init__(self, inchannel, outchannel, dropout, number_of_layers=2):
        super(InputBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=inchannel,
                out_channels=outchannel,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1,
            ),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.block2 = BasicBlock(outchannel, dropout, number_of_layers - 1)

    def forward(self, x):
        x = self.block1(x)
        return self.block2(x)


class MNIST13k(nn.Module):
    def __init__(self, dropout=0.0):
        super(MNIST13k, self).__init__()
        self.convb0 = InputBlock(1, 8, dropout, 3)

        self.bottleneck1 = nn.Conv2d(
            8, 8, kernel_size=3, stride=2, bias=False, padding=1
        )
        self.convb1 = BasicBlock(8, dropout, 3)

        self.bottleneck2 = nn.Conv2d(
            8, 16, kernel_size=3, stride=2, bias=False, padding=1
        )
        self.convb2 = BasicBlock(16, dropout, 3)

        self.convb3 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AvgPool2d(5, 5),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.convb0(x)

        x = self.bottleneck1(x)
        x = self.convb1(x)

        x = self.bottleneck2(x)
        x = self.convb2(x)

        return self.convb3(x)


class MNIST8k(nn.Module):
    def __init__(self, dropout=0.0):
        super(MNIST8k, self).__init__()
        input_channels = 8
        input_layers = 2

        b1_channels = 8
        b1_layers = 2
        nb1 = 2

        b2_channels = 8
        b2_layers = 2
        nb2 = 2

        self.convb0 = InputBlock(1, input_channels, dropout, input_layers)

        self.bottleneck1 = nn.Conv2d(
            input_channels, b1_channels, kernel_size=3, stride=2, bias=False, padding=1
        )

        self.convb1 = nn.ModuleList(
            [BasicBlock(b1_channels, dropout, b1_layers) for _ in range(nb1)]
        )

        self.bottleneck2 = nn.Conv2d(
            b1_channels, b2_channels, kernel_size=3, stride=2, bias=False, padding=1
        )

        self.convb2 = nn.ModuleList(
            [BasicBlock(b2_channels, dropout, b2_layers) for _ in range(nb2)]
        )

        self.convb3 = nn.Sequential(
            nn.Conv2d(b2_channels, 8, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(10, 10, kernel_size=1, stride=1, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.convb0(x)
        x = self.bottleneck1(x)

        for conv in self.convb1:
            x = conv(x)

        x = self.bottleneck2(x)

        for conv in self.convb2:
            x = conv(x)

        return self.convb3(x)
