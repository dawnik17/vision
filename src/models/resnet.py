"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.1):
        super(BasicBlock, self).__init__()
        cblock1 = nn.Sequential(
            nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        cblock2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.blocks = nn.Sequential(*[cblock1, cblock2])

        # skip connection
        self.skip = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.blocks(x)
        out += self.skip(x)

        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout=0.1):
        super(Bottleneck, self).__init__()
        cblock1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        cblock2 = nn.Sequential(
            nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        cblock3 = nn.Sequential(
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )

        self.blocks = nn.Sequential(*[cblock1, cblock2, cblock3])

        self.skip = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.blocks(x)
        out += self.skip(x)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_channels=[64, 64, 128, 256, 512],
        nclass=10,
        dropout=0.1,
        logit_layer="pooling",
    ):
        super(ResNet, self).__init__()
        start_planes = num_channels[0]

        self.in_planes = start_planes

        self.cblock1 = nn.Sequential(
            nn.Conv2d(3, start_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(start_planes),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer1 = self._make_layer(block, num_channels[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channels[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels[3], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channels[4], num_blocks[3], stride=2)

        if logit_layer == "linear":
            self.logit_layer = nn.Sequential(
                nn.AvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(num_channels[4] * block.expansion, nclass),
            )

        else:
            self.logit_layer = nn.Sequential(
                nn.Conv2d(self.in_planes, nclass, kernel_size=1),
                nn.AvgPool2d((4, 4)),
                nn.Flatten(),
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.cblock1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.logit_layer(out)
        return out


class ResNet18(ResNet):
    def __init__(self, nclass=10, dropout=0.1, logit_layer="pooling"):
        super(ResNet18, self).__init__(
            block=BasicBlock,
            num_blocks=[2, 2, 2, 2],
            nclass=nclass,
            dropout=dropout,
            logit_layer=logit_layer,
        )


class ResNet34(ResNet):
    def __init__(self, nclass=10, dropout=0.1, logit_layer="pooling"):
        super(ResNet34, self).__init__(
            block=BasicBlock,
            num_blocks=[3, 4, 6, 3],
            nclass=nclass,
            dropout=dropout,
            logit_layer=logit_layer,
        )


class ResNet50(ResNet):
    def __init__(self, nclass=10, dropout=0.1, logit_layer="pooling"):
        super(ResNet50, self).__init__(
            block=Bottleneck,
            num_blocks=[3, 4, 6, 3],
            nclass=nclass,
            dropout=dropout,
            logit_layer=logit_layer,
        )


class ResNet101(ResNet):
    def __init__(self, nclass=10, dropout=0.1, logit_layer="pooling"):
        super(ResNet101, self).__init__(
            block=Bottleneck,
            num_blocks=[3, 4, 23, 3],
            nclass=nclass,
            dropout=dropout,
            logit_layer=logit_layer,
        )


class ResNet152(ResNet):
    def __init__(self, nclass=10, dropout=0.1, logit_layer="pooling"):
        super(ResNet152, self).__init__(
            block=Bottleneck,
            num_blocks=[3, 8, 36, 3],
            nclass=nclass,
            dropout=dropout,
            logit_layer=logit_layer,
        )
