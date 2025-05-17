from torch import Tensor
import torch.nn as nn
from dataclasses import dataclass
from typing import Type


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super().__init__()

        stride = 2 if downsample else 1
        with_proj = in_channels != out_channels or downsample
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels)
        ) if with_proj else nn.Identity()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x += identity
        return self.relu(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super().__init__()
        stride = 2 if downsample else 1
        with_proj = in_channels != out_channels or downsample
        mid_channels = out_channels // 4

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels)
        ) if with_proj else nn.Identity()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x += identity
        return self.relu(x)


BlockClass = Type[BasicBlock | BottleneckBlock]


@dataclass
class BlockConfig:
    block_type: BlockClass
    in_channels: int
    out_channels: int
    downsample: bool

    def __iter__(self):
        yield self.in_channels
        yield self.out_channels
        yield self.downsample

    def __mul__(self, num: int):
        return [BlockConfig(self.block_type, *self) for _ in range(num)]


class ResNetBackBone(nn.Module):
    def __init__(self, block_configs: list[BlockConfig]):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.residual_blocks = nn.Sequential(*self._build_blocks(block_configs))

    def _build_blocks(self, block_configs: list[BlockConfig]) -> list[nn.Module]:
        blocks = []
        for cfg in block_configs:
            blocks.append(cfg.block_type(*cfg))
        return blocks

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.residual_blocks(x)
        return x


class ResNetHead(nn.Module):
    def __init__(self, num_classes: int, in_features: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc_layer = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block_configs: list[BlockConfig], num_classes: int):
        super().__init__()
        self.backbone = ResNetBackBone(block_configs)
        in_features = block_configs[-1].out_channels
        self.head = ResNetHead(num_classes, in_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


RESNET_18_CONFIG = [
    *BlockConfig(BasicBlock, 64, 64, False) * 2,

    BlockConfig(BasicBlock, 64, 128, True),
    BlockConfig(BasicBlock, 128, 128, False),

    BlockConfig(BasicBlock, 128, 256, True),
    BlockConfig(BasicBlock, 256, 256, False),

    BlockConfig(BasicBlock, 256, 512, True),
    BlockConfig(BasicBlock, 512, 512, False),
]

RESNET_34_CONFIG = [
    *BlockConfig(BasicBlock, 64, 64, False) * 3,

    BlockConfig(BasicBlock, 64, 128, True),
    *BlockConfig(BasicBlock, 128, 128, False) * 3,

    BlockConfig(BasicBlock, 128, 256, True),
    *BlockConfig(BasicBlock, 256, 256, False) * 5,

    BlockConfig(BasicBlock, 256, 512, True),
    *BlockConfig(BasicBlock, 512, 512, False) * 2,
]

RESNET_50_CONFIG = [
    BlockConfig(BottleneckBlock, 64, 256, False),
    *BlockConfig(BottleneckBlock, 256, 256, False) * 2,

    BlockConfig(BottleneckBlock, 256, 512, True),
    *BlockConfig(BottleneckBlock, 512, 512, False) * 3,

    BlockConfig(BottleneckBlock, 512, 1024, True),
    *BlockConfig(BottleneckBlock, 1024, 1024, False) * 5,

    BlockConfig(BottleneckBlock, 1024, 2048, True),
    *BlockConfig(BottleneckBlock, 2048, 2048, False) * 2,
]

RESNET_101_CONFIG = [
    BlockConfig(BottleneckBlock, 64, 256, False),
    *BlockConfig(BottleneckBlock, 256, 256, False) * 2,

    BlockConfig(BottleneckBlock, 256, 512, True),
    *BlockConfig(BottleneckBlock, 512, 512, False) * 3,

    BlockConfig(BottleneckBlock, 512, 1024, True),
    *BlockConfig(BottleneckBlock, 1024, 1024, False) * 22,

    BlockConfig(BottleneckBlock, 1024, 2048, True),
    *BlockConfig(BottleneckBlock, 2048, 2048, False) * 2,
]
RESNET_152_CONFIG = [
    BlockConfig(BottleneckBlock, 64, 256, False),
    *BlockConfig(BottleneckBlock, 256, 256, False) * 2,
    BlockConfig(BottleneckBlock, 256, 512, True),
    *BlockConfig(BottleneckBlock, 512, 512, False) * 7,
    BlockConfig(BottleneckBlock, 512, 1024, True),
    *BlockConfig(BottleneckBlock, 1024, 1024, False) * 35,
    BlockConfig(BottleneckBlock, 1024, 2048, True),
    *BlockConfig(BottleneckBlock, 2048, 2048, False) * 2,
]
