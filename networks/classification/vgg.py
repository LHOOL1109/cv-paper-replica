from dataclasses import dataclass
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class Conv3:
    in_channels: int
    out_channels: int


@dataclass
class Conv1:
    in_channels: int
    out_channels: int


@dataclass
class MaxPool:
    pass


VGG11A_CONFIG = [
    Conv3(3, 64),
    MaxPool(),
    Conv3(64, 128),
    MaxPool(),
    Conv3(128, 256), Conv3(256, 256),
    MaxPool(),
    Conv3(256, 512), Conv3(512, 512),
    MaxPool(),
    Conv3(512, 512), Conv3(512, 512),
    MaxPool()
    ]

VGG13B_CONFIG = [
    Conv3(3, 64), Conv3(64, 64),
    MaxPool(),
    Conv3(64, 128), Conv3(128, 128),
    MaxPool(),
    Conv3(128, 256), Conv3(256, 256),
    MaxPool(),
    Conv3(256, 512), Conv3(512, 512),
    MaxPool(),
    Conv3(512, 512), Conv3(512, 512),
    MaxPool()
    ]

VGG16C_CONFIG = [
    Conv3(3, 64), Conv3(64, 64),
    MaxPool(),
    Conv3(64, 128), Conv3(128, 128),
    MaxPool(),
    Conv3(128, 256), Conv3(256, 256), Conv1(256, 256),
    MaxPool(),
    Conv3(256, 512), Conv3(512, 512), Conv1(512, 512),
    MaxPool(),
    Conv3(512, 512), Conv3(512, 512), Conv1(512, 512),
    MaxPool()
    ]

VGG16D_CONFIG = [
    Conv3(3, 64), Conv3(64, 64),
    MaxPool(),
    Conv3(64, 128), Conv3(128, 128),
    MaxPool(),
    Conv3(128, 256), Conv3(256, 256), Conv3(256, 256),
    MaxPool(),
    Conv3(256, 512), Conv3(512, 512), Conv3(512, 512),
    MaxPool(),
    Conv3(512, 512), Conv3(512, 512), Conv3(512, 512),
    MaxPool()
    ]

VGG19E_CONFIG = [
    Conv3(3, 64), Conv3(64, 64),
    MaxPool(),
    Conv3(64, 128), Conv3(128, 128),
    MaxPool(),
    Conv3(128, 256), Conv3(256, 256), Conv3(256, 256), Conv3(256, 256),
    MaxPool(),
    Conv3(256, 512), Conv3(512, 512), Conv3(512, 512), Conv3(512, 512),
    MaxPool(),
    Conv3(512, 512), Conv3(512, 512), Conv3(512, 512), Conv3(512, 512),
    MaxPool()
    ]


class VGGNetBackBone(nn.Module):
    def __init__(self, layer_cfg: list[Conv3 | Conv1 | MaxPool], forward_collate_fn: Callable | None = None):
        super().__init__()
        self.forward_collate_fn = forward_collate_fn
        self.blocks = nn.ModuleDict()
        conv_idx = 1
        pool_idx = 1
        for cfg in layer_cfg:
            if isinstance(cfg, Conv3):
                conv = nn.Conv2d(cfg.in_channels, cfg.out_channels, 3, padding=1)
                self.blocks[f'conv3_{conv_idx}'] = conv
                self.blocks[f'relu3_{conv_idx}'] = nn.ReLU(inplace=True)
                conv_idx += 1
            elif isinstance(cfg, Conv1):
                conv = nn.Conv2d(cfg.in_channels, cfg.out_channels, 1)
                self.blocks[f'conv1_{conv_idx}'] = conv
                self.blocks[f'relu1_{conv_idx}'] = nn.ReLU(inplace=True)
                conv_idx += 1
            elif isinstance(cfg, MaxPool):
                self.blocks[f'pool{pool_idx}'] = nn.MaxPool2d(2, 2)
                pool_idx += 1
            else:
                raise ValueError

    def forward(self, x: Tensor) -> Tensor:
        if self.forward_collate_fn:
            return self.forward_collate_fn(self.blocks, x)
        for layer in self.blocks.values():
            x = layer(x)
        return x


class VGGNetHead(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.fc_layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc_layer3 = nn.Linear(4096, num_classes)

    def forward(self, x) -> Tensor:
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x


class VGGNet(nn.Module):
    def __init__(self,
                 layer_cfg: list[Conv3 | Conv1 | MaxPool],
                 num_classes: int = 1000
                 ):
        super().__init__()
        self.backbone = VGGNetBackBone(layer_cfg)
        self.head = VGGNetHead(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class LitVGG16D(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits: Tensor = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits: Tensor = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
