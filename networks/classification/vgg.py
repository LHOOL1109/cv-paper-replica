from dataclasses import dataclass

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


def build_layers(layer_cfg: list[Conv3 | Conv1 | MaxPool]) -> nn.Sequential:
    layers = []
    for layer_type in layer_cfg:
        if isinstance(layer_type, (Conv1)):
            layer = nn.Conv2d(layer_type.in_channels, layer_type.out_channels, 1, 1, 0)
            layers.append(layer)
            layers.append(nn.ReLU(inplace=True))

        elif isinstance(layer_type, (Conv3)):
            layer = nn.Conv2d(layer_type.in_channels, layer_type.out_channels, 3, 1, 1)
            layers.append(layer)
            layers.append(nn.ReLU(inplace=True))

        elif isinstance(layer_type, (MaxPool)):
            layer = nn.MaxPool2d(2, 2)
            layers.append(layer)

        else:
            raise ValueError

    return nn.Sequential(*layers)


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
        self.backbone = build_layers(layer_cfg)
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
