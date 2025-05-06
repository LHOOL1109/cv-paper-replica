import torch
from torch import Tensor
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels_1x1: int,
            out_channels_3x3_reduce: int,
            out_channel_3x3: int,
            out_channel_5x5_reduce: int,
            out_channel_5x5: int,
            out_channel_pool_proj: int,
            ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_1x1, 1, 1, 0),
            nn.ReLU(inplace=True),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_3x3_reduce, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_3x3_reduce, out_channel_3x3, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel_5x5_reduce, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel_5x5_reduce, out_channel_5x5, 5, 1, 2),
            nn.ReLU(inplace=True),
        )
        self.pooling = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channels, out_channel_pool_proj, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        out_conv1 = self.conv1(x)
        out_conv1_3 = self.conv1_3(x)
        out_conv1_5 = self.conv1_5(x)
        out_pooling = self.pooling(x)
        out = torch.cat([out_conv1, out_conv1_3, out_conv1_5, out_pooling], dim=1)
        return out


class GoogLeNetBackBone(nn.Module):
    def __init__(self, with_aux: bool = False):
        super().__init__()
        self.with_aux = with_aux
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            # nn.LocalResponseNorm(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            # nn.LocalResponseNorm(),
        )

        self.layer3 = nn.Sequential(
            InceptionModule(192, 64, 96, 128, 16, 32, 32),
            InceptionModule(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.layer4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.layer4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.layer4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.layer4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.layer4_pooling = nn.MaxPool2d(3, 2, 1)

        self.layer5 = nn.Sequential(
            InceptionModule(832, 256, 160, 320, 32, 128, 128),
            InceptionModule(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7, 1, 0),
        )

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4a(x)
        if self.with_aux:
            aux_out4a = x.clone()
        x = self.layer4b(x)
        x = self.layer4c(x)
        x = self.layer4d(x)
        if self.with_aux:
            aux_out4d = x.clone()
        x = self.layer4e(x)
        x = self.layer4_pooling(x)
        x = self.layer5(x)
        if self.with_aux:
            return aux_out4a, aux_out4d, x
        return x


class GoogLeNetHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc_layer(x)
        return x


class GoogLeNetAuxHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.aux_conv_layer = nn.Sequential(
            nn.AvgPool2d(5, 3),
            nn.Conv2d(in_channels, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
        )
        self.aux_fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.aux_conv_layer(x)
        x = self.aux_fc_layer(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int, with_aux: bool = False):
        super().__init__()
        self.with_aux = with_aux
        self.backbone = GoogLeNetBackBone(self.with_aux)
        self.head = GoogLeNetHead(num_classes)
        if self.with_aux:
            self.aux_head_4a = GoogLeNetAuxHead(512, num_classes)
            self.aux_head_4d = GoogLeNetAuxHead(528, num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        if self.with_aux:
            layer4a_out, layer4d_out, main_out = self.backbone(x)
            aux4a_logits = self.aux_head_4a(layer4a_out)
            aux4d_logits = self.aux_head_4d(layer4d_out)
            main_logits = self.head(main_out)
            return aux4a_logits, aux4d_logits, main_logits
        else:
            main_out = self.backbone(x)
            main_logits = self.head(main_out)
            return main_logits
