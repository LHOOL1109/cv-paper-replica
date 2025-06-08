from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


# pyramid pooling module
class PPMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_size: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class PPM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None,
                 pooling_size: list[int] = [1, 2, 3, 6]):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels // len(pooling_size)
        self.ppm = nn.ModuleList(
            [
               PPMBlock(in_channels, out_channels, size) for size in pooling_size
            ]
        )

        self._extra_out_channels = out_channels * len(pooling_size)

    def forward(self, feature: Tensor) -> Tensor:
        h, w = feature.shape[2:]
        output = [feature]
        for pool in self.ppm:
            out = pool(feature)
            out = nn.functional.interpolate(out, (h, w), mode="bilinear", align_corners=False)
            output.append(out)
        return torch.concat(output, dim=1)

    @property
    def extra_out_channels(self) -> int:
        return self._extra_out_channels


# pyramid scene parsing network
class PSPNet(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_out_channels: int, num_classes: int,
                 with_aux: bool = False, aux_in_channels: int | None = None):
        super().__init__()
        self.with_aux = with_aux
        self.backbone = backbone
        self.neck = PPM(backbone_out_channels)
        head_in_channels = backbone_out_channels + self.neck.extra_out_channels
        self.head = nn.Sequential(
            nn.Conv2d(head_in_channels, head_in_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_in_channels // 4),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(head_in_channels // 4, num_classes, 1)
        )

        if with_aux:
            assert aux_in_channels is not None
            self.aux_head = nn.Sequential(
                nn.Conv2d(aux_in_channels, aux_in_channels // 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(aux_in_channels // 4),
                nn.ReLU(True),
                nn.Dropout2d(0.1),
                nn.Conv2d(aux_in_channels // 4, num_classes, 1)
            )

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        feature = self.backbone(x)
        if self.training and self.with_aux:
            x, aux = feature
            aux_out = self.aux_head(aux)
        else:
            x = feature if not isinstance(feature, tuple) else feature[0]
            aux_out = None

        x = self.neck(x)
        x = self.head(x)
        return (x, aux_out) if aux_out is not None else x
