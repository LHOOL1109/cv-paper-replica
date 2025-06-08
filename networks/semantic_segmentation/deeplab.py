import torch
import torch.nn as nn
from torch import Tensor

# TODO: multi grid, add decoder (deeplabv3+)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list[int]):
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        )

        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, rate, rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )
        self.convs = nn.ModuleList(modules)

        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.proj = nn.Sequential(
            nn.Conv2d(out_channels * (len(modules) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        results = [conv(x) for conv in self.convs]
        global_pooling = self.image_pooling(x)
        global_pooling = nn.functional.interpolate(global_pooling, (h, w), mode="bilinear", align_corners=False)
        results.append(global_pooling)
        concat = torch.cat(results, dim=1)
        return self.proj(concat)


class DeepLabV3(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_out_channels: int,
                 aspp_out_channels: int, num_classes: int, atrous_rates=[12, 24, 36]):
        super().__init__()
        self.backbone = backbone
        self.neck = ASPP(backbone_out_channels, aspp_out_channels, atrous_rates)
        self.head = nn.Conv2d(aspp_out_channels, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
