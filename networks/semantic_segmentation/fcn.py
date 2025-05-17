from torch import Tensor
import torch.nn as nn
from networks.classification import vgg


def vggnet_forward_collate_fn(backbone_module_dict: nn.ModuleDict, x: Tensor) -> list[Tensor]:
    ret = []
    for layer_name, layer in backbone_module_dict.items():
        x = layer(x)
        if layer_name in [f"pool{i}" for i in (3, 4, 5)]:
            ret.append(x)
    return ret


class FCNHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.pool3_conv = nn.Conv2d(256, num_classes, 1)
        self.pool4_conv = nn.Conv2d(512, num_classes, 1)
        self.pool5_conv = nn.Conv2d(512, num_classes, 1)
        self.upsampling_conv2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1)
        self.upsampling_conv8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4)

    def forward(self, pool3: Tensor, pool4: Tensor, pool5: Tensor) -> Tensor:
        score5 = self.pool5_conv(pool5)
        upsampled_score5 = self.upsampling_conv2x(score5)

        score4 = self.pool4_conv(pool4)
        fuse4_5 = score4 + upsampled_score5
        upsampled_fuse_4_5 = self.upsampling_conv2x(fuse4_5)

        score3 = self.pool3_conv(pool3)
        fuse_3_4_5 = score3 + upsampled_fuse_4_5

        output = self.upsampling_conv8x(fuse_3_4_5)
        return output


class FCN8(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = vgg.VGGNetBackBone(vgg.VGG16C_CONFIG, vggnet_forward_collate_fn)
        self.head = FCNHead(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        pool3, pool4, pool5 = self.backbone(x)
        out = self.head(pool3, pool4, pool5)
        return out
