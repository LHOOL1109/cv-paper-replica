import torch
import torch.nn as nn
from torch import Tensor
from networks.transformer import TransformerEncoder


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, d_model: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.num_patches, d_model))

    def forward(self, x: Tensor) -> Tensor:
        # B, C, H, W
        batch_size = x.shape[0]
        x = self.proj(x)  # B, d_model, H / patch_size, W / patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, d_model
        cls = self.cls_token.expand(batch_size, -1, -1)  # B, 1, d_model
        x = torch.cat([cls, x], dim=1)  # B, num_patches + 1, d_model
        x += self.pos_embedding[: x.size(1)]
        return x


class VisionTransformerBackbone(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, d_model: int = 768):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.msa = TransformerEncoder(d_model, 12, 12, d_model * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedding(x)
        x = self.msa(x)
        return x


class VisionTransformerHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        cls_token = x[:, 0]
        return self.classifier(cls_token)


class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int, img_size: int = 224,
                 patch_size: int = 16, in_channels: int = 3, d_model: int = 768):
        super().__init__()
        self.backbone = VisionTransformerBackbone(img_size, patch_size, in_channels, d_model)
        self.head = VisionTransformerHead(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x
