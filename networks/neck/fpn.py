import torch.nn as nn
from torch import Tensor


# feature pyramid networks
class FPN(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 1) for in_channels in in_channels_list
            ]
        )
        self.top_down_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in in_channels_list
            ]
        )

        self._out_channels = out_channels
        self._num_depth = len(in_channels_list)

    def forward(self, features: dict[str, Tensor]) -> list[Tensor]:
        feature_names = list(features.keys())
        lateral_outputs = [
            lateral_conv(features[name]) for lateral_conv, name in zip(self.lateral_convs, feature_names)
        ]

        outputs = [self.top_down_convs[-1](lateral_outputs[-1])]
        for i in reversed(range(len(lateral_outputs) - 1)):
            upsampled = nn.functional.interpolate(outputs[0], size=lateral_outputs[i].shape[2:], mode="nearest")
            fused = upsampled + lateral_outputs[i]
            out = self.top_down_convs[i](fused)
            outputs.insert(0, out)
        return outputs

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def num_depth(self) -> int:
        return self._num_depth
