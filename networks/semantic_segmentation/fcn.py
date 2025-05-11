from torch import Tensor
import torch.nn as nn


class FCNHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        
