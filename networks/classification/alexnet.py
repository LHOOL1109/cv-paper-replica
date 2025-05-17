from torch import Tensor
import torch.nn as nn


class AlexNetBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=2,
                ),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2),
            nn.MaxPool2d(3, 2)
        )
        # layer 2
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2),
            nn.MaxPool2d(3, 2)
        )
        # layer 3
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )
        # layer 4
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True)
        )
        # layer 5
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )

    def forward(self, x: Tensor):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        return x


class AlexNetHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # layer 6
        self.fc_layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # layer 7
        self.fc_layer7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        # layer 8
        self.fc_layer8 = nn.Linear(4096, num_classes)

    def forward(self, x: Tensor):
        x = self.fc_layer6(x)
        x = self.fc_layer7(x)
        x = self.fc_layer8(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.backbone = AlexNetBackBone()
        self.head = AlexNetHead(num_classes=num_classes)

    def forward(self, x: Tensor):
        x = self.backbone(x)
        x = self.head(x)
        return x
