import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass


@dataclass
class Conv2dConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

    def __iter__(self):
        yield self.in_channels
        yield self.out_channels
        yield self.kernel_size
        yield self.stride
        yield self.padding


@dataclass
class MaxPoolConfig:
    kernel_size: int
    stride: int

    def __iter__(self):
        yield self.kernel_size
        yield self.stride


class ConvModule(nn.Module):
    def __init__(self, conv_configs: list[Conv2dConfig], max_pool_cfg: MaxPoolConfig | None = None):
        super().__init__()
        layers = []
        for cfg in conv_configs:
            layers.append(nn.Conv2d(*cfg))
            layers.append(nn.LeakyReLU(0.1, inplace=True))

        if max_pool_cfg:
            layers.append(nn.MaxPool2d(*max_pool_cfg))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_layers(x)


class YOLOV1BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = ConvModule([Conv2dConfig(3, 64, 7, 2, 3)], MaxPoolConfig(2, 2))
        self.conv_layer2 = ConvModule([Conv2dConfig(64, 192, 3, 1, 1)], MaxPoolConfig(2, 2))
        self.conv_layers3 = ConvModule(
            [
                Conv2dConfig(192, 128, 1, 1, 0),
                Conv2dConfig(128, 256, 3, 1, 1),
                Conv2dConfig(256, 256, 1, 1, 0),
                Conv2dConfig(256, 512, 3, 1, 1),
            ],
            MaxPoolConfig(2, 2)
        )
        self.conv_layers4 = ConvModule(
            [
                Conv2dConfig(512, 256, 1, 1, 0),
                Conv2dConfig(256, 512, 3, 1, 1),
                Conv2dConfig(512, 256, 1, 1, 0),
                Conv2dConfig(256, 512, 3, 1, 1),
                Conv2dConfig(512, 256, 1, 1, 0),
                Conv2dConfig(256, 512, 3, 1, 1),
                Conv2dConfig(512, 256, 1, 1, 0),
                Conv2dConfig(256, 512, 3, 1, 1),
                Conv2dConfig(512, 512, 1, 1, 0),
                Conv2dConfig(512, 1024, 3, 1, 1),
            ],
            MaxPoolConfig(2, 2)
        )
        self.conv_layers5 = ConvModule(
            [
                Conv2dConfig(1024, 512, 1, 1, 0),
                Conv2dConfig(512, 1024, 3, 1, 1),
                Conv2dConfig(1024, 512, 1, 1, 0),
                Conv2dConfig(512, 1024, 3, 1, 1),
                Conv2dConfig(1024, 1024, 3, 1, 1),
                Conv2dConfig(1024, 1024, 3, 2, 1),
            ]
        )
        self.conv_layers6 = ConvModule(
            [
                Conv2dConfig(1024, 1024, 3, 1, 1),
                Conv2dConfig(1024, 1024, 3, 1, 1),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layers3(x)
        x = self.conv_layers4(x)
        x = self.conv_layers5(x)
        x = self.conv_layers6(x)
        return x


class YOLOV1Head(nn.Module):
    def __init__(self, num_classes: int, box_per_grid: int):
        super().__init__()
        self.out_dim = (box_per_grid * 5 + num_classes)
        self.fc_layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(4096, 7 * 7 * (self.out_dim)),
            nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        return x.view(-1, 7, 7, self.out_dim)


class YOLOV1(nn.Module):
    def __init__(self, num_classes: int, box_per_grid: int):
        super().__init__()
        self.backbone = YOLOV1BackBone()
        self.head = YOLOV1Head(num_classes, box_per_grid)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


# class BBoxPredictor(nn.Module):
#     def __init__(self):
#         super().__init__()


# 7, 7, B *5 + C
class YOLOLoss(nn.Module):
    def __init__(self, num_classes: int, box_per_grid: int,
                 lambda_coord: float = 5, lambda_noobj: float = .5,
                 iou_threshold: float | None = None):
        super().__init__()
        self.C = num_classes
        self.B = box_per_grid
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.iou_threshold = iou_threshold
    
    def forward(self, pred: Tensor, gt: Tensor):
        
        pred_boxes, pred_confs, pred_classes = yolo_output_parser(pred, self.C, self.B)
        gt_boxes, gt_confs, gt_classes = yolo_output_parser(gt, self.C, 1)
        obj_mask = gt[..., 4] == 1
        noobj_mask = ~obj_mask


    def _box_loss(self, pred_boxes: Tensor, gt_boxes: Tensor, obj_mask: Tensor):

    
    def _conf_loss(self, pred_conf: Tensor, gt: Tensor, obj_mask: Tensor, noobj_mask: Tensor):
        
    
    def _class_loss(self, pred_class: Tensor, gt: Tensor, obj_mask: Tensor):
        ...
    
    def _get_responsible_box(self, pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
        for (pred_box, gt_box) in zip(pred_boxes, gt_boxes):
            pred_xmin = pred_boxes[..., 0]
            pred_xmin = pred_boxes[..., 0]
            pred_xmin = pred_boxes[..., 0]
            pred_xmin = pred_boxes[..., 0]

    



def yolo_output_parser(output: Tensor, num_classes: int, box_per_grid: int) -> tuple[Tensor, Tensor, Tensor]:
    box_data = output[..., :-num_classes]
    box_data = box_data.view(*output.shape[0:3], box_per_grid, 5)
    boxes = box_data[..., :4]
    confs = box_data[..., 4]
    classes = output[..., -num_classes:]
    return boxes, confs, classes


def yolo_ground_truth_processor():
    ...
