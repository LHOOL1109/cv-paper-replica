from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms
from utils.util import xywh_to_xyxy, get_iou

EPS = 1e-6


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


def yolov1_output_parser(output: Tensor, num_classes: int, box_per_grid: int) -> tuple[Tensor, Tensor, Tensor]:
    box_data = output[..., :-num_classes]
    box_data = box_data.view(*output.shape[0:3], box_per_grid, 5)
    boxes = box_data[..., :4]
    confs = box_data[..., 4]
    classes = output[..., -num_classes:]
    return boxes, confs, classes


def grid_coord_to_global_coord(boxes: Tensor) -> Tensor:
    device = boxes.device
    _, grid_ydim, grid_xdim, *_ = boxes.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_ydim, device=device),
        torch.arange(grid_xdim, device=device),
        indexing="ij"
    )
    grid_x = grid_x.unsqueeze(0).unsqueeze(-1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(-1)

    x_global = (boxes[..., 0] + grid_x) / grid_xdim
    y_global = (boxes[..., 1] + grid_y) / grid_ydim
    ret = torch.stack([x_global, y_global, boxes[..., 2], boxes[..., 3]], dim=-1)
    return ret


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
            nn.Dropout(0.5),
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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)


class YOLOV1(nn.Module):
    def __init__(self, num_classes: int, box_per_grid: int, ):
        super().__init__()
        self.backbone = YOLOV1BackBone()
        self.head = YOLOV1Head(num_classes, box_per_grid)
        self.head.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class YOLOV1Loss(nn.Module):
    def __init__(self, num_classes: int, box_per_grid: int,
                 lambda_coord: float = 5, lambda_noobj: float = .5,
                 ):
        super().__init__()
        self.C = num_classes
        self.B = box_per_grid
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred_boxes, pred_confs, pred_classes = yolov1_output_parser(pred, self.C, self.B)
        gt_boxes, gt_confs, gt_classes = yolov1_output_parser(gt, self.C, 1)
        grid_obj_mask = gt[..., 4] == 1
        res_box_mask = self._get_responsible_box_mask(pred_boxes, gt_boxes)
        box_loss = self._box_loss(pred_boxes, gt_boxes, res_box_mask)
        obj_conf_loss, noobj_conf_loss = self._conf_loss(pred_confs, gt_confs, res_box_mask)
        cls_loss = self._class_loss(pred_classes, gt_classes, grid_obj_mask)
        total_loss = (
            self.lambda_coord * box_loss +
            obj_conf_loss +
            self.lambda_noobj * noobj_conf_loss +
            cls_loss
        )
        return total_loss

    def _box_loss(self, pred_boxes: Tensor, gt_boxes: Tensor, res_box_mask: Tensor) -> Tensor:
        pred_obj_boxes = pred_boxes[res_box_mask]
        gt_obj_boxes = gt_boxes.expand_as(pred_boxes)[res_box_mask]

        xy_loss = ((gt_obj_boxes[..., 0:2] - pred_obj_boxes[..., 0:2])).pow(2).sum()

        pred_wh = pred_obj_boxes[..., 2:].clamp(min=1e-3)
        gt_wh = gt_obj_boxes[..., 2:].clamp(min=1e-3)
        wh_loss = ((gt_wh.sqrt() - pred_wh.sqrt()).pow(2)).sum()
        return xy_loss + wh_loss

    def _conf_loss(self, pred_confs: Tensor, gt_confs: Tensor, res_box_mask: Tensor) -> tuple[Tensor, Tensor]:
        gt_obj_confs = gt_confs.expand_as(pred_confs)[res_box_mask]
        pred_obj_confs = pred_confs[res_box_mask]
        gt_noobj_confs = gt_confs.expand_as(pred_confs)[~res_box_mask]
        pred_noobj_confs = pred_confs[~res_box_mask]
        obj_conf_loss = (gt_obj_confs - pred_obj_confs).pow(2).sum()
        noobj_conf_loss = (gt_noobj_confs - pred_noobj_confs).pow(2).sum()
        return obj_conf_loss, noobj_conf_loss

    def _class_loss(self, pred_classes: Tensor, gt_classes: Tensor, grid_obj_mask: Tensor):
        cls_loss = (gt_classes[grid_obj_mask] - pred_classes[grid_obj_mask]).pow(2).sum()
        return cls_loss

    def _get_responsible_box_mask(self, pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
        global_relative_pred_boxes = grid_coord_to_global_coord(pred_boxes)
        global_relative_gt_boxes = grid_coord_to_global_coord(gt_boxes)

        if global_relative_gt_boxes.shape[3] == 1 and global_relative_pred_boxes.shape[3] > 1:
            global_relative_gt_boxes = global_relative_gt_boxes.expand_as(global_relative_pred_boxes)

        iou = get_iou(global_relative_pred_boxes, global_relative_gt_boxes)
        max_iou_idx = iou.argmax(-1, keepdim=True)

        res_box_mask = torch.zeros_like(iou, dtype=torch.bool)
        res_box_mask.scatter_(-1, max_iou_idx, True)
        return res_box_mask


class YOLOV1LightningModel(L.LightningModule):
    def __init__(self, num_classes: int = 20, box_per_grid: int = 2,
                 lr: float = 1e-3, conf_thresh: float = 0.05, iou_thresh: float = 0.5,
                 lambda_coord: float = 5, lambda_noobj: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.box_per_grid = box_per_grid
        self.model = YOLOV1(num_classes, box_per_grid)
        self.criterion = YOLOV1Loss(num_classes, box_per_grid, lambda_coord, lambda_noobj)
        self.lr = lr
        self.metric = MeanAveragePrecision()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self._nms = nms

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)

        preds = self._parse_predictions(pred, x.shape[-1])
        targets = self._parse_targets(y, x.shape[-1])
        self.metric.update(preds, targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)

    def on_validation_epoch_end(self):
        result = self.metric.compute()
        self.log("val_map", result["map"], prog_bar=True)
        self.metric.reset()

    @torch.no_grad()
    def _parse_predictions(self, pred: Tensor, img_size: int):
        raw = pred[..., : self.box_per_grid * 5].reshape(*pred.shape[:3],
                                                         self.box_per_grid, 5)
        raw_xy = raw[..., :2].sigmoid()
        raw_wh = (raw[..., 2:4].clamp(min=1e-6)) ** 2
        corrected = torch.cat([raw_xy, raw_wh, raw[..., 4:5]], dim=-1)
        pred = torch.cat([corrected.reshape(*pred.shape[:3], -1),
                          pred[..., self.box_per_grid*5:]], dim=-1)
        boxes, confs, cls_probs = yolov1_output_parser(pred, self.num_classes, self.box_per_grid)
        cls_probs = cls_probs.unsqueeze(3).expand(-1, -1, -1, self.box_per_grid, -1)
        global_xywh = grid_coord_to_global_coord(boxes)
        global_xyxy = xywh_to_xyxy(global_xywh) * img_size

        bs = pred.size(0)
        out = []
        for b in range(bs):
            b_xyxy = global_xyxy[b].reshape(-1, 4)
            b_conf = confs[b].reshape(-1)
            b_cls = cls_probs[b].reshape(-1, self.num_classes)

            cls_conf, cls_lbl = b_cls.max(dim=-1)
            scores = b_conf * cls_conf
            mask = scores >= self.conf_thresh
            if not mask.any():
                out.append({"boxes": torch.empty((0, 4), device=pred.device),
                            "scores": torch.empty((0,), device=pred.device),
                            "labels": torch.empty((0,), dtype=torch.int64, device=pred.device)})
                continue
            boxes_f = b_xyxy[mask]
            scores_f = scores[mask]
            labels_f = cls_lbl[mask]
            x1 = torch.min(boxes_f[:, 0], boxes_f[:, 2])
            y1 = torch.min(boxes_f[:, 1], boxes_f[:, 3])
            x2 = torch.max(boxes_f[:, 0], boxes_f[:, 2])
            y2 = torch.max(boxes_f[:, 1], boxes_f[:, 3])
            boxes_f = torch.stack([x1, y1, x2, y2], dim=1)
            boxes_f = boxes_f.clamp(min=0, max=img_size)
            keep = self._nms(boxes_f, scores_f, self.iou_thresh)
            out.append({
                "boxes": boxes_f[keep],
                "scores": scores_f[keep],
                "labels": labels_f[keep]
            })
        return out

    @torch.no_grad()
    def _parse_targets(self, target: Tensor, img_size: int):
        gt_boxes, gt_confs, gt_classes = yolov1_output_parser(target, self.num_classes, 1)
        gt_xywh = grid_coord_to_global_coord(gt_boxes)
        gt_xyxy = xywh_to_xyxy(gt_xywh) * img_size

        bs = target.size(0)
        out = []
        for b in range(bs):
            conf = gt_confs[b].reshape(-1)
            mask = conf == 1
            b_boxes = gt_xyxy[b].reshape(-1, 4)[mask]
            cls_onehot = gt_classes[b].reshape(-1, self.num_classes)[mask]
            _, labels = cls_onehot.max(dim=-1)
            out.append({
                "boxes": b_boxes,
                "labels": labels
            })
        return out
