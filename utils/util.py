import torch
from torch import Tensor

EPS = 1e-6


def xywh_to_xyxy(coords: Tensor, dim=-1) -> Tensor:
    x, y, w, h = coords.unbind(dim)
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=dim)


def get_iou(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = xywh_to_xyxy(pred_boxes).unbind(-1)
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = xywh_to_xyxy(gt_boxes).unbind(-1)

    inter_xmin = torch.max(pred_xmin, gt_xmin)
    inter_ymin = torch.max(pred_ymin, gt_ymin)
    inter_xmax = torch.min(pred_xmax, gt_xmax)
    inter_ymax = torch.min(pred_ymax, gt_ymax)

    intersection = (inter_xmax - inter_xmin).clamp(0) * (inter_ymax - inter_ymin).clamp(0)
    pred_area = (pred_xmax - pred_xmin).clamp(0) * (pred_ymax - pred_ymin).clamp(0)
    gt_area = (gt_xmax - gt_xmin).clamp(0) * (gt_ymax - gt_ymin).clamp(0)

    union = pred_area + gt_area - intersection
    iou = intersection / union.clamp(EPS)
    return iou
