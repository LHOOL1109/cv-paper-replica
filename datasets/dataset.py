import os
import xml.etree.ElementTree as ET

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

FILE_DIR = os.path.dirname(__file__)


class YOLOV1VOCDataset(Dataset):
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    def __init__(self, root: str,
                 img_suffix: str, annot_suffix: str, img_set_txt: str,
                 transform: T.Compose | None = None,
                 grid_dim: tuple[int, int] = (7, 7),
                 ):
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(root, img_suffix)
        self.annot_dir = os.path.join(root, annot_suffix)
        self.num_classes = len(self.VOC_CLASSES)
        self.grid_dim = grid_dim
        with open(img_set_txt) as f:
            self.img_ids = [line.strip().split()[0] for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index) -> tuple[Tensor | Image.Image, Tensor]:
        img_id = self.img_ids[index]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        annot_path = os.path.join(self.annot_dir, f"{img_id}.xml")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self.encode_annotation(annot_path)
        return img, target

    def encode_annotation(self, annot_path: str) -> Tensor:
        target = torch.zeros([*self.grid_dim, self.num_classes + 5])
        annot_tree = ET.parse(annot_path)
        annot_root = annot_tree.getroot()
        img_w = int(annot_root.find("size/width").text)
        img_h = int(annot_root.find("size/height").text)

        for obj in annot_root.iter("object"):
            cls_name = obj.find("name").text
            cls_id = self.VOC_CLASSES.index(cls_name)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) / img_w
            ymin = float(bbox.find("ymin").text) / img_h
            xmax = float(bbox.find("xmax").text) / img_w
            ymax = float(bbox.find("ymax").text) / img_h

            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            box_width = xmax - xmin
            box_height = ymax - ymin

            grid_x = int(cx * self.grid_dim[1])
            grid_y = int(cy * self.grid_dim[0])
            local_x = cx * self.grid_dim[1] - grid_x
            local_y = cy * self.grid_dim[0] - grid_y

            target[grid_y, grid_x, 0: 4] = torch.tensor([local_x, local_y, box_width, box_height])
            target[grid_y, grid_x, 4] = 1
            target[grid_y, grid_x, 5 + cls_id] = 1
        return target


class YOLOV1DataModule(LightningDataModule):
    def __init__(self,
                 root: str, img_suffix: str, annot_suffix: str,
                 train_img_set_txt: str, val_img_set_txt: str,
                 batch_size: int = 16):
        super().__init__()
        self.root = root
        self.img_suffix = img_suffix
        self.annot_suffix = annot_suffix
        self.train_img_set_txt = train_img_set_txt
        self.val_img_set_txt = val_img_set_txt
        self.batch_size = batch_size
        self.transform = T.Compose(
            [
                T.Resize((448, 448)),
                T.ToTensor(),
            ]
        )

    def setup(self, stage):
        self.train_dataset = YOLOV1VOCDataset(
            self.root, self.img_suffix, self.annot_suffix, self.train_img_set_txt, self.transform
        )
        self.val_dataset = YOLOV1VOCDataset(
            self.root, self.img_suffix, self.annot_suffix, self.val_img_set_txt, self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


def get_cifar10_dataloaders(batch_size=64):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.CIFAR10(root=os.path.join(FILE_DIR, "data"), train=True,
                                download=True, transform=transform)
    val_ds = datasets.CIFAR10(root=os.path.join(FILE_DIR, "data"), train=False,
                              download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    return train_loader, val_loader
