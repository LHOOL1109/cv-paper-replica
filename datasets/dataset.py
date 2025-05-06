import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os

FILE_DIR = os.path.dirname(__file__)


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
