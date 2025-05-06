from datetime import datetime

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from datasets.dataset import get_cifar10_dataloaders
from models.classification.vgg import VGG16D_CONFIG, VGGNet


class LitVGG16D(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits: Tensor = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits: Tensor = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    base_model = VGGNet(VGG16D_CONFIG, 10)
    lit_model = LitVGG16D(base_model)

    wandb_logger = WandbLogger(project="vgg", name="vgg16d")
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        patience=100,
        mode="max",
        verbose=True
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        filename=f"vgg-{now}-{{epoch:02d}}-{{val_acc:.4f}}",
        verbose=True
        )
    trainer = L.Trainer(
        max_epochs=1000,
        precision="16-mixed",
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )

    train_loader, val_loader = get_cifar10_dataloaders()
    trainer.fit(lit_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                )
