from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


if __name__ == "__main__":
    from models.object_detection import yolov1
    from datasets.dataset import YOLOV1DataModule

    # torch.set_float32_matmul_precision('high')
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    datamodule = YOLOV1DataModule(
        "./datasets/data/VOCdevkit/VOC2012/",
        "JPEGImages",
        "Annotations",
        "./datasets/data/VOCdevkit/VOC2012/ImageSets/Main/train.txt",
        "./datasets/data/VOCdevkit/VOC2012/ImageSets/Main/val.txt",
        32,
        )
    lit_model = yolov1.YOLOV1LightningModel(conf_thresh=0.05, lr=1e-4)

    wandb_logger = WandbLogger(project="yolov1")
    early_stopping_callback = EarlyStopping(
        monitor="val_map",
        patience=100,
        mode="max",
        verbose=True
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/yolov1",
        monitor="val_map",
        mode="max",
        save_top_k=3,
        filename=f"yolov1-{now}-{{epoch:02d}}-{{val_map:.4f}}",
        verbose=True
        )
    trainer = L.Trainer(
        max_epochs=1000,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )
    trainer.fit(lit_model, datamodule)
