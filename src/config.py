from datetime import datetime
from functools import partial
import os

import albumentations as albu
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import LambdaLR
from segmentation_models_pytorch.losses import DiceLoss
import cv2

from src.base_config import Config

NUM_CLASSES = 1
N_EPOCHS = 40
BATCH_SIZE = 3
TRAIN_SIZE = 0.8
IMG_HEIGHT = 512
IMG_WIDTH = 512

date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

train_augmentation = albu.Compose(
    [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.HueSaturationValue(
            hue_shift_limit=20, 
            sat_shift_limit=30, 
            val_shift_limit=20, 
            p=0.5
        ),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5,
        ),
        albu.ShiftScaleRotate(),
        albu.GaussianBlur(),
        albu.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        albu.PadIfNeeded(
            min_height=int(1.5*IMG_HEIGHT), 
            min_width=int(1.5*IMG_WIDTH), 
            border_mode=cv2.BORDER_CONSTANT, 
            p=0.3,
        ),
        albu.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    ])


val_augmentation = albu.Compose(
    [
        albu.Resize(height=IMG_HEIGHT, width=IMG_WIDTH), 
    ])

        
config = Config(
    num_workers=6,
    seed=42,
    cls_losses={
        "name": "bce",
        "weight": 0.5,
        "loss_fn": BCEWithLogitsLoss(),
    },
    seg_losses={
        "name": "dice",
        "weight": 1.0,
        "loss_fn": DiceLoss(mode="binary", from_logits=True),
    },
    device="cuda",
    optimizer=torch.optim.AdamW,
    optimizer_kwargs={
        "lr": 1e-3,
        "weight_decay": 1e-5,
    },
    scheduler=LambdaLR,
    scheduler_kwargs={
        "lr_lambda": lambda epoch: 0.9 ** epoch,
    },
    train_size=TRAIN_SIZE,
    img_width=IMG_HEIGHT,
    img_height=IMG_WIDTH,
    train_augmentation=train_augmentation,
    val_augmentation=val_augmentation,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    early_stop_patience=20,
    model_kwargs={"encoder_name": "timm-efficientnet-b5", "encoder_weights": "imagenet"},
    log_metrics=["dice"],
    valid_metric="dice",
    minimize_metric=True,
    project_name="DetectionBbox",
    experiment_name=f"experiment_3_{date_time}",
)
