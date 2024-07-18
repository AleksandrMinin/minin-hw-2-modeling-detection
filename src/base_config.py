import json
import os
import typing as tp
from dataclasses import asdict, dataclass, field

import albumentations as albu
import torch
from torch.optim.optimizer import Optimizer


@dataclass
class Config:
    num_workers: int
    seed: int
    cls_losses: tp.Mapping
    seg_losses: tp.Mapping
    device: str
    optimizer: type(Optimizer)
    optimizer_kwargs: tp.Mapping
    scheduler: tp.Any
    scheduler_kwargs: tp.Mapping
    df_path: str
    train_images_path: str
    train_size: int
    img_width: int
    img_height: int
    train_augmentation: albu.Compose
    val_augmentation: albu.Compose
    batch_size: int
    n_epochs: int
    early_stop_patience: int
    experiment_name: str
    model_kwargs: tp.Mapping
    log_metrics: tp.List[str]
    valid_metric: str
    minimize_metric: bool
    project_name: str
    checkpoints_dir: str = field(init=False)

    def to_dict(self) -> dict:
        res = {}
        for k, v in asdict(self).items():
            try:
                if isinstance(v, torch.nn.Module):
                    res[k] = v.__class__.__name__
                elif isinstance(v, dict):
                    res[k] = json.dumps(v, indent=4)
                else:
                    res[k] = str(v)
            except Exception:
                res[k] = str(v)
        return res

    def __post_init__(self):
        self.checkpoints_dir = os.path.join("./weights", self.experiment_name)  # noqa: WPS601
