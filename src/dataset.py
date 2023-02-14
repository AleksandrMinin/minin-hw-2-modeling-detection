import logging
import typing as tp
from os import path as osp
from collections import OrderedDict
import json

import pandas as pd
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import Config
from src.constants import DF_PATH, TRAIN_IMAGES_PATH
from src.tools import read_rgb_img


class BboxDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: albu.Compose):
        self._df = df
        self._transforms = transforms

    def __getitem__(self, idx: int) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        imp_path = osp.join(TRAIN_IMAGES_PATH, self._df.iloc[idx][0])
        image = read_rgb_img(imp_path)
        bbox = _get_bbox_from_json(self._df.iloc[idx][1])
        mask = _get_mask_from_bbox(bbox, image)
        transformed = self._transforms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = mask.reshape(1,512,512).astype(np.float32)
        return {'image': image, 'mask': mask}

    def __len__(self) -> int:
        return len(self._df)


def _get_bbox_from_json(bbox: str) -> tp.Tuple[float, float, float, float]:
    bbox = bbox.replace('\\,', ', ')
    json_dict = json.loads(bbox)
    x_min, y_min, width, height  = [json_dict[key] for key in json_dict][1:]
    x_max = x_min + width
    y_max = y_min + height
    return (x_min, x_max, y_min, y_max)


def _get_mask_from_bbox(bbox: tp.Tuple[float, float, float, float], 
                        image: np.ndarray) -> np.ndarray:
    x_min, x_max, y_min, y_max = bbox
    h, w = image.shape[:2]
    x_min, x_max = int(x_min * w), int(x_max * w)
    y_min, y_max = int(y_min * h), int(y_max * h)
    mask = np.zeros((h,w),dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


def get_loaders(                                                           # noqa: WPS210
    config: Config,
) -> tp.Tuple[tp.OrderedDict[str, DataLoader], tp.Dict[str, DataLoader]]:  # noqa: WPS221
    train_dataset, valid_dataset, test_dataset = get_datasets(config)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return OrderedDict({"train": train_loader, "valid": valid_loader}), {"infer": test_loader}


def get_datasets(config: Config) -> tp.Tuple[Dataset, Dataset, Dataset]:  # noqa: WPS210
    train_df, valid_df, test_df = _get_dataframes(config)
    
    train_augs = config.train_augmentation
    test_augs = config.val_augmentation
    
    train_dataset = BboxDataset(train_df, transforms=train_augs)
    valid_dataset = BboxDataset(valid_df, transforms=test_augs)
    test_dataset = BboxDataset(test_df, transforms=test_augs)
    
    return train_dataset, valid_dataset, test_dataset


def _get_dataframes(config: Config) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger = logging.getLogger(__name__)
    df = pd.read_csv(DF_PATH, sep='\t')
    train_df, other_df = train_test_split(df, train_size=config.train_size, random_state=42, shuffle=True)
    valid_df, test_df = train_test_split(other_df, train_size=0.5, shuffle=False)
    
    logger.info(f"Train dataset: {len(train_df)}")  # noqa: WPS237
    logger.info(f"Valid dataset: {len(valid_df)}")  # noqa: WPS237
    logger.info(f"Test dataset: {len(test_df)}")    # noqa: WPS237

    return train_df, valid_df, test_df