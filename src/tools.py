import os
import random
import cv2
import numpy as np
import torch


def read_rgb_img(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image does not exist: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def set_global_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
