from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    # defined only for type hinting in IDE
    DATA = TypedDict("DATA", {
        "tensor_in": Tensor,
        "tensor_out": Tensor,
        "filepath_in": str | list[str],
        "filepath_out": str | list[str],
        # for debugging:
        'raw_in': np.ndarray | list[np.ndarray],
        'raw_out': np.ndarray | list[np.ndarray],
        'transformed_in': np.ndarray | list[np.ndarray],
        'transformed_out': np.ndarray | list[np.ndarray],
        'crop_box': list[int] | list[list[int]] # (i0, j0, i1, j1)
    })

import random
import numpy as np
import torch
from torch import Tensor

def to_device(data: DATA, device: str) -> DATA:
    return {k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v for k, v in data.items()}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # the following should be done before launching the interpreter
    # os.environ['PYTHONHASHSEED'] = str(3)

def initiate_reproducibility() -> None:
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) # which includes torch.backends.cudnn.deterministic = True

def apply_crop(img: np.ndarray, crop_box: list[int], return_img_w_bbox: bool = False) -> np.ndarray:
    img_cropped = img[crop_box[0]:crop_box[2], crop_box[1]:crop_box[3]]
    if return_img_w_bbox:
        img_w_bbox = img.copy()
        img_w_bbox[crop_box[0]:crop_box[2], crop_box[1]] = 0
        img_w_bbox[crop_box[0]:crop_box[2], crop_box[3]-1] = 0
        img_w_bbox[crop_box[0], crop_box[1]:crop_box[3]] = 0
        img_w_bbox[crop_box[2]-1, crop_box[1]:crop_box[3]] = 0
        return img_cropped, img_w_bbox
    return img_cropped