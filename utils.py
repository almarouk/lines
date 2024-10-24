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
from enum import Enum
from argparse import ArgumentTypeError

class HDR_TO_SDR(Enum):
    CLIP_HSL = "clip_hsl"
    SCALE_MAX = "scale_max"
    NO_CONVERSION = "no_conversion"

class BACKBONE(Enum):
    VGG_UNET = "vgg_unet"
    ATTENTION_UNET = "attention_unet"

def to_device(data: DATA, device: str) -> DATA:
    return {k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v for k, v in data.items()}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # the following should be done before launching the interpreter
    # os.environ['PYTHONHASHSEED'] = str(3)

def initiate_reproducibility(mode: bool = True) -> None:
    torch.backends.cudnn.benchmark = not mode
    torch.use_deterministic_algorithms(mode) # which includes torch.backends.cudnn.deterministic = True

def apply_crop(img: np.ndarray, crop_box: list[int], return_img_w_bbox: bool = False) -> np.ndarray:
    img_cropped = img[crop_box[0]:crop_box[2], crop_box[1]:crop_box[3]]
    if return_img_w_bbox:
        t = 5 # thickness
        c = 0 # color
        n_channels = img.shape[-1]
        img_w_bbox = img.copy()
        if n_channels > 4:
            img_w_bbox = np.expand_dims(img_w_bbox, -1)
            n_channels = 1
        elif n_channels == 4:
            n_channels = 3
            c = (0, 0, 0, 1)
        # img_w_bbox[..., :n_channels] /= np.maximum(img_w_bbox[..., :n_channels].max((-1, -2, -3), keepdims=True), 1)
        i1, j1 = crop_box[0], crop_box[1]
        i11, j11 = max(i1 - t, 0), max(j1 - t, 0)
        i2, j2 = crop_box[2], crop_box[3]
        i22, j22 = min(i2 + t, img_w_bbox.shape[-3]), min(j2 + t, img_w_bbox.shape[-2])
        img_w_bbox[..., i11:i1, j11:j22, :] = c
        img_w_bbox[..., i2:i22, j11:j22, :] = c
        img_w_bbox[..., i11:i22, j11:j1, :] = c
        img_w_bbox[..., i11:i22, j2:j22, :] = c
        if len(img_w_bbox.shape) > len(img.shape):
            img_w_bbox = img_w_bbox.squeeze(-1)
        return img_cropped, img_w_bbox
    return img_cropped

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v = v.lower()
        if v in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise ArgumentTypeError('Boolean value expected.')