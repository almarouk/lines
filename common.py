from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import HDR_TO_SDR, BACKBONE

from dataset import BoxesDataset
from model import LineDetector

def get_dataset(
        data_path: str,
        to_sdr: HDR_TO_SDR,
        max_distance: float,
        batch_size: int,
        num_workers: int,
        **kwargs
    ) -> BoxesDataset:
    return BoxesDataset(data_path, to_sdr, max_distance, batch_size, num_workers)

def get_model(
        size: int,
        max_distance: float,
        clamp_output: bool,
        backbone: BACKBONE,
        **kwargs
    ) -> LineDetector:
    return LineDetector(size, max_distance, clamp_output, backbone)