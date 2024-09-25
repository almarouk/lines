from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias
if TYPE_CHECKING:
    from torch.nn import Module
    from torch import Tensor
    _DATA : TypeAlias = dict[str, Tensor]

import os
# should be placed BEFORE importing opencv
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.cuda import is_available as is_torch_cuda_available
from Dataset import BoxesDataset
from Model import LineDetector
from argparse import ArgumentParser, Namespace
import cv2

def get_dataset(data_path: str, batch_size: int, max_distance: float, num_workers: int):
    return BoxesDataset(data_path, batch_size, max_distance, num_workers)

def get_model(max_distance: float, clamp_output: bool) -> Module:
    return LineDetector(max_distance, clamp_output)

def to_device(data: _DATA, device: str) -> _DATA:
    return {k: v.to(device, non_blocking=True) for k, v in data.items()}

def run(args: Namespace) -> None:
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    max_distance = ckpt['args']['max_distance']
    clamp_output = ckpt['args']['clamp_output']

    dataset = get_dataset(args.data_path, args.batch_size, max_distance, args.num_workers)
    data_loader = dataset.get_data_loader('val')

    model = get_model(max_distance, clamp_output)
    model.load_state_dict(ckpt['model'])
    device = 'cuda' if is_torch_cuda_available() else 'cpu'
    model = model.to(device, non_blocking=True)
    model.eval()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    data: _DATA
    for data in data_loader:
        images = data['in'].detach().cpu().numpy()
        ground_truths = data['out'].detach().cpu().numpy()
        with torch.no_grad():
            data = to_device(data, device)
            predictions : Tensor = model(data['in'])
            predictions = predictions.detach().cpu().numpy()
        images = images.transpose((0, 2, 3, 1))
        for i in range(len(images)):
            image = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
            ground_truth = ground_truths[i] / max_distance
            prediction = predictions[i] / max_distance
            cv2.imwrite(os.path.join(args.output_path, f"{i:04d}_img.exr"), image)
            cv2.imwrite(os.path.join(args.output_path, f"{i:04d}_gt.png"), ground_truth)
            cv2.imwrite(os.path.join(args.output_path, f"{i:04d}_pred.png"), prediction)

if __name__ == '__main__':
    # TODO:

    parser = ArgumentParser()

    parser.add_argument("--data-path", type=str)
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    run(args)