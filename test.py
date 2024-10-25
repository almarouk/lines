from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import DATA
    from torch.nn import Module
    from torch import Tensor

import os
# should be placed BEFORE importing opencv
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import sys
import torch
from torch.cuda import is_available as is_torch_cuda_available
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from utils import to_device, str2bool
from common import get_dataset, get_model

def run(args: Namespace) -> None:
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    for k, v in ckpt['args'].items():
        if k not in args:
            setattr(args, k, v)

    args.testing_path = os.path.join(
        args.output_path,
        "testing",
        ckpt['args']['tag'],
        os.path.splitext(os.path.basename(args.ckpt_path))[0],
        args.tag
    )
    os.makedirs(args.testing_path, exist_ok=True)
    max_distance = args.max_distance

    dataset = get_dataset(**vars(args))
    data_loader = dataset.get_data_loader('test')

    model = get_model(**vars(args))
    model.load_state_dict(ckpt['model'])
    device = 'cuda' if is_torch_cuda_available() else 'cpu'
    model = model.to(device, non_blocking=True)
    model.eval()

    offset = 5 # between patches

    data: DATA
    for data in data_loader:
        tensor_in = data['tensor_in']
        n = tensor_in.shape[0]
        tensor_in = tensor_in.reshape(n * 4, *tensor_in.shape[2:])
        with torch.no_grad():
            tensor_in = tensor_in.to(device, non_blocking=True)
            predictions : Tensor = model(tensor_in)
            print(predictions.shape)
            predictions = predictions.reshape(n, 4, *predictions.shape[1:])
            predictions = predictions.detach().cpu().numpy()
        for l in range(n):
            prediction = predictions[l]
            img = np.zeros((
                prediction.shape[1] * 2 + offset,
                prediction.shape[2] * 2 + offset
            ), dtype=np.uint8)
            for k in range(4):
                i = (k // 2) * (prediction.shape[1] + offset)
                j = (k % 2) * (prediction.shape[2] + offset)
                img[i:i+prediction.shape[1], j:j+prediction.shape[2]] = (prediction[k] * 255 / max_distance).astype(np.uint8)
            pred_path = os.path.join(
                args.testing_path,
                os.path.splitext(os.path.basename(data['filepath_out'][l][0]))[0] + ".png"
            )
            cv2.imwrite(pred_path, prediction)

def get_args_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("--data-path", type=str)
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", type=str, default='')
    parser.add_argument("--suppress-exit", type=str2bool, default=False)

    return parser

def process_args(args: Namespace) -> None:
    import time
    args.tag = f"{args.tag}_{time.strftime("%Y%m%d-%H%M%S")}"
    if args.seed is None:
        args.seed = 42 # or torch.initial_seed() % 2 ** 32
    args.batch_size = max(1, args.batch_size // 4)

if __name__ == '__main__':
    # TODO:

    args = None
    try:
        print(sys.argv)
        args = get_args_parser().parse_args()
        process_args(args)
        run(args)
    except:
        if (args is not None and args.suppress_exit) or "--suppress-exit" in sys.argv[1:]:
            import traceback
            traceback.print_exc()
            print("\n---------- The above exception has been suppressed on exit ----------\n")
            sys.exit(0)
        else:
            raise