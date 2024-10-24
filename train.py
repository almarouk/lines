from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import DATA
    from torch.optim.optimizer import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.nn import Module
    from torch.utils.data import DataLoader
    # from torch.nn.modules.loss import _Loss
    from torch import Tensor
    from argparse import Namespace
    from torch.utils.tensorboard import SummaryWriter

import os
# should be placed BEFORE importing opencv
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dataset import BoxesDataset
from model import LineDetector
from loss import Loss
from utils import to_device, set_seed, initiate_reproducibility, HDR_TO_SDR, str2bool, BACKBONE

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD, RMSprop
from torch.cuda import is_available as is_torch_cuda_available
from torch.nn.utils import clip_grad_norm_
from argparse import ArgumentParser
# from pathlib import Path
import sys
import glob
import re
from contextlib import nullcontext
import numpy as np

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

def get_loss_fn(
        loss: str,
        max_distance: float,
        reduction: str = "mean",
        **kwargs
    ) -> Loss:
    return Loss(loss, max_distance, reduction)

def get_optimizer_fn(optimizer: str) -> type[Optimizer]:
    return {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}[optimizer]

def get_scheduler(args: Namespace, optimizer: Optimizer) -> None | LRScheduler:
    scheduler: None | ReduceLROnPlateau
    if args.scheduler == "none":
        scheduler = None
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, factor=args.scheduler_factor, mode='min')
    else:
        sys.exit(f"Unrecognized option for argument scheduler: {args.scheduler}")
    return scheduler

def get_profiler(profiling: bool, tb_path: str):
    if not profiling:
        return nullcontext()

    from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(skip_first=10, wait=0, warmup=3, active=6, repeat=1),
        on_trace_ready=tensorboard_trace_handler(tb_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)

def save_training(
        ckpt_path: str,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        args: Namespace,
        epoch: int,
        global_step: int,
        best_val: float):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": None if scheduler is None else scheduler.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val
    }
    torch.save(ckpt, ckpt_path)

def remove_old_ckpts(args: Namespace, tag: str = ''):
    ckpts_all = glob.glob(f"ckpt{tag}_*.tar", root_dir=args.training_path)
    ckpts_filtered = [re.match(f"ckpt{tag}_(\\d+).tar", ckpt) for ckpt in ckpts_all]
    ckpts_filtered = sorted([(int(x.groups()[0]), x.string) for x in ckpts_filtered if x is not None], key=lambda x: x[0])
    if len(ckpts_filtered) > args.keep_last_ckpts:
        for _, ckpt in ckpts_filtered[:-args.keep_last_ckpts]:
            os.remove(os.path.join(args.training_path, ckpt))

@torch.no_grad()
def do_validation(
        args: Namespace,
        model: Module,
        data_loader: DataLoader,
        device: str) -> float:
    loss_fn = get_loss_fn(**vars(args), reduction="none")
    running_loss: Tensor = 0
    num_samples: int = 0
    training = model.training
    if training:
        model.eval()
    for data in data_loader:
        data = to_device(data, device)
        loss = loss_fn(model(data['tensor_in']), data['tensor_out'])
        running_loss += torch.sum(torch.mean(loss, dim=list(range(1, loss.dim()))))
        num_samples += len(data['tensor_in'])
    running_loss /= num_samples
    if training:
        model.train()
    return running_loss.item()

def main(args: Namespace) -> None:
    set_seed(args.seed)
    initiate_reproducibility(args.reproducible)
    if args.detect_anomaly:
        torch.autograd.detect_anomaly(check_nan=True)

    best_val = float("inf")
    dataset = get_dataset(**vars(args))
    train_loader = dataset.get_data_loader('train')
    val_loader = dataset.get_data_loader('val')
    model = get_model(**vars(args))
    if args.clip_grad_value is not None:
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip_grad_value, args.clip_grad_value))
    loss_fn = get_loss_fn(**vars(args))
    device = 'cuda' if is_torch_cuda_available() else 'cpu'
    model = model.to(device, non_blocking=True)
    optimizer_fn = get_optimizer_fn(args.optimizer)
    optimizer = optimizer_fn(params=model.parameters(), lr=args.lr)
    scheduler = get_scheduler(args, optimizer)

    data: DATA
    model.train()

    # initialize TensorBoard SummaryWriter
    writer : SummaryWriter | None = None
    if args.debug:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.tb_path)
        writer.add_custom_scalars({
            "Loss": {
                "Total Loss": ["Multiline", ["loss/train", "loss/val"]]
            }
        })
        with open(args.trace_dev_path, 'r') as f:
            writer.add_text("dev info", f.read())

        data = to_device(dataset.get_samples("train", 5, args.seed), device)
        writer.add_graph(model, data['tensor_in'])
        writer.add_image("Sample/Train/Raw Input", np.concatenate(data["raw_in"], axis=0), dataformats="HWC")
        writer.add_image("Sample/Train/Raw Output", np.concatenate(data["raw_out"], axis=0), dataformats="HW")
        writer.add_image("Sample/Train/Transformed Input", np.concatenate(data["transformed_in"], axis=0), dataformats="HWC")
        writer.add_image("Sample/Train/Transformed Output", np.concatenate(data["transformed_out"], axis=0), dataformats="HW")
        del data

        data = to_device(dataset.get_samples("val", 5, args.seed), device)
        writer.add_image("Sample/Validation/Raw Input", np.concatenate(data["raw_in"], axis=0), dataformats="HWC")
        writer.add_image("Sample/Validation/Raw Output", np.concatenate(data["raw_out"], axis=0), dataformats="HW")
        writer.add_image("Sample/Validation/Transformed Input", np.concatenate(data["transformed_in"], axis=0), dataformats="HWC")
        writer.add_image("Sample/Validation/Transformed Output", np.concatenate(data["transformed_out"], axis=0), dataformats="HW")
        del data

    # torch.backends.cudnn.benchmark = True

    running_loss = 0.
    global_batch_index = 0
    global_step = 0
    epoch = 0
    data: DATA
    for epoch in range(args.epochs):
        set_seed(args.seed + epoch)
        model.train()
        profiling = args.profiling and epoch == 0
        with get_profiler(profiling, args.tb_path) as prof:
            for data in train_loader:
                optimizer.zero_grad()
                data = to_device(data, device)
                preds = model(data['tensor_in'])
                loss: Tensor = loss_fn(preds, data['tensor_out'])
                loss.backward()
                if args.clip_grad_norm is not None:
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                global_batch_index += 1
                global_step += len(data['tensor_in'])
                if args.debug:
                    running_loss += loss.item()
                    if global_batch_index % args.log_every_iters == 0:
                        writer.add_scalar("loss/train", running_loss / args.log_every_iters, global_step)
                        running_loss = 0.
                del preds, data, loss
                if profiling:
                    prof.step()

        if args.debug and (epoch + 1) % args.val_every_epochs == 0:
            loss_val = do_validation(args, model, val_loader, device)
            writer.add_scalar("loss/val", loss_val, global_step)

            def write_prediction(split: str, tag: str) -> None:
                data = to_device(dataset.get_samples(split, 5, args.seed), device)
                if (epoch + 1) == args.val_every_epochs:
                    writer.add_image(
                        f"Prediction/{tag}/Input",
                        np.concatenate(data["transformed_in"], axis=0),
                        global_step,
                        dataformats="HWC")
                    writer.add_image(
                        f"Prediction/{tag}/Ground Truth",
                        np.concatenate(data["transformed_out"], axis=0),
                        global_step,
                        dataformats="HW")
                with torch.no_grad():
                    training = model.training
                    model.eval()
                    preds = model(data['tensor_in'])
                    model.train(training)
                writer.add_image(
                    f"Prediction/{tag}/Output",
                    torch.cat(list(preds / args.max_distance), dim=0),
                    global_step,
                    dataformats="HW")
                # TODO visualize non-reduced loss as heatmap
                del data

            write_prediction("train", "Train")
            write_prediction("val", "Validation")

            if args.ckpt_best_val and loss_val <= best_val:
                best_val = loss_val
                ckpt_path = os.path.join(args.training_path, f"ckpt_best_{epoch:04d}.tar")
                save_training(
                    ckpt_path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    epoch=epoch,
                    global_step=global_step,
                    best_val=best_val)
                remove_old_ckpts(args, "_best")
        torch.cuda.empty_cache()

        if (epoch + 1) % args.ckpt_every_epochs == 0 or epoch + 1 == args.epochs:
            ckpt_path = os.path.join(args.training_path, f"ckpt_{epoch:04d}.tar")
            save_training(
                ckpt_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_val=best_val)
            remove_old_ckpts(args)

def get_args_parser() -> ArgumentParser:
    parser = ArgumentParser()

    group = parser.add_argument_group("Data")
    group.add_argument("--data-path", type=str, required=True) # Path)
    group.add_argument("--to-sdr", type=str, default=None, choices=[e.value for e in HDR_TO_SDR])
    group.add_argument("--batch-size", type=int, default=8)
    group.add_argument("--num-workers", type=int, default=4)
    group.add_argument("--max-distance", type=float, default=10)

    group = parser.add_argument_group("Model")
    group.add_argument("--backbone", type=str, default=None, choices=[e.value for e in BACKBONE])
    group.add_argument("--clamp-output", type=str2bool, default=False)
    group.add_argument("--size", type=int, default=32)

    group = parser.add_argument_group("Optimizer")
    group.add_argument("--optimizer", type=str, default='adam', choices=["adam", "sgd", "rmsprop"])
    group.add_argument("--learning-rate", dest="lr", type=float, default=0.01)

    group = parser.add_argument_group("Scheduler")
    group.add_argument("--scheduler", type=str, default="none", choices=["none", "ReduceLROnPlateau"])
    group.add_argument("--scheduler-patience", type=int, default=5)
    group.add_argument("--scheduler-factor", type=float, default=0.2)

    group = parser.add_argument_group("Training")
    group.add_argument("--loss", type=str, default="l1", choices=["l1", "l2"])
    group.add_argument("--output-path", type=str, required=True) # Path)
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--clip-grad-norm", type=float, default=None) # default = 1.0
    group.add_argument("--clip-grad-value", type=float, default=None)

    group.add_argument("--val-every-epochs", type=int, default=1)
    group.add_argument("--ckpt-best-val", type=str2bool, default=True)
    group.add_argument("--ckpt-every-epochs", type=int, default=1)
    group.add_argument("--keep-last-ckpts", type=int, default=1)
    group.add_argument("--log-every-iters", type=int, default=10)

    group = parser.add_argument_group("Other")
    group.add_argument("--reproducible", type=str2bool, default=False)
    group.add_argument("--seed", type=int, default=None)
    group.add_argument("--tag", type=str, default='')
    group.add_argument("--debug", type=str2bool, default=False)
    group.add_argument("--detect-anomaly", type=str2bool, default=False)
    group.add_argument("--profiling", type=str2bool, default=False)
    group.add_argument("--suppress-exit", type=str2bool, default=False)

    return parser

def process_args(args: Namespace) -> None:
    # if using Path from pathlib:
    # args.output_path = args.output_path.resolve()
    import time
    args.tag = f"{args.tag}_{time.strftime("%Y%m%d-%H%M%S")}"
    args.training_path = os.path.join(args.output_path, "training", args.tag)
    args.tb_path = os.path.join(args.output_path, "tensorboard", args.tag)
    args.trace_dev_path = os.path.join(args.training_path, "dev_info.txt")
    os.makedirs(args.training_path, exist_ok=True)
    os.makedirs(args.tb_path, exist_ok=True)
    if args.seed is None:
        args.seed = 42 # or torch.initial_seed() % 2 ** 32
    args.to_sdr = HDR_TO_SDR.NO_CONVERSION if args.to_sdr is None else HDR_TO_SDR(args.to_sdr)
    args.backbone = BACKBONE.VGG_UNET if args.backbone is None else BACKBONE(args.backbone)

def trace_dev(args: Namespace) -> None:
    import sys
    import subprocess
    import json
    with open(args.trace_dev_path, 'w') as f:
        f.write("\n\n---------- tag with datetime: ----------\n\n")
        f.write(args.tag)
        f.write("\n\n---------- sys.argv ----------\n\n")
        f.write(" ".join(sys.argv))
        f.write("\n\n---------- args ----------\n\n")
        f.write(json.dumps(vars(args), default=lambda x: str(x), indent=4))
        cwd = os.path.abspath(os.path.dirname(sys.argv[0]))
        f.write("\n\n---------- cwd ----------\n\n")
        f.write(cwd)
        git = f"git -c safe.directory=\"{cwd}\""
        for cmd in [
            "whoami",
            "echo $USER",
            "hostname",
            f"{git} rev-parse HEAD",
            f"{git} status",
            f"{git} log --max-count=10",
            "printenv"
        ]:
            f.write(f"\n\n---------- {cmd} ----------\n\n")
            f.flush()
            subprocess.run(cmd, encoding='ascii', stdout=f, stderr=f, text=True, cwd=cwd, shell=True)
            f.flush()

if __name__ == '__main__':
    # TODO:
    # - add data augmentation
    # - add reproducibility by specifying fixed seeds for random algos
    # - add training resuming support
    # - add finetuning support?
    # - add configuration support
    # - add stop signal handling
    # - add AMP training (Automatic Mixed Precision)
    #   - in this case, watch out for GradScaler
    #     - if gradient clipping or gradient norm clipping is used,
    #       it should be done on the unscaled gradients
    # - log normalized loss relative to max_distance (for comparability)
    # - Tensorboard
    # - Normalize input data (e.g. color values between -1 and 1)
    # - add __all__ to python modules
    # - handle logging to multiple outputs/channels

    args = None
    try:
        args = get_args_parser().parse_args()
        process_args(args)
        trace_dev(args)
        main(args)
    except:
        if (args is not None and args.suppress_exit) or "--suppress-exit" in sys.argv[1:]:
            import traceback
            traceback.print_exc()
            print("\n---------- The above exception has been suppressed on exit ----------\n")
            sys.exit(0)
        else:
            raise