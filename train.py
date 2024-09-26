from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias
if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.nn import Module
    from torch.utils.data import DataLoader
    # from torch.nn.modules.loss import _Loss
    from torch import Tensor
    _DATA : TypeAlias = dict[str, Tensor]

import os
# should be placed BEFORE importing opencv
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD, RMSprop
from torch.nn import L1Loss, MSELoss
from torch.cuda import is_available as is_torch_cuda_available
from Dataset import BoxesDataset
from Model import LineDetector
from argparse import ArgumentParser, Namespace
# from pathlib import Path
import sys
from time import ctime
import glob
import re

def get_dataset(data_path: str, batch_size: int, max_distance: float, num_workers: int):
    return BoxesDataset(data_path, batch_size, max_distance, num_workers)

def get_model(max_distance: float, clamp_output: bool) -> Module:
    return LineDetector(max_distance, clamp_output)

def get_loss_fn(loss: str) -> Module:
    return {'l1': L1Loss, 'l2': MSELoss}[loss]

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

def to_device(data: _DATA, device: str) -> _DATA:
    return {k: v.to(device, non_blocking=True) for k, v in data.items()}

def save_training(
        ckpt_path: str,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        args: Namespace,
        epoch: int,
        it_total: int,
        best_val: float):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": None if scheduler is None else scheduler.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "it_total": it_total,
        "best_val": best_val
    }
    torch.save(ckpt, ckpt_path)

def remove_old_ckpts(args: Namespace):
    ckpts_all = glob.glob("ckpt_*.tar", root_dir=args.output_path)
    ckpts_filtered = [re.match(r"ckpt_(\d+).tar", ckpt) for ckpt in ckpts_all]
    ckpts_filtered = sorted([int(x.groups()[0]) for x in ckpts_filtered if x is not None])
    if len(ckpts_filtered) > args.keep_last_ckpts:
        for ckpt in ckpts_filtered[:-args.keep_last_ckpts]:
            os.remove(os.path.join(args.output_path, f"ckpt_{ckpt:04d}.tar"))

def do_validation(
        args: Namespace,
        model: Module,
        data_loader: DataLoader,
        device: str) -> float:
    loss_fn = get_loss_fn(args.loss)(reduction='sum')
    loss: Tensor = 0
    num_samples: int = 0
    training = model.training
    if training:
        model.eval()
    for data in data_loader:
        data = to_device(data, device)
        with torch.no_grad():
            loss += loss_fn(model(data['in']), data['out'])
            num_samples += len(data['in'])
    loss /= num_samples
    if training:
        model.train()
    return loss.item()

def do_training(args: Namespace) -> None:
    best_val = float("inf")
    dataset = get_dataset(args.data_path, args.batch_size, args.max_distance, args.num_workers)
    train_loader = dataset.get_data_loader('train')
    val_loader = dataset.get_data_loader('val')
    model = get_model(args.max_distance, args.clamp_output)
    loss_fn = get_loss_fn(args.loss)()
    device = 'cuda' if is_torch_cuda_available() else 'cpu'
    model = model.to(device, non_blocking=True)
    optimizer_fn = get_optimizer_fn(args.optimizer)
    optimizer = optimizer_fn(params=model.parameters(), lr=args.lr)
    scheduler = get_scheduler(args, optimizer)

    epoch = 0
    it_total = 0
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, "log.txt"), "w", buffering=1) as writer:
        while epoch < args.epochs:
            data: _DATA
            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                data = to_device(data, device)
                preds = model(data['in'])
                loss: Tensor = loss_fn(preds, data['out'])
                loss.backward()
                optimizer.step()
                if it_total % args.log_every_iters == 0:
                    writer.write(f"[{ctime()} | epoch {epoch:04d} | iteration {it_total:04d}] train loss {loss.item():.8f}\n")
                del preds, data, loss
                it_total += 1
            epoch += 1

            if epoch % args.val_every_epochs == 0:
                loss_val = do_validation(args, model, val_loader, device)
                writer.write(f"[{ctime()} | epoch {epoch:04d} | iteration {it_total:04d}] val loss {loss_val:.8f}\n")
                if scheduler is not None:
                    scheduler.step(loss_val)
                if args.ckpt_best_val and loss_val <= best_val:
                    ckpts_best_old = glob.glob("ckpt_*_best.tar", root_dir=args.output_path)
                    if len(ckpts_best_old) > 0:
                        for ckpt in ckpts_best_old:
                            os.remove(os.path.join(args.output_path, ckpt))
                    best_val = loss_val
                    ckpt_path = os.path.join(args.output_path, f"ckpt_{epoch:04d}_best.tar")
                    save_training(
                        ckpt_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        epoch=epoch,
                        it_total=it_total,
                        best_val=best_val)
                # torch.cuda.empty_cache()

            if epoch % args.ckpt_every_epochs == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(args.output_path, f"ckpt_{epoch:04d}.tar")
                save_training(
                    ckpt_path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    epoch=epoch,
                    it_total=it_total,
                    best_val=best_val)
                remove_old_ckpts(args)

if __name__ == '__main__':
    # TODO:
    # - add reproducibility by specifying fixed seeds for random algos
    # - add training resuming support
    # - add finetuning support?
    # - add configuration support
    # - add stop signal handling
    # - add AMP training (Automatic Mixed Precision)

    parser = ArgumentParser()

    group = parser.add_argument_group("Data")
    group.add_argument("--data-path", type=str) # Path)
    group.add_argument("--batch-size", type=int, default=16)
    group.add_argument("--num-workers", type=int, default=4)
    group.add_argument("--max-distance", type=float, default=10)

    group = parser.add_argument_group("Optimizer")
    group.add_argument("--optimizer", type=str, default='adam', choices=["adam", "sgd", "rmsprop"])
    group.add_argument("--learning-rate", dest="lr", type=float, default=0.01)

    group = parser.add_argument_group("Scheduler")
    group.add_argument("--scheduler", type=str, default="none", choices=["none", "ReduceLROnPlateau"])
    group.add_argument("--scheduler-patience", type=int, default=5)
    group.add_argument("--scheduler-factor", type=float, default=0.2)

    group = parser.add_argument_group("Training")
    group.add_argument("--loss", type=str, default="l1", choices=["l1", "l2"])
    group.add_argument("--output-path", type=str) # Path)
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--val-every-epochs", type=int, default=1)
    group.add_argument("--ckpt-best-val", type=bool, default=True)
    group.add_argument("--ckpt-every-epochs", type=int, default=1)
    group.add_argument("--keep-last-ckpts", type=int, default=1)
    group.add_argument("--log-every-iters", type=int, default=10)

    group = parser.add_argument_group("Other")
    group.add_argument("--seed", type=int, default=-1)
    group.add_argument("--clamp-output", type=bool, default=False)

    args = parser.parse_args()
    # args.output_path = args.output_path.resolve()

    do_training(args)