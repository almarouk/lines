from __future__ import annotations
from typing import TYPE_CHECKING, override, Callable, List, Iterator

if TYPE_CHECKING:
    from utils import DATA

import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset, default_collate, Sampler
from torch import Tensor
from torchvision.transforms import v2
from torch.nn import ModuleList
import os
import cv2
import numpy as np
import random
from utils import apply_crop, set_seed

class BoxesDataset:
    _scene_ids = {
        "train": list(range(1, 20)),
        "val": [20]
    }

    def __init__(self, data_path: str, to_sdr: bool, max_distance: float, batch_size: int, num_workers: int = 0):
        super().__init__()
        self._rootpath = data_path
        self._to_sdr = to_sdr
        self._max_distance = max_distance
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._files = {}
        self._init()

    def _init(self):
        for split, scene_ids in self._scene_ids.items():
            split_files = self._files[split] = []
            for id in scene_ids:
                scene_files = {}
                _, _, filenames = next(os.walk(os.path.join(self._rootpath, str(id))))
                for filename in filenames:
                    view_id, _, suffix = os.path.splitext(filename)[0].rpartition('_')
                    view_files = scene_files.get(view_id, None)
                    if not view_files:
                        view_files = scene_files[view_id] = [''] * 2
                    if suffix == "beauty":
                        view_files[0] = os.path.join(str(id), filename)
                    elif suffix == "wireframe":
                        view_files[1] = os.path.join(str(id), filename)
                split_files.extend(scene_files.values())
            split_files.sort()

    def get_data_loader(self, split: str) -> DataLoader:
        assert split in self._scene_ids
        is_train = split == "train"
        # TODO replace list of string by numpy array in case of multiple workers
        # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        return DataLoader(
            dataset=_Dataset(
                files=self._files[split],
                root_path=self._rootpath,
                to_sdr=self._to_sdr,
                max_distance=self._max_distance,
                augment=is_train
            ),
            batch_size=self._batch_size,
            shuffle=is_train,
            drop_last=is_train,
            num_workers=self._num_workers,
            pin_memory=True,
            worker_init_fn=get_worker_init_fn(),
            collate_fn=collate_wrapper,
            # to reduce worker creation overhead, especially on Windows where 'spawn' method
            # is used in contrast with 'fork' method on Linux to create worker processes
            persistent_workers=is_train and self._num_workers > 0
        )
    
    def get_samples(self, split: str, n_samples: int, seed: int) -> DATA:
        assert split in self._scene_ids
        is_train = split == "train"
        loader = DataLoader(
            dataset=_Dataset(
                files=self._files[split],
                root_path=self._rootpath,
                to_sdr=self._to_sdr,
                max_distance=self._max_distance,
                augment=is_train
            ),
            batch_sampler=OneBatchSampler(n_samples, len(self._files[split]), True, seed),
            num_workers=0,
            pin_memory=True,
            worker_init_fn=get_worker_init_fn(seed),
            collate_fn=collate_wrapper,
            persistent_workers=False
        )
        for data in loader:
            break
        return data

class OneBatchSampler(Sampler[List[int]]):

    def __init__(self, n_samples: int, n_data, debug: bool, seed: int) -> None:
        self.n_samples = n_samples
        self.n_data = n_data
        self.debug = debug
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        yield [(int(i), self.debug) for i in torch.randperm(self.n_data, generator=generator)[:self.n_samples]]

    def __len__(self) -> int:
        return 1

def get_worker_init_fn(seed: int | None = None) -> Callable[[int], None]:

    def worker_init_fn(worker_id: int):
        # worker seed is initialized by worker's parent and can be accessed by either of:
        # - torch.initial_seed()
        # - torch.utils.data.get_worker_info().seed
        # PS. 1. note that torch.utils.data.get_worker_info() returns None in main process
        # PS. 2. worker_init_fn will never be called by main process, even if num_workers=0 or 1
        # PS. 3. if persistent_workers=True, worker_init_fn is called only once per worker 
        # for the whole lifespan of the worker
        worker_seed = torch.initial_seed() if worker_init_fn.seed is None else worker_init_fn.seed
        worker_seed = worker_seed % 2 ** 32
        set_seed(worker_seed)
        # or only set seed of numpy and python since pytorch's seed is already set correctly

    fn = worker_init_fn
    fn.seed = seed
    return fn

def collate_wrapper(batch: list[DATA]) -> DATA:
    # custom collate function that doesn't cast non-tensor values to tensors
    batch_collated : DATA = {}
    elem = batch[0]
    for key in elem.keys():
        batch_collated[key] = [d[key] for d in batch]
        if isinstance(elem[key], Tensor):
            batch_collated[key] = default_collate(batch_collated[key])
    return batch_collated

class _Dataset(Dataset):
    def __init__(self,
            files: list[str], root_path: str,
            to_sdr: bool, max_distance: float,
            augment: bool
        ):
        super().__init__()
        self._files = files
        self._root_path = root_path
        self._to_sdr = to_sdr
        self._max_distance = max_distance
        self._augment = augment
        self._init()

    def _init(self):
        # TODO initialise random seed here?
        # TODO make jit compatible
        self.preprocess = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self._augment:
            self.transforms_augment = v2.RandomOrder([
                # v2.RandomApply(ModuleList([
                #     v2.JPEG((5, 50))
                # ]), p=.5),
                v2.RandomApply(ModuleList([
                    v2.ColorJitter(brightness=.3, hue=.2)
                ]), p=.2),
                v2.RandomApply(ModuleList([
                    v2.GaussianBlur(kernel_size=(5, 9), sigma=(.1, 5.))
                ]), p=.5),
                v2.RandomApply(ModuleList([
                    v2.GaussianNoise(mean=0., sigma=0.1)
                ]), p=.2),
                v2.RandomInvert(p=.2),
                v2.RandomAutocontrast(p=.2),
                v2.RandomEqualize(p=.2)
            ])

    @override
    def __getitem__(self, index, debug=False) -> DATA:
        if isinstance(index, tuple):
            debug = index[1]
            index = index[0]
        item : DATA = {}

        # IMREAD_UNCHANGED to read alpha channel
        flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
        filepath_in = os.path.join(self._root_path, self._files[index][0])
        filepath_out = os.path.join(self._root_path, self._files[index][1])
        img_in = cv2.cvtColor(cv2.imread(filepath_in, flags), cv2.COLOR_BGRA2RGBA)
        # line information is saved in alpha channel
        img_out = cv2.imread(filepath_out, flags)[..., -1]

        if debug:
            item['raw_in'] = img_in
            item['raw_out'] = img_out

        # TODO use RandomCrop in data augmentation instead?
        # crop image
        crop_factor = 0.5
        initial_shape = np.array(img_in.shape[:2])
        crop_shape = np.floor(crop_factor * initial_shape).astype(int)
        i, j = np.random.randint(0, np.ceil((1 - crop_factor) * initial_shape))
        crop_box = [i, j, i + crop_shape[0], j + crop_shape[1]]
        if debug:
            item['crop_box'] = crop_box
            img_in, item['raw_in'] = apply_crop(img_in, crop_box, True)
            img_out, item['raw_out'] = apply_crop(img_out, crop_box, True)
        else:
            img_in = apply_crop(img_in, crop_box)
            img_out = apply_crop(img_out, crop_box)

        # add random uniform background and apply alpha blending
        # note that the foreground is pre-multiplied by alpha in EXR format
        n_channels : int = img_in.shape[-1] - 1
        background_color = np.random.random_sample(n_channels).astype(np.float32)
        img_in = img_in[..., :n_channels] + (1 - img_in[..., n_channels:]) * background_color

         # HDR to SDR
        if self._to_sdr:
            img_in = cv2.cvtColor(
                np.clip(
                    cv2.cvtColor(img_in, cv2.COLOR_RGB2HLS),
                    (0, 0, 0), (360, 1, 1),
                    dtype=img_in.dtype
                ),
                cv2.COLOR_HLS2RGB
            )

        # channels-first order
        img_in = np.transpose(img_in, (2, 0, 1))

        # transform data
        img_in = Tensor(img_in)
        if self._augment:
            img_in = self.transforms_augment(img_in)
        if debug:
            item['transformed_in'] = np.transpose(img_in.numpy(), (1, 2, 0))
        img_in : Tensor = self.preprocess(img_in)

        # compute distance field
        img_out = np.round(img_out * 255).astype(np.uint8)
        _, img_out = cv2.threshold(img_out, 0, 255 / 2, cv2.THRESH_BINARY_INV)
        img_out = cv2.distanceTransform(img_out, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _, img_out = cv2.threshold(img_out, self._max_distance, None, cv2.THRESH_TRUNC)
        # img_out = np.clip(img_out, 0, max_distance)
        if debug:
            item['transformed_out'] = img_out
        img_out = Tensor(img_out)

        item.update({
            'tensor_in': img_in,
            'tensor_out': img_out,
            'filepath_in': filepath_in,
            'filepath_out': filepath_out
        })

        return item

    @override
    def __len__(self) -> int:
        return len(self._files)