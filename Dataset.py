from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import os
import cv2
import numpy as np
from typing import override

class BoxesDataset:
    _scene_ids = {
        "train": list(range(1, 20)),
        "val": [20]
    }

    def __init__(self, data_path: str, batch_size: int, max_distance: float, num_workers: int = 0):
        super().__init__()
        self._rootpath = data_path
        self._batch_size = batch_size
        self._max_distance = max_distance
        self._num_workers = num_workers
        self._files_in = {}
        self._files_out = {}
        self._init()

    def _init(self):
        for split, scene_ids in self._scene_ids.items():
            files_in = self._files_in[split] = []
            files_out = self._files_out[split] = []
            for id in scene_ids:
                scene_path = os.path.join(self._rootpath, str(id))
                _, _, filenames = next(os.walk(scene_path))
                for filename in filenames:
                    suffix = os.path.splitext(filename)[0].split('_')[-1]
                    if suffix == "beauty":
                        files_in.append(os.path.join(scene_path, filename))
                    elif suffix == "wireframe":
                        files_out.append(os.path.join(scene_path, filename))

    def get_data_loader(self, split) -> DataLoader:
        assert split in self._scene_ids
        # TODO replace list of string by numpy array in case of multiple workers
        # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        return DataLoader(
            dataset=_Dataset(self._files_in[split], self._files_out[split], self._max_distance),
            batch_size=self._batch_size,
            shuffle=split == "train",
            num_workers=self._num_workers,
            pin_memory=True,
            # TODO
            # worker_init_fn=,
            # collate_fn=,
            # See https://pytorch.org/docs/stable/data.html
        )

class _Dataset(Dataset):
    def __init__(self, files_in: list[str], files_out: list[str], max_distance: float):
        super().__init__()
        assert len(files_in) == len(files_out)
        self._files_in = files_in
        self._files_out = files_out
        self._max_distance = max_distance
        self._init()

    def _init(self):
        # TODO initialise random seed here?
        pass
    
    @override
    def __getitem__(self, index) -> dict[str, Tensor]:
        # IMREAD_UNCHANGED to read alpha channel
        flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
        img_in = cv2.imread(self._files_in[index], flags)
        # line information is saved in alpha channel
        img_out = cv2.imread(self._files_out[index], flags)[..., -1]

        # crop image
        crop_factor = 0.25
        initial_shape = np.array(img_in.shape[:2])
        crop_shape = np.floor(crop_factor * initial_shape).astype(int)
        i, j = np.random.randint(0, np.ceil((1 - crop_factor) * initial_shape))
        img_in = img_in[i:i+crop_shape[0], j:j+crop_shape[1]]
        img_out = img_out[i:i+crop_shape[0], j:j+crop_shape[1]]

        # add random uniform background and apply alpha blending
        # note that the foreground is pre-multiplied by alpha in EXR format
        n_channels = img_in.shape[-1] - 1
        background_color = np.random.random_sample(n_channels).astype(np.float32)
        img_in = img_in[..., :n_channels] + (1 - img_in[..., n_channels:]) * background_color
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        # TODO HDR to SDR?

        # compute distance field
        img_out = np.round(img_out * 255).astype(np.uint8)
        _, img_out = cv2.threshold(img_out, 0, 255 / 2, cv2.THRESH_BINARY_INV)
        img_out = cv2.distanceTransform(img_out, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _, img_out = cv2.threshold(img_out, self._max_distance, None, cv2.THRESH_TRUNC)
        # img_out = np.clip(img_out, 0, max_distance)
        # channels-first order
        img_in = img_in.transpose((2, 0, 1))

        return {'in': Tensor(img_in), 'out': Tensor(img_out)}
        
    
    @override
    def __len__(self) -> int:
        return len(self._files_in)