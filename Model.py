import torch
from torch.nn import Module, Sequential, AvgPool2d, Conv2d, BatchNorm2d, ReLU
import torch.nn.functional as F


class VGGUNet(Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.pool = AvgPool2d(kernel_size=2, stride=2)
        sizes = [size, size * 2, size * 4, size * 8]
        
        # Encoder blocks
        self.block1 = Sequential(
            Conv2d(3, sizes[0], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[0]),
            Conv2d(sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[0]),
        )
        self.block2 = Sequential(
            Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[1]),
            Conv2d(sizes[1], sizes[1], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[1]),
        )
        self.block3 = Sequential(
            Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[2]),
            Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[2]),
        )
        self.block4 = Sequential(
            Conv2d(sizes[2], sizes[3], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[3]),
            Conv2d(sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[3]),
        )

        # Decoder blocks
        self.deblock4 = Sequential(
            Conv2d(sizes[3], sizes[2], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[2]),
            Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[2]),
        )
        self.deblock3 = Sequential(
            Conv2d(sizes[3], sizes[2], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[2]),
            Conv2d(sizes[2], sizes[1], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[1]),
        )
        self.deblock2 = Sequential(
            Conv2d(sizes[2], sizes[1], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[1]),
            Conv2d(sizes[1], sizes[0], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[0]),
        )
        self.deblock1 = Sequential(
            Conv2d(sizes[1], sizes[0], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[0]),
            Conv2d(sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
            ReLU(),
            BatchNorm2d(sizes[0]),
        )

    def forward(self, inputs):
        # Encoding
        features = [self.block1(inputs)]
        for block in [self.block2, self.block3, self.block4]:
            features.append(block(self.pool(features[-1])))

        # Decoding
        out = self.deblock4(features[-1])
        for deblock, feat in zip(
            [self.deblock3, self.deblock2, self.deblock1], features[:-1][::-1]):
            out = deblock(torch.cat([
                F.interpolate(out, feat.shape[2:4], mode='bilinear'),
                feat], dim=1))

        return out

class LineDetector(Module):
    def __init__(self, size: int, max_distance: float, clamp_output: bool):
        super().__init__()
        self.max_distance = max_distance
        self.clamp_output = clamp_output
        self.size = size
        self.base = VGGUNet(size)
        self.head = Sequential(
            Conv2d(size, 1, kernel_size=1),
            ReLU(),
        )

    def forward(self, x):
        preds = self.head(self.base(x)).squeeze(1)
        if self.clamp_output:
            preds = torch.clamp(preds, max=self.max_distance)
        return preds
