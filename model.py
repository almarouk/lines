import torch
from torch.nn import Module, Sequential, AvgPool2d, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Upsample, Sigmoid
import torch.nn.functional as F

from utils import BACKBONE

class AttentionBlock(Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = Sequential(
            Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(F_int),
        )

        self.W_x = Sequential(
            Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(F_int),
        )

        self.psi = Sequential(
            Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(1),
            Sigmoid(),
        )

        self.relu = ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = Sequential(
            Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(ch_out),
            ReLU(),
            Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(ch_out),
            ReLU()
        )

    def forward(self,x):
        return self.conv(x)

class UpConv(Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = Sequential(
            Upsample(scale_factor=2, mode="bilinear"),
            Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(ch_out),
            ReLU(),
        )

    def forward(self, x):
        return self.up(x)

class AttentionUNet(Module):
    def __init__(self, size: int, img_ch: int = 3):
        super().__init__()
        self.size = size
        self.img_ch = img_ch

        # [64, 128, 256, 512, 1024]
        sizes = [size, size * 2, size * 4, size * 8, size * 16]

        self.pool = MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(img_ch, sizes[0])
        self.conv2 = ConvBlock(sizes[0], sizes[1])
        self.conv3 = ConvBlock(sizes[1], sizes[2])
        self.conv4 = ConvBlock(sizes[2], sizes[3])
        self.conv5 = ConvBlock(sizes[3], sizes[4])

        self.up5 = UpConv(sizes[4], sizes[3])
        self.att5 = AttentionBlock(F_g=sizes[3], F_l=sizes[3], F_int=sizes[2])
        self.up_conv5 = ConvBlock(sizes[4], sizes[3])

        self.up4 = UpConv(sizes[3], sizes[2])
        self.att4 = AttentionBlock(F_g=sizes[2], F_l=sizes[2], F_int=sizes[1])
        self.up_conv4 = ConvBlock(sizes[3], sizes[2])

        self.up3 = UpConv(sizes[2], sizes[1])
        self.att3 = AttentionBlock(F_g=sizes[1], F_l=sizes[1], F_int=sizes[0])
        self.up_conv3 = ConvBlock(sizes[2], sizes[1])

        self.up2 = UpConv(sizes[1], sizes[0])
        self.att2 = AttentionBlock(F_g=sizes[0], F_l=sizes[0], F_int=sizes[0] // 2)
        self.up_conv2 = ConvBlock(sizes[1], sizes[0])

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)

        x3 = self.pool(x2)
        x3 = self.conv3(x3)

        x4 = self.pool(x3)
        x4 = self.conv4(x4)

        x5 = self.pool(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        return d2

class VGGUNet(Module):
    def __init__(self, size: int, img_ch: int = 3):
        super().__init__()
        self.size = size
        self.img_ch = img_ch

        sizes = [size, size * 2, size * 4, size * 8]

        self.pool = AvgPool2d(kernel_size=2, stride=2)
        
        # Encoder blocks
        self.block1 = Sequential(
            Conv2d(img_ch, sizes[0], kernel_size=3, stride=1, padding=1),
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
    def __init__(self, size: int, max_distance: float, clamp_output: bool, backbone: BACKBONE):
        super().__init__()
        self.max_distance = max_distance # TODO remove this variable
        self.clamp_output = clamp_output
        self.size = size
        if backbone == BACKBONE.VGG_UNET:
            self.base = VGGUNet(size)
        elif backbone == BACKBONE.ATTENTION_UNET:
            self.base = AttentionUNet(size)
        
        self.head = Sequential(
            Conv2d(size, 1, kernel_size=1),
            BatchNorm2d(1), # TODO check if it's valid to batch norm at the end
            ReLU(),
        )

    def forward(self, x):
        preds = self.head(self.base(x)).squeeze(1)
        if self.clamp_output:
            preds = torch.clamp(preds, max=1)
        return preds
