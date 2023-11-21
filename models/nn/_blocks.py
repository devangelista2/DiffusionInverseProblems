import torch
import torch.nn as nn

from ._utils import Normalize, get_timestep_embedding, swish


class ResidualBlock(nn.Module):
    """
    Implements a Simple Convolutional Block with Residual Connection between input and output
    """

    def __init__(self, n_ch):
        super().__init__()

        # Infos
        self.n_ch = n_ch

        # Layers
        self.bn = nn.BatchNorm2d(n_ch)
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=3, padding="same")

    def forward(self, x):
        h = x
        h = self.bn(h)
        h = swish(self.conv1(h))
        h = swish(self.conv2(h))
        return x + h


class DownBlock(nn.Module):
    """
    Implements an Encoding level of UNet.
    """

    def __init__(self, n_ch, n_blocks):
        super().__init__()

        # Infos
        self.n_ch = n_ch
        self.n_blocks = n_blocks

        # Layers
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(n_ch) for _ in range(n_blocks)]
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(n_ch, 2 * n_ch, kernel_size=3, padding="same")

    def forward(self, x):
        h = x
        skips = []
        for layer in self.residual_blocks:
            h = layer(h)
            skips.append(h)
        h = self.avg_pool(h)
        h = swish(self.out_conv(h))
        return h, skips


class UpBlock(nn.Module):
    """
    Implements an Encoding level of UNet.
    """

    def __init__(self, n_ch, n_blocks):
        super().__init__()

        # Infos
        self.n_ch = n_ch
        self.n_blocks = n_blocks

        # Layers
        self.in_conv = nn.Conv2d(2 * n_ch, n_ch, kernel_size=3, padding="same")
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_blocks = nn.ModuleList(
            [
                nn.Conv2d(2 * n_ch, n_ch, kernel_size=1, padding="same")
                for _ in range(n_blocks)
            ]
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(n_ch) for _ in range(n_blocks)]
        )

    def forward(self, x):
        h, skips = x
        h = self.upsampling(h)
        h = self.in_conv(h)
        for i in range(self.n_blocks):
            h = torch.cat((h, skips.pop()), dim=1)
            h = self.conv_blocks[i](h)
            h = self.residual_blocks[i](h)
        return h
