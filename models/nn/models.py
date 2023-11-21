import math

import torch
import torch.nn as nn

from ._blocks import DownBlock, ResidualBlock, UpBlock
from ._utils import Normalize, get_timestep_embedding, sinusoidal_embedding, swish


class UNet(nn.Module):
    """
    Implements a UNet model that takes as input the image x_t and the value alpha_t to predict the noise epsilon_t such that
    x_t = sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) epsilon_t.
    """

    def __init__(self, in_ch, n_ch, n_conv_per_level, L):
        """
        in_ch: Number of channels for the input images
        n_ch:  Number of channels at the first convolutional layer. After that, it doubles every time a MaxPool is used.
        n_conv_per_level: Number of convolutional layers per level.
        L: Number of levels.
        """
        super().__init__()

        # Infos
        self.in_ch = in_ch
        self.n_ch = n_ch
        self.n_conv_per_level = n_conv_per_level
        self.L = L

        self.n_channels = [n_ch * 2**l for l in range(L)]

        # Layers
        self.preprocess_conv = nn.Conv2d(in_ch, self.n_channels[0], kernel_size=1)
        self.preprocess_conv2 = nn.Conv2d(
            2 * self.n_channels[0], self.n_channels[0], kernel_size=1
        )
        self.down_blocks = nn.ModuleList(
            [DownBlock(n_ch, n_conv_per_level) for n_ch in self.n_channels[:-1]]
        )
        self.up_blocks = nn.ModuleList(
            [UpBlock(n_ch, n_conv_per_level) for n_ch in reversed(self.n_channels[:-1])]
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(self.n_channels[-1]) for _ in range(n_conv_per_level)]
        )
        self.out_conv = nn.Conv2d(self.n_ch, self.in_ch, kernel_size=1, padding="same")

    def forward(self, x_t, alpha_t):
        # Embed alpha_t
        e = sinusoidal_embedding(alpha_t, self.n_ch)
        e = nn.UpsamplingNearest2d(size=x_t.shape[2:])(e)

        # Preprocess x_t to have the right number of channels
        h = self.preprocess_conv(x_t)

        # Concatenate with noise
        h = torch.cat((h, e), dim=1)

        # Process h to have the right number of channels
        h = self.preprocess_conv2(h)

        ## ENCODER
        skips = []
        for down_layer in self.down_blocks:
            h, s = down_layer(h)
            skips.append(s)

        ## BOTTOM
        for layer in self.residual_blocks:
            h = layer(h)

        ## DECODER
        for up_layer in self.up_blocks:
            h = up_layer((h, skips.pop()))

        return self.out_conv(h)
