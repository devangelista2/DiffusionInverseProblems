import torch
from torch import nn
from models.nn.models import UNet
from models.nn._blocks import DownBlock, ResidualBlock, UpBlock


################################
# GENERATOR
class Generator(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        latent_dim: int,
        n_ch: int = 32,
    ) -> None:
        r"""
        input_shape: tuple[int] -> (c, h, w), the shape of input noise
        """
        super(Generator, self).__init__()

        self.input_shape = input_shape
        self.c, self.h, self.w = self.input_shape

        self.latent_dim = latent_dim
        self.n_ch = n_ch

        # Define forward model
        self.layers = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, n_ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_ch * 8),
            nn.ReLU(True),
            # Output: (n_ch * 8) x 4 x 4
            
            nn.ConvTranspose2d(n_ch * 8, n_ch * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_ch * 4),
            nn.ReLU(True),
            # Output: (n_ch * 4) x 7 x 7
            
            nn.ConvTranspose2d(n_ch * 4, n_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_ch * 2),
            nn.ReLU(True),
            # Output: (n_ch * 2) x 14 x 14
            
            nn.ConvTranspose2d(n_ch * 2, self.c, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: img_channels x 28 x 28
        )

    def forward(self, z):
        return self.model(z)


################################
# DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        n_ch: int = 32,
        n_conv_per_level: int = 2,
        L: int = 4,
    ):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        self.c, self.h, self.w = self.input_shape

        self.n_ch = n_ch
        self.n_conv_per_level = n_conv_per_level
        self.L = L

        self.n_channels = [n_ch * 2**l for l in range(L)]

        # Define discriminator layers
        self.preprocess_conv = nn.Conv2d(self.c, self.n_channels[0], kernel_size=1)

        self.down_blocks = nn.ModuleList(
            [DownBlock(c, n_conv_per_level) for c in self.n_channels[:-1]]
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(self.n_channels[-1]) for _ in range(n_conv_per_level)]
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.h * self.w // (2 ** (L + 2)) * self.n_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Preprocess the input
        h = self.preprocess_conv(x)

        # Downscale
        for down_layer in self.down_blocks:
            h, _ = down_layer(h)

        # Apply residual blocks
        for layer in self.residual_blocks:
            h = layer(h)

        # Flatten
        h = self.flatten(h)

        # FC
        h = self.fc(h)
        return self.sigmoid(h)
