import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# PixelNorm Layer
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + epsilon)

# Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, n_mapping_layers):
        super(MappingNetwork, self).__init__()
        layers = [PixelNorm()]
        for _ in range(n_mapping_layers):
            layers.append(spectral_norm(nn.Linear(z_dim, w_dim)))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)

# Adaptive Instance Normalization
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)

    def forward(self, x, style):
        scale = self.style_scale(style).view(-1, x.size(1), 1, 1)
        shift = self.style_shift(style).view(-1, x.size(1), 1, 1)
        return scale * self.norm(x) + shift

# Modulated Convolution Layer
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim):
        super(ModulatedConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.style_scale = nn.Linear(style_dim, in_channels)

    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        scale = self.style_scale(style).view(batch, in_channels, 1, 1)
        x = x * scale
        return self.conv(x)