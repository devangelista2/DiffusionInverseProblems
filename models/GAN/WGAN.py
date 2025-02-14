import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn


# Gradient Penalty Function
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(
        True
    )
    critic_interpolates = critic(interpolates)

    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Generator
class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(WGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


# WGANCritic (instead of Discriminator)
class WGANCritic(nn.Module):
    def __init__(self):
        super(WGANCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LayerNorm([128, 32, 32]),  # Replace BatchNorm with LayerNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LayerNorm([256, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LayerNorm([512, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # No Sigmoid
        )

    def forward(self, x):
        return self.model(x).view(-1)


class WGAN(object):
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.image_size = config.data.image_size
        self.channels = config.data.channels
        self.latent_dim = config.DCGAN.latent_dim

        # Initialize generator and discriminator
        self.generator = WGANGenerator(self.latent_dim).to(self.device)
        self.discriminator = WGANCritic().to(self.device)

    def train(self, dataset):
        pass

    def load(self, path):
        self.generator.load_state_dict(torch.load(path))

    def test_generation(self, path, n_samples=16, *args, **kwargs):
        """
        NOTE: n must be a perfect square!
        """
        n = int(np.sqrt(n_samples))

        z = torch.randn((n_samples, self.latent_dim)).to(self.device)
        x_gen = self.G(z)

        # Move x_gen to cpu() and normalize
        x_gen = x_gen.cpu().detach().numpy()
        x_gen = (x_gen - x_gen.min()) / (x_gen.max() - x_gen.min())

        # Results
        #### Create results folder if required
        if not os.path.exists(path):
            os.makedirs(path)

        # Save images
        plt.figure()
        for i in range(n_samples):
            plt.subplot(n, n, i + 1)
            if self.config.data.channels == 1:
                plt.imshow(x_gen[i, 0], cmap="gray")
            elif self.config.data.channels == 3:
                plt.imshow(np.transpose(x_gen[i], axes=(1, 2, 0)))
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"{path}/generation.png",
            dpi=400,
        )
        plt.close()

    def G(self, z, *args, **kwargs):
        x = self.generator(z)

        with torch.no_grad():
            m = x.min()
            M = x.max()

        # Normalize x
        x = (x - m) / (M - m)
        return x
