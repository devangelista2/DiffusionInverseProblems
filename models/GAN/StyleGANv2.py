import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

from miscellaneous.losses import path_length_regularization, r1_reg

from ._blocks import *


###########################################
# GENERATOR
###########################################
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(GenBlock, self).__init__()
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim)
        self.adain1 = AdaIN(style_dim, out_channels)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim)
        self.adain2 = AdaIN(style_dim, out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, style):
        x = self.upsample(x)
        x = self.lrelu(self.adain1(self.conv1(x, style), style))
        x = self.lrelu(self.adain2(self.conv2(x, style), style))
        return x

# StyleGANv2 Generator
class StyleGANv2Generator(nn.Module):
    def __init__(self, z_dim, w_dim, n_mapping_layers, img_channels):
        super(StyleGANv2Generator, self).__init__()
        self.mapping = MappingNetwork(z_dim, w_dim, n_mapping_layers)
        self.initial_const = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.initial_adain = AdaIN(w_dim, 512)
        self.initial_conv = ModulatedConv2d(512, 512, 3, w_dim)

        self.blocks = nn.ModuleList([
            GenBlock(512, 512, w_dim), # 8x8
            GenBlock(512, 256, w_dim), # 16x16
            GenBlock(256, 128, w_dim), # 32x32
            GenBlock(128, 64, w_dim),  # 64x64
            GenBlock(64, 32, w_dim),   # 128x128
        ])

        self.to_rgb = nn.Conv2d(32, img_channels, 1)

    def forward(self, z):
        w = self.mapping(z)
        x = self.initial_const.expand(z.size(0), -1, -1, -1)
        x = self.initial_adain(x, w)
        x = self.initial_conv(x, w)

        for block in self.blocks:
            x = block(x, w)

        return torch.tanh(self.to_rgb(x))

###########################################
# DISCRIMINATOR
###########################################
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.downsample(x)
        return x

# StyleGANv2 Discriminator
class StyleGANv2Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(StyleGANv2Discriminator, self).__init__()
        self.initial_conv = nn.Conv2d(img_channels, 16, 1)
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(16, 32),   # 128x128 to 64x64
            DiscriminatorBlock(32, 64),   # 64x64 to 32x32
            DiscriminatorBlock(64, 128),  # 32x32 to 16x16
            DiscriminatorBlock(128, 256), # 16x16 to 8x8
            DiscriminatorBlock(256, 512), # 8x8 to 4x4
        ])
        self.final_conv = nn.Conv2d(512, 1, 4) # Output single scalar

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x.view(-1)

###########################################
# STYLEGANv2
###########################################
class StyleGANv2(object):
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.image_size = config.data.image_size
        self.channels = config.data.channels

        self.z_dim = config.StyleGANv2.z_dim
        self.w_dim = config.StyleGANv2.w_dim
        self.n_mapping_layers = config.StyleGANv2.n_mapping_layers
        self.lambda_r1 = config.StyleGANv2.lambda_r1
        self.lambda_pl = config.StyleGANv2.lambda_pl

        # Initialize generator and discriminator
        self.generator = StyleGANv2Generator(self.z_dim, self.w_dim, 
                                             self.n_mapping_layers, 
                                             self.channels).to(self.device)
        self.discriminator = StyleGANv2Discriminator(self.channels).to(self.device)

    def train(self, dataset):
        # Load data
        train_loader = data.DataLoader(
            dataset, batch_size=self.config.training.batch_size, shuffle=True
        )

        # Define the optimizer(s)
        optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                       lr=self.config.optimizer.lr,
                                       betas=(self.config.optimizer.beta1, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                       lr=self.config.optimizer.lr,
                                       betas=(self.config.optimizer.beta1, 0.999))

        # Initialize path length loss
        mean_path_length = 0.0

        # Start training
        for epoch in range(self.config.training.n_epochs):
            total_gen_loss = 0
            total_disc_loss = 0

            # Initialize tqdm
            loop = tqdm(train_loader)
            loop.set_description(f"Epoch: {epoch+1}/{self.config.training.n_epochs} ->")

            # Batch steps
            for i, x in enumerate(loop):
                if self.config.data.dataset == "MNIST":
                    x, _ = x

                # Configure input
                x_true = x.to(self.device)

                # Train Discriminator
                z = torch.randn(x.size(0), self.z_dim).to(self.device)
                x_fake = self.generator(z).detach()

                d_real = self.discriminator(x_true)
                d_fake = self.discriminator(x_fake)

                real_loss = F.softplus(-d_real).mean()  # -log(sigmoid(d_real))
                fake_loss = F.softplus(d_fake).mean()  # -log(sigmoid(1 - d_fake))
                d_loss = real_loss + fake_loss

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # R1 Regularization
                x_true.requires_grad_(True)
                d_real = self.discriminator(x_true)
                r1_loss = r1_reg(d_real, x_true)
                
                optimizer_D.zero_grad()
                (self.lambda_r1 * r1_loss).backward()
                optimizer_D.step()

                # Train Generator
                z = torch.randn(x.size(0), self.z_dim).to(self.device)
                x_fake = self.generator(z)

                g_loss = F.softplus(-self.discriminator(x_fake)).mean()  # -log(sigmoid(d_fake))

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                total_disc_loss = total_disc_loss + d_loss.item()
                total_gen_loss = total_gen_loss + g_loss.item()
                
                # Add to tqdm
                loop.set_postfix(loss = (round(total_gen_loss / (i+1), 4), round(total_disc_loss / (i+1), 4)))

            # Every 10 epochs, save the model weights
            if (epoch % 10) == 0:
                weights_path = f"./model_weights/StyleGANv2/{self.config.data.dataset}_{self.config.training.loss}.pth"
                torch.save(self.generator.state_dict(), weights_path)

                self.test_generation(
                    path=f"./results/{self.config.data.dataset}_{self.config.training.loss}/StyleGANv2",
                    n_samples=16,
                    )

    def load(self, path):
        self.generator.load_state_dict(torch.load(path))

    def test_generation(self, path, n_samples=16, diffusion_steps=20):
            """
            NOTE: n must be a perfect square!
            """
            n = int(np.sqrt(n_samples))

            z = torch.randn((n_samples, self.z_dim)).to(self.device)
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

        # Normalize x
        x = (x - x.min()) / (x.max() - x.min())
        return x
