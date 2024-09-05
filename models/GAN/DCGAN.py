import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim, image_size, channels):
        super(Generator, self).__init__()

        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim

        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, latent_dim, image_size, channels):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            *self.discriminator_block(channels, 16, bn=False),
            *self.discriminator_block(16, 32),
            *self.discriminator_block(32, 64),
            *self.discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def discriminator_block(self, in_filters, out_filters, bn=True):
        block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class DCGAN(object):
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.image_size = config.data.image_size
        self.channels = config.data.channels
        self.latent_dim = config.DCGAN.latent_dim

        # Initialize generator and discriminator
        self.generator = Generator(self.latent_dim, self.image_size, self.channels).to(self.device)
        self.discriminator = Discriminator(self.latent_dim, self.image_size, self.channels).to(self.device)

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

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
                                       
        # Define the loss function
        adversarial_loss = torch.nn.BCELoss()

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

                # Adversarial ground truths
                true = torch.Tensor(x.shape[0], 1).fill_(1.0).requires_grad_(False).to(self.device)
                fake = torch.Tensor(x.shape[0], 1).fill_(0.0).requires_grad_(False).to(self.device)

                # Configure input
                x_true = x.to(self.device)

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = torch.randn((x.shape[0], self.latent_dim)).to(self.device)

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(self.discriminator(gen_imgs), true)

                # Print out result
                total_gen_loss = total_gen_loss + g_loss.item()

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(self.discriminator(x_true), true)
                fake_loss = adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                # Print out result
                total_disc_loss = total_disc_loss + d_loss.item()

                loop.set_postfix(loss = (round(total_gen_loss / (i+1), 4), round(total_disc_loss / (i+1), 4)))

                d_loss.backward()
                optimizer_D.step()

            # Every 10 epochs, save the model weights
            if (epoch % 10) == 0:
                weights_path = f"./model_weights/DCGAN/{self.config.data.dataset}_{self.config.training.loss}.pth"
                torch.save(self.generator.state_dict(), weights_path)

    def load(self, path):
        self.generator.load_state_dict(torch.load(path))

    def test_generation(self, path, n_samples=16, diffusion_steps=20):
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

    def G(self, z):
        x = self.generator(z)

        # Normalize x
        x = (x - x.min()) / (x.max() - x.min())
        return x