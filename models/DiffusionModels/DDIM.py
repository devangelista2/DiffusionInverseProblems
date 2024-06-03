import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from miscellaneous import schedules
from models.nn.ema import EMAHelper
from models.nn.models import ConditionedUNet


class DDIM(object):
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Initialize the schedule
        self.alpha = schedules.improved_cosine

        # Define the model
        self.model = ConditionedUNet(
            config.model.in_ch,
            config.model.n_ch,
            config.model.n_conv_per_level,
            config.model.L,
        ).to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        # Define EMA Model
        self.ema_helper = EMAHelper(mu=self.config.model.ema_rate)
        self.ema_helper.register(self.model)

    def train(self, dataset):
        # Load data
        train_loader = data.DataLoader(
            dataset, batch_size=self.config.training.batch_size, shuffle=True
        )

        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.beta1, 0.999),
            eps=self.config.optimizer.eps,
        )

        # Define the loss function
        if self.config.training.loss.lower() == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif self.config.training.loss.lower() == "mae":
            self.loss_fn = torch.nn.L1Loss()

        # Compute the total number of steps per epoch
        steps_per_epoch = len(dataset) // self.config.training.batch_size

        # Start training
        for epoch in range(self.config.training.n_epochs):
            total_loss = 0
            step = 0
            start_time = time.time()
            print(f"Epoch: {epoch+1}/{self.config.training.n_epochs}:")

            # Batch steps
            for i, x_0 in enumerate(train_loader):
                # Put the model in training mode (for BatchNormalization and Dropout layers)
                self.model.train()

                # Send the data to the device, get random noise realization and get beta/alpha values
                x_0 = x_0.to(self.device)
                eps_t = torch.randn_like(x_0).to(self.device)

                # sample time in the interval [0, 1]
                t = torch.rand((x_0.size(0), 1, 1, 1)).to(self.device)

                # Compute corrupted x
                alpha_t = self.alpha(t, self.config).to(self.device)
                x_t = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * eps_t

                # Recover e from x_t
                eps_pred = self.model(x_t, alpha_t)

                # Compute loss
                loss = self.loss_fn(eps_pred, eps_t)

                # Print out result
                total_loss = total_loss + loss.item()
                time_elapsed = format_seconds(seconds=time.time() - start_time)
                print(
                    f"Iteration: {i+1}/{steps_per_epoch}, Time elapsed: {time_elapsed}, Loss: {total_loss/(i+1):0.4f}.",
                    end="\r",
                )

                # Setp gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.ema_helper.update(self.model)
                data_start = time.time()

            # Every 10 epochs, save the model weights
            if (epoch % 10) == 0:
                weights_path = f"./model_weights/{self.config.data.dataset}_{self.config.training.loss}.pth"
                torch.save(self.model.state_dict(), weights_path)

    def reverse_diffusion(self, x_T, diffusion_steps=None, training=False):
        """
        NOTE: the diffusion steps can be different from the diffusion steps used in training (accellerated diffusion).
              See DDIM paper for details.
        """
        # Put the model in evaluation mode
        self.model.eval()

        # If not training
        if not training:
            # Disable weights gradient memorization to avoid memory issues.
            for param in self.model.parameters():
                param.requires_grad = False

        if diffusion_steps is None:
            diffusion_steps = self.num_timesteps

        # Get the step size (to move backward)
        delta_t = 1 / diffusion_steps

        # Get the batch size
        batch_size = x_T.size(0)

        x_t = x_T.to(self.device)
        for step in range(diffusion_steps):
            # Define the time t
            t = torch.ones((batch_size, 1, 1, 1)) - step * delta_t

            # Get alpha(t) and alpha(t-1)
            alpha_t = self.alpha(t, self.config).to(self.device)
            alpha_t_pred = self.alpha(t - delta_t, self.config).to(self.device)

            # Predict the noise of xt by UNet
            e_pred = self.model(x_t, alpha_t)

            # From the noise, predict x0
            x_pred = (x_t - e_pred * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

            # Compute x_{t-1} by x_0
            x_t = alpha_t_pred.sqrt() * x_pred + (1 - alpha_t_pred).sqrt() * e_pred

        return x_t

    def test_generation(self, path, n_samples=16, diffusion_steps=20):
        """
        NOTE: n must be a perfect square!
        """
        n = int(np.sqrt(n_samples))

        x_T = torch.randn(
            (
                n_samples,
                self.config.data.channels,
                self.config.data.image_size,
                self.config.data.image_size,
            )
        ).to(self.config.device)
        x_0 = self.reverse_diffusion(x_T, diffusion_steps=diffusion_steps)

        # Move x_0 to cpu() and normalize
        x_0 = x_0.cpu().detach().numpy()
        x_0 = (x_0 - x_0.min()) / (x_0.max() - x_0.min())

        # Results
        #### Create results folder if required
        if not os.path.exists(path):
            os.makedirs(path)

        # Save images
        plt.figure()
        for i in range(n_samples):
            plt.subplot(n, n, i + 1)
            if self.config.data.channels == 1:
                plt.imshow(x_0[i, 0], cmap="gray")
            elif self.config.data.channels == 3:
                plt.imshow(np.transpose(x_0[i], axes=(1, 2, 0)))
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"{path}/generation.png",
            dpi=400,
        )
        plt.close()


def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
