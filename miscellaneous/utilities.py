from typing import Any

import matplotlib.pyplot as plt
import torch

from models.DDIM import DDIM


class ImageGenerator:
    def __init__(self, config, diffusion_steps=50):
        self.config = config
        self.diffusion_steps = diffusion_steps

        # Load trained DDIM Model
        weights_path = (
            f"./model_weights/{config.data.dataset}_{config.training.loss}.pth"
        )
        self.ddim_model = DDIM(config)
        self.ddim_model.model.load_state_dict(torch.load(weights_path))

        # Disable weights gradient memorization to avoid memory issues.
        for param in self.ddim_model.model.parameters():
            param.requires_grad = False

    def __call__(self, x_T):
        # Send x_T to device
        x_T = x_T.to(self.config.device)

        # Generate x_0
        x_0 = self.ddim_model.reverse_diffusion(
            x_T, diffusion_steps=self.diffusion_steps
        )

        # Normalize x_0
        x_0 = (x_0 - x_0.min()) / (x_0.max() - x_0.min())

        return x_0
