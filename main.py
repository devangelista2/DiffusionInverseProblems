import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from miscellaneous import configurations, data, schedules
from models.DDIM import DDIM
from models.nn import models

# SET PARAMETERS
CONFIG_PATH = "./configs/CIFAR10.yml"
TRAIN = True

SAVE_RESULTS = True

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Load the data
x_train, _ = data.load_data(config)

# Define DDIM model
ddim_model = DDIM(config)

# Train if required, else load weights
if TRAIN:
    ddim_model.train(x_train)
else:
    weights_path = f"./model_weights/{config.data.dataset}_{config.training.loss}.pth"
    ddim_model.model.load_state_dict(torch.load(weights_path))

# Generate 9 images and visualize them
x_T = torch.randn(
    (9, config.data.channels, config.data.image_size, config.data.image_size)
).to(config.device)
x_0 = ddim_model.reverse_diffusion(x_T, diffusion_steps=20)

# Results
#### Create results folder if required
if not os.path.exists(f"./results/{config.data.dataset}_{config.training.loss}"):
    os.makedirs(f"./results/{config.data.dataset}_{config.training.loss}")

# Save image if required
if SAVE_RESULTS:
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_0.cpu().detach().numpy()[i, 0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"./results/{config.data.dataset}_{config.training.loss}/generation.png",
        dpi=400,
    )
    plt.close()
