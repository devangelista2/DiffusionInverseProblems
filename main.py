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

# Generate 6 images and visualize them
x_T = torch.randn((6, 1, 28, 28)).to(config.device)
x_0 = ddim_model.reverse_diffusion(x_T, diffusion_steps=20)

# Save image if required
if SAVE_RESULTS:
    plt.imsave("gen.png", x_0.cpu().detach().numpy()[0, 0], cmap="gray")
