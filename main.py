import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from miscellaneous import configurations, data, schedules
from models.ddim import DDIM
from models.nn import models

# SET PARAMETERS
CONFIG_PATH = "./configs/simple_celeba.yml"
TRAIN = False

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Load the data
x_train = data.ImageDataset(config)

# Define DDIM model
ddim_model = DDIM(config)

# Train if required, else load weights
if TRAIN:
    ddim_model.train(x_train)
else:
    weights_path = f"./model_weights/{config.data.dataset}.pth"
    ddim_model.model.load_state_dict(torch.load(weights_path))

# Generate 6 images and visualize them
x_T = torch.randn((6, 1, 64, 64)).to(config.device)
x_0 = ddim_model.reverse_diffusion(x_T, diffusion_steps=20)

plt.figure(figsize=(10, 15))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_0.cpu().detach().numpy()[i, 0], cmap="gray")
plt.show()
