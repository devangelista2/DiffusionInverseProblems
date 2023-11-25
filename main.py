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

SAVE_RESULTS = False

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

# Schedule
alpha = schedules.improved_cosine
alpha_t = alpha(0.02 * torch.ones((6, 1, 1, 1)), config)

# Test reconstruction
x_0 = x_train[1].unsqueeze(0)
eps_t = torch.randn_like(x_0)
x_t = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * eps_t
x_t = x_t.to(config.device)
eps_t = eps_t.to(config.device)
alpha_t = alpha_t.to(config.device)
e_pred = ddim_model.model(x_t, alpha_t)
x_pred = (x_t - (1 - alpha_t.sqrt()) * e_pred) / alpha_t.sqrt()

plt.imsave('a.png', x_t.cpu().detach().numpy()[0, 0], cmap='gray')
plt.imsave('b.png', x_pred.cpu().detach().numpy()[0, 0], cmap='gray')

print(nn.MSELoss()(x_0.cpu(), x_pred.cpu()))

# Generate 6 images and visualize them
#x_T = torch.randn((6, 1, 64, 64)).to(config.device)
# x_0 = ddim_model.reverse_diffusion(x_T, diffusion_steps=40)

# Save image if required
if SAVE_RESULTS:
    plt.imsave('a.png', x_0.cpu().detach().numpy()[0, 0], cmap='gray')