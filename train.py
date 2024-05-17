import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from miscellaneous import configurations, data, schedules

# from models.DDIM import DDIM
from models.DiffusionModels.DDIM import DDIM
from models.nn import models

# SET PARAMETERS
CONFIG_PATH = "./configs/SimpleShapes.yml"
TRAIN = True

SAVE_RESULTS = True

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Load the data
x_train, _ = data.load_data(config)
print(x_train[0].shape)

# Define DDIM model
ddim_model = DDIM(config)

# Train if required, else load weigsazhts
if TRAIN:
    ddim_model.train(x_train)
else:
    weights_path = f"./model_weights/{config.data.dataset}_{config.training.loss}.pth"
    ddim_model.model.load_state_dict(torch.load(weights_path))

# Test the model if required
if SAVE_RESULTS:
    ddim_model.test_generation(
        path=f"./results/{config.data.dataset}_{config.training.loss}",
        n_samples=16,
        diffusion_steps=200,
    )
