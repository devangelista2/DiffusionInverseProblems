import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from miscellaneous import configurations, data, schedules
from models.DiffusionModels.DDIM import DDIM
from models.nn import models

# SET PARAMETERS
CONFIG_PATH = "./configs/SimpleShapes.yml"
SAVE_RESULTS = True

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Load trained DDIM model
weights_path = f"./model_weights/{config.data.dataset}_{config.training.loss}.pth"

ddim_model = DDIM(config)
ddim_model.model.load_state_dict(torch.load(weights_path))

# Test the model generation
ddim_model.test_generation(
    path=f"./results/{config.data.dataset}_{config.training.loss}",
    n_samples=16,
    diffusion_steps=200,
)
