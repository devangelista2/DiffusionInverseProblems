import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from miscellaneous import configurations, data, schedules, utilities
from models.DDIM import DDIM
from models.nn import models
from variational import operators, solvers

# SET PARAMETERS
CONFIG_PATH = "./configs/SimpleCelebA.yml"

OPERATOR = "GaussianBlur"
NOISE_LEVEL = 0.01

# Select starting point in: "zeros", "random"
STARTING_POINT = "random"

# Load the test image.
# If int -> select the corresponding test set image. If it is a path, it loads the corresponding image.
TEST_IMAGE = 10

# SPECIFY OPERATOR SETTINGS (NOT ALL OF THEM ARE REQUIRED FOR ALL THE EXPERIMENTS)
kernel_size = 7
kernel_variance = 1


###################################################################################
# FROM HERE, DO NOT MODIFY.

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Get knowledge from config
c, nx, ny = config.data.channels, config.data.image_size, config.data.image_size

# Get operator
if OPERATOR == "GaussianBlur":
    K = operators.GaussianBlur(shape=(c, nx, ny))

# Load test image
if isinstance(TEST_IMAGE, int):
    # Load the data
    x_test, _ = data.load_data(config)
    x_true = x_test[TEST_IMAGE : TEST_IMAGE + 1]
else:
    x_true = plt.imread(TEST_IMAGE)
x_true = x_true.to(config.device)  # Send x_true to device

# Compute corrupted data
y = K(x_true)
e = torch.randn_like(y)
y_delta = y + e / torch.norm(e, p="fro") * torch.norm(y, p="fro") * NOISE_LEVEL

# Get Generator
G = utilities.ImageGenerator(config, diffusion_steps=50)

# Define starting point
if STARTING_POINT == "zeros":
    x_T = torch.zeros_like(x_true, requires_grad=True)
elif STARTING_POINT == "random":
    x_T = torch.randn_like(x_true, requires_grad=True)

# Compute solution by GD
GDSolver = solvers.GD(K, G, config)
z_sol = GDSolver(y_delta, lmbda=1e-6, z0=x_T, maxit=50, x_true=x_true, alpha=1)
x_sol = G(z_sol).detach().cpu().numpy()

plt.imshow(x_sol[0, 0], cmap="gray")
plt.show()
