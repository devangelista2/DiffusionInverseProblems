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
CONFIG_PATH = "./configs/Mayo128.yml"

# Select operatore in: "Identity", "GaussianBlur"
OPERATOR = "Identity"
NOISE_LEVEL = 0

# Select starting point in: "zeros", "random"
STARTING_POINT = "random"

# Load the test image.
# If int -> select the corresponding test set image. If it is a path, it loads the corresponding image.
TEST_IMAGE = 10

# SPECIFY OPERATOR SETTINGS (NOT ALL OF THEM ARE REQUIRED FOR ALL THE EXPERIMENTS)
kernel_size = 3
kernel_variance = 1

# Reconstructor settings
DIFFUSION_STEPS = 10
LAMBDA = 0  # Regularization parameter
MAXIT = 300
ALPHA = 0.01  # Step-size for the optimizer

# Other parameters
SAVE_RESULT = True


###################################################################################
# FROM HERE, DO NOT MODIFY.

# Load config file
config = configurations.load_config(CONFIG_PATH)
BASE_PATH = f"./results/{config.data.dataset}_{config.training.loss}"

# Get knowledge from config
c, nx, ny = config.data.channels, config.data.image_size, config.data.image_size

# Get operator
if OPERATOR == "GaussianBlur":
    K = operators.GaussianBlur(shape=(c, nx, ny))
elif OPERATOR == "Identity":
    K = operators.Identity(shape=(c, nx, ny))

# Load test image
if isinstance(TEST_IMAGE, int):
    # Load the data
    _, x_test = data.load_data(config)

    # If the data is MNIST, then x_test is a list of tuples.
    # The data element is the 0-th element of the TEST_IMAGE-th
    # element of that tuple
    if config.data.dataset in ["MNIST", "CIFAR10"]:
        x_true = x_test[TEST_IMAGE][0].unsqueeze(0)
    elif config.data.dataset in ["Mayo256", "Mayo128"]:
        x_true = x_test[TEST_IMAGE].unsqueeze(0)
    else:
        x_true = x_test[TEST_IMAGE : TEST_IMAGE + 1]
else:
    x_true = plt.imread(TEST_IMAGE)

# Normalize x_true
x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min())
x_true = x_true.to(config.device)  # Send x_true to device

# Verbose
print(f"Image loaded from {config.data.dataset} dataset. Shape: {x_true.shape}.")

# Generate solution
z_sol = torch.tensor(np.load("z.npy"))
x_sol = (
    utilities.ImageGenerator(config, diffusion_steps=200)(z_sol).detach().cpu().numpy()
)

plt.figure(figsize=(25, 9))
plt.subplot(1, 2, 1)
plt.imshow(x_true.detach().cpu().numpy()[0, 0])
plt.gray()
plt.title(r"$x_{true}$", fontsize=20)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(x_sol[0, 0])
plt.gray()
plt.axis("off")
plt.title(r"$x_{DGP}$", fontsize=20)

plt.show()
