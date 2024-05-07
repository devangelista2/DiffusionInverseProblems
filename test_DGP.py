import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch

from miscellaneous import configurations, data, utilities
from variational import operators, solvers

# SET PARAMETERS
CONFIG_PATH = "./configs/Mayo256.yml"

# Select operatore in: "Identity", "GaussianBlur", "Radon"
OPERATOR = "Radon"
NOISE_LEVEL = 0.01

# Select starting point in: "zeros", "random"
STARTING_POINT = "random"

# Load the test image.
# If int -> select the corresponding test set image. If it is a path, it loads the corresponding image.
TEST_IMAGE = 10

# SPECIFY OPERATOR SETTINGS (NOT ALL OF THEM ARE REQUIRED FOR ALL THE EXPERIMENTS)
kernel_size = 3
kernel_variance = 1

angular_range = [0, 180]
n_angles = 180

# Reconstructor settings
DIFFUSION_STEPS = 10
LAMBDA = 0  # Regularization parameter
MAXIT = 300
ALPHA = 0.01  # Step-size for the optimizer
REGULARIZER = "Tik_z"  # in {Tik_z, TV_z, Tik_x, TV_x}

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
    K = operators.GaussianBlur(
        shape=(c, nx, ny), kernel_size=kernel_size, sigma=kernel_variance
    )
elif OPERATOR == "Radon":
    angles = np.linspace(
        np.deg2rad(angular_range[0]), np.deg2rad(angular_range[1]), n_angles
    )
    K = operators.Radon(input_shape=(1, c, nx, ny), angles=angles, geometry="fanflat")
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

# Compute corrupted data
y = K(x_true)
e = torch.randn_like(y)
y_delta = y + e / torch.norm(e, p="fro") * torch.norm(y, p="fro") * NOISE_LEVEL

# Get Generator
G = utilities.ImageGenerator(config, diffusion_steps=DIFFUSION_STEPS)

# Define starting point
if STARTING_POINT == "zeros":
    x_T = torch.zeros_like(x_true, requires_grad=True)
elif STARTING_POINT == "random":
    x_T = torch.randn_like(x_true, requires_grad=True)

# Compute solution by GD
GDSolver = solvers.GD(K, G, REGULARIZER, config, optimizer="adam")
z_sol, metrics = GDSolver(
    y_delta,
    lmbda=LAMBDA,
    z0=x_T,
    maxit=MAXIT,
    x_true=x_true,
    alpha=ALPHA,
    return_metrics=True,
)

with torch.no_grad():
    x_sol = (
        utilities.ImageGenerator(config, diffusion_steps=10)(z_sol)
        .detach()
        .cpu()
        .numpy()
    )

# Saving
for metric_name in metrics.keys():
    np.save(
        f"{BASE_PATH}/{metric_name}_{OPERATOR}_DS_{DIFFUSION_STEPS}_NL_{NOISE_LEVEL}_lmbda_{LAMBDA}_alpha_{ALPHA}.npy",
        metrics[metric_name].detach().numpy(),
    )

if SAVE_RESULT:
    plt.figure(figsize=(25, 9))
    plt.subplot(1, 3, 1)
    plt.imshow(x_true.detach().cpu().numpy()[0, 0])
    plt.gray()
    plt.title(r"$x_{true}$", fontsize=20)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(x_true.detach().cpu().numpy()[0, 0])
    plt.gray()
    plt.title(r"$y^{\delta}$", fontsize=20)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(x_sol[0, 0])
    plt.gray()
    plt.axis("off")
    plt.title(r"$x_{DGP}$", fontsize=20)

    plt.tight_layout()
    plt.savefig(
        f"{BASE_PATH}/recon_{OPERATOR}_DS_{DIFFUSION_STEPS}_NL_{NOISE_LEVEL}_lmbda_{LAMBDA}_alpha_{ALPHA}.png"
    )
    plt.close()
