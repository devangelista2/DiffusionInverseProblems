import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from miscellaneous import configurations, data, schedules, utilities
from models.DiffusionModels.DDIM import DDIM
from models.nn import models
from variational import operators, solvers


def update_step(x_t, alpha_t, alpha_t_pred):
    # Send to device
    x_t = x_t.to(device)
    alpha_t = alpha_t.to(device)
    alpha_t_pred = alpha_t_pred.to(device)

    # Predict the noise of xt by UNet
    e_pred = ddim.model(x_t, alpha_t)

    # From the noise, predict x0
    x_pred = (x_t - e_pred * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

    # Compute x_{t-1} by x_0
    return alpha_t_pred.sqrt() * x_pred + (1 - alpha_t_pred).sqrt() * e_pred


# SET PARAMETERS
CONFIG_PATH = "./configs/Mayo256.yml"

# Load config file
config = configurations.load_config(CONFIG_PATH)
BASE_PATH = f"./results/{config.data.dataset}_{config.training.loss}"

# Load trained DDIM Model
weights_path = f"./model_weights/{config.data.dataset}_{config.training.loss}.pth"

ddim = DDIM(config)
ddim.model.load_state_dict(torch.load(weights_path))

# Load test image
TEST_IMAGE = 10
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

# Put the model in evaluation mode
ddim.model.eval()

# Disable weights gradient memorization to avoid memory issues.
for param in ddim.model.parameters():
    param.requires_grad = False

diffusion_steps = 50

# Get the step size (to move backward)
delta_t = 1 / diffusion_steps

# Define starting point
x_T = torch.randn_like(x_true, requires_grad=False)

# Get the batch size
batch_size = x_T.size(0)

# Initialize the schedule
alpha = schedules.improved_cosine

device = "cuda"
x_t = x_T
x_t.requires_grad_(True)
for step in range(diffusion_steps):
    # Verbose
    print(f"Time: {step}.")

    # Define the time t
    t = torch.ones((batch_size, 1, 1, 1)) - step * delta_t

    # Get alpha(t) and alpha(t-1)
    alpha_t = alpha(t, config)
    alpha_t_pred = alpha(t - delta_t, config)

    # Update step
    x_t_new = update_step(x_t, alpha_t, alpha_t_pred)

    # Update gradient
    # x_t_new.backward(x_t.to(device))
    x_t = x_t.to(device).view(-1)
    x_t_new = x_t_new.view(-1)
    grad_t = torch.autograd.grad(
        x_t_new, x_t, x_t, retain_graph=True, create_graph=True, allow_unused=True
    )

    # Compute Jacobian
    jacobian = torch.cat([grad.view(-1) for grad in grad_t])
    print(jacobian.shape)

    # Prepare for next step
    x_t = x_t_new.view((1, 1, 256, 256))
    del x_t_new

    # Reset gradient
    x_t = x_t.detach().requires_grad_(True)
x_t = x_t.cpu().detach().numpy()

plt.imsave("x_gen.png", x_t[0, 0], cmap="gray")
