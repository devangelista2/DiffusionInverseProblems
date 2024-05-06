import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from miscellaneous import configurations, data, schedules, utilities

# SET PARAMETERS
CONFIG_PATH = "./configs/SimpleCelebA.yml"

# Select operatore in: "Identity", "GaussianBlur"
OPERATOR = "GaussianBlur"
NOISE_LEVEL = 0

# Select starting point in: "zeros", "random"
STARTING_POINT = "random"

# Reconstructor settings
DIFFUSION_STEPS = 50
LAMBDA = 0.01    # Regularization parameter
MAXIT = 300
ALPHA = 0.05     # Step-size for the optimizer

###################################################################################
# FROM HERE, DO NOT MODIFY.
# Load config file
config = configurations.load_config(CONFIG_PATH)
BASE_PATH = f"./results/{config.data.dataset}_{config.training.loss}"

data_fname = glob.glob(f"{BASE_PATH}/*.npy")
data_array = []
for fname in data_fname:
    # Remove the ".npy" at the end
    fname_split = fname.split('.npy')[0]

    # Split by "_"
    fname_split = fname_split.split('_')

    # Get infos
    operator = fname_split[2]
    ds = int(fname_split[fname_split.index('DS') + 1])
    nl = float(fname_split[fname_split.index('NL') + 1])
    lmbda = float(fname_split[fname_split.index('lmbda') + 1])
    alpha = float(fname_split[fname_split.index('alpha') + 1])
    
    if operator == OPERATOR and alpha == ALPHA and nl == NOISE_LEVEL:
        data_array.append([np.load(fname), lmbda, ds])

# Plot
lgnd = []
for infos in data_array:
    lgnd.append(r"$\lambda = $" + f"{str(infos[1])}")
    plt.plot(infos[0])
plt.grid()
plt.xlabel("k")
plt.ylabel("SSIM")
plt.legend(lgnd)
plt.savefig('a.png')
plt.close()

