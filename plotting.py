import os

import matplotlib.pyplot as plt
import numpy as np

from miscellaneous import configurations

# SET PARAMETERS
CONFIG_PATH = "./configs/Mayo128.yml"

# Select operator in: "Identity", "GaussianBlur", "Radon"
OPERATOR = "Radon"
NOISE_LEVEL = 0.03

# Load config file
config = configurations.load_config(CONFIG_PATH)
BASE_PATH = f"./results/{config.data.dataset}_{config.training.loss}_180_120"

# Reconstructor settings (NOTE: The length of all the tuples HAS to be the same).
GENERATIVE_MODEL = ("DDIM", "DDIM", "DDIM", "DDIM") # in {"DDIM", "DCGAN", "StyleGANv2"}
ALPHA = (1e-2, 1e-2, 1e-2, 1e-2)  # Step-size for the optimizer

DIFFUSION_STEPS = (10, 10, 10, 10)
LAMBDATik       = (0, 0, 0, 1e-2)
LAMBDATV        = (0, 1e-1, 1e-2, 0)

METRIC = "SSIM"

# LOAD FILES AND DEFINE LEGEND
metrics = []
legends = []
for i in range(len(DIFFUSION_STEPS)):
    # Append the i-th file to metrics
    fname = f"{METRIC}_{OPERATOR}_DS_{DIFFUSION_STEPS[i]}_NL_{NOISE_LEVEL}_" + \
            f"lmbdaTik_{LAMBDATik[i]}_lmbdaTV_{LAMBDATV[i]}_alpha_{ALPHA[i]}.npy"
    metrics.append(np.load(f"{BASE_PATH}/{GENERATIVE_MODEL[i]}/{fname}"))
    legends.append(r"$\lambda_{Tik}$:" + str(LAMBDATik[i]) + r", $\lambda_{TV}$:" + str(LAMBDATV[i]))
# Convert metrics to array
metrics = np.array(metrics)

# Plot
plt.plot(metrics.T)
plt.legend(legends)
plt.grid()
plt.xlabel("k")
plt.title(f"{METRIC} at " + r"$\delta = $" + str(NOISE_LEVEL))
plt.savefig(f"{METRIC}_NL_{NOISE_LEVEL}.png", dpi=400)
plt.close()

# Verbose
for i in range(len(legends)):
    print(f"{METRIC} {legends[i]}: {metrics[i, -1]}.")