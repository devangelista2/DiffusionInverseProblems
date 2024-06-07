import matplotlib.pyplot as plt
import numpy as np
import torch

from miscellaneous import configurations, data, utilities
from variational import solvers

# SET PARAMETERS
CONFIG_PATH = "./configs/Mayo128.yml"
GENERATIVE_MODEL = "StyleGANv2" # in {"DDIM", "DCGAN", "StyleGANv2"}

# Select operatore in: "Identity", "GaussianBlur", "Radon"
OPERATOR = "GaussianBlur"
NOISE_LEVEL = 0.02

# SPECIFY OPERATOR SETTINGS (NOT ALL OF THEM ARE REQUIRED FOR ALL THE EXPERIMENTS)
settings = {
    "kernel_size": 3,
    "kernel_variance": 1,
    "angular_range": [0, 180],
    "n_angles": 120,
}

# Reconstructor settings
DIFFUSION_STEPS = 10
LAMBDA = 0

# Regularization parameter
MAXIT = 800
ALPHA = 1e-4  # Step-size for the optimizer
REGULARIZER = "Tik_z"  # in {Tik_z, TV_z, Tik_x, TV_x}

# Load the test image.
# If int -> select the corresponding test set image. If it is a path, it loads the corresponding image.
TEST_IMAGE = 10

###################################################################################
# FROM HERE, DO NOT MODIFY.

# Load config file
config = configurations.load_config(CONFIG_PATH)
BASE_PATH = f"./results/{config.data.dataset}_{config.training.loss}"

# Get operator
settings["shape"] = (config.data.channels, config.data.image_size, config.data.image_size)
K = utilities.get_operator(OPERATOR, settings)

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
y_delta = y + utilities.gaussian_noise(y, NOISE_LEVEL)

# Get Generator
weights_path = f"./model_weights/{GENERATIVE_MODEL}/{config.data.dataset}_{config.training.loss}.pth"
model = utilities.get_model(GENERATIVE_MODEL, config, weights_path=weights_path)
G = model.G

# Define starting point
if GENERATIVE_MODEL == "DCGAN":
    latent_shape = (1, config.DCGAN.latent_dim)
if GENERATIVE_MODEL == "StyleGANv2":
    latent_shape = (1, config.StyleGANv2.z_dim)
elif GENERATIVE_MODEL == "DDIM":
    latent_shape = x_true.shape

# Define starting point
torch.manual_seed(42)
z0 = torch.randn(latent_shape, requires_grad=True, device=config.device)

# Compute solution by GD
GDSolver = solvers.GD(K, G, REGULARIZER, config, optimizer="adam")
z_sol, metrics = GDSolver(
    y_delta,
    lmbda=LAMBDA,
    z0=z0,
    maxit=MAXIT,
    x_true=x_true,
    alpha=ALPHA,
    return_metrics=True,
)

with torch.no_grad():
    x_sol = G(z_sol).cpu().numpy()

# Saving metrics over iterations
for metric_name in metrics.keys():
    np.save(
        f"{BASE_PATH}/{GENERATIVE_MODEL}/{metric_name}_{OPERATOR}_DS_{DIFFUSION_STEPS}_NL_{NOISE_LEVEL}_lmbda_{LAMBDA}_alpha_{ALPHA}.npy",
        metrics[metric_name].detach().numpy(),
    )

# Saving reconstruction
plt.figure(figsize=(25, 9))
plt.subplot(1, 3, 1)
plt.imshow(x_true.detach().cpu().numpy()[0, 0])
plt.gray()
plt.title(r"$x_{true}$", fontsize=20)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(y_delta.detach().cpu().numpy()[0, 0])
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
    f"{BASE_PATH}/{GENERATIVE_MODEL}/recon_{OPERATOR}_DS_{DIFFUSION_STEPS}_NL_{NOISE_LEVEL}_lmbda_{LAMBDA}_alpha_{ALPHA}.png"
)
plt.close()
