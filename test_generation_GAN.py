from miscellaneous import configurations
from models.GAN.DCGAN import Generator

import torch
import matplotlib.pyplot as plt
import numpy as np

# SET PARAMETERS
CONFIG_PATH = "./configs/Mayo128.yml"
SAVE_RESULTS = True

n_samples = 16

# Load config file
config = configurations.load_config(CONFIG_PATH)

# Load trained DDIM model
weights_path = f"./model_weights/GAN/{config.data.dataset}_{config.training.loss}.pth"

# Set device
device = config.device

# Define GAN model
G = Generator(
    input_shape=(config.data.channels, config.data.image_size, config.data.image_size)
).to(device)
G.load_state_dict(torch.load(weights_path))

# Test the model generation
z = torch.randn(
    (n_samples, config.data.channels, config.data.image_size, config.data.image_size),
    device=device,
)
x_gen = G(z).cpu().detach().numpy()

# Save images
plt.figure()
for i in range(n_samples):
    plt.subplot(4, 4, i + 1)
    if config.data.channels == 1:
        plt.imshow(x_gen[i, 0], cmap="gray")
    elif config.data.channels == 3:
        plt.imshow(np.transpose(x_gen[i], axes=(1, 2, 0)))
    plt.axis("off")
plt.tight_layout()
plt.savefig(
    f"./results/{config.data.dataset}_{config.training.loss}/GAN/generation.png",
    dpi=400,
)
plt.close()
