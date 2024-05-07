import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch

from variational import operators

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}.")

MAXIT = 200
ALPHA = 1

# Define the image
x_true = skimage.data.camera()

# Normalize
x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min())

# Extend its shape
x_true = np.expand_dims(x_true, (0, 1))
N, c, h, w = x_true.shape

# Convert to tensor
x_true = torch.tensor(x_true, dtype=torch.float, device=DEVICE)

# Define the forward operator
angles = np.linspace(0, np.pi, 180)
det_size = int(max(h, w) * np.sqrt(2))

K = operators.Radon(
    input_shape=(N, c, h, w), angles=angles, det_size=det_size, geometry="fanflat"
)

# Compute y
y = K(x_true)

# Define x_0
x_0 = torch.zeros_like(x_true, requires_grad=True, device=DEVICE)

# Cycle
for k in range(MAXIT):
    # Compute loss
    loss = torch.nn.MSELoss()(y, K(x_0))
    print(f"Loss at: {k=}: {loss.item():.4f}.")

    # Compute gradient
    loss.backward()

    # Update x_0
    x = x_0 - ALPHA * x_0.grad
    x_0 = x.clone().detach()
    x_0 = x_0.requires_grad_(True)

x_pred = x_0.cpu().detach().numpy()[0, 0]

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(x_true.cpu().detach().numpy()[0, 0], cmap="gray")

plt.subplot(1, 2, 2)
plt.imshow(x_pred, cmap="gray")
plt.show()
