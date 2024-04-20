import torch
import deepinv
import skimage
import matplotlib.pyplot as plt
import math
import numpy as np

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}.")

# Define the image 
x = skimage.data.camera()

# Normalize
x = (x - x.min()) / (x.max() - x.min())

# Extend its shape
x = np.expand_dims(x, (0, 1))
N, c, h, w = x.shape

# Convert to tensor
x = torch.tensor(x, requires_grad=True, dtype=torch.float, device=DEVICE)

# Define the forward operator
angles = torch.linspace(0, 180, 180)

R = deepinv.physics.Tomography(img_width=w,
                               angles=angles,
                               circle=False,
                               device=DEVICE)

# Compute y
y = R(x)
x_back = R.A_dagger(y)

# Compute loss
loss = torch.nn.MSELoss()(x, x_back)

# Compute gradient
loss.backward()