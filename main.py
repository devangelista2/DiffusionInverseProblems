import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from experiments import model_based, deep_generative_prior, generation
from miscellaneous import iputils, metrics

import config

# Load config
kernel_type = config.problem_params['kernel_type']
k = config.problem_params['k']
sigma = config.problem_params['sigma']
noise_std = config.problem_params['noise_std']

data_path = config.problem_params['data_path']

# Load (test) data
x_test = np.load(data_path + 'test.npy')

# Normalize
x_test = (x_test / 255.).astype(np.float32)

# Choose the ground-truth image.
idx = 20
x_true = x_test[idx:idx+1]
_, m, n, _ = x_true.shape

# Define the corruption model
K = iputils.tf_K(k, sigma)

# Generate the corrupted image
y = K(x_true)
y_delta = y + noise_std * np.random.normal(0, 1, y.shape)

# Visualize
plt.imsave('./results/x_true.png', x_true[0, :, :, 0], cmap='gray')
plt.imsave('./results/y.png', y[0, :, :, 0], cmap='gray')
plt.imsave('./results/y_delta.png', y_delta[0, :, :, 0], cmap='gray')

# Naive inversion
compute_naive = False
if compute_naive:
    x_naive = model_based.NaiveSolution(y_delta.numpy())
    print(x_naive.shape)

    # Compute metrics
    print(f"SSIM(x_naive, x_true): {metrics.SSIM(x_true.reshape((m, n)), x_naive.reshape((m, n)))}")

    # Visualize Naive solution
    plt.imsave('./results/x_naive.png', x_naive.reshape((m, n)), cmap='gray')

# Regularized inversion (Tikhonov)
compute_tik = False
if compute_tik:
    lmbda = 0.12
    x_tik = model_based.TikhonovSolution(y_delta.numpy(), lmbda)

    # Compute metrics
    print(f"SSIM(x_tik, x_true): {metrics.SSIM(x_true.reshape((m, n)), x_tik.reshape((m, n)))}")

    # Visualize Naive solution
    plt.imsave('./results/x_tik.png', x_tik.reshape((m, n)), cmap='gray')

# DiDeCoNN solution
compute_dideconn_random = True
if compute_dideconn_random:
    # Choose starting iterate
    z0 = tf.random.normal(y_delta.shape)

    # Choose weights for the diffusion network
    mode = 'convex_unet'
    weights_path = "convex_diffusion_celeba_grayscale.hdf5"

    # Solve
    lmbda = 1e-3
    x_dideconn, z_dideconn, loss_vec = deep_generative_prior.DiDeCoNN_Hard(y_delta, lmbda, z0, weights_path, mode, x_true)

    # Compute metrics
    print(f"SSIM(x_dideconn, x_true): {metrics.SSIM(x_true.reshape((m, n)), x_dideconn.numpy().reshape((m, n)))}")

    # Visualize Naive solution
    plt.imsave('./results/x_dideconn_random.png', x_dideconn.numpy().reshape((m, n)), cmap='gray')

    # Save the error
    np.save('./results/error_plots/dideconn_random_ssim.npy', loss_vec)

# DiDeCoNN_blurred
compute_dideconn_blurred = False
if compute_dideconn_blurred:
    # Choose starting iterate
    z0 = tf.Variable(y_delta)

    # Choose weights for the diffusion network
    weights_path = "diffusion_celeba_grayscale.hdf5"

    # Solve
    lmbda = 1e-3
    x_dideconn, z_dideconn, loss_vec = deep_generative_prior.DiDeCoNN_Hard(y_delta, lmbda, z0, weights_path, x_true)

    # Compute metrics
    print(f"SSIM(x_dideconn, x_true): {metrics.SSIM(x_true.reshape((m, n)), x_dideconn.numpy().reshape((m, n)))}")

    # Visualize Naive solution
    plt.imsave('./results/x_dideconn_blurred.png', x_dideconn.numpy().reshape((m, n)), cmap='gray')

    # Save the error
    np.save('./results/error_plots/dideconn_blurred_ssim.npy', loss_vec)

# DiDeCoNN_true
compute_dideconn_true = False
if compute_dideconn_true:
    # Choose starting iterate
    z0 = tf.Variable(x_true)

    # Choose weights for the diffusion network
    weights_path = "diffusion_celeba_grayscale.hdf5"

    # Solve
    lmbda = 1e-3
    x_dideconn, z_dideconn, loss_vec = deep_generative_prior.DiDeCoNN_Hard(y_delta, lmbda, z0, weights_path, x_true)

    # Compute metrics
    print(f"SSIM(x_dideconn, x_true): {metrics.SSIM(x_true.reshape((m, n)), x_dideconn.numpy().reshape((m, n)))}")

    # Visualize Naive solution
    plt.imsave('./results/x_dideconn_true.png', x_dideconn.numpy().reshape((m, n)), cmap='gray')

    # Save the error
    np.save('./results/error_plots/dideconn_true_ssim.npy', loss_vec)
