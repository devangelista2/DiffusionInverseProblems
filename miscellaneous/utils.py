import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
from IPPy import utils as iputils
import models.DDIM as DDIM

from models.nn import diffusion_models
from . import schedules
import config

# Set config parameters
input_shape = config.model_params['input_shape']
depths = config.model_params['depths']
block_depth = config.model_params['block_depth']

learning_rate = config.training_params['learning_rate']

def RGB_to_gray(rgb_path, gray_path):
    import skimage
    from skimage.color import rgb2gray

    # Load the dataset
    data_rgb = np.load(rgb_path)

    # Convert it to Gray-scale
    data_gray = 0.2125 * data_rgb[:, :, :, 0] + 0.7154 * data_rgb[:, :, :, 1] + 0.0721 * data_rgb[:, :, :, 2]
    data_gray = data_gray.astype(np.uint8)
    data_gray = np.expand_dims(data_gray, -1)

    np.save(gray_path, data_gray)

def get_trained_model(weights_filename, x_test, mode='unet'):
    # Get the model
    if mode == 'unet':
        network = diffusion_models.DiffusionUNet(input_shape, depths, block_depth)
    elif mode == 'convex_unet':
        network = diffusion_models.DiffusionConvexUNet(input_shape, depths, block_depth)
    diffusion_schedule = schedules.improved_cosine

    ddim_model = DDIM.DiffusionModel(input_shape, network, diffusion_schedule)

    ddim_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.mean_absolute_error,
    )

    # calculate mean and variance of training dataset for normalization
    ddim_model.normalizer.adapt(x_test)

    # Load the weights
    ddim_model.load_weights("model_weights/" + weights_filename)
    return ddim_model

def generate_samples(n_samples, data_path, weights_path, out_path):
    # Load (test) data
    x_test = np.load(data_path + 'test.npy')

    # Normalize
    x_test = (x_test / 255.).astype(np.float32)

    # Get the trained model
    model = get_trained_model(weights_path, x_test)

    # Generate one image
    z = tf.random.normal((n_samples, 64, 64, 1))
    x_gen = model.reverse_diffusion(z, 10)

    # Save it
    plt.figure(figsize=(6*n_samples, 6))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(x_gen[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=500)