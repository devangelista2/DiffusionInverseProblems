import numpy as np
import tensorflow as tf

from miscellaneous import utils

import config

data_path = config.problem_params['data_path']

def generate_from_model(weights_path, n_images, mode='unet'):
    # Load diffusion network
    x_test = np.load(data_path + 'test.npy')
    x_test = (x_test / 255.).astype(np.float32)
    
    model = utils.get_trained_model(weights_path, x_test, mode)

    # Get latent variables
    z = tf.random.normal((n_images, ) + x_test.shape[1:])

    # Generate
    x_gen = model.reverse_diffusion(z, diffusion_steps=20)
    # x_gen = model.denormalize(x_gen)

    return x_gen