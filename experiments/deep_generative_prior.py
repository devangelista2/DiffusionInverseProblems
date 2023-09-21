import numpy as np
import tensorflow as tf

from miscellaneous import iputils, utils, metrics
from variational import solvers

import config


data_path = config.problem_params['data_path']
k = config.problem_params['k']
sigma = config.problem_params['sigma']

def DiDeCoNN_Hard(y_delta, lmbda, z0, weights_path, mode='unet', x_true=None, maxit=200):
    # Load x_test
    x_test = np.load(data_path + 'test.npy')

    # Normalize
    x_test = (x_test / 255.).astype(np.float32)
    
    # Load diffusion network
    model = utils.get_trained_model(weights_path, x_test, mode)
    G = iputils.generator_from_model(model)

    # Convert x_true to tf Tensor
    if x_true is not None:
        x_true = tf.convert_to_tensor(x_true)

    # Compute the solution
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    tf_K = iputils.tf_K(k, sigma)

    solver = solvers.HardGD(G, tf_K, lmbda=lmbda, optimizer=optimizer, metric=metrics.SSIM, verbose=True)

    if x_true is None:
        z_dideconn = solver(y_delta, z0)
        return G(z_dideconn), z_dideconn
    else:
        z_dideconn, loss_vec = solver(y_delta, z0, x_true)
        return G(z_dideconn), z_dideconn, loss_vec