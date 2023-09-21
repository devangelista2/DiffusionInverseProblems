import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

from models.nn import diffusion_models
import models.DDIM as DDIM
from miscellaneous import schedules
import config

# Set config parameters
input_shape = config.model_params['input_shape']
depths = config.model_params['depths']
block_depth = config.model_params['block_depth']

learning_rate = config.training_params['learning_rate']
num_epochs = config.training_params['num_epochs']
batch_size = config.training_params['batch_size']

# Load data
data_path = './data/celeba_grayscale/'
x_train = np.load(data_path + 'train.npy')

# Normalize
x_train = (x_train / 255.).astype(np.float32)

# To make the training faster, consider just a subset of the training set
N = 50_000
idx = np.random.choice(np.arange(N), N)

x_train = x_train[idx]

# Get the DDIM Model. A DDIM model is composed by a neural network architecture and a given diffusion schedule.
network = diffusion_models.DiffusionConvexUNet(input_shape, depths, block_depth)
diffusion_schedule = schedules.improved_cosine

ddim_model = DDIM.DiffusionModel(input_shape, network, diffusion_schedule)

# Compile the model
ddim_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.mean_absolute_error,
)

# calculate mean and variance of training dataset for normalization
ddim_model.normalizer.adapt(x_train)

### Run training
weights_filename = "convex_diffusion_celeba_grayscale.hdf5"

ddim_model.fit(
    x_train,
    epochs=num_epochs,
    batch_size=batch_size
)
ddim_model.save_weights("model_weights/" + weights_filename)