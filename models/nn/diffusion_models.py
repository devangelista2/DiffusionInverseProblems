import tensorflow as tf
from tensorflow import keras

from . import _blocks, _utils

def DiffusionUNet(input_shape, depths, block_depth):
    noisy_images = keras.Input(shape=input_shape)
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = keras.layers.Lambda(_utils.sinusoidal_embedding)(noise_variances)
    e = keras.layers.UpSampling2D(size=input_shape[0], interpolation="nearest")(e)

    x = keras.layers.Conv2D(depths[0], kernel_size=1)(noisy_images)
    x = keras.layers.Concatenate()([x, e])

    skips = []
    for depth in depths[:-1]:
        x = _blocks.DownBlock(depth, block_depth)([x, skips])

    for _ in range(block_depth):
        x = _blocks.ResidualBlock(depths[-1])(x)

    for depth in reversed(depths[:-1]):
        x = _blocks.UpBlock(depth, block_depth)([x, skips])

    x = keras.layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="denoising")


def DiffusionConvexUNet(input_shape, depths, block_depth):
    noisy_images = keras.Input(shape=input_shape)
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = keras.layers.Lambda(_utils.sinusoidal_embedding)(noise_variances)
    e = keras.layers.UpSampling2D(size=input_shape[0], interpolation="nearest")(e)

    x = keras.layers.Conv2D(depths[0], kernel_size=1, kernel_constraint=keras.constraints.NonNeg())(noisy_images)
    x = keras.layers.Concatenate()([x, e])

    skips = []
    for depth in depths[:-1]:
        x = _blocks.ConvexDownBlock(depth, block_depth)([x, skips])

    for _ in range(block_depth):
        x = _blocks.ConvexResidualBlock(depths[-1])(x)

    for depth in reversed(depths[:-1]):
        x = _blocks.ConvexUpBlock(depth, block_depth)([x, skips])

    x = keras.layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros", kernel_constraint=keras.constraints.NonNeg())(x)

    return keras.Model([noisy_images, noise_variances], x, name="denoising")