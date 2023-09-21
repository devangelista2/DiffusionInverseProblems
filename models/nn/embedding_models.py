import tensorflow as tf
from tensorflow import keras

from . import _blocks

def get_Unet(image_size, depths, block_depth):
    input_images = keras.Input(shape=(image_size, image_size, 1))

    x = keras.layers.Conv2D(depths[0], kernel_size=1)(input_images)

    skips = []
    for depth in depths[:-1]:
        x = _blocks.DownBlock(depth, block_depth)([x, skips])

    for _ in range(block_depth):
        x = _blocks.ResidualBlock(depths[-1])(x)

    for depth in reversed(depths[:-1]): 
        x = _blocks.UpBlock(depth, block_depth)([x, skips])

    x = keras.layers.Conv2D(1, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model(input_images, x, name="unet")