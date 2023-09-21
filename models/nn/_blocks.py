import tensorflow as tf
from tensorflow import keras

def ResidualBlock(depth):
    def apply(x):
        input_depth = x.shape[-1]
        if input_depth == depth:
            residual = x
        else:
            residual = keras.layers.Conv2D(depth, kernel_size=1)(x)
        x = keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = keras.layers.Conv2D(
            depth, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = keras.layers.Conv2D(depth, kernel_size=3, padding="same")(x)
        x = keras.layers.Add()([x, residual])
        return x

    return apply


def DownBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(depth)(x)
            skips.append(x)
        x = keras.layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(depth)(x)
        return x

    return apply


def ConvexResidualBlock(depth):
    def apply(x):
        input_depth = x.shape[-1]
        if input_depth == depth:
            residual = x
        else:
            residual = keras.layers.Conv2D(depth, kernel_size=1, kernel_constraint=keras.constraints.NonNeg())(x)
        x = keras.layers.BatchNormalization(center=False, scale=False, gamma_constraint=keras.constraints.NonNeg())(x)
        x = keras.layers.Conv2D(
            depth, kernel_size=3, padding="same", activation="relu", kernel_constraint=keras.constraints.NonNeg()
        )(x)
        x = keras.layers.Conv2D(depth, kernel_size=3, padding="same", kernel_constraint=keras.constraints.NonNeg())(x)
        x = keras.layers.Add()([x, residual])
        return x

    return apply


def ConvexDownBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ConvexResidualBlock(depth)(x)
            skips.append(x)
        x = keras.layers.MaxPooling2D()(x)
        return x

    return apply


def ConvexUpBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        x = keras.layers.UpSampling2D()(x)
        for _ in range(block_depth):
            x = keras.layers.Concatenate()([x, skips.pop()])
            x = ConvexResidualBlock(depth)(x)
        return x

    return apply