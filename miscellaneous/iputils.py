import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

def viz(x, title=None):
    """
    Visualize one (or more) n x n array x
    """
    if isinstance(x, tuple):
        l = len(x)
        plt.figure(figsize=(8*l, 8))

        for i in range(l):
            plt.subplot(1, l, i+1)
            plt.imshow(x[i])
            plt.gray()
            if title==None:
                plt.title(f"Shape of x: {x[i].shape}")
            else:
                plt.title(title[i])
        plt.show()
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(x)
        plt.gray()
        if title==None:
            plt.title(f"Shape of x: {x.shape}")
        else:
            plt.title(title)
        plt.show()

def get_gaussian_kernel(k, sigma):
    """
    Creates gaussian kernel with kernel size 'k' and a variance of 'sigma'
    """
    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def get_motion_blur_kernel(k):
    """
    Creates motion blur kernel with kernel size 'k'
    """
    kernel_motion_blur = np.zeros((k, k))

    for i in range(k):
        kernel_motion_blur[i, k-i-1] = 1
        
        if i > 0:
            kernel_motion_blur[i, k-i] = 0.5
        if i < k-1:
            kernel_motion_blur[i, k-i-2] = 0.5
    kernel_motion_blur = kernel_motion_blur / np.sum(kernel_motion_blur)
    return kernel_motion_blur

def fft_convolve(x, K):
    """
    1 - Pad the kernel K to match the shape of x
    2 - Lunch fft_convolve between x and K
    """
    import scipy.signal

    n = x.shape[0]
    k = K.shape[0]

    K_full = np.zeros_like(x)
    K_full[(n-k)//2:(n+k)//2, (n-k)//2:(n+k)//2] = K

    return scipy.signal.fftconvolve(x, K_full, 'same')

# Define the generator
def generator_from_model(model):
    def G(z):
        x_gen = model.reverse_diffusion(z, diffusion_steps=10)
        return model.denormalize(x_gen)
    return G

def tf_K(k, sigma):
    def K(x):
        x = tf.cast(x, tf.float32)
        kernel = get_gaussian_kernel(k, sigma)
        kernel = tf.convert_to_tensor(kernel)
        kernel = tf.reshape(kernel, kernel.shape + (1, 1))
        kernel = tf.cast(kernel, tf.float32)
        return tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
    return K