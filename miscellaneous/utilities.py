import torch
import numpy as np

from variational import operators

from models.DiffusionModels.DDIM import DDIM
from models.GAN.DCGAN import DCGAN
from models.GAN.StyleGANv2 import StyleGANv2
import time

########################
# MODEL UTILITIES
########################
def get_model(model_name, config, weights_path=None):
    # Define generative model
    if model_name == "DDIM":
        model = DDIM(config)
    elif model_name == "DCGAN":
        model = DCGAN(config)
    elif model_name == "StyleGANv2":
        model = StyleGANv2(config)
    else:
        raise NotImplementedError
    
    if weights_path is not None:
        # Load trained model
        model.load(weights_path)
    return model


def get_operator(operator_name, operator_setup):
    operator_shape = operator_setup["shape"]

    if operator_name == "GaussianBlur":
        kernel_size = operator_setup["kernel_size"]
        kernel_variance = operator_setup["kernel_variance"]

        K = operators.GaussianBlur(
            shape=operator_shape, kernel_size=kernel_size, sigma=kernel_variance
        )
    elif operator_name == "Radon":
        angular_range = operator_setup["angular_range"]
        n_angles = operator_setup["n_angles"]

        angles = np.linspace(
            np.deg2rad(angular_range[0]), np.deg2rad(angular_range[1]), n_angles
        )
        K = operators.Radon(input_shape=(1,) + operator_shape, angles=angles, geometry="fanflat")
    elif operator_name == "Identity":
        K = operators.Identity(shape=operator_shape)
    
    return K

def gaussian_noise(y, noise_level, seed=42):
    torch.manual_seed(seed)
    e = torch.randn_like(y)
    return e / torch.norm(e, p="fro") * torch.norm(y, p="fro") * noise_level

########################
# PYTORCH UTILITIES
########################
class CustomNumpyOperator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, K, x):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.

        K -> Operator that can be applied to Numpy version of x. It requires a __call__ method and a .T.
        x -> Pytorch array to which K has to be applied.
        """
        ctx.save_for_backward(x)
        ctx.K = K

        x_npy = x.detach().numpy()
        y_npy = ctx.K(x_npy)
        y = torch.from_numpy(y_npy)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (x,) = ctx.saved_tensors

        grad_output_npy = grad_output.numpy()
        KT_grad_output_npy = ctx.K.T(grad_output_npy)
        KT_grad_output = torch.from_numpy(KT_grad_output_npy)
        return None, KT_grad_output


########################
# IMAGE OPERATIONS
########################
def gradient(x):
    """
    Computes the gradient of x, where x is an (N, c, h, w) Torch tensor.
    """
    D_h = torch.diff(x, n=1, dim=3, prepend=torch.zeros_like(x)[:, :, :, :1])
    D_v = torch.diff(x, n=1, dim=2, prepend=torch.zeros_like(x)[:, :, :1, :])
    return D_h, D_v


########################
# COMMON UTILITIES
########################
def format_time(start_time):
    # End the timer
    end_time = time.time()

    # Calculate the time difference
    elapsed_time = end_time - start_time

    # Convert elapsed time to hours, minutes, and seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format using an f-string with %H:%M:%S style
    formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    return formatted_time

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def project(x):
    x[x<0] = 0
    return x
