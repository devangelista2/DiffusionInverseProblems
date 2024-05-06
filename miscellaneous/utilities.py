import math

import lpips
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim_fn

from models.DDIM import DDIM


class ImageGenerator:
    def __init__(self, config, diffusion_steps=50):
        self.config = config
        self.diffusion_steps = diffusion_steps

        # Load trained DDIM Model
        weights_path = (
            f"./model_weights/{config.data.dataset}_{config.training.loss}.pth"
        )
        self.ddim_model = DDIM(config)
        self.ddim_model.model.load_state_dict(torch.load(weights_path))

        # Disable weights gradient memorization to avoid memory issues.
        for param in self.ddim_model.model.parameters():
            param.requires_grad = False

    def __call__(self, x_T):
        # Send x_T to device
        x_T = x_T.to(self.config.device)

        # Generate x_0
        x_0 = self.ddim_model.reverse_diffusion(
            x_T, diffusion_steps=self.diffusion_steps
        )

        # Normalize x_0
        x_0 = (x_0 - x_0.min()) / (x_0.max() - x_0.min())

        return x_0
    
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
        return KT_grad_output


########################
# METRICS
########################
def ssim(x_true, x_pred):
    """
    Computes and returns the SSIM between x_true and x_pred. Both are assumed to be pytorch Tensors of
    shape (N, c, h, w), where:
        N -> Number of samples.
        c -> Number of channels.
        h,w -> Shape of the image.
    Moreover, both x_true and x_pred are assumed to be normalized in [0, 1]. The function output is a Tensor
    of shape (N, ) containing at each position the value of the SSIM of the i-th image.
    """
    # Get shape
    N, c, h, w = x_true.shape

    # Initalize output
    ssim_vec = torch.zeros((N, ))

    # Move both to cpu and than numpy if required.
    x_true_det = x_true.detach().cpu().numpy()
    x_pred_det = x_pred.detach().cpu().numpy()

    # Cycle on the samples
    for i in range(N):
        if c == 1:
            # B-W images
            ssim_vec[i] = ssim_fn(x_true_det[i, 0], x_pred_det[i, 0], data_range=1)
        else:
            # RGB images
            ssim_vec[i] = ssim_fn(x_true_det[i], x_pred_det[i], data_range=1)
        
    # If N == 1 -> Return just its value
    if N == 1:
        return ssim_vec[0]
    return ssim_vec

def psnr(x_true, x_pred):
    """
    Computes and returns the psnr between x_true and x_pred. Both are assumed to be pytorch Tensors of
    shape (N, c, h, w), where:
        N -> Number of samples.
        c -> Number of channels.
        h,w -> Shape of the image.
    Moreover, both x_true and x_pred are assumed to be normalized in [0, 1]. The function output is a Tensor
    of shape (N, ) containing at each position the value of the psnr of the i-th image.
    """
    # Get shape
    N, c, h, w = x_true.shape

    # Initalize output
    psnr_vec = torch.zeros((N, ))

    # Move both to cpu and than numpy if required.
    x_true_det = x_true.detach().cpu()
    x_pred_det = x_pred.detach().cpu()

    # Cycle on the samples
    for i in range(N):
        # Measure max pixel value
        max_pixel = x_true_det.max()
        
        # Compute mse
        mse = torch.mean(torch.square(x_true_det[i] - x_pred_det[i]))

        # Compute PSNR
        psnr_vec[i] = 20 * math.log10(max_pixel / math.sqrt(mse)) 
        
    # If N == 1 -> Return just its value
    if N == 1:
        return psnr_vec[0]
    return psnr_vec

# Setup the LPIPS metric (done just once)
lpips_alex = lpips.LPIPS(net='alex')
def LPIPS(x_true, x_pred):
    """
    Computes and returns the LPIPS between x_true and x_pred. Both are assumed to be pytorch Tensors of
    shape (N, c, h, w), where:
        N -> Number of samples.
        c -> Number of channels.
        h,w -> Shape of the image.
    Moreover, both x_true and x_pred are assumed to be normalized in [0, 1]. The function output is a Tensor
    of shape (N, ) containing at each position the value of the LPIPS of the i-th image.
    """

    # Get shape
    N, c, h, w = x_true.shape

    # Initalize output
    LPIPS_vec = torch.zeros((N, ))

    # Move both to cpu and than numpy if required.
    x_true_det = x_true.detach().cpu()
    x_pred_det = x_pred.detach().cpu()

    # Normalize in [-1, 1]
    x_true_det = 2 * x_true_det - 1
    x_pred_det = 2 * x_pred_det - 1

    # Cycle on the samples
    for i in range(N):
        if c == 1:
            # B-W image
            x_true_det = x_true_det.repeat(1, 3, 1, 1)
            x_pred_det = x_pred_det.repeat(1, 3, 1, 1)
        # Compute LPIPS
        LPIPS_vec[i] = lpips_alex(x_true_det, x_pred_det)
        
    # If N == 1 -> Return just its value
    if N == 1:
        return LPIPS_vec[0]
    return LPIPS_vec


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
