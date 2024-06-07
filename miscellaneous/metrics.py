import math

import lpips
import torch
from skimage.metrics import structural_similarity as ssim_fn

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
    ssim_vec = torch.zeros((N,))

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
    psnr_vec = torch.zeros((N,))

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
lpips_alex = lpips.LPIPS(net="alex")


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
    LPIPS_vec = torch.zeros((N,))

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