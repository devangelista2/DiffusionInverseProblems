import matplotlib.pyplot as plt
import torch

from models.DDIM import DDIM
from skimage.metrics import structural_similarity as ssim_fn

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