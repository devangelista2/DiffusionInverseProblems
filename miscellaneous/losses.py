import torch
from torch.autograd import grad
import math

###########################################
# DIFFUSION MODELS
###########################################
def noise_estimation_loss(
    model,
    x0: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

###########################################
# GAN
###########################################
def gradient_penalty(discriminator, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
    interpolated_samples = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated_samples.requires_grad_(True)
    d_interpolated = discriminator(interpolated_samples)

    gradients = grad(
        outputs=d_interpolated,
        inputs=interpolated_samples,
        grad_outputs=torch.ones(d_interpolated.size(), device=real_samples.device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

# R1 Regularization
def r1_reg(d_out, real_samples):
    batch_size = real_samples.size(0)
    grad_real = grad(
        outputs=d_out.sum(), 
        inputs=real_samples, 
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )[0]
    grad_penalty = (grad_real.view(batch_size, -1).norm(2, dim=1) ** 2).mean()
    return grad_penalty

# Path Length Regularization
def path_length_regularization(fake_images, w, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_images) / math.sqrt(fake_images.shape[2] * fake_images.shape[3])
    G = grad(
        outputs=(fake_images * noise).sum(), 
        inputs=w, 
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )[0]
    path_lengths = G.norm(2, dim=2).mean(dim=1)
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = ((path_lengths - path_mean) ** 2).mean()
    return path_penalty, path_mean
