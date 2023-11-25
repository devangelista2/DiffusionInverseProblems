import math

import numpy as np
import torch


def improved_cosine(t, config):
    # Get max and min alpha
    alpha_min, alpha_max = config.diffusion.alpha_min, config.diffusion.alpha_max

    # diffusion times -> angles
    start_angle = math.acos(alpha_max)
    end_angle = math.acos(alpha_min)

    diffusion_angles = start_angle + t * (end_angle - start_angle)

    # angles -> signal and noise rates
    alpha_t = torch.cos(diffusion_angles) ** 2

    return alpha_t
