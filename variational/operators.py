from typing import Any

import torch


class GaussianBlur:
    def __init__(self, shape, kernel_size=7, sigma=1) -> None:
        """
        Defines the GaussianBlur operator with the kernel size and variance (sigma) specified. Shape is a tuple representing
        the shape of the image to which the kernel has to be applied. In particular, it has to be a 3-dimensional tuple
        containing (c, h, w), where c: number of channels, h: height and w: width.
        """
        self.shape = shape
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Get device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Define the Gaussian Kernel
        self.kernel = self.get_gaussian_kernel()
        self.kernel = self.kernel.to(self.device)

    def __call__(self, x):
        return torch.conv2d(x, self.kernel, padding="same")

    def get_gaussian_kernel(self):
        """
        Creates gaussian kernel with kernel size 'k' and a variance of 'sigma'
        """
        ax = torch.linspace(
            -(self.kernel_size - 1) / 2.0,
            (self.kernel_size - 1) / 2.0,
            self.kernel_size,
        )
        gauss = torch.exp(-0.5 * torch.square(ax) / self.sigma**2)
        kernel = torch.outer(gauss, gauss)
        kernel = kernel / torch.sum(kernel)
        return kernel.unsqueeze(0).unsqueeze(0).repeat((1, self.shape[0], 1, 1))
