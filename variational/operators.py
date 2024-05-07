import math
import os

import torch

from miscellaneous import utilities


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


class Identity:
    def __init__(self, shape) -> None:
        """
        Defines the Identity operator. Shape is a tuple representing the shape of the image
        In particular, it has to be a 3-dimensional tuple containing (c, h, w), where
            c: number of channels,
            h: height and w: width.
        """
        self.shape = shape

        # Get device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def __call__(self, x):
        return x


class Radon:
    def __init__(self, input_shape, angles, det_size=None, geometry="parallel") -> None:
        # Input setup
        self.input_shape = input_shape
        self.N, self.c, self.h, self.w = input_shape

        # Geometry
        self.geometry = geometry

        # Projector setup
        if det_size is None:
            self.det_size = int(self.h * math.sqrt(2))
        else:
            self.det_size = det_size
        self.angles = angles
        self.n_angles = len(angles)

        # Define projector
        self.proj = self.get_astra_projection_operator()
        self.shape = self.proj.shape

    def __call__(self, x):
        # Assert the number of channels of x is 1
        assert x.shape[1] == 1

        # Save x device
        x_device = x.device

        self.K = utilities.CustomNumpyOperator()
        y = self.K.apply(self.proj, x[0, 0].cpu().flatten())
        return y.to(x_device)

    def get_astra_projection_operator(self):
        import astra

        # create geometries and projector
        if self.geometry == "parallel":
            proj_geom = astra.create_proj_geom(
                "parallel", 1.0, self.det_size, self.angles
            )
            vol_geom = astra.create_vol_geom(self.h, self.w)
            proj_id = astra.create_projector("linear", proj_geom, vol_geom)

        elif self.geometry == "fanflat":
            proj_geom = astra.create_proj_geom(
                "fanflat", 1.0, self.det_size, self.angles, 1800, 500
            )
            vol_geom = astra.create_vol_geom(self.h, self.w)
            proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

        else:
            raise NotImplementedError("Geometry (still) undefined.")

        return astra.OpTomo(proj_id)
