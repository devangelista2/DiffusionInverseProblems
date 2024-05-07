import torch

from miscellaneous import utilities
from variational import regularizers


class GD:
    def __init__(self, K, G, R, config, optimizer=None):
        self.K = K
        self.G = G
        self.R = R
        self.config = config

        self.optimizer = optimizer

    def __call__(
        self,
        y_delta,
        lmbda,
        z0,
        x_true=None,
        alpha=1,
        maxit=200,
        tolf=1e-4,
        tolx=1e-5,
        verbose=False,
        metrics=None,
        return_obj=False,
        return_metrics=False,
    ):
        # Define starting point
        z = z0
        x = self.G(z)

        # Define optimizer (if not None)
        if self.optimizer is not None and self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam([z], lr=alpha)

        # Initialization
        obj = torch.zeros((maxit + 1,))
        obj[0] = self.obj_function(z, x, y_delta, lmbda)

        if x_true is not None:
            with torch.no_grad():
                # PSNR
                psnr_vec = torch.zeros((maxit + 1,), requires_grad=False)
                psnr_vec[0] = utilities.psnr(x, x_true)

                # LPIPS
                lpips_vec = torch.zeros((maxit + 1,), requires_grad=False)
                lpips_vec[0] = utilities.LPIPS(x, x_true)

                # SSIM
                ssim_vec = torch.zeros((maxit + 1,), requires_grad=False)
                ssim_vec[0] = utilities.ssim(x, x_true)

        k = 0
        stopping = False
        while not stopping:
            # Update z_old
            z_old = torch.clone(z)

            # Update z
            obj_k = torch.sum(torch.square(self.K(x) - y_delta)) + lmbda * torch.sum(
                torch.square(z)
            )
            obj_k.backward()

            if self.optimizer is None:
                z = z_old - alpha * z.grad
            else:
                optimizer.step()
                optimizer.zero_grad()

            # Compute x from z
            x = self.G(z)

            # Update k
            k = k + 1

            # Update objective function
            obj[k] = self.obj_function(z, x, y_delta, lmbda)

            # Compute distance between iterates
            dist = torch.norm(z - z_old) / (torch.norm(z) + 1e-6)
            if x_true is not None:
                with torch.no_grad():
                    psnr_vec[k] = utilities.psnr(x_true, x)
                    lpips_vec[k] = utilities.LPIPS(x_true, x)
                    ssim_vec[k] = utilities.ssim(x_true, x)
                    print(
                        f"k = {k}. PSNR: {psnr_vec[k]:0.4f}, LPIPS: {lpips_vec[k]:0.4f}, SSIM: {ssim_vec[k]:0.4f}."
                    )

            # Check convergence
            stopping = (k >= maxit - 1) or (dist < tolx) or (obj[k] < tolf)
        if return_obj:
            return z, obj[:k]
        if return_metrics:
            metrics = {
                "PSNR": psnr_vec[:k],
                "LPIPS": lpips_vec[:k],
                "SSIM": ssim_vec[:k],
            }
            return z, metrics
        return z

    def obj_function(self, z, x, y_delta, lmbda):
        # Compute the residual
        res = torch.sum(torch.square(self.K(x) - y_delta))

        # Compute the regularization term
        if self.R == "Tik_z":
            reg = regularizers.Tik(z)
        elif self.R == "TV_z":
            reg = regularizers.TV(z)
        elif self.R == "Tik_x":
            reg = regularizers.Tik(x)
        elif self.R == "TV_x":
            reg = regularizers.TV(x)

        f_k = 0.5 * (res + lmbda * reg)

        return f_k
