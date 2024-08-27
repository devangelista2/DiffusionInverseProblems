import torch

import miscellaneous.metrics
from variational import regularizers


class Adam:
    def __init__(self, K, G, config):
        self.K = K
        self.G = G
        self.config = config

    def __call__(
        self,
        y_delta,
        lmbdaTik,
        lmbdaTV,
        z0,
        x_true=None,
        alpha=1,
        maxit=200,
        tolf=1e-4,
        tolx=1e-5,
        verbose=False,
        return_obj=False,
        return_metrics=False,
    ):
        # Define starting point
        z = z0
        x = self.G(z)

        # Define optimizer
        optimizer = torch.optim.Adam([z], lr=alpha)

        # Initialization
        obj = torch.zeros((maxit + 1,))
        obj[0] = self.obj_function(z, x, y_delta, lmbdaTik, lmbdaTV)

        if x_true is not None:
            with torch.no_grad():
                # PSNR
                psnr_vec = torch.zeros((maxit + 1,), requires_grad=False)
                psnr_vec[0] = miscellaneous.metrics.psnr(x, x_true)

                # LPIPS
                lpips_vec = torch.zeros((maxit + 1,), requires_grad=False)
                lpips_vec[0] = miscellaneous.metrics.LPIPS(x, x_true)

                # SSIM
                ssim_vec = torch.zeros((maxit + 1,), requires_grad=False)
                ssim_vec[0] = miscellaneous.metrics.ssim(x, x_true)

                # GNorm
                gnorm_vec = torch.zeros((maxit + 1,), requires_grad=False)
                gnorm_vec[0] = 0.0

        k = 0
        stopping = False
        while not stopping:
            # Update z_old
            z_old = torch.clone(z)

            # Update z
            obj_k = self.obj_function(z, x, y_delta, lmbdaTik, lmbdaTV)
            obj_k.backward()

            # Compute gradient norm
            gnorm = torch.norm(z.grad)

            optimizer.step()
            optimizer.zero_grad()

            # Compute x from z
            x = self.G(z)

            # Update k
            k = k + 1

            # Update objective function
            obj[k] = obj_k

            # Compute distance between iterates
            dist = torch.norm(z - z_old) / (torch.norm(z) + 1e-6)
            if x_true is not None:
                with torch.no_grad():
                    psnr_vec[k] = miscellaneous.metrics.psnr(x_true, x)
                    lpips_vec[k] = miscellaneous.metrics.LPIPS(x_true, x)
                    ssim_vec[k] = miscellaneous.metrics.ssim(x_true, x)
                    gnorm_vec[k] = gnorm
                    print(
                        f"k = {k}. PSNR: {psnr_vec[k]:0.4f}, LPIPS: {lpips_vec[k]:0.4f}, SSIM: {ssim_vec[k]:0.4f}, gnorm: {gnorm:0.4f}."
                    )

            # Check convergence
            stopping = (k >= maxit - 1) or (dist < tolx) or (obj[k] < tolf)
        if return_metrics:
            metrics = {
                "PSNR": psnr_vec[:k],
                "LPIPS": lpips_vec[:k],
                "SSIM": ssim_vec[:k],
                "GNorm": gnorm_vec[:k],
                "fval": obj[:k],
            }
            return z, metrics
        return z

    def obj_function(self, z, x, y_delta, lmbdaTik, lmbdaTV):
        # Compute the residual
        res = torch.sum(torch.square(self.K(x) - y_delta))
        regTik = regularizers.Tik(z)
        regTV = regularizers.TV_beta(x)

        f_k = res + lmbdaTik * regTik + lmbdaTV * regTV

        return f_k


class GradientDescent:
    def __init__(self, K, model, config):
        self.K = K
        self.model = model
        self.config = config

    def __call__(
        self,
        y_delta: torch.Tensor,
        lmbdaTik: float,
        lmbdaTV: float,
        z0: torch.Tensor,
        x_true: torch.Tensor = None,
        alpha: float = 1,
        maxit: int = 200,
        tolf: float = 1e-4,
        tolx: float = 1e-5,
        verbose: bool = False,
        return_obj: bool = False,
        return_metrics: bool = False,
    ):
        # Define starting point
        z0 = z0.requires_grad_(False)
        z: torch.Tensor = z0.clone()
        res, x, grad = self.compute_res(
            z, y_delta, diffusion_steps=10, return_grad=True
        )

        # Normalize gradient
        gnorm = torch.norm(grad)
        grad = grad / torch.norm(grad)

        # Initialization
        obj = torch.zeros((maxit + 1,))
        obj[0] = res + lmbdaTik * torch.sum(torch.square(z))

        if x_true is not None:
            with torch.no_grad():
                # PSNR
                psnr_vec = torch.zeros((maxit + 1,), requires_grad=False)
                psnr_vec[0] = miscellaneous.metrics.psnr(x, x_true)

                # LPIPS
                lpips_vec = torch.zeros((maxit + 1,), requires_grad=False)
                lpips_vec[0] = miscellaneous.metrics.LPIPS(x, x_true)

                # SSIM
                ssim_vec = torch.zeros((maxit + 1,), requires_grad=False)
                ssim_vec[0] = miscellaneous.metrics.ssim(x, x_true)

                # Gradient Norm
                gnorm_vec = torch.zeros((maxit + 1,), requires_grad=False)
                gnorm_vec[0] = gnorm

        k = 0
        stopping = False
        while not stopping:
            # Update z_old
            z_old = z.clone()
            print(torch.norm(z_old))

            # Update z
            z = z_old - alpha * grad

            # Update x
            res, x, grad = self.compute_res(
                z, y_delta, diffusion_steps=10, return_grad=True
            )

            # Normalize gradient
            gnorm = torch.norm(grad)

            # Update k
            k = k + 1

            # Update objective function
            obj_k = res + lmbdaTik * torch.sum(torch.square(z))
            obj[k] = obj_k

            # Compute distance between iterates
            dist = torch.norm(z - z_old) / (torch.norm(z) + 1e-6)
            if x_true is not None:
                with torch.no_grad():
                    psnr_vec[k] = miscellaneous.metrics.psnr(x_true, x)
                    lpips_vec[k] = miscellaneous.metrics.LPIPS(x_true, x)
                    ssim_vec[k] = miscellaneous.metrics.ssim(x_true, x)
                    gnorm_vec[k] = gnorm
                    print(
                        f"k = {k}. PSNR: {psnr_vec[k]:0.4f}, LPIPS: {lpips_vec[k]:0.4f}, SSIM: {ssim_vec[k]:0.4f}, gnorm: {gnorm_vec[k]:0.4f}."
                    )

            # Check convergence
            stopping = (k >= maxit - 1) or (dist < tolx) or (obj[k] < tolf)
        if return_metrics:
            metrics = {
                "PSNR": psnr_vec[:k],
                "LPIPS": lpips_vec[:k],
                "SSIM": ssim_vec[:k],
                "GNorm": gnorm_vec[:k],
                "fval": obj[:k],
            }
            return z, metrics
        return z

    def compute_res(
        self,
        z: torch.Tensor,
        y_delta: torch.Tensor,
        diffusion_steps: int,
        return_grad: bool = False,
    ):
        # Initialization
        x_t = z.clone()
        delta_t = 1 / diffusion_steps

        with torch.no_grad():
            # Forward pass: Compute x_0 through the reverse diffusion, saving intermediate steps.
            xs = [x_t]
            for step in range(diffusion_steps + 1):
                # update the time t
                t = torch.ones((1, 1, 1, 1)) - step * delta_t

                # update x_t
                x_t = self.model.reverse_diffusion_step(xs[-1], t, delta_t)
                xs.append(x_t.clone())
            x = x_t.clone()

            # Compute the residual
            res = torch.sum(torch.square(self.K(x) - y_delta))

            if return_grad:
                # Initialize the gradient to propagate back
                grad = self.K.T(self.K(x) - y_delta)

                # Backward pass: Propagate gradients back through the loop using JVP
                for step in reversed(range(diffusion_steps + 1)):
                    # update the time t
                    t = torch.ones((1, 1, 1, 1)) - step * delta_t

                    v = grad
                    _, grad = torch.autograd.functional.jvp(
                        lambda x: self.model.reverse_diffusion_step(x, t, delta_t),
                        (xs[step],),
                        (v,),
                    )

                return res, x, grad
            return res
