from typing import Any

import torch


class GD:
    def __init__(self, K, G, config):
        self.K = K
        self.G = G
        self.config = config

    def __call__(
        self,
        y_delta,
        lmbda,
        z0,
        x_true=None,
        alpha=1,
        maxit=200,
        tolf=1e-3,
        tolx=5e-4,
        verbose=False,
        return_obj=False,
    ):
        # Define starting point
        z = z0
        x = self.G(z)

        # Define optimizer
        optimizer = torch.optim.Adam([z], lr=1e-3)

        # Initialization
        obj = torch.zeros((maxit + 1,))
        obj[0] = self.obj_function(z, x, y_delta, lmbda)

        k = 0
        stopping = False
        while not stopping:
            # Update z_old
            z_old = torch.clone(z)

            # Compute gradient
            # optimizer.zero_grad()
            # obj_k = torch.sum(torch.square(self.K(x) - y_delta)) + lmbda * torch.sum(
            #    torch.square(z)
            # )
            # obj_k.backward(inputs=[z], retain_graph=True)
            # optimizer.step()

            # Update z
            obj_k = torch.sum(torch.square(self.K(x) - y_delta)) + lmbda * torch.sum(
                torch.square(z)
            )
            grad_z = torch.autograd.grad(obj_k, z)[0]
            z.grad = grad_z
            # z = z_old - alpha * grad_z
            optimizer.step()
            z.grad = None

            # Compute x from z
            x = self.G(z)

            # Update k
            k = k + 1

            # Update objective function
            obj[k] = self.obj_function(z, x, y_delta, lmbda)

            # Compute distance between iterates
            dist = torch.norm(z - z_old) / (torch.norm(z) + 1e-6)
            if x_true is not None:
                RE = torch.norm(x - x_true) / torch.norm(x_true)
                print(f"k = {k}. Relative Error: {RE:0.4f}.")

            # Check convergence
            stopping = (k >= maxit - 1) and (dist > tolx) and (obj[k] > tolf)
        if return_obj:
            return z, obj[:k]
        return z

    def obj_function(self, z, x, y_delta, lmbda):
        # Compute the residual and the regularization term
        res = torch.sum(torch.square(self.K(x) - y_delta))
        reg = torch.sum(torch.square(z))

        f_k = 0.5 * (res + lmbda * reg)

        return f_k
