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
        maxit=200,
        tolf=1e-3,
        tolx=5e-4,
        verbose=False,
        return_obj=False,
    ):
        # Initialization
        if return_obj:
            obj = torch.zeros((maxit + 1,))
            obj[0] = self.obj_function(z0, y_delta, lmbda)

        # Define starting point
        z = z0.reshape((self.mx, self.nx))

        k = 0
        stopping = False
        while not stopping:
            # Update z and x
            x = z

            # Update k
            k = k + 1

            # Update objective function (if required)
            if return_obj:
                obj[k] = self.obj_function(z, y_delta, lmbda)
            # Print rel.err.
            if verbose and (x_true is not None):
                rel_err = torch.linalg.norm(
                    x.flatten() - x_true.flatten()
                ) / torch.linalg.norm(x_true.flatten())
                print(f"{k=}: Rel. Err. = {rel_err:0.4f}.")

            # Compute residual and iterates distance for stopping conditions
            res = torch.linalg.norm(self.K(x) - y_delta) / torch.max(y_delta)
            dist = torch.linalg.norm(x_old.flatten() - x.flatten()) / (
                torch.linalg.norm(x.flatten()) + 1e-6
            )

            # Update stopping condition
            stopping = (
                (dist < tolx)
                or ((res / torch.sqrt(len(y_delta))) < tolf)
                or (k >= maxit)
            )

        if verbose:
            print("\n---------------------------")
            if k >= maxit:
                print("Algorithm didn't converged.")
            elif dist < tolx:
                print("Algorithm converged with x-condition.")
            elif (res / torch.sqrt(len(y_delta))) < tolf:
                print("Algorithm converged with f-condition.")
            print(f"Iterations: \t   {k}.")
            print(f"Relative Distance: {dist:0.5f}.")
            print(f"Residual: \t   {res / torch.sqrt(len(y_delta)):0.5f}.")
            print("---------------------------\n")

        if return_obj:
            return x, obj[:k]
        return x

    def obj_function(self, z, y_delta, lmbda):
        # Compute the residual and the regularization term
        x = self.G(z)  # Generate x from z
        res = torch.norm(self.K(x) - y_delta, p="fro")
        reg = torch.norm(z, p="fro")

        return 0.5 * res**2 + 0.5 * lmbda * reg**2
