from IPPy import operators, solvers, stabilizers, metrics
from miscellaneous import iputils

import config

# Load config
k = config.problem_params['k']
sigma = config.problem_params['sigma']

def NaiveSolution(y_delta):
    # Get the shape for the problem
    _, m, n, _ = y_delta.shape

    # Get the forward operator K
    kernel = iputils.get_gaussian_kernel(k, sigma)
    K = operators.ConvolutionOperator(kernel, (m, n))

    # Get the CGLS solver
    naive_solver = solvers.CGLS(K)

    # Compute the solution and return it.
    x_naive = naive_solver(y_delta.flatten(), y_delta.flatten()).reshape(y_delta.shape)
    return x_naive

def TikhonovSolution(y_delta, lmbda):
    # Get the shape for the problem
    _, m, n, _ = y_delta.shape

    # Get the forward kernel
    kernel = iputils.get_gaussian_kernel(k, sigma)
    
    # Compute the solution and return it
    tik_solver = stabilizers.Tik_CGLS_stabilizer(kernel, lmbda, k=200)
    x_tik = tik_solver(y_delta[0, :, :, 0]).reshape((1, m, n, 1))
    return x_tik