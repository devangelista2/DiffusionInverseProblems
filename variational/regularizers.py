import torch

from miscellaneous import utilities


def Tik(x):
    return torch.sum(torch.square(x))


def TV(x):
    # Get the directional derivatives of x
    Dh_x, Dv_x = utilities.gradient(x)

    # Anisotropic TV -> Dx = sqrt(Dh_x^2 + Dv_x^2)
    Dx = torch.sqrt(torch.square(Dh_x) + torch.square(Dv_x))

    # Return the || Dx ||_1
    return torch.sum(torch.abs(Dx))

def TV_beta(x, beta=1e-6):
    # Get the directional derivatives of x
    Dh_x, Dv_x = utilities.gradient(x)

    # Anisotropic TV -> Dx = sqrt(Dh_x^2 + Dv_x^2)
    Dx = torch.sqrt(torch.square(Dh_x) + torch.square(Dv_x) + beta**2)

    # Return the || Dx ||_1
    return torch.sum(torch.abs(Dx))
