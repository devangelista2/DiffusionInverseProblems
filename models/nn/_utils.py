import math

import torch


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """

    half_dim = embedding_dim // 2
    emb = math.log(10_000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def sinusoidal_embedding(t, embedding_dim):
    embedding_min_frequency = 1.0
    embedding_max_frequency = 10_000
    frequencies = torch.exp(
        torch.linspace(
            math.log(embedding_min_frequency),
            math.log(embedding_max_frequency),
            embedding_dim // 2,
        )
    ).to(t.device)
    angular_speeds = 2.0 * math.pi * frequencies
    emb = t.float() * angular_speeds[None, :, None, None]
    embeddings = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return embeddings


def swish(x):
    """
    Define the swish activation function (commonly used in DDIM models)
    """
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    """
    Group normalization layer.
    """
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )
