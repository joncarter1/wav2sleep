import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class ConvLayerNorm(nn.Module):
    """Layer norm for convolutional layers with channels first."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.eps = eps

    def forward(self, x_BCT: Tensor) -> Tensor:
        mu_BCT = x_BCT.mean(1, keepdim=True)
        sigma_BCT = (x_BCT - mu_BCT).pow(2).mean(1, keepdim=True)
        x_BCT = (x_BCT - mu_BCT) / torch.sqrt(sigma_BCT + self.eps)
        x_BCT = self.weight * x_BCT + self.bias
        return x_BCT


class ConvRMSNorm(nn.Module):
    """RMS normalization for convolutional layers."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.eps = eps

    def forward(self, x_BCT: Tensor) -> Tensor:
        sigma_BCT = x_BCT.pow(2).mean(1, keepdim=True)
        x_BCT = x_BCT / torch.sqrt(sigma_BCT + self.eps)
        x_BCT = self.weight * x_BCT
        return x_BCT


class ConvGroupNorm(nn.Module):
    """Group norm implementation."""

    def __init__(
        self, num_features: int, num_groups: int = 8, channels_per_group: int | None = None, eps: float = 1e-5
    ):
        super().__init__()
        if channels_per_group is not None:  # Channels per group takes precedence if set.
            num_groups = num_features // channels_per_group
        if num_features < num_groups:
            logger.warning(f'{num_features=} is less than {num_groups=}. Will function as instance norm.')
            num_groups = num_features  # Revert to instance norm.
        if num_features % num_groups != 0:
            raise ValueError(f'{num_features=} must be divisible by {num_groups=}.')
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, eps=eps)

    def forward(self, x_NCT: Tensor) -> Tensor:
        return self.norm(x_NCT)


def get_activation(name: str, **kwargs):
    """Return an activation function from its name."""
    if name == 'relu':
        return nn.ReLU(**kwargs)
    elif name == 'leaky':
        return nn.LeakyReLU(**kwargs)
    elif name == 'gelu':
        return nn.GELU(**kwargs)
    elif name == 'silu' or name == 'swish':
        return nn.SiLU(**kwargs)
    elif name == 'linear':
        return nn.Identity()
    else:
        raise ValueError(f'{name=} is unsupported.')


def get_norm(name: str | None = 'batch', causal: bool = False, *args, **kwargs) -> nn.Module:
    # Extract norm_eps - only used by instance norm, but may be passed for any norm type
    norm_eps = kwargs.pop('norm_eps', None)

    if name == 'batch':
        return nn.BatchNorm1d(*args, **kwargs)
    elif name == 'layer':
        return ConvLayerNorm(*args, **kwargs)
    elif name == 'rms':
        return ConvRMSNorm(*args, **kwargs)
    elif name is None:
        return nn.Identity()
    elif name == 'instance':
        if norm_eps is not None:
            kwargs['eps'] = norm_eps
        return nn.InstanceNorm1d(*args, **kwargs)
    elif name == 'group':
        return ConvGroupNorm(*args, **kwargs)
    else:
        raise ValueError(f'Normalisation with {name=} and {causal=} unknown.')
