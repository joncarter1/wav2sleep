import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class ConvLayerNorm(nn.Module):
    """Layer norm for convolutional layers.

    Permute the input to have the channel dimension last, apply layer norm, and permute back."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=True)

    def forward(self, x_NCT: Tensor) -> Tensor:
        x_NTC = x_NCT.permute(0, 2, 1)
        x_NTC_norm = self.norm(x_NTC)
        return x_NTC_norm.permute(0, 2, 1)


def get_activation(name: str, **kwargs):
    """Return an activation function from its name."""
    if name == 'relu':
        return nn.ReLU(**kwargs)
    elif name == 'leaky':
        return nn.LeakyReLU(**kwargs)
    elif name == 'gelu':
        return nn.GELU(**kwargs)
    elif name == 'linear':
        return nn.Identity()
    else:
        raise ValueError(f'{name=} is unsupported.')


def get_norm(name: str | None = 'batch', *args, **kwargs) -> nn.Module:
    if name == 'batch':
        return nn.BatchNorm1d(*args, **kwargs)
    elif name == 'instance':
        return nn.InstanceNorm1d(*args, **kwargs)
    elif name == 'layer':
        return ConvLayerNorm(*args, **kwargs)
    elif name is None:
        return nn.Identity()
    else:
        raise ValueError(f'Normalisation with {name=} unknown.')


def embed_ignore_inf(x_BT: torch.Tensor, embedder: nn.Module) -> torch.Tensor:
    """Apply embedder, handling where input is infinite along a batch dimension.

    During training, neg inf tensors are provided in place of unavailable signals.
    This function returns 0s for infinite values in the input tensor.

    Note: This probably shouldn't be used with architectures that use batch normalisation,
    because it will cause the batch norm to be applied to the zeroed out values.
    """
    inf_samples = torch.isinf(x_BT[:, 0])[:, None]  # Infinite samples in batch. (just need to check first element)
    x_BT_in = torch.where(torch.isinf(x_BT), 0.0, x_BT)  # Set inf. values to zero for stability.
    x_BSF = embedder(x_BT_in)
    return ~inf_samples[:, None] * x_BSF  # Fill original infinite values with zeros.
