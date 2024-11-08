"""PyTorch implementation of SleepPPG-Net.

Implemented from Fig. 2 of:

K. Kotzen et al., ‘SleepPPG-Net: A Deep Learning Algorithm for Robust Sleep Staging From Continuous Photoplethysmography’, IEEE J. Biomed. Health Inform., Feb. 2023.

The SleepPPG-Net architecture is an adaptation of the model proposed in:
N. Sridhar et al., ‘Deep learning for automated sleep staging using instantaneous heart rate’, npj Digit. Med., Aug. 2020.
"""

__all__ = ('SleepPPGNet',)
import torch
from torch import Tensor, nn

from .blocks import ConvBlock1D, ConvLayer1D
from .utils import get_activation


class SleepPPGNet(nn.Module):
    INPUT_LENGTH: int = 1228800  # Model makes hard assumption on input dimension (10h @ [1024/30 ~ 34.13] Hz)

    def __init__(
        self,
        n_classes: int = 4,
        feature_dim: int = 128,
        dropout: float = 0.2,
        activation: str = 'leaky',
        norm: str = 'batch',
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.conv_block = WindowEncoder(activation=activation, norm=norm)
        self.dense = DenseBlock(in_dim=1024, out_dim=feature_dim)
        self.dilated_convs = nn.Sequential(
            *[
                DilatedConvBlock(feature_dim=feature_dim, dropout=dropout, activation=activation, norm=norm),
                DilatedConvBlock(feature_dim=feature_dim, dropout=dropout, activation=activation, norm=norm),
            ]
        )
        self.classifier = nn.Linear(in_features=feature_dim, out_features=n_classes)

    def encode(self, x_BT: Tensor) -> Tensor:
        """Encode a PPG waveform into a feature representation.

        Args:
            x (Tensor): shape [N, 1228800]
        Returns:
            Tensor: shape [N, 256, 4800]
        """
        if x_BT.size(1) != self.INPUT_LENGTH:
            raise ValueError(f'Input tensor had unexpected shape: {x_BT.size()}')
        # Add channel dim for conv layers.
        x_B1T = torch.unsqueeze(x_BT, dim=1)
        x_BCT = self.conv_block(x_B1T)
        # Reshape + time-distributed dense layer.
        x_BFS = self.dense(x_BCT)
        x_BFS = self.dilated_convs(x_BFS)
        x_BSF = x_BFS.transpose(-1, -2)
        return x_BSF

    def forward(self, x_BT: Tensor) -> Tensor:
        """Apply SleepPPG-Net to an input sequence, producing logits.

        Assumes a fixed 10-hour input sequence, producing 1200 classifications
        for each batch element.

        Args:
            x (Tensor): shape [N, 1228800]
        Returns:
            Tensor: shape [N, 1200, 4]
        """
        x_BSF = self.encode(x_BT)
        return self.classifier(x_BSF)


class WindowEncoder(nn.Module):
    """Sleep window encoder layer.

    Progressively downsamples the input waveform.
    """

    CHANNELS = [16, 16, 32, 32, 64, 64, 128, 256]

    def __init__(self, activation: str = 'leaky', norm: str = 'batch') -> None:
        super().__init__()
        blocks = []
        in_channels = 1
        for out_channels in self.CHANNELS:
            blocks.append(ConvBlock1D(in_channels, out_channels, activation=activation, norm=norm))
            in_channels = out_channels
        self.model = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [N, 1, 1228800]

        Returns:
            Tensor: shape [N, 256, 4800]
        """
        return self.model(x)


class DenseBlock(nn.Module):
    """Time-distributed dense layer."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 128, activation: str = 'leaky') -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [N, 256, 4800]

        Returns:
            Tensor: shape [N, F, 1200]
        """
        # Reshape for the dense layer.
        out = x.transpose(-1, -2).reshape(-1, 1200, 1024)
        # Apply FC layer.
        out = self.linear(out)
        out = self.activation(out)
        # Re-shape back to channel-first for dilated convolutions.
        return out.transpose(-1, -2)


class DilatedConvBlock(nn.Module):
    """Dilated Convolutional Block.

    Uses a consistent channel dimension and progressively wider dilations to increase the context length for each epoch.
    """

    DILATIONS = [1, 2, 4, 8, 16, 32]

    def __init__(
        self,
        feature_dim: int = 128,
        dropout: float = 0.2,
        activation: str = 'leaky',
        norm: str = 'batch',
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        blocks = []
        self.kernel_size = kernel_size
        for dilation in self.DILATIONS:
            # Calculate effective kernel size to pad correctly.
            k_eff = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = k_eff // 2
            blocks.append(
                ConvLayer1D(
                    input_dim=feature_dim,
                    output_dim=feature_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    activation=activation,
                    norm=norm,
                )
            )
        self.dropout = nn.Dropout(p=dropout)
        self.conv_layers = nn.Sequential(*blocks)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [N, F, S]
        Returns:
            Tensor: shape [N, F, S]
        """
        out = self.conv_layers(x)
        out = self.dropout(out)
        out = out + x
        out = self.activation(out)
        return out
