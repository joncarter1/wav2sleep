"""Basic neural network building blocks in PyTorch."""

from torch import Tensor, nn

from .utils import get_activation, get_norm


class ConvBlock1D(nn.Module):
    """Three-layer convolutional blockß."""

    def __init__(
        self, input_dim: int, output_dim: int, dropout: float = 0.0, activation: str = 'leaky', norm: str = 'batch'
    ) -> None:
        super().__init__()
        self.activation = get_activation(activation)
        self.conv1 = ConvLayer1D(
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        self.conv2 = ConvLayer1D(
            input_dim=output_dim,
            output_dim=output_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        self.conv3 = ConvLayer1D(
            input_dim=output_dim,
            output_dim=output_dim,
            kernel_size=3,
            padding=1,
            stride=1,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), padding=0)
        # Conventional ResNet downsampling - linear transformation with matching stride and output channels.
        self.downsample = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [N, Cin, L]

        Returns:
            Tensor: shape [N, Cout, L]
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = out + self.downsample(x)
        out = self.activation(out)
        return out


class ConvLayer1D(nn.Module):
    """Generic 1D Convolutional layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        causal: bool = False,
        groups: int = 1,
        activation: str = 'relu',
        bias: bool = False,
        norm: str | None = 'batch',
    ) -> None:
        super().__init__()
        self.activation = get_activation(activation)
        self.norm = get_norm(norm, num_features=output_dim)
        self.causal = causal
        if causal:  # Create causal padding.
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = padding
        bias = bias or norm is None  # Make sure bias is used if no normalization is applied.
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=groups,
            bias=bias or norm is None,
            dilation=dilation,
        )
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.causal and self.padding != 0:
            out = out[:, :, : -self.padding]
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out
