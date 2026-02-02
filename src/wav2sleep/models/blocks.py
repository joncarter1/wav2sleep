"""Basic neural network building blocks in PyTorch."""

from torch import Tensor, nn

from .utils import get_activation, get_norm


class ConvBlock1D(nn.Module):
    """Three-layer convolutional block with downsampling by a factor of 2."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        activation: str = 'leaky',
        norm: str = 'batch',
        causal: bool = False,
        norm_eps: float | None = None,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual

        def create_conv(input_dim: int, output_dim: int, stride: int = 1) -> ConvLayer1D:
            return ConvLayer1D(
                input_dim=input_dim,
                output_dim=output_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                activation=activation,
                norm=norm,
                dropout=dropout,
                causal=causal,
                norm_eps=norm_eps,
            )

        self.conv1 = create_conv(input_dim, output_dim)
        self.conv2 = create_conv(output_dim, output_dim)
        # Downsample within the convolutional path using a stride-2 conv for proper causal alignment.
        self.conv3 = create_conv(output_dim, output_dim, stride=2)
        self.activation = get_activation(activation)
        # Conventional ResNet downsampling - linear transformation with matching stride and output channels.
        if self.use_residual:
            self.downsample = nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            )
        else:
            self.register_parameter('downsample', None)

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
        if self.use_residual:
            out = out + self.downsample(x)
        out = self.activation(out)
        return out


class DilatedConvBlock(nn.Module):
    """Dilated Convolutional Block.

    Uses a consistent channel dimension and progressively wider dilations to increase the context length for each epoch.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        dropout: float = 0.2,
        activation: str = 'leaky',
        norm: str = 'batch',
        kernel_size: int = 7,
        causal: bool = False,
        num_dilations: int = 6,
    ) -> None:
        super().__init__()
        blocks = []
        self.kernel_size = kernel_size
        self.dilations = [2**i for i in range(num_dilations)]
        for dilation in self.dilations:
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
                    causal=causal,
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
        norm_eps: float | None = None,
    ) -> None:
        super().__init__()
        self.causal = causal
        if causal:  # Create causal padding.
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = padding
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=groups,
            bias=bias or norm is None,  # Make sure bias is used if no normalization is applied.
            dilation=dilation,
        )
        if norm == 'weight':  # Directly apply weight norm to the conv layer.
            self.conv = nn.utils.parametrizations.weight_norm(self.conv)
            self.norm = nn.Identity()
        else:
            norm_kwargs = {'norm_eps': norm_eps} if norm_eps is not None else {}
            self.norm = get_norm(norm, causal=causal, num_features=output_dim, **norm_kwargs)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        # In causal mode, remove the right-padding immediately after convolution to avoid skewing
        # normalization stats. When using stride > 1, trim slightly less to preserve correct
        # downsampled length so that residual paths (e.g. 1x1, stride-2) align.
        if self.causal and self.padding > 0:
            stride = self.conv.stride[0] if isinstance(self.conv.stride, tuple) else self.conv.stride
            right_trim = max(self.padding - (stride - 1), 0)
            if right_trim > 0:
                out = out[:, :, :-right_trim]
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out
