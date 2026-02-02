"""wav2sleep model."""

import logging
import math

import torch
from torch import Tensor, nn

from ..settings import COLS_TO_SAMPLES_PER_EPOCH
from .blocks import ConvBlock1D, DilatedConvBlock
from .utils import get_activation

logger = logging.getLogger(__name__)


class Wav2Sleep(nn.Module):
    """Model for sleep staging.

    This model is used to classify sleep stages from time-series inputs.

    The network works as follows:
        1. Each input signal is passed through a signal encoder.
        2. Embeddings are mixed for each sleep epoch level.
        3. Sequence model mixes epoch features.
        4. Classifier is applied to output features.
    """

    def __init__(
        self,
        signal_encoders: 'SignalEncoders',
        epoch_mixer: 'MultiModalAttentionEmbedder',
        sequence_mixer: 'SequenceCNN',
        num_classes: int,
    ):
        super().__init__()
        self.signal_encoders = signal_encoders
        self.epoch_mixer = epoch_mixer
        self.sequence_mixer = sequence_mixer
        self.feature_dim = self.epoch_mixer.feature_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(in_features=self.feature_dim, out_features=num_classes)

    @property
    def valid_signals(self) -> list[str]:
        """Return list of signals that can be used by the model."""
        return list(self.signal_encoders.signal_map.keys())

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Classify sleep stages from dictionary of input signals.

        Returns logit probabilities of each sleep stage.

        Args:
            x: Dictionary of input tensors.
                Each tensor has shape [batch_size, seq_len, patch_len]
        Returns:
            Tensor, shape [batch_size, seq_len, num_classes]
        """
        # Create feature vectors for each signal
        z_dict = self.signal_encoders(x)
        # Mix features for each sleep epoch from different modalities.
        z_BSF = self.epoch_mixer(z_dict)
        # Mix unified features across the sequence.
        z_BSF = self.sequence_mixer(z_BSF)
        # Apply a classifier to features for each element in the sequence(s).
        logits_BSF = self.classifier(z_BSF)
        return logits_BSF

    def predict(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Classify sequence using transformer encoder + classifier.
        Returns the class with the highest probability for each element.

        Args:
            x: Dictionary of input tensors.
                Each tensor has shape [batch_size, seq_len, patch_len]
        Returns:
            Tensor, shape [batch_size, seq_len]
        """
        logits_BSF = self(x)
        return logits_BSF.argmax(axis=2)


class SignalEncoders(nn.Module):
    """Class that handles multiple signal encoders."""

    def __init__(
        self,
        signal_map: dict[str, str],
        feature_dim: int,
        activation: str,
        norm: str = 'instance',
        causal: bool = False,
        chunk_causal: bool = True,
        embed_signals: bool = False,
        initial_channels: int = 16,
        max_channels: int = 128,
        output_norm: bool = False,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.signal_map = signal_map
        self.causal = causal
        encoders = {}
        # Create encoders for the signals.
        # Multiple signals *can* map to the same encoder.
        for signal_name, encoder_name in self.signal_map.items():
            if encoder_name in encoders:
                continue
            if signal_name not in COLS_TO_SAMPLES_PER_EPOCH:
                raise ValueError(f"Column {signal_name} unrecognised. Doesn't have a sampling rate.")
            samples_per_epoch = COLS_TO_SAMPLES_PER_EPOCH[signal_name]
            encoders[encoder_name] = SignalEncoder(
                input_dim=1,
                feature_dim=feature_dim,
                samples_per_epoch=samples_per_epoch,
                activation=activation,
                norm=norm,
                causal=causal,
                chunk_causal=chunk_causal,
                initial_channels=initial_channels,
                max_channels=max_channels,
                output_norm=output_norm,
                use_residual=use_residual,
            )
        self.encoders = nn.ModuleDict(encoders)
        # Optionally add embedding to indicate signal source.
        self.embed_signals = embed_signals
        self.sig_to_embedding_idx = {sig: i for i, sig in enumerate(sorted(signal_map.keys()))}
        if self.embed_signals:
            self.embedder = nn.Embedding(num_embeddings=len(signal_map), embedding_dim=self.feature_dim)
        else:
            self.register_parameter('embedder', None)

    def __len__(self) -> int:
        return len(self.encoders)

    def get_encoder(self, signal_name: str) -> 'SignalEncoder':
        """Get the signal encoder for a signal."""
        if self.signal_map is not None:
            encoder_name = self.signal_map[signal_name]
        else:
            encoder_name = signal_name
        return self.encoders[encoder_name]  # type: ignore

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_dict: dict[str, torch.Tensor] = {}
        # Apply signal encoder for each input signal.
        for signal_name, x_BT in x.items():
            mask_B = torch.isinf(x_BT[:, 0])  # Identify inf. batch elements (missing signals during training).
            x_BT = torch.where(torch.isinf(x_BT), 0.0, x_BT)  # Set inf. values to zero for stability.
            z_BSF = self.get_encoder(signal_name)(x_BT)
            # Re-fill output batch elements with neg. inf. values so embedder knows what is missing.
            z_BSF = torch.where(mask_B[:, None, None], float('-inf'), z_BSF)
            if self.embed_signals:  # Add embedding to indicate signal source. (For shared encoders)
                e_key = torch.tensor([self.sig_to_embedding_idx[signal_name]], device=z_BSF.device, dtype=torch.int64)
                e_1F = self.embedder(e_key)
                e_BSF = e_1F[None, :, :].repeat(z_BSF.size(0), z_BSF.size(1), 1)
                z_BSF += e_BSF
            z_dict[signal_name] = z_BSF
        return z_dict


class SignalEncoder(nn.Module):
    """Signal encoder layer.

    Progressively downsamples the input waveform, resulting in feature vector sequence for each sleep epoch.
    Then applies time-distributed dense layer to produce the final feature vector.

    When causal=True, each 30-second epoch is processed independently (quasi-causal),
    ensuring predictions for epoch t only use data from epoch t. This allows using
    standard instance normalization with improved training stability.
    """

    def __init__(
        self,
        input_dim: int = 1,
        feature_dim: int = 256,
        activation: str = 'gelu',
        samples_per_epoch: int = 1024,
        norm: str = 'instance',
        initial_channels: int = 16,
        max_channels: int = 128,
        causal: bool = False,
        chunk_causal: bool = True,
        output_norm: bool = False,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.samples_per_epoch = samples_per_epoch
        self.causal = causal
        self.chunk_causal = chunk_causal
        # Check that samples_per_epoch is a power of 2
        if samples_per_epoch & (samples_per_epoch - 1) != 0:
            raise ValueError(f'samples_per_epoch must be a power of 2, got {samples_per_epoch}')
        # Calculate number of convolutional blocks to downsample the input to four feature vectors per sleep epoch (30 seconds).
        num_blocks = int(math.log2(samples_per_epoch)) - 2
        logger.debug(f'Creating {num_blocks=} convolutional blocks.')
        # Double number of channels every other block.
        channels = [min(initial_channels * 2 ** (i // 2), max_channels) for i in range(num_blocks)]
        blocks = []
        # Note: When causal=True, we can achieve causality by chunking the input into 30-second epochs OR by using causal convolutions.
        _causal_conv_mode = causal and not chunk_causal
        for i, output_dim in enumerate(channels):
            if norm == 'auto':
                if i < 2:  # Use instance norm in early layers.
                    _norm_i = 'instance'
                else:
                    _norm_i = 'layer'
            else:
                _norm_i = norm
            # Use a larger epsilon for instance norm to prevent NaN from low-variance feature maps.
            # This applies to both causal and non-causal modes for numerical stability.
            _norm_eps = 1e-2 if _norm_i == 'instance' else None
            blocks.append(
                ConvBlock1D(
                    input_dim,
                    output_dim,
                    activation=activation,
                    norm=_norm_i,
                    norm_eps=_norm_eps,
                    causal=_causal_conv_mode,
                    use_residual=use_residual,
                )
            )
            input_dim = output_dim
        self.cnn = nn.Sequential(*blocks)
        self.epoch_dim = channels[-1] * 4  # Flattened dimension for time-distributed dense layer.
        self.linear = nn.Linear(self.epoch_dim, feature_dim)
        self.activation = get_activation(activation)
        # Optional output normalization - normalizes across feature_dim (causal-friendly).
        self.output_norm = nn.LayerNorm(feature_dim) if output_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [B, T]

        Returns:
            Tensor: shape [B, S, f_dim]
        """
        if x.size(-1) % self.samples_per_epoch:
            raise ValueError(f'Input length {x.size(-1)} must be divisible by {self.samples_per_epoch=}.')
        B = x.size(0)
        S = x.size(-1) // self.samples_per_epoch

        if self.causal and self.chunk_causal:
            # Process each epoch independently for quasi-causal operation.
            # Reshape: [B, T] -> [B*S, 1, samples_per_epoch]
            y = x.view(B, S, self.samples_per_epoch)
            y = y.reshape(B * S, 1, self.samples_per_epoch)
            y = self.cnn(y)
            # y shape: [B*S, C, 4] -> [B*S, 4, C] -> [B, S, epoch_dim]
            y = y.transpose(-1, -2).reshape(B, S, self.epoch_dim)
        else:
            # Non-causal path: process entire sequence at once.
            y = x.unsqueeze(1)  # Add channel dim [B, 1, T]
            y = self.cnn(y)
            # Re-shape for time-distributed dense layer.
            y = y.transpose(-1, -2).reshape(B, -1, self.epoch_dim)

        # Apply FC layer.
        y = self.linear(y)
        y = self.activation(y)
        y = self.output_norm(y)
        return y


class MultiModalAttentionEmbedder(nn.Module):
    """Block that combines feature vectors from multiple signals using attention."""

    def __init__(
        self,
        feature_dim: int,
        layers: int = 4,
        dropout: float = 0.0,
        dim_ff: int = 512,
        activation: str = 'gelu',
        norm_first: bool = True,
        nhead: int = 4,
        register_tokens: int = 0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            dim_feedforward=dim_ff,
            activation=get_activation(activation),
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            norm_first=norm_first,
        )
        self.num_layers = layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # Learnable matrix of the CLS token + any register tokens.
        self.num_register_tokens = register_tokens
        self.register_tokens = nn.Parameter(torch.randn(1, 1, self.feature_dim, register_tokens + 1))

    def forward(self, z_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Turn multi-modal inputs into tokens.

        Patches is a dictionary mapping from each signal patch to the
        corresponding embedder. Signal patches may be of different sizes
        e.g. due to different sampling rates. However, each embedder must
        transform the patch to the same feature dimension.
        """
        z_stack: list[torch.Tensor] = []  # Feature vectors for each modality
        m_stack: list[torch.Tensor] = []  # Mask vectors for each modality
        signals = [signal_name for signal_name in sorted(z_dict.keys())]
        if len(signals) == 0:
            raise ValueError('No signals provided to MultiModalAttentionEmbedder.')
        # Embedding different modalities.
        for signal_name in signals:
            z_BSF = z_dict[signal_name]
            B, S, *_ = z_BSF.size()
            # Find where channels are missing within a batch, set to zero for stability.
            m_B = (torch.isinf(z_BSF)).any(axis=[1, 2])
            z_BSF = torch.where(m_B[:, None, None], 0.0, z_BSF)
            # Concat.
            z_stack.append(z_BSF)
            m_stack.append(m_B)
        z_BSFC = torch.stack(z_stack, dim=-1)
        m_BC = torch.stack(m_stack, dim=-1)  # True where signals unavailable.
        B, S, F, C = z_BSFC.size()
        if F != self.feature_dim:
            raise ValueError(f'Feature dimension {F} does not match {self.feature_dim=}.')
        # Add CLS and register tokens per signal patch. D = C + R (no. register_tokens + 1)
        z_BSFD = torch.cat([self.register_tokens.repeat(B, S, 1, 1), z_BSFC], dim=-1)
        B, S, F, D = z_BSFD.size()
        # Create padding mask so transformer can't attend to unavailable signals.
        # Can always attend to CLS and any register tokens.
        m_BR = torch.zeros_like(m_BC[:, 0]).bool()[:, None].repeat(1, self.num_register_tokens + 1)
        m_BD = torch.cat([m_BR, m_BC], dim=-1)
        # Reshape features for signal-wise per-patch attention.
        z_NDF = z_BSFD.flatten(start_dim=0, end_dim=1).permute(dims=(0, 2, 1))
        # Reshape mask for signal-wise per-patch attention.
        m_BSD = m_BD[:, None, :].repeat(1, S, 1)
        m_ND = m_BSD.flatten(start_dim=0, end_dim=1)
        # Apply transformer to signal-wise per-patch features. Reshape.
        z_NDF = self.transformer_encoder(z_NDF, src_key_padding_mask=m_ND)
        z_BSFD = z_NDF.permute(dims=(0, 2, 1)).reshape(B, S, F, D)
        # Return just the CLS token per-patch as the feature vector.
        z_BSF = z_BSFD[:, :, :, 0]
        return z_BSF


class SequenceCNN(nn.Module):
    """Simple dilated CNN model for sequence mixing."""

    def __init__(
        self,
        feature_dim: int = 128,
        dropout: float = 0.2,
        num_layers: int = 2,
        activation: str = 'gelu',
        norm: str = 'batch',
        causal: bool = False,
        num_dilations: int = 6,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.dilated_convs = nn.Sequential(
            *[
                DilatedConvBlock(
                    feature_dim=feature_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    causal=causal,
                    num_dilations=num_dilations,
                    kernel_size=kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_BSF: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (Tensor): shape [N, S, F]
        Returns:
            Tensor: shape [N, S, F]
        """
        # Re-shape back to channel-first for dilated convolutions.
        x_BFS = x_BSF.transpose(-1, -2)
        x_BFS = self.dilated_convs(x_BFS)
        return x_BFS.transpose(-1, -2)
