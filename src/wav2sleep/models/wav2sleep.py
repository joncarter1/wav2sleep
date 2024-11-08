"""wav2sleep model."""

import random

import torch
from torch import Tensor, nn

from ..settings import COL_MAP, HIGH_FREQ_LEN
from .ppgnet import ConvBlock1D, DilatedConvBlock
from .utils import embed_ignore_inf, get_activation

CHANNEL_CONFIGS = {
    'high': {'small': [16, 16, 32, 32, 32, 32, 32, 32], 'large': [16, 16, 32, 32, 64, 64, 128, 128]},
    'low': {'small': [16, 16, 32, 32, 32, 32], 'large': [16, 32, 64, 64, 128, 128]},
}


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


class SignalEncoder(nn.Module):
    """Signal encoder layer.

    Progressively downsamples the input waveform, resulting in feature vector sequence for each sleep epoch.
    Then applies time-distributed dense layer to produce the final feature vector.
    """

    def __init__(
        self,
        input_dim: int = 1,
        feature_dim: int = 256,
        activation: str = 'relu',
        frequency: str = 'high',
        size: str = 'large',
        norm: str = 'instance',
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        channels = CHANNEL_CONFIGS[frequency][size]
        blocks = []
        for output_dim in channels:
            blocks.append(ConvBlock1D(input_dim, output_dim, activation=activation, norm=norm))
            input_dim = output_dim
        self.cnn = nn.Sequential(*blocks)
        self.epoch_dim = channels[-1] * 4  # Flattenedß dimension for time-distributed dense layer.
        self.linear = nn.Linear(self.epoch_dim, feature_dim)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape [B, T]

        Returns:
            Tensor: shape [B, S, f_dim]
        """
        B = x.size(0)
        y = x.unsqueeze(1)  # Add channel dim
        y = self.cnn(y)
        # Re-shape for time-distributed dense layer.
        y = y.transpose(-1, -2).reshape(B, -1, self.epoch_dim)
        # Apply FC layer.
        y = self.linear(y)
        return self.activation(y)


class SignalEncoders(nn.Module):
    """Class that handles multiple signal encoders."""

    def __init__(
        self,
        signal_map: dict[str, str],
        feature_dim: int,
        activation: str,
        norm: str = 'instance',
        size='large',
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.signal_map = signal_map
        encoders = {}
        # Create encoders for the signals.
        # Multiple signals *can* map to the same encoder, though we found this didn't work as well.
        for signal_name, encoder_name in self.signal_map.items():
            if encoder_name in encoders:
                continue
            if signal_name not in COL_MAP:
                raise ValueError(f'Column {signal_name} unrecognised.')
            frequency = 'high' if COL_MAP[signal_name] == HIGH_FREQ_LEN else 'low'
            encoders[encoder_name] = SignalEncoder(
                input_dim=1, feature_dim=feature_dim, frequency=frequency, activation=activation, size=size, norm=norm
            )
        self.encoders = nn.ModuleDict(encoders)

    def __len__(self) -> int:
        return len(self.encoders)

    def get_encoder(self, signal_name: str) -> SignalEncoder:
        """Get the signal encoder for a signal."""
        if self.signal_map is not None:
            encoder_name = self.signal_map[signal_name]
        else:
            encoder_name = signal_name
        return self.encoders[encoder_name]  # type: ignore

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_dict: dict[str, torch.Tensor] = {}
        # Apply signal encoder for each input signal.
        # Slow (sequential) embedding of different modalities, perhaps e.g. torch.compile could speed this up.
        for signal_name, x_BT in x.items():
            x_BT = x[signal_name]
            z_BSF = embed_ignore_inf(x_BT, self.get_encoder(signal_name))
            z_dict[signal_name] = z_BSF
        return z_dict


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
        permute_signals: bool = True,
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
        self.permute_signals = permute_signals

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
        # Randomly permute signal ordering for transformer.
        # Shouldn't make a difference but no harm... (see NoPE paper)
        if self.training:
            random.shuffle(signals)
        # Slow (sequential) embedding of different modalities.
        # Perhaps e.g. torch.compile could speed this up...
        for signal_name in signals:
            z_BSF = z_dict[signal_name]
            B, S, *_ = z_BSF.size()
            # Find where entire channel was missing within a batch (feature output will be all zero).
            m_B = (z_BSF == 0).all(axis=[1, 2])
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
    ) -> None:
        super().__init__()
        self.dilated_convs = nn.Sequential(
            *[
                DilatedConvBlock(feature_dim=feature_dim, dropout=dropout, activation=activation, norm=norm)
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
