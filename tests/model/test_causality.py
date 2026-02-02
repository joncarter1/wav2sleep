import torch

from wav2sleep.models.wav2sleep import (
    MultiModalAttentionEmbedder,
    SequenceCNN,
    SignalEncoders,
    Wav2Sleep,
)


def test_causality():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    causal = True
    norm = 'batch'
    encoders = SignalEncoders(
        signal_map={
            'ECG': 'ECG',
            'PPG': 'PPG',
        },
        feature_dim=16,
        activation='relu',
        norm=norm,
        causal=causal,
    )
    model = Wav2Sleep(
        signal_encoders=encoders,
        epoch_mixer=MultiModalAttentionEmbedder(feature_dim=16),
        sequence_mixer=SequenceCNN(feature_dim=16, causal=causal, norm=norm),
        num_classes=4,
    )
    model = model.to(device)
    model.eval()
    L = 1_228_800
    x = torch.randn(1, L, device=device)
    x2 = x[:, : L // 2]
    y = model({'ECG': x, 'PPG': x})
    y2 = model({'ECG': x2, 'PPG': x2})
    L_out = y2.shape[1]
    assert torch.allclose(y[:, :L_out], y2[:, :L_out])
