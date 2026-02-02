import torch

from wav2sleep.models.wav2sleep import (
    MultiModalAttentionEmbedder,
    SequenceCNN,
    SignalEncoders,
    Wav2Sleep,
)


def test_compile():
    """Check model is still compileable."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_dim = 16
    encoders = SignalEncoders(
        signal_map={
            'ECG': 'ECG',
            'PPG': 'PPG',
        },
        feature_dim=feature_dim,
        activation='relu',
        norm='instance',
    )
    encoders = encoders.to(device)
    encoders.compile(fullgraph=True)
    model = Wav2Sleep(
        signal_encoders=encoders,
        epoch_mixer=MultiModalAttentionEmbedder(feature_dim=feature_dim),
        sequence_mixer=SequenceCNN(feature_dim=feature_dim),
        num_classes=1,
    )
    model = model.to(device)

    torch._functorch.config.activation_memory_budget = 0.2
    model.compile(mode='max-autotune', fullgraph=True)

    x = torch.randn(1, 1_228_800, device=device)
    encoders({'ECG': x, 'PPG': x})
    model({'ECG': x, 'PPG': x})
    return
