import torch
from torch.distributions.one_hot_categorical import OneHotCategorical


class SignalMasker:
    def __init__(self, dropouts: dict[str, float], backups: list[str] | None = None):
        self.channel_dropouts = dropouts
        self.backup_channels = backups

    def __call__(self, signals):
        channel_dropout_probs = []
        channel_onehot_probs = []
        z_stack: list[torch.Tensor] = []  # Mask vectors for each modality
        for signal_name, x_BT in signals.items():
            z_B = torch.isinf(x_BT[:, 0])
            channel_dropout = self.channel_dropouts.get(signal_name, 0.0)
            if channel_dropout < 0.0 or channel_dropout > 1:
                raise ValueError(f'{channel_dropout=} is not a valid probability.')
            channel_dropout_probs.append(channel_dropout)
            # Create vector to sample backup when all are stochastically masked in first pass.
            if self.backup_channels is not None:
                channel_onehot_probs.append(~z_B if signal_name in self.backup_channels else torch.zeros_like(z_B))
            else:
                channel_onehot_probs.append(~z_B * (1 - channel_dropout))
            z_stack.append(z_B)
        z_BC = torch.stack(z_stack, dim=-1)  # True where signals unavailable.
        if z_BC.all(dim=-1).any():
            raise ValueError('Found batch element with all signals unavailable.')
        B = z_BC.size(0)
        # Create probability vector for sampling channels to mask out.
        p_BC = torch.Tensor(channel_dropout_probs)[None, :].to(x_BT.device).repeat(B, 1)
        if (p_BC == 1).all(dim=-1).any():
            raise ValueError('Dropout probability equal to 1 for all channels.')
        # Create one-hot vector so we don't mask out all channels.
        p_min_m_BC = torch.stack(channel_onehot_probs, dim=-1).to(x_BT.device)
        # If specific back up channels specified, check they're available.
        if (p_min_m_BC == 0).all(dim=-1).any():
            raise ValueError('No backup channels for stochastic sampling were available')
        min_m_BC = OneHotCategorical(p_min_m_BC).sample().bool()
        # Create multiplicative mask to remove some channels. (1 = Keep)
        m_BC = (1 - p_BC).bernoulli().bool()
        # Make sure at least one channel is available and mask is non-zero.
        all_zero = torch.logical_or(z_BC, ~m_BC).all(dim=-1)
        m_BC[all_zero] = min_m_BC[all_zero]
        # Shouldn't be possible to trigger error below, retaining for safety.
        if torch.logical_or(z_BC, ~m_BC).all(dim=-1).any():
            raise ValueError('Masking will result in no available channels for a batch element.')
        # Mask signals by setting to neg. infinity.
        for signal_name, m_B in zip(signals.keys(), m_BC.T):
            signals[signal_name][~m_B] = float('-inf')
        return signals
