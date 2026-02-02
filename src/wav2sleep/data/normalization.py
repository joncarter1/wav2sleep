"""Causal normalization utilities for physiological signals.

This module provides causal (online) normalization using exponential moving average (EMA)
to track baseline and variance without using future information. This is critical for
real-time or streaming applications where predictions must be made with only past data.

The main function `causal_rolling_normalize()` implements EMA-based z-score normalization
with outlier clipping to prevent variance corruption from extreme values.
"""

from typing import Union

import numpy as np
import torch
from numba import njit


@njit(cache=True)
def _ema_normalize_loop(
    signal: np.ndarray,
    alpha_baseline: float,
    alpha_variance: float,
    mu_init: float,
    sigma_sq_init: float,
    outlier_threshold_sigma: float,
    min_sigma_sq: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-optimized EMA normalization loop.

    This is the performance-critical inner loop that computes running mean and variance
    using exponential moving average. Compiled with Numba for ~50-100x speedup over
    pure Python.

    Args:
        signal: 1D array of signal values (float64)
        alpha_baseline: Smoothing factor for mean tracking (dt / baseline_tau)
        alpha_variance: Smoothing factor for variance tracking (dt / tau)
        mu_init: Initial mean estimate from warm-up period
        sigma_sq_init: Initial variance estimate from warm-up period
        outlier_threshold_sigma: Number of standard deviations for outlier detection
        min_sigma_sq: Minimum variance floor (squared) to prevent division by near-zero
        eps: Small epsilon to prevent division by zero

    Returns:
        Tuple of (mu, sigma_sq, outlier_mask) arrays, each with same length as signal
    """
    n = len(signal)
    mu = np.empty(n, dtype=np.float64)
    sigma_sq = np.empty(n, dtype=np.float64)
    outlier_mask = np.zeros(n, dtype=np.bool_)

    mu[0] = mu_init
    # Apply floor to initial sigma
    sigma_sq[0] = max(sigma_sq_init, min_sigma_sq, eps)

    for t in range(1, n):
        # Update mean: mu_t = alpha_baseline * x_t + (1 - alpha_baseline) * mu_{t-1}
        mu[t] = alpha_baseline * signal[t] + (1.0 - alpha_baseline) * mu[t - 1]

        # Compute residual after baseline correction
        residual = signal[t] - mu[t]
        # Use floored sigma for outlier detection
        sigma_prev = np.sqrt(max(sigma_sq[t - 1], min_sigma_sq))

        # Outlier detection and clipping
        threshold = outlier_threshold_sigma * sigma_prev
        if abs(residual) > threshold:
            outlier_mask[t] = True
            # Clip residual to prevent variance corruption
            if residual > threshold:
                residual = threshold
            else:
                residual = -threshold

        # Update variance: sigma^2_t = alpha_var * residual^2 + (1 - alpha_var) * sigma^2_{t-1}
        sigma_sq[t] = alpha_variance * (residual * residual) + (1.0 - alpha_variance) * sigma_sq[t - 1]

    return mu, sigma_sq, outlier_mask


def compute_sampling_freq_from_epoch_samples(
    samples_per_epoch: int,
    epoch_duration_seconds: float = 30.0,
) -> float:
    """Compute sampling frequency from samples per epoch.

    Args:
        samples_per_epoch: Number of samples in one epoch (default 30-second epochs)
        epoch_duration_seconds: Duration of epoch in seconds (default 30)

    Returns:
        Sampling frequency in Hz

    Examples:
        >>> compute_sampling_freq_from_epoch_samples(1024)  # ECG/PPG
        34.13333...
        >>> compute_sampling_freq_from_epoch_samples(256)  # ABD/THX
        8.53333...
        >>> compute_sampling_freq_from_epoch_samples(4096)  # EOG
        136.53333...
    """
    return samples_per_epoch / epoch_duration_seconds


def causal_rolling_normalize(
    signal: Union[np.ndarray, torch.Tensor],
    sampling_freq: float,
    tau_seconds: float = 900.0,
    eps: float = 1e-6,
    outlier_threshold_sigma: float = 4.0,
    return_outlier_mask: bool = False,
    baseline_tau_seconds: float | None = None,
    min_sigma: float = 0.1,
) -> Union[np.ndarray, torch.Tensor, tuple]:
    """Apply causal exponential moving average (EMA) normalization to a 1D signal.

    Uses exponential moving average to compute running mean and variance, then normalizes
    the signal as (x_t - mu_t) / max(sigma_t, min_sigma). This is a causal operation that only
    uses past samples, making it suitable for real-time processing or when future information
    must not leak into predictions.

    Supports two-stage normalization with separate time constants for baseline and variance:
    - baseline_tau_seconds: Controls how fast the mean tracks baseline drift (faster = smaller tau)
    - tau_seconds: Controls how fast variance adapts (slower = larger tau for stability)

    For signals with baseline drift (like CFS/CCSHS PPG), use baseline_tau_seconds=120-300
    with tau_seconds=900 to track drift quickly while keeping variance estimation stable.

    Args:
        signal: 1D array of signal values (numpy array or torch tensor)
        sampling_freq: Sampling frequency in Hz
        tau_seconds: Time constant in seconds for variance tracking.
            Larger values = slower adaptation (smoother variance estimate).
            Default 900s (15 minutes) provides stable variance estimation.
        eps: Small epsilon value to prevent division by zero
        outlier_threshold_sigma: Number of standard deviations to consider a residual an outlier.
            Outliers are detected after baseline correction (residual = x_t - mu_t).
            Default 4.0 provides robust outlier rejection.
        return_outlier_mask: If True, also return boolean mask indicating outlier samples
        baseline_tau_seconds: Time constant for baseline (mean) tracking. If None, uses tau_seconds.
            Use smaller values (e.g., 120-300s) to track baseline drift faster while keeping
            variance tracking slow for stability. This is useful for signals with significant
            baseline wander like PPG from CFS/CCSHS datasets.
        min_sigma: Minimum sigma floor to prevent division by near-zero variance.
            Default 0.1 handles flat/saturated signal segments without over-damping.

    Returns:
        Normalized signal with same shape and type as input. If return_outlier_mask is True,
        returns tuple of (normalized_signal, outlier_mask).

    Notes:
        - Input type (numpy/torch) is preserved in output
        - For torch tensors, device and dtype are preserved
        - Baseline EMA update: mu_t = alpha_baseline * x_t + (1 - alpha_baseline) * mu_{t-1}
        - Variance EMA update: sigma^2_t = alpha_var * residual^2 + (1 - alpha_var) * sigma^2_{t-1}
        - Alpha = dt / tau where dt = 1 / sampling_freq
        - Smaller alpha = slower adaptation (larger time constant)
    """
    # Detect input type and convert torch to numpy if needed
    is_torch = isinstance(signal, torch.Tensor)
    if is_torch:
        device = signal.device
        dtype = signal.dtype
        signal_np = signal.cpu().numpy()
    else:
        signal_np = signal

    # Handle edge case: empty signal
    if len(signal_np) == 0:
        if is_torch:
            if return_outlier_mask:
                return signal, torch.zeros(0, dtype=torch.bool, device=device)
            return signal
        else:
            if return_outlier_mask:
                return signal_np, np.zeros(0, dtype=bool)
            return signal_np

    # Compute alpha (smoothing factor) from time constant
    # alpha = dt / tau where dt = 1 / sampling_freq
    dt = 1.0 / sampling_freq

    # Use separate time constants for baseline and variance if specified
    baseline_tau = baseline_tau_seconds if baseline_tau_seconds is not None else tau_seconds
    alpha_baseline = dt / baseline_tau  # For mean tracking
    alpha_variance = dt / tau_seconds  # For variance tracking

    # Precompute min_sigma_sq for efficiency
    min_sigma_sq = min_sigma * min_sigma

    # First N samples to use for initial mean and variance estimation
    # Use the faster tau for warm-up to get a good initial estimate
    warm_up_tau = min(baseline_tau, tau_seconds)
    warm_up_samples = int(warm_up_tau * sampling_freq)
    warm_up_samples = min(warm_up_samples, len(signal_np) // 10)  # Cap at 10% of signal
    warm_up_samples = max(1, warm_up_samples)
    warm_up_signal = signal_np[:warm_up_samples]
    mu_init = np.mean(warm_up_signal)
    # Apply floor to warm-up variance
    sigma_sq_init = max(np.var(warm_up_signal), min_sigma_sq)

    # Convert to float64 for Numba (required for consistent behavior)
    signal_f64 = signal_np.astype(np.float64)

    # Call Numba-optimized loop
    mu, sigma_sq, outlier_mask = _ema_normalize_loop(
        signal_f64,
        alpha_baseline,
        alpha_variance,
        float(mu_init),
        float(sigma_sq_init),
        outlier_threshold_sigma,
        min_sigma_sq,
        eps,
    )

    # Normalize: (x - mu) / sigma with floor applied
    sigma = np.sqrt(np.maximum(sigma_sq, min_sigma_sq))
    normalized = (signal_np - mu) / sigma

    # Convert back to torch if needed
    if is_torch:
        result = torch.from_numpy(normalized).to(device=device, dtype=dtype)
        if return_outlier_mask:
            mask = torch.from_numpy(outlier_mask).to(device=device)
            return result, mask
        return result
    else:
        if return_outlier_mask:
            return normalized, outlier_mask
        return normalized
