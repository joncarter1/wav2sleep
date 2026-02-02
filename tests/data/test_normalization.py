"""Tests for causal normalization functions."""

import numpy as np
import pytest
import torch

from wav2sleep.data.normalization import causal_rolling_normalize


# Reference implementation (pure Python, no Numba) for comparison
def _reference_causal_normalize(
    signal: np.ndarray,
    sampling_freq: float,
    tau_seconds: float = 900.0,
    eps: float = 1e-6,
    outlier_threshold_sigma: float = 4.0,
    baseline_tau_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure Python reference implementation for testing."""
    signal_np = signal.astype(np.float64)

    dt = 1.0 / sampling_freq
    baseline_tau = baseline_tau_seconds if baseline_tau_seconds is not None else tau_seconds
    alpha_baseline = dt / baseline_tau
    alpha_variance = dt / tau_seconds

    warm_up_tau = min(baseline_tau, tau_seconds)
    warm_up_samples = int(warm_up_tau * sampling_freq)
    warm_up_samples = min(warm_up_samples, len(signal_np) // 10)
    warm_up_samples = max(1, warm_up_samples)
    warm_up_signal = signal_np[:warm_up_samples]
    mu_init = np.mean(warm_up_signal)
    sigma_sq_init = np.var(warm_up_signal)

    mu = np.zeros_like(signal_np)
    sigma_sq = np.zeros_like(signal_np)
    outlier_mask = np.zeros(len(signal_np), dtype=bool)

    mu[0] = mu_init
    sigma_sq[0] = max(sigma_sq_init, eps)

    for t in range(1, len(signal_np)):
        mu[t] = alpha_baseline * signal_np[t] + (1.0 - alpha_baseline) * mu[t - 1]
        residual = signal_np[t] - mu[t]
        sigma_prev = np.sqrt(sigma_sq[t - 1])
        threshold = outlier_threshold_sigma * sigma_prev

        if abs(residual) > threshold:
            outlier_mask[t] = True
            if residual > threshold:
                residual = threshold
            else:
                residual = -threshold

        sigma_sq[t] = alpha_variance * (residual * residual) + (1.0 - alpha_variance) * sigma_sq[t - 1]

    sigma = np.sqrt(np.maximum(sigma_sq, eps))
    normalized = (signal_np - mu) / sigma
    return normalized, outlier_mask


class TestCausalNormalizationCorrectness:
    """Tests to verify the optimized implementation matches reference behavior."""

    def test_matches_reference_random_signal(self):
        """Optimized implementation should match reference on random data."""
        np.random.seed(42)
        signal = np.random.randn(10000).astype(np.float32)
        sampling_freq = 34.0  # ECG/PPG rate

        result, mask = causal_rolling_normalize(
            signal, sampling_freq, return_outlier_mask=True, baseline_tau_seconds=120.0
        )
        expected, expected_mask = _reference_causal_normalize(signal, sampling_freq, baseline_tau_seconds=120.0)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_matches_reference_with_outliers(self):
        """Should correctly detect and handle outliers."""
        np.random.seed(123)
        signal = np.random.randn(5000).astype(np.float32)
        # Inject some outliers
        signal[1000] = 50.0
        signal[2000] = -50.0
        signal[3000] = 100.0
        sampling_freq = 34.0

        result, mask = causal_rolling_normalize(
            signal, sampling_freq, return_outlier_mask=True, baseline_tau_seconds=120.0
        )
        expected, expected_mask = _reference_causal_normalize(signal, sampling_freq, baseline_tau_seconds=120.0)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(mask, expected_mask)
        # Verify outliers were detected
        assert mask[1000] or mask[1001]  # Outlier detection may lag by 1 sample
        assert mask[2000] or mask[2001]
        assert mask[3000] or mask[3001]

    def test_matches_reference_different_tau(self):
        """Should work with different tau values."""
        np.random.seed(456)
        signal = np.random.randn(8000).astype(np.float32)
        sampling_freq = 136.0  # EOG rate

        for tau in [300.0, 600.0, 1200.0]:
            for baseline_tau in [60.0, 120.0, 300.0]:
                result = causal_rolling_normalize(
                    signal, sampling_freq, tau_seconds=tau, baseline_tau_seconds=baseline_tau
                )
                expected, _ = _reference_causal_normalize(
                    signal, sampling_freq, tau_seconds=tau, baseline_tau_seconds=baseline_tau
                )
                np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)


class TestCausalNormalizationEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_signal(self):
        """Should handle empty signal gracefully."""
        signal = np.array([], dtype=np.float32)
        result = causal_rolling_normalize(signal, sampling_freq=34.0)
        assert len(result) == 0

    def test_single_sample(self):
        """Should handle single-sample signal."""
        signal = np.array([1.0], dtype=np.float32)
        result = causal_rolling_normalize(signal, sampling_freq=34.0)
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_constant_signal(self):
        """Constant signal should normalize to near-zero."""
        signal = np.ones(1000, dtype=np.float32) * 5.0
        result = causal_rolling_normalize(signal, sampling_freq=34.0)
        # After warm-up, constant signal should be ~0 (mean subtracted, divided by small std)
        # The exact value depends on eps, but should be finite
        assert np.all(np.isfinite(result))

    def test_very_short_signal(self):
        """Should handle very short signals (< warm-up period)."""
        signal = np.random.randn(10).astype(np.float32)
        result = causal_rolling_normalize(signal, sampling_freq=34.0)
        assert len(result) == 10
        assert np.all(np.isfinite(result))


class TestCausalNormalizationTypePreservation:
    """Tests for input/output type preservation."""

    def test_numpy_input_returns_numpy(self):
        """NumPy input should return NumPy output."""
        signal = np.random.randn(1000).astype(np.float32)
        result = causal_rolling_normalize(signal, sampling_freq=34.0)
        assert isinstance(result, np.ndarray)

    def test_torch_input_returns_torch(self):
        """Torch input should return Torch output."""
        signal = torch.randn(1000, dtype=torch.float32)
        result = causal_rolling_normalize(signal, sampling_freq=34.0)
        assert isinstance(result, torch.Tensor)

    def test_torch_dtype_preserved(self):
        """Torch dtype should be preserved."""
        for dtype in [torch.float32, torch.float64]:
            signal = torch.randn(1000, dtype=dtype)
            result = causal_rolling_normalize(signal, sampling_freq=34.0)
            assert result.dtype == dtype

    def test_torch_device_preserved(self):
        """Torch device should be preserved (CPU test only)."""
        signal = torch.randn(1000, dtype=torch.float32, device='cpu')
        result = causal_rolling_normalize(signal, sampling_freq=34.0)
        assert result.device == signal.device

    def test_torch_outlier_mask_type(self):
        """Outlier mask should be torch tensor when input is torch."""
        signal = torch.randn(1000, dtype=torch.float32)
        result, mask = causal_rolling_normalize(signal, sampling_freq=34.0, return_outlier_mask=True)
        assert isinstance(result, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool


class TestCausalNormalizationCausality:
    """Tests to verify the causality property (no future information leakage)."""

    def test_causality_suffix_independent(self):
        """Appending different suffixes should not change the prefix output.

        Note: The warm-up period uses 10% of the signal, so we need signals
        of the same length to have identical warm-up. This test verifies that
        changing the suffix (after warm-up) doesn't affect earlier outputs.
        """
        np.random.seed(789)
        # Use same prefix
        prefix = np.random.randn(8000).astype(np.float32)
        # Different suffixes
        suffix_a = np.random.randn(2000).astype(np.float32)
        suffix_b = np.random.randn(2000).astype(np.float32) * 10  # Very different scale

        signal_a = np.concatenate([prefix, suffix_a])
        signal_b = np.concatenate([prefix, suffix_b])

        result_a = causal_rolling_normalize(signal_a, sampling_freq=34.0, baseline_tau_seconds=120.0)
        result_b = causal_rolling_normalize(signal_b, sampling_freq=34.0, baseline_tau_seconds=120.0)

        # The prefix (first 8000 samples) should be identical regardless of suffix
        np.testing.assert_allclose(result_a[:8000], result_b[:8000], rtol=1e-6, atol=1e-8)


class TestCausalNormalizationDeterminism:
    """Tests to verify deterministic behavior."""

    def test_deterministic_same_input(self):
        """Same input should always produce same output."""
        signal = np.random.randn(1000).astype(np.float32)

        result1 = causal_rolling_normalize(signal.copy(), sampling_freq=34.0)
        result2 = causal_rolling_normalize(signal.copy(), sampling_freq=34.0)

        np.testing.assert_array_equal(result1, result2)

    def test_deterministic_multiple_calls(self):
        """Multiple calls should produce identical results."""
        np.random.seed(101)
        signal = np.random.randn(2000).astype(np.float32)

        results = [causal_rolling_normalize(signal.copy(), sampling_freq=34.0) for _ in range(5)]

        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)


class TestCausalNormalizationRealisticSignals:
    """Tests with realistic signal sizes and parameters."""

    @pytest.mark.parametrize(
        'samples_per_epoch,signal_name',
        [
            (1024, 'ECG/PPG'),
            (256, 'ABD/THX'),
            (4096, 'EOG'),
        ],
    )
    def test_realistic_signal_sizes(self, samples_per_epoch, signal_name):
        """Should handle realistic signal sizes for each modality."""
        # Simulate 1 hour of data (120 epochs)
        n_epochs = 120
        n_samples = samples_per_epoch * n_epochs
        sampling_freq = samples_per_epoch / 30.0

        np.random.seed(42)
        signal = np.random.randn(n_samples).astype(np.float32)

        result = causal_rolling_normalize(signal, sampling_freq=sampling_freq, baseline_tau_seconds=120.0)

        assert result.shape == signal.shape
        assert np.all(np.isfinite(result))
        # Normalized signal should have reasonable statistics
        assert abs(np.mean(result)) < 1.0  # Mean should be near zero
        assert 0.1 < np.std(result) < 10.0  # Std should be reasonable
