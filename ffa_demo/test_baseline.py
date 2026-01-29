#!/usr/bin/env python3
"""
Test suite for FFA baseline functions.

Run this before evolution to ensure the baseline implementation is correct
and the evaluator is working properly.

Usage:
    python -m pytest ffa_demo/test_baseline.py -v
    
    # Or run directly:
    python ffa_demo/test_baseline.py
"""

import numpy as np
import pytest
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffa_demo.data_generator import (
    PulsarParams,
    generate_pulsar_timeseries,
    generate_gaussian_pulse_profile,
    generate_test_dataset,
)
from ffa_demo.baseline_functions import (
    boxcar_matched_filter,
    boxcar_matched_filter_reference,
    fold_timeseries,
    compute_profile_snr,
    compute_filtered_snr,
)
from ffa_demo.evaluator import (
    evaluate_filter_function,
    evaluate_on_pulsar_detection,
    EvaluationResult,
)


class TestDataGenerator:
    """Tests for synthetic pulsar data generation."""

    def test_pulsar_params_pulse_width(self):
        """Test PulsarParams.pulse_width property."""
        params = PulsarParams(period=0.5, duty_cycle=0.1, snr=10)
        assert params.pulse_width == 0.05  # 0.5 * 0.1

    def test_generate_pulse_profile_shape(self):
        """Test pulse profile generation has correct shape."""
        profile = generate_gaussian_pulse_profile(128, duty_cycle=0.05)
        assert profile.shape == (128,)

    def test_generate_pulse_profile_normalized(self):
        """Test pulse profile is normalized to unit sum of squares."""
        profile = generate_gaussian_pulse_profile(64, duty_cycle=0.1)
        sum_sq = np.sum(profile**2)
        assert np.isclose(sum_sq, 1.0, atol=1e-6)

    def test_generate_timeseries_shape(self):
        """Test timeseries generation has correct shape."""
        params = PulsarParams(period=0.5, duty_cycle=0.05, snr=10)
        data, meta = generate_pulsar_timeseries(
            params, duration=10, sample_rate=1000, seed=42
        )
        expected_samples = 10 * 1000
        assert data.shape == (expected_samples,)
        assert meta["n_samples"] == expected_samples

    def test_generate_timeseries_reproducible(self):
        """Test that seed makes generation reproducible."""
        params = PulsarParams(period=0.5, duty_cycle=0.05, snr=10)
        data1, _ = generate_pulsar_timeseries(
            params, duration=5, sample_rate=1000, seed=123
        )
        data2, _ = generate_pulsar_timeseries(
            params, duration=5, sample_rate=1000, seed=123
        )
        np.testing.assert_array_equal(data1, data2)

    def test_generate_test_dataset_length(self):
        """Test dataset generation returns correct number of pulsars."""
        dataset = generate_test_dataset(n_pulsars=5, duration=10, sample_rate=1000)
        assert len(dataset) == 5
        for data, params, meta in dataset:
            assert isinstance(data, np.ndarray)
            assert isinstance(params, PulsarParams)
            assert isinstance(meta, dict)


class TestBaselineFunctions:
    """Tests for baseline FFA functions."""

    def test_boxcar_filter_matches_reference(self):
        """Test baseline matches reference implementation."""
        np.random.seed(42)
        profile = np.random.randn(128)

        for width in [1, 4, 8, 16, 32]:
            result = boxcar_matched_filter(profile, width)
            reference = boxcar_matched_filter_reference(profile, width)
            np.testing.assert_allclose(result, reference, atol=1e-10)

    def test_boxcar_filter_width_one(self):
        """Test width=1 returns normalized input."""
        profile = np.array([1.0, 2.0, 3.0, 4.0])
        result = boxcar_matched_filter(profile, 1)
        expected = profile / np.sqrt(1)  # sqrt(1) = 1
        np.testing.assert_allclose(result, expected)

    def test_boxcar_filter_circular(self):
        """Test circular convolution wraps around correctly."""
        profile = np.array([1.0, 0.0, 0.0, 0.0])
        result = boxcar_matched_filter(profile, 2)
        # Width 2: [0]+[1], [1]+[2], [2]+[3], [3]+[0]
        expected = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        np.testing.assert_allclose(result, expected)

    def test_fold_timeseries_basic(self):
        """Test basic folding operation."""
        # Simple periodic signal (use floats to avoid type issues)
        data = np.tile([1.0, 2.0, 3.0, 4.0], 10)  # 40 samples, period 4
        folded = fold_timeseries(data, period_samples=4)
        # Sum of 10 periods, normalized by sqrt(10)
        expected = np.array([1.0, 2.0, 3.0, 4.0]) * 10 / np.sqrt(10)
        np.testing.assert_allclose(folded, expected)

    def test_fold_timeseries_snr_scaling(self):
        """Test that folding increases SNR by sqrt(N)."""
        np.random.seed(42)
        # Create signal with known SNR
        n_periods = 100
        period = 10
        n_samples = n_periods * period
        
        signal = np.zeros(n_samples)
        for i in range(n_periods):
            signal[i * period] = 1.0  # Unit pulse at start of each period
        
        noise = np.random.randn(n_samples) * 0.1
        data = signal + noise
        
        folded = fold_timeseries(data, period)
        # Peak should be roughly n_periods / sqrt(n_periods) = sqrt(n_periods)
        assert folded.max() > 5  # Should be ~10 for 100 periods

    def test_compute_profile_snr_pulse(self):
        """Test SNR computation on profile with pulse."""
        np.random.seed(42)
        profile = np.random.randn(64) * 0.1
        profile[30:35] += 5.0  # Add strong pulse
        
        snr = compute_profile_snr(profile)
        assert snr > 10  # Should detect strong pulse

    def test_compute_profile_snr_noise(self):
        """Test SNR on pure noise is moderate."""
        np.random.seed(42)
        profile = np.random.randn(64)
        
        snr = compute_profile_snr(profile)
        # Pure noise can have SNR up to ~10 due to random peaks
        # Main check: it's much lower than a real pulse (which would be >20)
        assert snr < 15  # Pure noise, moderate SNR

    def test_compute_filtered_snr_improves(self):
        """Test that matched filter improves SNR for boxcar pulse."""
        np.random.seed(42)
        profile = np.random.randn(128) * 0.5
        profile[60:68] += 3.0  # Width-8 boxcar pulse
        
        snr_unfiltered = compute_profile_snr(profile)
        snr_filtered, best_width = compute_filtered_snr(profile, boxcar_matched_filter)
        
        assert snr_filtered > snr_unfiltered
        assert best_width in [4, 8, 12]  # Should find width near 8


class TestEvaluator:
    """Tests for the fitness evaluator."""

    def test_evaluate_baseline_passes(self):
        """Test baseline implementation passes evaluation."""
        result = evaluate_filter_function(
            boxcar_matched_filter,
            enforce_runtime_constraint=False,
        )
        
        assert result.is_correct
        assert result.max_diff < 1e-8
        assert result.snr_error < 0.01
        assert result.speedup > 0

    def test_evaluate_reference_is_baseline(self):
        """Test reference vs reference gives speedup ~1."""
        result = evaluate_filter_function(
            boxcar_matched_filter_reference,
            baseline_func=boxcar_matched_filter_reference,
            enforce_runtime_constraint=False,
        )
        
        assert result.is_correct
        # Allow wider range due to timing variance and system noise
        assert 0.5 < result.speedup < 2.0  # Should be ~1.0

    def test_evaluate_wrong_output_fails(self):
        """Test that incorrect implementation fails."""
        def bad_filter(profile, width):
            return np.zeros_like(profile)  # Wrong!
        
        result = evaluate_filter_function(bad_filter)
        
        assert not result.is_correct or not result.is_valid

    def test_evaluate_on_pulsar_detection(self):
        """Test pulsar detection evaluation."""
        result = evaluate_on_pulsar_detection(
            boxcar_matched_filter,
            n_pulsars=3,
            duration=15,
            sample_rate=5000,
        )
        
        assert "detection_rate" in result
        assert "avg_snr_ratio" in result
        assert result["n_total"] == 3


class TestPerformance:
    """Performance benchmarks."""

    def test_baseline_performance(self):
        """Benchmark baseline implementation."""
        np.random.seed(42)
        profile = np.random.randn(256)
        
        # Warm up
        for _ in range(3):
            _ = boxcar_matched_filter(profile, 16)
        
        # Time
        n_iter = 100
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = boxcar_matched_filter(profile, 16)
        elapsed = time.perf_counter() - start
        
        avg_ms = elapsed * 1000 / n_iter
        print(f"\nBaseline: {avg_ms:.3f} ms per call (256 bins, width 16)")
        
        # Should complete in reasonable time
        assert avg_ms < 10  # Less than 10ms

    def test_reference_performance(self):
        """Benchmark reference implementation."""
        np.random.seed(42)
        profile = np.random.randn(256)
        
        # Warm up
        for _ in range(3):
            _ = boxcar_matched_filter_reference(profile, 16)
        
        # Time
        n_iter = 100
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = boxcar_matched_filter_reference(profile, 16)
        elapsed = time.perf_counter() - start
        
        avg_ms = elapsed * 1000 / n_iter
        print(f"\nReference: {avg_ms:.3f} ms per call (256 bins, width 16)")


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("  FFA Baseline Test Suite")
    print("=" * 60)
    
    # Run pytest programmatically
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ])
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("  ✅ All tests passed!")
        print("  Baseline is ready for evolution.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  ❌ Some tests failed.")
        print("  Please fix issues before running evolution.")
        print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(run_all_tests())
