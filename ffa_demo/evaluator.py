"""
Fitness Evaluator for FFA Evolution

Evaluates candidate `boxcar_matched_filter` implementations on:
1. Correctness: Results must match reference within tolerance
2. Speed: Execution time relative to baseline
3. S/N preservation: Detected SNR must be within 1% of reference

The overall fitness score combines these metrics with constraints.
"""

import time
import numpy as np
from typing import Callable
from dataclasses import dataclass

from .data_generator import PulsarParams, generate_pulsar_timeseries
from .baseline_functions import (
    boxcar_matched_filter_reference,
    fold_timeseries,
    compute_profile_snr,
)


@dataclass
class EvaluationResult:
    """Results from evaluating a candidate implementation."""

    fitness: float  # Overall fitness score (higher is better)
    speedup: float  # Speedup vs baseline (>1 means faster)
    snr_error: float  # Relative SNR error (0 = perfect)
    max_diff: float  # Maximum absolute difference from reference
    is_correct: bool  # Passes correctness check
    is_valid: bool  # Passes all constraints
    runtime_ms: float  # Average runtime in milliseconds
    baseline_runtime_ms: float  # Baseline runtime for comparison
    details: dict  # Additional metrics


# Runtime constraint: evolved code must be within ±5% of baseline runtime
# to ensure "constrained folding evolution" behavior
RUNTIME_TOLERANCE = 0.05  # 5%


def evaluate_filter_function(
    filter_func: Callable[[np.ndarray, int], np.ndarray],
    baseline_func: Callable[[np.ndarray, int], np.ndarray] | None = None,
    n_trials: int = 5,
    profile_sizes: list[int] | None = None,
    widths: list[int] | None = None,
    correctness_tol: float = 1e-8,
    snr_error_tol: float = 0.01,  # 1% SNR error tolerance
    enforce_runtime_constraint: bool = True,
    seed: int = 42,
) -> EvaluationResult:
    """
    Evaluate a candidate boxcar_matched_filter implementation.

    Args:
        filter_func: Candidate filter function to evaluate
        baseline_func: Baseline for comparison (default: reference impl)
        n_trials: Number of timing trials per configuration
        profile_sizes: List of profile sizes to test (default: [64, 128, 256, 512])
        widths: List of boxcar widths to test (default: [1, 4, 8, 16, 32])
        correctness_tol: Tolerance for numerical correctness
        snr_error_tol: Maximum allowed relative SNR error
        enforce_runtime_constraint: If True, reject if runtime exceeds ±5% of baseline
        seed: Random seed for reproducibility

    Returns:
        EvaluationResult with fitness score and metrics
    """
    np.random.seed(seed)

    if baseline_func is None:
        baseline_func = boxcar_matched_filter_reference

    if profile_sizes is None:
        profile_sizes = [64, 128, 256, 512]

    if widths is None:
        widths = [1, 4, 8, 16, 32]

    # Generate test profiles
    test_profiles = []
    for size in profile_sizes:
        profile = np.random.randn(size)
        # Add synthetic pulse
        pulse_start = size // 4
        pulse_width = max(size // 16, 2)
        profile[pulse_start : pulse_start + pulse_width] += 5.0
        test_profiles.append(profile)

    # Correctness check
    max_diff = 0.0
    total_snr_error = 0.0
    n_snr_tests = 0

    for profile in test_profiles:
        for width in widths:
            if width > len(profile) // 2:
                continue

            try:
                result = filter_func(profile, width)
                reference = baseline_func(profile, width)
            except Exception as e:
                # Function crashed - not valid
                return EvaluationResult(
                    fitness=0.0,
                    speedup=0.0,
                    snr_error=float("inf"),
                    max_diff=float("inf"),
                    is_correct=False,
                    is_valid=False,
                    runtime_ms=float("inf"),
                    baseline_runtime_ms=0.0,
                    details={"error": str(e)},
                )

            # Check shape
            if result.shape != reference.shape:
                return EvaluationResult(
                    fitness=0.0,
                    speedup=0.0,
                    snr_error=float("inf"),
                    max_diff=float("inf"),
                    is_correct=False,
                    is_valid=False,
                    runtime_ms=0.0,
                    baseline_runtime_ms=0.0,
                    details={"error": f"Shape mismatch: {result.shape} vs {reference.shape}"},
                )

            # Check numerical accuracy
            diff = np.max(np.abs(result - reference))
            max_diff = max(max_diff, diff)

            # Check SNR preservation
            snr_result = compute_profile_snr(result)
            snr_reference = compute_profile_snr(reference)
            if snr_reference > 1e-6:
                rel_error = abs(snr_result - snr_reference) / snr_reference
                total_snr_error += rel_error
                n_snr_tests += 1

    is_correct = max_diff < correctness_tol
    avg_snr_error = total_snr_error / n_snr_tests if n_snr_tests > 0 else 0.0
    snr_ok = avg_snr_error < snr_error_tol

    if not is_correct:
        return EvaluationResult(
            fitness=0.0,
            speedup=0.0,
            snr_error=avg_snr_error,
            max_diff=max_diff,
            is_correct=False,
            is_valid=False,
            runtime_ms=0.0,
            baseline_runtime_ms=0.0,
            details={"reason": "Numerical accuracy below threshold"},
        )

    # Timing benchmark
    # Use largest profile and medium width for representative timing
    bench_profile = test_profiles[-1]  # Largest
    bench_width = min(16, len(bench_profile) // 4)

    # Warm-up
    for _ in range(3):
        _ = filter_func(bench_profile, bench_width)
        _ = baseline_func(bench_profile, bench_width)

    # Time candidate
    times_candidate = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for _ in range(100):  # Multiple iterations for stability
            _ = filter_func(bench_profile, bench_width)
        elapsed = time.perf_counter() - start
        times_candidate.append(elapsed / 100)

    # Time baseline
    times_baseline = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for _ in range(100):
            _ = baseline_func(bench_profile, bench_width)
        elapsed = time.perf_counter() - start
        times_baseline.append(elapsed / 100)

    avg_candidate_ms = np.median(times_candidate) * 1000
    avg_baseline_ms = np.median(times_baseline) * 1000

    speedup = avg_baseline_ms / avg_candidate_ms if avg_candidate_ms > 0 else 0.0

    # Runtime constraint check (±5%)
    runtime_in_bounds = True
    if enforce_runtime_constraint:
        min_allowed = avg_baseline_ms * (1 - RUNTIME_TOLERANCE)
        max_allowed = avg_baseline_ms * (1 + RUNTIME_TOLERANCE)
        runtime_in_bounds = min_allowed <= avg_candidate_ms <= max_allowed

    is_valid = is_correct and snr_ok and runtime_in_bounds

    # Compute fitness score
    # Fitness = speedup * accuracy_factor * snr_factor
    # If invalid (constraint violation), fitness is heavily penalized

    if not is_valid:
        # Penalized fitness for invalid solutions
        fitness = 0.1 * speedup * (1.0 - avg_snr_error)
    else:
        # Valid solution: reward speedup
        accuracy_factor = 1.0 - min(max_diff / correctness_tol, 1.0) * 0.1
        snr_factor = 1.0 - min(avg_snr_error / snr_error_tol, 1.0) * 0.1
        fitness = speedup * accuracy_factor * snr_factor

    return EvaluationResult(
        fitness=fitness,
        speedup=speedup,
        snr_error=avg_snr_error,
        max_diff=max_diff,
        is_correct=is_correct,
        is_valid=is_valid,
        runtime_ms=avg_candidate_ms,
        baseline_runtime_ms=avg_baseline_ms,
        details={
            "profile_sizes": profile_sizes,
            "widths": widths,
            "n_snr_tests": n_snr_tests,
            "runtime_in_bounds": runtime_in_bounds,
        },
    )


def evaluate_on_pulsar_detection(
    filter_func: Callable[[np.ndarray, int], np.ndarray],
    n_pulsars: int = 5,
    duration: float = 30.0,
    sample_rate: float = 10000.0,
    seed: int = 123,
) -> dict:
    """
    Evaluate filter function on synthetic pulsar detection task.

    This is a secondary evaluation to ensure evolved code works
    in the full FFA pipeline, not just isolated benchmarks.

    Args:
        filter_func: Filter function to test
        n_pulsars: Number of synthetic pulsars to test
        duration: Observation duration in seconds
        sample_rate: Sampling rate in Hz
        seed: Random seed

    Returns:
        Dict with detection metrics
    """
    np.random.seed(seed)

    detections = 0
    total_snr_ratio = 0.0

    for i in range(n_pulsars):
        # Generate pulsar with random parameters
        params = PulsarParams(
            period=np.random.uniform(0.1, 0.5),
            duty_cycle=np.random.uniform(0.03, 0.08),
            snr=np.random.uniform(10, 20),
            initial_phase=np.random.uniform(0, 1),
        )

        data, meta = generate_pulsar_timeseries(
            params,
            duration=duration,
            sample_rate=sample_rate,
            seed=seed + i,
        )

        # Fold at true period
        period_samples = meta["period_samples"]
        profile = fold_timeseries(data, period_samples, n_bins=128)

        # Apply matched filter with various widths
        best_snr = 0.0
        for width in [1, 2, 4, 8, 16]:
            try:
                filtered = filter_func(profile, width)
                snr = compute_profile_snr(filtered)
                best_snr = max(best_snr, snr)
            except Exception:
                pass

        # Count detection if SNR > 5
        if best_snr > 5.0:
            detections += 1

        # Compare to expected SNR
        if params.snr > 0:
            total_snr_ratio += best_snr / params.snr

    detection_rate = detections / n_pulsars
    avg_snr_ratio = total_snr_ratio / n_pulsars

    return {
        "detection_rate": detection_rate,
        "avg_snr_ratio": avg_snr_ratio,
        "n_detected": detections,
        "n_total": n_pulsars,
    }


def create_fitness_function(
    enforce_runtime_constraint: bool = True,
) -> Callable[[Callable], float]:
    """
    Create a fitness function for use with AlphaEvolve.

    Args:
        enforce_runtime_constraint: Whether to enforce ±5% runtime bound

    Returns:
        Fitness function that takes a filter function and returns score
    """

    def fitness(filter_func: Callable[[np.ndarray, int], np.ndarray]) -> float:
        result = evaluate_filter_function(
            filter_func,
            enforce_runtime_constraint=enforce_runtime_constraint,
        )
        return result.fitness

    return fitness


if __name__ == "__main__":
    # Test the evaluator with baseline implementation
    from .baseline_functions import boxcar_matched_filter

    print("Evaluating baseline boxcar_matched_filter...")
    result = evaluate_filter_function(
        boxcar_matched_filter,
        enforce_runtime_constraint=False,  # Baseline should pass without constraint
    )

    print(f"\nResults:")
    print(f"  Fitness:      {result.fitness:.3f}")
    print(f"  Speedup:      {result.speedup:.2f}x")
    print(f"  SNR Error:    {result.snr_error:.4f}")
    print(f"  Max Diff:     {result.max_diff:.2e}")
    print(f"  Is Correct:   {result.is_correct}")
    print(f"  Is Valid:     {result.is_valid}")
    print(f"  Runtime:      {result.runtime_ms:.3f} ms")
    print(f"  Baseline:     {result.baseline_runtime_ms:.3f} ms")

    print("\nTesting on pulsar detection...")
    detection_result = evaluate_on_pulsar_detection(boxcar_matched_filter)
    print(f"  Detection Rate: {detection_result['detection_rate']:.1%}")
    print(f"  Avg SNR Ratio:  {detection_result['avg_snr_ratio']:.2f}")
