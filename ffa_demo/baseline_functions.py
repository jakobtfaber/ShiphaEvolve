"""
Baseline FFA Functions with EVOLVE-BLOCK Targets

This module contains the core FFA functions that AlphaEvolve will attempt
to optimize through LLM-based evolutionary search.

The primary optimization target is `boxcar_matched_filter`, which is the
computational bottleneck in FFA periodicity searches.

EVOLVE-BLOCK markers indicate code regions that the LLM may modify.
"""

import numpy as np

# =============================================================================
# EVOLUTION TARGET: Boxcar Matched Filter
# =============================================================================

# EVOLVE-BLOCK-START: boxcar_matched_filter
def boxcar_matched_filter(profile: np.ndarray, width: int) -> np.ndarray:
    """
    Apply boxcar matched filter to a folded pulse profile.

    This function convolves the profile with a boxcar kernel of the given width,
    implementing circular (wrap-around) convolution for proper phase handling.

    The matched filter maximizes SNR for box-shaped pulses of the given width.

    Args:
        profile: 1D array of folded pulse profile values
        width: Width of boxcar kernel in bins (1 to len(profile)//2)

    Returns:
        Filtered profile with same length as input

    Performance target: O(n) time complexity
    Accuracy requirement: Results must match naive convolution within 1e-10

    # HINT (disabled by default):
    # Consider using cumulative sums for O(n) complexity
    # Consider FFT-based convolution for large profiles
    # NumPy's stride tricks can enable vectorized window operations
    """
    n = len(profile)
    width = min(width, n)

    # Baseline implementation: direct convolution with wrap-around
    # This is O(n * width) - can be improved to O(n)
    result = np.zeros(n)
    for i in range(n):
        total = 0.0
        for j in range(width):
            idx = (i + j) % n
            total += profile[idx]
        result[i] = total

    # Normalize for unit response to unit impulse
    result /= np.sqrt(width)

    return result
# EVOLVE-BLOCK-END: boxcar_matched_filter


# =============================================================================
# SUPPORTING FUNCTIONS (not evolved, but used in evaluation)
# =============================================================================

def fold_timeseries(
    data: np.ndarray, period_samples: int, n_bins: int | None = None
) -> np.ndarray:
    """
    Fold time series at specified period to create pulse profile.

    Args:
        data: 1D time series array
        period_samples: Period in number of samples
        n_bins: Number of output phase bins (default: period_samples)

    Returns:
        Folded profile array normalized by sqrt(n_periods)
    """
    if n_bins is None:
        n_bins = period_samples

    n_samples = len(data)
    n_periods = n_samples // period_samples

    if n_periods < 2:
        return np.zeros(n_bins)

    # Truncate to complete periods
    truncated = data[: n_periods * period_samples]

    # Reshape and sum across periods
    reshaped = truncated.reshape(n_periods, period_samples)
    folded = np.sum(reshaped, axis=0)

    # Rebin if needed
    if n_bins != period_samples:
        samples_per_bin = period_samples // n_bins
        binned = np.zeros(n_bins)
        for i in range(n_bins):
            start = i * samples_per_bin
            end = start + samples_per_bin
            binned[i] = np.sum(folded[start:end])
        folded = binned

    # Normalize by sqrt(n_periods) for proper SNR scaling
    folded /= np.sqrt(n_periods)

    return folded


def compute_profile_snr(profile: np.ndarray) -> float:
    """
    Compute signal-to-noise ratio of a folded profile.

    Uses off-pulse baseline estimation from the lowest quartile of bins.

    Args:
        profile: Folded pulse profile

    Returns:
        Signal-to-noise ratio (peak - baseline) / noise_std
    """
    if len(profile) < 8:
        return 0.0

    # Estimate baseline from lowest quartile
    sorted_profile = np.sort(profile)
    n_baseline = max(len(profile) // 4, 2)
    baseline = sorted_profile[:n_baseline]

    mean_baseline = np.mean(baseline)
    std_baseline = np.std(baseline)

    if std_baseline < 1e-10:
        return 0.0

    # SNR = (peak - baseline) / noise
    peak = np.max(profile)
    snr = (peak - mean_baseline) / std_baseline

    return snr


def compute_filtered_snr(
    profile: np.ndarray,
    filter_func: callable,
    widths: list[int] | None = None,
) -> tuple[float, int]:
    """
    Compute best SNR across multiple boxcar filter widths.

    Args:
        profile: Folded pulse profile
        filter_func: Matched filter function (profile, width) -> filtered
        widths: List of widths to try (default: geometric series)

    Returns:
        Tuple of (best_snr, best_width)
    """
    n = len(profile)

    if widths is None:
        # Default: geometric series of widths
        widths = [1]
        w = 1
        while w < n // 2:
            w = int(np.ceil(w * 1.5))
            widths.append(min(w, n // 2))

    best_snr = 0.0
    best_width = 1

    for w in widths:
        filtered = filter_func(profile, w)
        snr = compute_profile_snr(filtered)
        if snr > best_snr:
            best_snr = snr
            best_width = w

    return best_snr, best_width


# =============================================================================
# REFERENCE IMPLEMENTATION (for correctness validation)
# =============================================================================

def boxcar_matched_filter_reference(profile: np.ndarray, width: int) -> np.ndarray:
    """
    Reference implementation using NumPy convolve.

    This is the ground truth for validating evolved implementations.
    NOT used in timing - only for correctness checks.
    """
    n = len(profile)
    width = min(width, n)

    # Create boxcar kernel
    kernel = np.ones(width)

    # Circular convolution via padding
    extended = np.concatenate([profile, profile[:width - 1]])
    result = np.convolve(extended, kernel, mode='valid')[:n]

    # Normalize
    result /= np.sqrt(width)

    return result


# =============================================================================
# OPTIONAL ADVANCED HINTS (enabled with --enable-hints flag)
# =============================================================================

# When hints are enabled, uncomment the following in the EVOLVE-BLOCK docstring:
#
# ADVANCED HINTS:
# 1. Prefix-sum approach:
#    prefix = np.zeros(2*n + 1)
#    prefix[1:] = np.cumsum(np.tile(profile, 2))
#    result[i] = prefix[i + width] - prefix[i]
#
# 2. FFT convolution (best for large width):
#    kernel_fft = np.fft.fft(np.concatenate([np.ones(width), np.zeros(n-width)]))
#    result = np.real(np.fft.ifft(np.fft.fft(profile) * kernel_fft))
#
# 3. NumPy stride tricks for vectorized windows:
#    from numpy.lib.stride_tricks import sliding_window_view
#    padded = np.concatenate([profile, profile[:width-1]])
#    windows = sliding_window_view(padded, width)
#    result = np.sum(windows, axis=1)


if __name__ == "__main__":
    # Quick validation
    np.random.seed(42)
    test_profile = np.random.randn(128)
    test_profile[60:68] += 5  # Add fake pulse

    for width in [1, 4, 8, 16]:
        result = boxcar_matched_filter(test_profile, width)
        reference = boxcar_matched_filter_reference(test_profile, width)
        max_diff = np.max(np.abs(result - reference))
        print(f"Width {width:2d}: max diff = {max_diff:.2e}, SNR = {compute_profile_snr(result):.1f}")

    print("\nBaseline validation passed!")
