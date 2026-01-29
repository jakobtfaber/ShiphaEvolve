"""
Synthetic Pulsar Timeseries Generator

Generates realistic pulsar signals embedded in Gaussian noise for
testing FFA (Fast Folding Algorithm) implementations.

Based on standard pulsar timing techniques - see Lorimer & Kramer (2012)
"Handbook of Pulsar Astronomy" for theoretical background.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PulsarParams:
    """Parameters describing a synthetic pulsar signal."""

    period: float  # Period in seconds
    duty_cycle: float  # Pulse width / period (0 to 1)
    snr: float  # Signal-to-noise ratio (folded)
    initial_phase: float = 0.0  # Initial phase (0 to 1)

    @property
    def pulse_width(self) -> float:
        """Pulse width in seconds."""
        return self.period * self.duty_cycle


def generate_gaussian_pulse_profile(
    n_bins: int, duty_cycle: float, center_phase: float = 0.5
) -> np.ndarray:
    """
    Generate a normalized Gaussian pulse profile.

    Args:
        n_bins: Number of phase bins
        duty_cycle: FWHM as fraction of period
        center_phase: Phase of pulse peak (0 to 1)

    Returns:
        1D array of pulse profile values (normalized to unit sum of squares)
    """
    phase = np.linspace(0, 1, n_bins, endpoint=False)

    # Convert FWHM duty cycle to Gaussian sigma
    sigma = duty_cycle / (2 * np.sqrt(2 * np.log(2)))

    # Gaussian profile with wrap-around handling
    profile = np.exp(-((phase - center_phase) ** 2) / (2 * sigma**2))
    profile += np.exp(-((phase - center_phase + 1) ** 2) / (2 * sigma**2))
    profile += np.exp(-((phase - center_phase - 1) ** 2) / (2 * sigma**2))

    # Normalize to unit sum of squares
    profile /= np.sqrt(np.sum(profile**2))

    return profile


def generate_pulsar_timeseries(
    params: PulsarParams,
    duration: float,
    sample_rate: float,
    add_red_noise: bool = False,
    red_noise_index: float = 2.0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Generate synthetic pulsar time series with Gaussian pulses in white noise.

    The SNR parameter represents the signal-to-noise ratio that would be
    achieved after optimal folding at the true period.

    Args:
        params: PulsarParams object defining the pulsar
        duration: Total observation duration in seconds
        sample_rate: Sampling rate in Hz
        add_red_noise: Whether to add power-law red noise (radio-frequency interference)
        red_noise_index: Spectral index of red noise (P(f) ∝ f^-α)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (time_series, metadata_dict)

    Example:
        >>> params = PulsarParams(period=0.5, duty_cycle=0.05, snr=10)
        >>> data, meta = generate_pulsar_timeseries(params, duration=60, sample_rate=10000)
        >>> print(f"Generated {meta['n_samples']} samples with {meta['n_periods']:.0f} periods")
    """
    if seed is not None:
        np.random.seed(seed)

    # Create time array
    n_samples = int(duration * sample_rate)
    sample_time = 1.0 / sample_rate
    t = np.arange(n_samples) * sample_time

    # Calculate phase at each sample
    phase = ((t / params.period) + params.initial_phase) % 1.0

    # Generate pulse profile using Gaussian shape
    sigma = params.duty_cycle / (2 * np.sqrt(2 * np.log(2)))
    pulse = np.exp(-((phase - 0.5) ** 2) / (2 * sigma**2))

    # Handle wrap-around for pulses centered at phase 0.5
    pulse += np.exp(-((phase - 0.5 + 1) ** 2) / (2 * sigma**2))
    pulse += np.exp(-((phase - 0.5 - 1) ** 2) / (2 * sigma**2))

    # Calculate amplitude for desired SNR after folding
    # SNR scales with sqrt(N_pulses * duty_cycle) under coherent folding
    n_periods = duration / params.period
    amplitude = params.snr / np.sqrt(n_periods * params.duty_cycle)

    # Generate signal
    signal = amplitude * pulse

    # Generate white Gaussian noise (unit variance)
    noise = np.random.randn(n_samples)

    # Optionally add red noise (simulates RFI / 1/f noise)
    if add_red_noise:
        freqs = np.fft.fftfreq(n_samples, sample_time)
        freqs[0] = 1e-10  # Avoid division by zero
        power_spectrum = np.abs(freqs) ** (-red_noise_index / 2)
        power_spectrum[0] = 0  # Remove DC component

        random_phases = np.random.uniform(0, 2 * np.pi, n_samples)
        red_noise_fft = power_spectrum * np.exp(1j * random_phases)
        red_noise = np.real(np.fft.ifft(red_noise_fft))
        red_noise /= np.std(red_noise)

        # Mix white and red noise
        noise = 0.7 * noise + 0.3 * red_noise
        noise /= np.std(noise)

    # Combine signal and noise
    timeseries = signal + noise

    metadata = {
        "n_samples": n_samples,
        "sample_time": sample_time,
        "sample_rate": sample_rate,
        "duration": duration,
        "n_periods": n_periods,
        "period_samples": int(params.period * sample_rate),
        "true_period": params.period,
        "true_snr": params.snr,
        "duty_cycle": params.duty_cycle,
        "has_red_noise": add_red_noise,
    }

    return timeseries, metadata


def generate_test_dataset(
    n_pulsars: int = 10,
    period_range: tuple[float, float] = (0.1, 1.0),
    snr_range: tuple[float, float] = (8, 20),
    duty_cycle_range: tuple[float, float] = (0.03, 0.10),
    duration: float = 60.0,
    sample_rate: float = 10000.0,
    seed: int = 42,
) -> list[tuple[np.ndarray, PulsarParams, dict]]:
    """
    Generate a dataset of synthetic pulsars for evaluation.

    Args:
        n_pulsars: Number of pulsars to generate
        period_range: (min, max) period in seconds
        snr_range: (min, max) SNR
        duty_cycle_range: (min, max) duty cycle
        duration: Observation duration in seconds
        sample_rate: Sampling rate in Hz
        seed: Random seed

    Returns:
        List of (timeseries, params, metadata) tuples
    """
    np.random.seed(seed)
    dataset = []

    for i in range(n_pulsars):
        params = PulsarParams(
            period=np.random.uniform(*period_range),
            duty_cycle=np.random.uniform(*duty_cycle_range),
            snr=np.random.uniform(*snr_range),
            initial_phase=np.random.uniform(0, 1),
        )

        data, meta = generate_pulsar_timeseries(
            params,
            duration=duration,
            sample_rate=sample_rate,
            add_red_noise=(i % 3 == 0),  # Add red noise to 1/3 of samples
            seed=seed + i,
        )

        dataset.append((data, params, meta))

    return dataset


if __name__ == "__main__":
    # Quick test
    params = PulsarParams(period=0.5, duty_cycle=0.05, snr=12)
    data, meta = generate_pulsar_timeseries(
        params, duration=60, sample_rate=10000, seed=42
    )
    print(f"Generated timeseries: {meta['n_samples']} samples, {meta['n_periods']:.0f} periods")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}], std={data.std():.2f}")
