import numpy as np
from numpy.typing import ArrayLike, NDArray


# HTK mel scale
def hz_to_mel(hz: ArrayLike) -> NDArray[np.float64]:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: ArrayLike) -> NDArray[np.float64]:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def melspace(start_hz: float, end_hz: float, n: int) -> NDArray[np.float64]:
    """Evenly space n points between start_hz and end_hz on the HTK mel scale,
    endpoints included."""

    if n < 2:
        raise ValueError(f"melspace: n should be at least 2, got {n}")

    mels = np.linspace(hz_to_mel(start_hz), hz_to_mel(end_hz), n, dtype=np.float64)
    hz = mel_to_hz(mels)

    # Hz to mel and back is inexact, so set the endpoints directly.
    hz[0], hz[-1] = start_hz, end_hz
    return hz


def chirp(
    freqs: ArrayLike,
    amplitude: float,
    sample_rate: int = 16_000,
) -> NDArray[np.float32]:
    """Sine sweep through freqs (Hz), one sample per entry."""

    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.ndim != 1:
        raise ValueError(
            f"chirp: freqs should be one-dimensional, got shape {freqs.shape}"
        )
    if not np.all((freqs >= 0.0) & (freqs < sample_rate / 2)):
        raise ValueError(
            f"chirp: freqs should be in [0, {sample_rate / 2}) Hz, "
            f"got min {np.min(freqs)}, max {np.max(freqs)}"
        )

    # Integrate frequency to get phase, in float64. Round to float32 only at output.
    phase_cycles = (np.cumsum(freqs) - freqs) / sample_rate

    return (amplitude * np.sin(2.0 * np.pi * phase_cycles)).astype(np.float32)
