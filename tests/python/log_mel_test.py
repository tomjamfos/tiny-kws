import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from signals import chirp, melspace

from tiny_kws import _native


def reference_log_mel(clip: np.ndarray) -> np.ndarray:
    mel_spec = T.MelSpectrogram(
        sample_rate=16_000,
        n_fft=512,
        win_length=480,
        hop_length=160,
        f_min=20.0,
        f_max=8_000.0,
        n_mels=32,
        window_fn=torch.hann_window,
        power=2.0,
        center=False,
        norm=None,
        mel_scale="htk",
    )
    waveform = torch.from_numpy(clip)
    waveform = F.pad(waveform, (0, 16_000 - len(waveform)))

    # _native.log_mel() frames by win_length. T.MelSpectrogram() frames by n_fft
    # and centres the window in each frame, leaving (n_fft - win_length) / 2 = 16
    # samples either side. Zero-pad 16 at the start to align the windows,
    # and 16 at the end to give 98 frames.
    waveform = F.pad(waveform, (16, 16))

    return torch.clamp(mel_spec(waveform), min=1e-6).log().numpy()


def test_matches_torchaudio() -> None:
    rng = np.random.default_rng(seed=23)
    clip = rng.uniform(-1.0, 1.0, 16_000).astype(dtype=np.float32)

    actual = _native.log_mel(clip)
    expected = reference_log_mel(clip)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.max(np.abs(actual - expected)) < 1e-4


def test_short_clip_is_zero_padded() -> None:
    rng = np.random.default_rng(seed=23)
    clip = rng.uniform(-1.0, 1.0, 12_000).astype(dtype=np.float32)
    clip_padded = np.pad(clip, (0, 16_000 - len(clip)))

    assert np.array_equal(_native.log_mel(clip), _native.log_mel(clip_padded))


def test_silence_equals_log_floor() -> None:
    features = _native.log_mel(np.zeros(16_000, dtype=np.float32))

    assert features.shape == (32, 98)
    assert np.allclose(features, np.log(1e-6), rtol=0.0, atol=1e-6)


def test_matches_torchaudio_on_chirp() -> None:
    clip = chirp(melspace(20.0, 7_900.0, 16_000), amplitude=0.5)

    actual = np.exp(_native.log_mel(clip))
    expected = np.exp(reference_log_mel(clip))

    # Rounding error scales with each frame's peak, so compare linear power with a
    # tolerance scaled to that peak. A chirp leaves most cells far below the frame
    # peak, where comparing logs would magnify the error.
    assert np.all(np.abs(actual - expected) < 1e-5 * np.max(expected, axis=0))


def test_chirp_peak_rises_through_mel_bins() -> None:
    clip = chirp(melspace(20.0, 7_900.0, 16_000), amplitude=0.5)
    peak_bins = np.argmax(_native.log_mel(clip), axis=0)

    assert np.all(np.isin(np.diff(peak_bins), (0, 1)))
    assert peak_bins[0] == 0
    assert peak_bins[-1] == 31


def test_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        _native.log_mel(np.zeros(16_001, dtype=np.float32))
    with pytest.raises(ValueError):
        _native.log_mel(np.zeros((2, 16_000), dtype=np.float32))


def test_rejects_non_float_dtypes() -> None:
    with pytest.raises(TypeError):
        _native.log_mel(np.zeros(16_000, dtype=np.int16))  # type: ignore
    with pytest.raises(TypeError):
        _native.log_mel(np.zeros(16_000, dtype=np.complex64))  # type: ignore


def test_accepts_float64() -> None:
    rng = np.random.default_rng(seed=23)
    clip = rng.uniform(-1.0, 1.0, 16_000).astype(dtype=np.float32)
    clip_float64 = clip.astype(dtype=np.float64)

    assert np.array_equal(_native.log_mel(clip_float64), _native.log_mel(clip))
