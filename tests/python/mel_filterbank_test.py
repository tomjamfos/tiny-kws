import numpy as np
import torchaudio.functional as F

from tiny_kws import _native


def reference_filterbank() -> np.ndarray:
    return F.melscale_fbanks(
        257, 20.0, 8_000.0, 32, 16_000, norm=None, mel_scale="htk"
    ).T.numpy()


def test_matches_torchaudio() -> None:
    actual = _native.mel_filterbank()
    expected = reference_filterbank()

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.max(np.abs(actual - expected)) < 1e-5
    assert np.array_equal(actual != 0, expected != 0)  # check non-zero support is equal
