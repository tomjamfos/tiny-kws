from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tiny_kws._native import load_wav
from tiny_kws.labels import TARGET_WORDS, UNKNOWN_WORDS

CLIP_SAMPLES = 16_000  # one second at 16 kHz, kClipSamples in src/log_mel.h

# Noise level anchor: the median RMS of all train word
# clips padded to CLIP_SAMPLES, from scripts/dataset_stats.py.
MEDIAN_WORD_RMS_DBFS = -24.47

# One array per _background_noise_ wav in file name order.
# Each is scaled to unit RMS and at least CLIP_SAMPLES long.
type NoiseSources = tuple[NDArray[np.float32], ...]


@dataclass(frozen=True, kw_only=True)
class Splits:
    """
    Clip keys per official split,
    as root-relative "word/file.wav" paths in sorted order.
    """

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def build_splits(dataset_root: Path) -> Splits:
    """
    Assign each wav in the word folders to a split,
    using the official validation and test lists.
    Wavs named in neither are assigned to train.

    Raises FileNotFoundError if a list file, a word folder or a listed wav is missing.
    """

    listed_val = set((dataset_root / "validation_list.txt").read_text().split())
    listed_test = set((dataset_root / "testing_list.txt").read_text().split())

    val, test, train = [], [], []
    for word in (*TARGET_WORDS, *UNKNOWN_WORDS):
        for path in (dataset_root / word).iterdir():
            if path.suffix != ".wav":
                continue
            clip_key = f"{word}/{path.name}"
            if clip_key in listed_val:
                val.append(clip_key)
            elif clip_key in listed_test:
                test.append(clip_key)
            else:
                train.append(clip_key)

    if missing := sorted((listed_val | listed_test) - set(val) - set(test)):
        raise FileNotFoundError(
            f"{len(missing)} validation and test clips missing under "
            f"{dataset_root}: {', '.join(missing[:3])}"
            f"{', ...' if len(missing) > 3 else ''}"
        )

    # Sorted so a key's index doesn't depend on the filesystem's iterdir() order.
    return Splits(
        train=tuple(sorted(train)),
        validation=tuple(sorted(val)),
        test=tuple(sorted(test)),
    )


def rms(x: NDArray[np.float32]) -> float:
    """
    Root mean square of x.
    Computed in float64 and returned as float.
    """
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def pad_clip(clip: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Zero-pad clip at its end to CLIP_SAMPLES. Always returns a new array.

    Raises ValueError if clip is longer than CLIP_SAMPLES.
    """
    if len(clip) > CLIP_SAMPLES:
        raise ValueError(
            f"expected no more than {CLIP_SAMPLES:,} samples, got {len(clip):,} samples"
        )

    out = np.zeros(CLIP_SAMPLES, np.float32)
    out[: len(clip)] = clip
    return out


def load_noise(dataset_root: Path) -> NoiseSources:
    """
    Load every wav in _background_noise_ scaled to unit RMS.
    Sorted by file name so that order of noise sources is deterministic.

    Raises FileNotFoundError if _background_noise_ doesn't contain any wavs,
    ValueError if any of the wavs are shorter than CLIP_SAMPLES.
    """
    noise_root = dataset_root / "_background_noise_"
    if not (noise_paths := sorted(noise_root.glob("*.wav"))):
        raise FileNotFoundError(f"no wav files under {noise_root}")

    noise_sources = []
    for path in noise_paths:
        samples = load_wav(path)
        if len(samples) < CLIP_SAMPLES:
            raise ValueError(
                f"{path} is {CLIP_SAMPLES - len(samples)} samples short of one clip"
            )
        noise_sources.append(samples / rms(samples))

    return tuple(noise_sources)


def mix_noise(
    clip: NDArray[np.float32],
    noise_sources: NoiseSources,
    rng: np.random.Generator,
    range_db: tuple[float, float],
) -> NDArray[np.float32]:
    """
    Mix a random crop of a noise source into clip,
    at a level drawn from range_db relative to the median word RMS.

    Draws the source, then the crop start, then the level in dB, each uniformly,
    so a short noise source is as likely to be picked as a long one.
    Seeded mixes depend on the draw sequence.

    clip must be CLIP_SAMPLES long. Returns a new array clamped to [-1, 1].
    """
    noise_source = noise_sources[rng.integers(len(noise_sources))]
    start_index = rng.integers(len(noise_source) - CLIP_SAMPLES, endpoint=True)
    noise_clip = noise_source[start_index : start_index + CLIP_SAMPLES]

    # Sample the noise source's RMS in dBFS,
    # uniformly in dB over range_db relative to the median word clip RMS.
    noise_dbfs = MEDIAN_WORD_RMS_DBFS + rng.uniform(*range_db)

    # Convert the RMS from dB to amplitude, scale the crop of
    # the unit RMS source by it, and mix the result into clip.
    # Mixed in place into a copy of clip, since NumPy's stubs type float32
    # array arithmetic as NDArray[floating] but in-place ops keep float32.
    mixed = clip.copy()
    mixed += 10 ** (noise_dbfs / 20.0) * noise_clip

    return np.clip(mixed, -1.0, 1.0)


def time_shift(
    clip: NDArray[np.float32], rng: np.random.Generator, max_shift_samples: int
) -> NDArray[np.float32]:
    """
    Shift clip in time by a sample offset drawn uniformly from
    [-max_shift_samples, max_shift_samples]. The shifted clip is
    truncated to [0, CLIP_SAMPLES) and zero-padded to fill that range.

    clip must be CLIP_SAMPLES long. max_shift_samples must not exceed CLIP_SAMPLES.
    Returns a new array.
    """
    shift = rng.integers(-max_shift_samples, max_shift_samples, endpoint=True)

    shifted_clip = np.zeros(CLIP_SAMPLES, np.float32)
    if shift >= 0:
        shifted_clip[shift:] = clip[: CLIP_SAMPLES - shift]
    else:
        shifted_clip[: CLIP_SAMPLES + shift] = clip[-shift:]

    return shifted_clip
