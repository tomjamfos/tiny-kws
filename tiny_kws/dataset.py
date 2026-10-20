import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tiny_kws._native import load_wav, log_mel
from tiny_kws.labels import (
    SILENCE_CLASS_INDEX,
    TARGET_WORDS,
    UNKNOWN_WORDS,
    WORD_TO_CLASS_INDEX,
)

CLIP_SAMPLES = 16_000  # one second at 16 kHz, kClipSamples in src/log_mel.h

# Noise level anchor: the median RMS of all train word
# clips padded to CLIP_SAMPLES, from scripts/dataset_stats.py.
MEDIAN_WORD_RMS_DBFS = -24.47

# Noise level range in dB relative to MEDIAN_WORD_RMS_DBFS, drawn uniformly.
# Used for the "silence" class and noise mixed with word clips.
# At the lower bound the log-mel floor clamps nearly all the noise,
# leaving a word clip's features largely unchanged and a "silence"
# clip close to all-zero. The upper bound is the median word level.
NOISE_RANGE_DB = (-70.0, 0.0)


@enum.unique
class DrawStream(enum.IntEnum):
    """
    The first element of every SeedSequence spawn_key.

    Generators built from the same seed differ only in their spawn_key,
    so leading with a member keeps keys unique across streams.

    Members must keep their values, since every draw in a stream depends on them.
    """

    EPOCH_PLAN = 0
    WORD_AUGMENTATION = 1
    TRAIN_SILENCE = 2
    EVALUATION_SILENCE = 3


def stream_rng(seed: int, stream: DrawStream, *key: int) -> np.random.Generator:
    """
    Generator built from seed, stream and key.

    Nothing is shared between calls, so worker count,
    call order and resuming a run never change the draws.
    """
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream, *key)))


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


def clip_class_index(clip_key: str) -> int:
    """
    Class index of the word in clip_key.
    Words in UNKNOWN_WORDS map to UNKNOWN_CLASS_INDEX.

    Raises KeyError if the word isn't in TARGET_WORDS or UNKNOWN_WORDS.
    """
    return WORD_TO_CLASS_INDEX[clip_key.partition("/")[0]]


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


def load_word_clip(dataset_root: Path, clip_key: str) -> NDArray[np.float32]:
    """
    Load the word clip at clip_key under dataset_root, zero-padded to CLIP_SAMPLES.

    Raises RuntimeError if the file is missing or unreadable,
    ValueError if it isn't a 16 kHz mono 16-bit PCM wav or is longer than CLIP_SAMPLES.
    """
    return pad_clip(load_wav(dataset_root / clip_key))


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


def generate_silence_clip(
    noise_sources: NoiseSources, rng: np.random.Generator
) -> NDArray[np.float32]:
    """
    Generate a "silence" clip from a random crop of a noise source,
    at a level drawn from NOISE_RANGE_DB relative to the median word RMS.

    Draws the source, then the crop start, then the level in dB, each uniformly.
    Returns a new array, CLIP_SAMPLES long and clamped to [-1, 1].
    """
    return mix_noise(
        np.zeros(CLIP_SAMPLES, np.float32), noise_sources, rng, NOISE_RANGE_DB
    )


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


@dataclass(frozen=True, kw_only=True)
class AugmentationConfig:
    """
    Time-domain augmentation of word clips,
    applied before the log-mel frontend.

    max_shift_fraction: the largest time shift in either
    direction as a fraction of CLIP_SAMPLES. 0 is no time shift.

    noise_range_db: bounds of the noise level mixed with word clips,
    drawn uniformly in dB relative to MEDIAN_WORD_RMS_DBFS.
    Equal bounds give a fixed level. None is no noise mixing.

    Raises ValueError if max_shift_fraction is outside [0, 1]
    or noise_range_db's lower bound exceeds its upper bound.
    """

    max_shift_fraction: float = 0.0
    noise_range_db: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_shift_fraction <= 1.0:
            raise ValueError(
                f"max_shift_fraction must be in [0, 1], got {self.max_shift_fraction}"
            )
        if (
            self.noise_range_db is not None
            and self.noise_range_db[0] > self.noise_range_db[1]
        ):
            raise ValueError(
                f"noise_range_db lower bound must not exceed upper bound, "
                f"got {self.noise_range_db}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Request:
    """
    Specifies one training example for TrainDataset.

    index is the word clip's position in splits.train,
    or the "silence" clip's number within the epoch.
    Draws depend on the epoch as well as the index.
    """

    silence: bool
    epoch: int
    index: int


class TrainDataset:
    """
    Builds the training example each Request specifies.

    Each example's draws depend only on seed and its request.
    clip_keys must be splits.train, where Request.index gives each word clip's position.
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        clip_keys: tuple[str, ...],
        noise_sources: NoiseSources,
        augmentation: AugmentationConfig,
        seed: int,
    ) -> None:

        self.dataset_root = dataset_root
        self.clip_keys = clip_keys
        self.noise_sources = noise_sources
        self.max_shift_samples = round(augmentation.max_shift_fraction * CLIP_SAMPLES)
        self.noise_range_db = augmentation.noise_range_db
        self.seed = seed

    def __getitem__(self, request: Request) -> tuple[NDArray[np.float32], int]:
        if request.silence:
            rng = stream_rng(
                self.seed, DrawStream.TRAIN_SILENCE, request.epoch, request.index
            )
            clip = generate_silence_clip(self.noise_sources, rng)
            return log_mel(clip), SILENCE_CLASS_INDEX

        rng = stream_rng(
            self.seed, DrawStream.WORD_AUGMENTATION, request.epoch, request.index
        )
        shift_rng, noise_rng = rng.spawn(2)

        clip_key = self.clip_keys[request.index]
        clip = load_word_clip(self.dataset_root, clip_key)

        # Apply time-domain augmentations. Noise is mixed after
        # the time shift, so that it also covers the zero-filled gap.
        clip = time_shift(clip, shift_rng, self.max_shift_samples)
        if self.noise_range_db is not None:
            clip = mix_noise(clip, self.noise_sources, noise_rng, self.noise_range_db)

        return log_mel(clip), clip_class_index(clip_key)
