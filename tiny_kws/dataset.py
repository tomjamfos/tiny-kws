from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tiny_kws.labels import TARGET_WORDS, UNKNOWN_WORDS

CLIP_SAMPLES = 16_000  # one second at 16 kHz, kClipSamples in src/log_mel.h


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

    listed_validation = set((dataset_root / "validation_list.txt").read_text().split())
    listed_test = set((dataset_root / "testing_list.txt").read_text().split())

    validation, test, train = [], [], []
    for word in (*TARGET_WORDS, *UNKNOWN_WORDS):
        for path in (dataset_root / word).iterdir():
            if path.suffix != ".wav":
                continue
            clip_key = f"{word}/{path.name}"
            if clip_key in listed_validation:
                validation.append(clip_key)
            elif clip_key in listed_test:
                test.append(clip_key)
            else:
                train.append(clip_key)

    listed_missing = sorted(
        (listed_validation | listed_test) - set(validation) - set(test)
    )
    if listed_missing:
        raise FileNotFoundError(
            f"{len(listed_missing)} validation and test clips missing under "
            f"{dataset_root}: {', '.join(listed_missing[:3])}"
            f"{', ...' if len(listed_missing) > 3 else ''}"
        )

    # Sorted so a key's index doesn't depend on the filesystem's iterdir() order.
    return Splits(
        train=tuple(sorted(train)),
        validation=tuple(sorted(validation)),
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
