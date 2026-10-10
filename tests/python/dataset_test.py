import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from tiny_kws import dataset, labels


class FixedDrawGenerator(np.random.Generator):
    """
    Let tests choose each draw and assert exact equalities.
    Returns values from integers() and uniform() in call order.
    """

    def __init__(self, *draws: float) -> None:
        super().__init__(np.random.PCG64())
        self._draws = iter(draws)

    def integers(self, *args: Any, **kwargs: Any) -> Any:
        return next(self._draws)

    def uniform(self, *args: Any, **kwargs: Any) -> Any:
        return next(self._draws)


@pytest.fixture(scope="module")
def splits(dataset_root: Path) -> dataset.Splits:
    return dataset.build_splits(dataset_root)


@pytest.fixture
def stub_dataset_root(tmp_path: Path) -> Path:
    """
    Minimal fake dataset for any test that writes or deletes files.
    Such tests must never take dataset_root, which points at the real dataset.
    """
    for word in (*labels.TARGET_WORDS, *labels.UNKNOWN_WORDS):
        (tmp_path / word).mkdir()

    # One target word (down) and one unknown word (bed).
    # Target words are collected first, so down is found before bed, and the expected
    # train split in test_splits_match_list_files only matches after sorting.
    for key in ("down/c.wav", "down/a.wav", "down/b.wav", "bed/e.wav", "bed/d.wav"):
        (tmp_path / key).touch()
    (tmp_path / "validation_list.txt").write_text("down/b.wav\n")
    (tmp_path / "testing_list.txt").write_text("bed/e.wav\n")

    (noise_dir := tmp_path / "_background_noise_").mkdir()

    # b.wav is the shortest valid length.
    for name, length in (("b.wav", 16_000), ("a.wav", 20_000)):
        sf.write(noise_dir / name, np.full(length, 1_000, np.int16), 16_000, "PCM_16")

    # Mirrors speech-commands-v0.02/_background_noise_,
    # load_noise must skip non-wav files.
    (noise_dir / "README.md").touch()

    return tmp_path


@pytest.fixture(scope="module")
def noise_sources(dataset_root: Path) -> dataset.NoiseSources:
    return dataset.load_noise(dataset_root)


def test_split_wav_counts(splits: dataset.Splits) -> None:

    # speech-commands-v0.02/README.md:19, "105,829 audio files"
    num_total_keys = 105_829

    # line count of speech-commands-v0.02/validation_list.txt
    num_validation_keys = 9_981

    # line count of speech-commands-v0.02/testing_list.txt
    num_test_keys = 11_005

    assert len(splits.validation) == num_validation_keys
    assert len(splits.test) == num_test_keys
    assert len(splits.train) == num_total_keys - num_validation_keys - num_test_keys


def test_keys_unique_across_splits(splits: dataset.Splits) -> None:
    keys = (*splits.train, *splits.validation, *splits.test)
    assert len(set(keys)) == len(keys)


def test_keys_sorted(splits: dataset.Splits) -> None:
    for split_keys in (splits.train, splits.validation, splits.test):
        assert list(split_keys) == sorted(split_keys)


def test_splits_match_list_files(stub_dataset_root: Path) -> None:
    assert dataset.build_splits(stub_dataset_root) == dataset.Splits(
        train=("bed/d.wav", "down/a.wav", "down/c.wav"),
        validation=("down/b.wav",),
        test=("bed/e.wav",),
    )


def test_stray_folder_skipped(stub_dataset_root: Path) -> None:
    (stub_dataset_root / "extra").mkdir()
    (stub_dataset_root / "extra/f.wav").touch()

    splits = dataset.build_splits(stub_dataset_root)
    assert "extra/f.wav" not in (*splits.train, *splits.validation, *splits.test)


def test_non_wav_files_skipped(stub_dataset_root: Path) -> None:
    (stub_dataset_root / "down/.DS_Store").touch()
    (stub_dataset_root / "down/notes.txt").touch()

    splits = dataset.build_splits(stub_dataset_root)
    assert all(key.endswith(".wav") for key in splits.train)


def test_missing_listed_clip_raises(stub_dataset_root: Path) -> None:
    (stub_dataset_root / "testing_list.txt").write_text("bed/e.wav\nbed/missing.wav\n")

    with pytest.raises(FileNotFoundError, match="bed/missing.wav"):
        dataset.build_splits(stub_dataset_root)


def test_missing_word_folder_raises(stub_dataset_root: Path) -> None:
    (stub_dataset_root / "zero").rmdir()

    with pytest.raises(FileNotFoundError, match="zero"):
        dataset.build_splits(stub_dataset_root)


def test_rms() -> None:
    assert dataset.rms(np.full(1_000, -0.25, np.float32)) == 0.25


def test_pad_clip_short() -> None:
    padded_clip = dataset.pad_clip(np.ones(8, np.float32))
    assert padded_clip.dtype == np.float32
    assert len(padded_clip) == dataset.CLIP_SAMPLES
    assert np.all(padded_clip[:8] == 1.0)
    assert np.all(padded_clip[8:] == 0.0)


def test_pad_clip_full_length_unchanged() -> None:
    clip = np.arange(dataset.CLIP_SAMPLES, dtype=np.float32)
    assert np.array_equal(dataset.pad_clip(clip), clip)


def test_pad_clip_returns_copy() -> None:
    clip = np.ones(dataset.CLIP_SAMPLES, dtype=np.float32)
    assert not np.shares_memory(dataset.pad_clip(clip), clip)


def test_pad_clip_long_raises() -> None:
    with pytest.raises(ValueError, match="16,001"):
        dataset.pad_clip(np.zeros(dataset.CLIP_SAMPLES + 1, np.float32))


def test_noise_source_count(noise_sources: dataset.NoiseSources) -> None:
    assert len(noise_sources) == 6


def test_noise_normalised(noise_sources: dataset.NoiseSources) -> None:
    for source in noise_sources:
        source_rms = np.sqrt(np.mean(np.square(source)))
        assert source_rms == pytest.approx(1.0, rel=1e-6)


def test_noise_source_arrays(noise_sources: dataset.NoiseSources) -> None:
    for source in noise_sources:
        assert source.dtype == np.float32
        assert source.ndim == 1
        assert len(source) >= dataset.CLIP_SAMPLES


def test_noise_short_file_raises(stub_dataset_root: Path) -> None:
    sf.write(
        stub_dataset_root / "_background_noise_/b.wav",
        np.full(15_999, 1_000, np.int16),
        16_000,
        "PCM_16",
    )
    with pytest.raises(ValueError, match="b.wav"):
        dataset.load_noise(stub_dataset_root)


def test_noise_missing_folder_raises(stub_dataset_root: Path) -> None:
    shutil.rmtree(stub_dataset_root / "_background_noise_")
    with pytest.raises(FileNotFoundError, match="_background_noise_"):
        dataset.load_noise(stub_dataset_root)


def test_mix_noise_level() -> None:
    rng = np.random.default_rng(seed=0)

    # level_db cancels the median word RMS anchor in mix_noise,
    # so noise mixed with a silent clip is at -6 dBFS (amplitude: 0.5).
    # One full-length source and a zero-width range fix every draw.
    source = np.ones(dataset.CLIP_SAMPLES, np.float32)
    level_db = 20.0 * np.log10(0.5) - dataset.MEDIAN_WORD_RMS_DBFS
    mixed = dataset.mix_noise(
        np.zeros(dataset.CLIP_SAMPLES, np.float32), (source,), rng, (level_db, level_db)
    )
    assert mixed == pytest.approx(0.5)


def test_mix_noise_clamps() -> None:
    rng = np.random.default_rng(seed=0)

    # clip and noise are in phase, so their sum exceeds ±1.
    # Alternating signs exercise the clamp at both bounds.
    even = np.arange(dataset.CLIP_SAMPLES) % 2 == 0
    source = np.where(even, -1.0, 1.0).astype(np.float32)
    clip = 0.9 * source

    # Noise at -6 dBFS (amplitude: 0.5) before mixing with clip.
    # Anything above -20 dBFS (amplitude: 0.1) also passes.
    level_db = 20.0 * np.log10(0.5) - dataset.MEDIAN_WORD_RMS_DBFS

    mixed = dataset.mix_noise(clip, (source,), rng, (level_db, level_db))
    assert np.array_equal(mixed, source)


def test_mix_noise_draw_order() -> None:

    # Source index, crop start,
    # and a level offset in dB that cancels the median word RMS.
    rng = FixedDrawGenerator(1, 5, -dataset.MEDIAN_WORD_RMS_DBFS)

    sources = (
        # A wrong source index would give all -1s.
        -np.ones(dataset.CLIP_SAMPLES, np.float32),
        # mix_noise doesn't pad, so a non-zero crop start needs samples beyond one clip.
        np.ones(dataset.CLIP_SAMPLES + 5, np.float32),
    )
    sources[1][5] = -1.0  # -1 acts as a sentinel for the expected crop start

    mixed = dataset.mix_noise(
        np.zeros(dataset.CLIP_SAMPLES, np.float32), sources, rng, (-70.0, -5.0)
    )
    expected = np.ones(dataset.CLIP_SAMPLES, np.float32)
    expected[0] = -1.0

    assert np.array_equal(mixed, expected)


def test_mix_noise_returns_new_array() -> None:
    rng = np.random.default_rng(seed=0)
    clip = np.zeros(dataset.CLIP_SAMPLES, np.float32)
    source = np.ones(dataset.CLIP_SAMPLES, np.float32)
    mixed = dataset.mix_noise(clip, (source,), rng, (-70.0, -5.0))

    assert mixed.dtype == np.float32
    assert len(mixed) == dataset.CLIP_SAMPLES
    assert not np.shares_memory(mixed, clip)
    assert not np.shares_memory(mixed, source)
    assert np.all(clip == 0.0)


def test_time_shift_zero_is_identity() -> None:
    rng = np.random.default_rng(seed=0)
    clip = np.arange(dataset.CLIP_SAMPLES, dtype=np.float32)
    shifted_clip = dataset.time_shift(clip, rng, max_shift_samples=0)

    assert shifted_clip.dtype == np.float32
    assert np.array_equal(shifted_clip, clip)
    assert not np.shares_memory(shifted_clip, clip)


# Identity, the smallest possible shift each way, and a full clip each way.
@pytest.mark.parametrize("shift", [0, -1, 1, -16_000, 16_000])
def test_time_shift_shifts_and_fills(shift: int) -> None:

    # Start at 1 so a zero can only come from the fill.
    clip = np.arange(1, dataset.CLIP_SAMPLES + 1, dtype=np.float32)

    shifted_clip = dataset.time_shift(
        clip, FixedDrawGenerator(shift), max_shift_samples=16_000
    )
    expected = np.roll(clip, shift)
    if shift >= 0:
        expected[:shift] = 0.0
    else:
        expected[shift:] = 0.0

    assert np.array_equal(shifted_clip, expected)
