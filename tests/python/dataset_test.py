import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from tiny_kws import _native, dataset, labels


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


def test_draw_stream_values() -> None:
    assert [(stream.name, stream.value) for stream in dataset.DrawStream] == [
        ("EPOCH_PLAN", 0),
        ("WORD_AUGMENTATION", 1),
        ("TRAIN_SILENCE", 2),
        ("EVALUATION_SILENCE", 3),
    ]


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


def test_clip_class_index_target_word() -> None:
    assert dataset.clip_class_index("yes/a.wav") == labels.CLASS_NAMES.index("yes")


def test_clip_class_index_unknown_word() -> None:
    assert dataset.clip_class_index("bed/a.wav") == labels.UNKNOWN_CLASS_INDEX


def test_clip_class_index_non_word_raises() -> None:
    with pytest.raises(KeyError, match="_background_noise_"):
        dataset.clip_class_index("_background_noise_/a.wav")


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


def test_load_word_clip_pads(stub_dataset_root: Path) -> None:

    # load_wav converts PCM_16 to float by dividing by 32_768,
    # so the 8 samples load as exactly 0.5. The rest of the clip is zero padding.
    sf.write(
        stub_dataset_root / "down/a.wav", np.full(8, 16_384, np.int16), 16_000, "PCM_16"
    )
    expected = np.zeros(dataset.CLIP_SAMPLES, np.float32)
    expected[:8] = 0.5

    clip = dataset.load_word_clip(stub_dataset_root, "down/a.wav")
    assert np.array_equal(clip, expected)


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


def test_generate_silence_clip_level(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(seed=0)

    # Patch NOISE_RANGE_DB to a fixed level that cancels the median word RMS anchor,
    # putting the clip at -6 dBFS (amplitude: 0.5).
    # The unpatched upper bound puts the clip at the anchor, below -6 dBFS,
    # so 0.5 can only come from the patched one.
    # One full-length source and a zero-width range fix every draw.
    source = np.ones(dataset.CLIP_SAMPLES, np.float32)
    level_db = 20.0 * np.log10(0.5) - dataset.MEDIAN_WORD_RMS_DBFS
    monkeypatch.setattr(dataset, "NOISE_RANGE_DB", (level_db, level_db))

    clip = dataset.generate_silence_clip((source,), rng)

    assert clip.dtype == np.float32
    assert len(clip) == dataset.CLIP_SAMPLES
    assert clip == pytest.approx(0.5)


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


def test_config_accepts_bounds() -> None:
    dataset.AugmentationConfig()
    dataset.AugmentationConfig(max_shift_fraction=1.0)
    dataset.AugmentationConfig(noise_range_db=(-6.0, -6.0))


@pytest.mark.parametrize("shift_fraction", [-1e-5, 2.0])
def test_config_shift_out_of_range_raises(shift_fraction: float) -> None:
    with pytest.raises(ValueError, match="max_shift_fraction"):
        dataset.AugmentationConfig(max_shift_fraction=shift_fraction)


def test_config_noise_range_reversed_raises() -> None:
    with pytest.raises(ValueError, match="noise_range_db"):
        dataset.AugmentationConfig(noise_range_db=(-5.0, -70.0))


type TrainDatasetFactory = Callable[[dataset.AugmentationConfig], dataset.TrainDataset]

AUGMENTATION_ON = dataset.AugmentationConfig(
    max_shift_fraction=0.1, noise_range_db=dataset.NOISE_RANGE_DB
)


@pytest.fixture(scope="module")
def make_train_dataset(
    dataset_root: Path, splits: dataset.Splits, noise_sources: dataset.NoiseSources
) -> TrainDatasetFactory:
    def make(augmentation: dataset.AugmentationConfig) -> dataset.TrainDataset:
        return dataset.TrainDataset(
            dataset_root=dataset_root,
            clip_keys=splits.train,
            noise_sources=noise_sources,
            augmentation=augmentation,
            seed=0,
        )

    return make


@pytest.mark.parametrize("silence", [False, True])
def test_train_dataset_reproducible(
    make_train_dataset: TrainDatasetFactory, silence: bool
) -> None:

    request = dataset.Request(silence=silence, epoch=1, index=2)

    # Build this request, then one with silence flipped,
    # so that any draw state shared between calls would change the next build.
    used = make_train_dataset(AUGMENTATION_ON)
    used[request]
    used[dataset.Request(silence=not silence, epoch=0, index=0)]

    # Built again, the example must match a fresh dataset's.
    features, class_index = used[request]
    fresh_features, fresh_class_index = make_train_dataset(AUGMENTATION_ON)[request]

    assert np.array_equal(features, fresh_features)
    assert class_index == fresh_class_index


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            dataset.Request(silence=False, epoch=0, index=0),
            dataset.Request(silence=False, epoch=1, index=0),
        ),
        (
            dataset.Request(silence=False, epoch=0, index=0),
            dataset.Request(silence=False, epoch=0, index=1),
        ),
        (
            dataset.Request(silence=True, epoch=0, index=0),
            dataset.Request(silence=True, epoch=1, index=0),
        ),
        (
            dataset.Request(silence=True, epoch=0, index=0),
            dataset.Request(silence=True, epoch=0, index=1),
        ),
    ],
    ids=["word epoch", "word index", "silence epoch", "silence index"],
)
def test_train_dataset_key_changes_draws(
    dataset_root: Path,
    splits: dataset.Splits,
    noise_sources: dataset.NoiseSources,
    monkeypatch: pytest.MonkeyPatch,
    first: dataset.Request,
    second: dataset.Request,
) -> None:

    # Compare clips rather than features as the log-mel
    # floor can flatten two quiet draws to the same features.
    monkeypatch.setattr(dataset, "log_mel", lambda clip: clip)

    # Both positions hold the same clip, so only the key can make word clips differ.
    train_dataset = dataset.TrainDataset(
        dataset_root=dataset_root,
        clip_keys=(splits.train[0], splits.train[0]),
        noise_sources=noise_sources,
        augmentation=AUGMENTATION_ON,
        seed=0,
    )

    assert not np.array_equal(train_dataset[first][0], train_dataset[second][0])


def test_train_dataset_augmentation_off(
    make_train_dataset: TrainDatasetFactory, dataset_root: Path, splits: dataset.Splits
) -> None:

    train_dataset = make_train_dataset(dataset.AugmentationConfig())
    clip_key = splits.train[2]

    features, class_index = train_dataset[
        dataset.Request(silence=False, epoch=0, index=2)
    ]
    clip = dataset.load_word_clip(dataset_root, clip_key)

    assert np.array_equal(features, _native.log_mel(clip))
    assert features.dtype == np.float32
    assert class_index == dataset.clip_class_index(clip_key)


def test_train_dataset_silence_class(make_train_dataset: TrainDatasetFactory) -> None:
    train_dataset = make_train_dataset(AUGMENTATION_ON)
    features, class_index = train_dataset[
        dataset.Request(silence=True, epoch=0, index=0)
    ]

    assert features.shape == (32, 98)
    assert features.dtype == np.float32
    assert class_index == labels.SILENCE_CLASS_INDEX


def test_train_dataset_max_shift_rounds(
    make_train_dataset: TrainDatasetFactory,
) -> None:

    # 0.0001 of CLIP_SAMPLES is 1.6, which rounds to 2 but truncates to 1.
    augmentation = dataset.AugmentationConfig(max_shift_fraction=1e-4)

    assert make_train_dataset(augmentation).max_shift_samples == 2


def test_train_dataset_noise_after_shift(
    make_train_dataset: TrainDatasetFactory, monkeypatch: pytest.MonkeyPatch
) -> None:

    # Instead of applying time shift, zero the whole clip.
    # Noise mixed after this survives, noise mixed before doesn't.
    # A fixed level at the median word RMS keeps the noise above the log-mel floor.
    monkeypatch.setattr(
        dataset, "time_shift", lambda clip, rng, max_shift_samples: np.zeros_like(clip)
    )
    train_dataset = make_train_dataset(
        dataset.AugmentationConfig(noise_range_db=(0.0, 0.0))
    )
    features, _ = train_dataset[dataset.Request(silence=False, epoch=0, index=0)]
    silent = _native.log_mel(np.zeros(dataset.CLIP_SAMPLES, np.float32))

    assert not np.array_equal(features, silent)
