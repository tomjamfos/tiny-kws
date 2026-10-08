from pathlib import Path

import pytest

from tiny_kws import dataset, labels


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
    return tmp_path


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
