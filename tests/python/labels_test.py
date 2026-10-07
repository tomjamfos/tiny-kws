from pathlib import Path

from tiny_kws import labels


def test_class_names_order() -> None:
    assert labels.CLASS_NAMES == (
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes",
        "unknown",
        "silence",
    )


def test_word_lists_unique() -> None:
    assert len(labels.TARGET_WORDS) == len(set(labels.TARGET_WORDS)) == 10
    assert len(labels.UNKNOWN_WORDS) == len(set(labels.UNKNOWN_WORDS)) == 25


def test_word_lists_disjoint() -> None:
    assert not set(labels.TARGET_WORDS) & set(labels.UNKNOWN_WORDS)


def test_word_to_class_index_covers_every_word() -> None:
    assert labels.WORD_TO_CLASS_INDEX.keys() == set(labels.TARGET_WORDS) | set(
        labels.UNKNOWN_WORDS
    )


def test_word_lists_match_dataset_folders(dataset_root: Path) -> None:
    folders = {p.name for p in dataset_root.iterdir() if p.is_dir()} - {
        "_background_noise_"
    }
    assert folders == set(labels.TARGET_WORDS) | set(labels.UNKNOWN_WORDS)
