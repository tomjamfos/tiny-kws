"""
Speech Commands v0.02 statistics, written to data/derived/speech_commands_stats.json

Used to compute the values of hardcoded constants in tiny_kws/dataset.py.

archive_sha256 restates the hash recorded in CLAUDE.md for the archive the
dataset was extracted from. The hash isn't checked against the dataset on disk.
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tiny_kws.dataset import build_splits, load_word_clip, rms

DATA_ROOT = Path(__file__).parents[1] / "data"
DATASET_ROOT = DATA_ROOT / "datasets/speech-commands-v0.02"
OUTPUT_PATH = DATA_ROOT / "derived/speech_commands_stats.json"


def main() -> None:

    splits = build_splits(DATASET_ROOT)

    per_clip_rms = []
    for key in tqdm(splits.train, unit="clip", leave=False):
        # RMS over full padded clips, as this is what the model sees.
        per_clip_rms.append(rms(load_word_clip(DATASET_ROOT, key)))

    median_rms = float(np.median(per_clip_rms))
    median_rms_dbfs = 20.0 * np.log10(median_rms)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(
            {
                "num_train_word_clips": len(splits.train),
                "num_validation_word_clips": len(splits.validation),
                "num_test_word_clips": len(splits.test),
                "train_padded_clip_median_rms": median_rms,
                "train_padded_clip_median_rms_dbfs": median_rms_dbfs,
                "archive_sha256": (
                    "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58"
                ),
            },
            f,
            indent=2,
        )
    print(
        f"Word clips: {len(splits.train):,} train, "
        f"{len(splits.validation):,} validation, {len(splits.test):,} test\n"
        f"Median RMS of padded train clips: {median_rms_dbfs:.4f} dBFS\n"
        f"Written to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
