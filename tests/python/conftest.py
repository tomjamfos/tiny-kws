from pathlib import Path

import pytest

DATASET_ROOT = Path(__file__).parents[2] / "data/datasets/speech-commands-v0.02"


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    if not DATASET_ROOT.is_dir():
        pytest.skip(f"dataset not found at {DATASET_ROOT}")
    return DATASET_ROOT
