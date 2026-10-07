from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tiny_kws import _native

PCM_EDGES = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)

VALID_PCM = [
    pytest.param(PCM_EDGES, id="edges"),
    pytest.param(np.arange(-32768, 32768, dtype=np.int16), id="every_int16"),
]


@pytest.mark.parametrize("pcm", VALID_PCM)
def test_decodes_pcm_to_float(tmp_path: Path, pcm: np.ndarray) -> None:
    path = tmp_path / "clip.wav"
    sf.write(path, pcm, 16_000, subtype="PCM_16")

    samples = _native.load_wav(path)

    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, pcm / np.float32(32768))


def test_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    sf.write(path, PCM_EDGES, 16_000, subtype="PCM_16")

    np.testing.assert_array_equal(_native.load_wav(path), _native.load_wav(str(path)))


def chunk_ids(path: Path) -> list[bytes]:
    """Chunk IDs of a wav file, in order of appearance"""
    data, ids = path.read_bytes(), []
    offset = 12  # skip "RIFF", size and "WAVE"

    while offset + 8 <= len(data):
        ids.append(data[offset : offset + 4])
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        offset += 8 + chunk_size + (chunk_size & 1)  # chunks are padded to even length
    return ids


def test_skips_leading_metadata_chunks(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"

    with sf.SoundFile(path, "w", 16_000, 1, "PCM_16") as f:
        f.title = "Clip 0001"  # dummy title metadata string
        f.write(PCM_EDGES)
    assert chunk_ids(path) == [b"fmt ", b"LIST", b"data"]

    samples = _native.load_wav(path)

    np.testing.assert_array_equal(samples, PCM_EDGES / np.float32(32768))


def test_skips_trailing_metadata_chunks(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"

    with sf.SoundFile(path, "w", 16_000, 1, "PCM_16") as f:
        f.write(PCM_EDGES)
        f.software = "Lavf58.76.100"  # ffmpeg's encoder tag, typical wav metadata
    assert chunk_ids(path) == [b"fmt ", b"data", b"LIST"]

    samples = _native.load_wav(path)

    np.testing.assert_array_equal(samples, PCM_EDGES / np.float32(32768))


def test_no_samples_raises(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    sf.write(path, np.array([], dtype=np.int16), 16_000, subtype="PCM_16")

    with pytest.raises(ValueError, match="contains no samples"):
        _native.load_wav(path)


def test_missing_wav_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="cannot open"):
        _native.load_wav(tmp_path / "clip.wav")


def test_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        _native.load_wav(tmp_path)


DATASET_CLIPS = [
    # Speech clips all have just fmt and data chunks
    "right/bb05582b_nohash_3.wav",
    # The only files in the dataset with chunks after data
    "_background_noise_/doing_the_dishes.wav",
    "_background_noise_/dude_miaowing.wav",
    "_background_noise_/exercise_bike.wav",
    "_background_noise_/running_tap.wav",
]


@pytest.mark.parametrize("name", DATASET_CLIPS)
def test_dataset_matches_soundfile(dataset_root: Path, name: str) -> None:
    path = dataset_root / name
    reference, _ = sf.read(path, dtype="float32")
    np.testing.assert_array_equal(_native.load_wav(path), reference)
