import pytest
import torch

from tiny_kws.labels import CLASS_NAMES
from tiny_kws.models import baseline


@pytest.mark.parametrize("batch_size", [1, 3])
def test_baseline_output_shape(batch_size: int) -> None:

    features = torch.zeros(batch_size, 32, 98)
    logits = baseline.Baseline().eval()(features)

    assert logits.shape == (batch_size, len(CLASS_NAMES))


def test_baseline_ignores_per_bin_offset() -> None:

    features = torch.zeros(1, 32, 98)
    offset = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1)

    model = baseline.Baseline().eval()

    assert torch.equal(model(features + offset), model(features))
