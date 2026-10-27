import torch

from tiny_kws.models import normalisation


def test_per_bin_mean_normalisation_contents() -> None:

    # Averaging over the mel bins would give [2, 1.5, 1, -0.5] for the constant bin.
    # Averaging over the whole clip would give [1, 1, 1, 1] for the constant bin.
    clip = torch.tensor([[[1.0, 2.0, 3.0, 6.0], [5.0, 5.0, 5.0, 5.0]]])
    normalised = normalisation.PerBinMeanNormalisation()(clip)
    expected = torch.tensor([[[-2.0, -1.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0]]])

    assert torch.equal(normalised, expected)


def test_per_bin_mean_normalisation_does_not_depend_on_batch() -> None:

    batch = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 6.0], [5.0, 5.0, 5.0, 5.0]],
            [[7.0, 7.0, 7.0, 7.0], [3.0, 3.0, 3.0, 3.0]],
        ]
    )
    normalise = normalisation.PerBinMeanNormalisation()
    normalised_separately = torch.cat([normalise(clip) for clip in batch.split(1)])

    assert torch.equal(normalise(batch), normalised_separately)
