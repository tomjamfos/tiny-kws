import torch
from torch import nn

from tiny_kws import analysis


def test_num_parameters_includes_submodules() -> None:

    # Counting parameter tensors instead of their elements would give 3.
    # Counting without recursing into submodules would give 0.
    # Counting a bias for each Linear would give 46.
    model = nn.Sequential(nn.Linear(8, 4, bias=False), nn.Linear(4, 2))
    count = analysis.num_parameters(model)

    assert type(count) is int
    assert count == 42


def test_num_parameters_includes_frozen() -> None:

    # Counting only parameters with requires_grad=True
    # would give 8, the elements in the bias.
    model = nn.Linear(8, 8)
    model.weight.requires_grad_(False)

    assert analysis.num_parameters(model) == 72


def test_num_parameters_counts_shared_parameter_once() -> None:

    # Counting the shared weight for each Linear would give 144.
    first, second = nn.Linear(8, 8), nn.Linear(8, 8)
    first.weight = second.weight
    model = nn.Sequential(first, second)

    assert analysis.num_parameters(model) == 80


def test_num_parameters_excludes_buffers() -> None:

    # Counting buffers too, as the state dict does, would give 104.
    model = nn.Linear(8, 8)
    model.register_buffer("buffer", torch.zeros(32))

    assert analysis.num_parameters(model) == 72
