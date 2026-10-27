import torch


def num_parameters(model: torch.nn.Module) -> int:
    """
    Number of parameters in model, including those with requires_grad=False.
    A shared parameter is counted once, however many times the model uses it.
    Buffers aren't counted.
    """
    return sum(parameter.numel() for parameter in model.parameters())
