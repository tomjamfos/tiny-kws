import torch
from torch import nn


class PerBinMeanNormalisation(nn.Module):
    """
    Subtracts each mel bin's mean over the last axis, which must be time.

    Each clip is normalised individually, so the result doesn't depend on the batch.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=-1, keepdim=True)
