from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from tiny_kws import metrics
from tiny_kws.dataset import EvaluationSet
from tiny_kws.labels import CLASS_NAMES


@dataclass(frozen=True, kw_only=True)
class Evaluation:
    """
    A model's class-balanced loss and confusion matrix on an EvaluationSet.
    """

    loss: float
    confusion: NDArray[np.int64]


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    evaluation_set: EvaluationSet,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> Evaluation:
    """
    Run model over evaluation_set and return its loss and confusion matrix.
    Leaves model in eval mode.

    The loss is the class-balanced cross-entropy (the mean of per-class means).
    Each class is weighted equally, however many clips it has.
    """

    model.eval()
    batches = torch.from_numpy(evaluation_set.features).split(batch_size)
    logits = torch.cat([model(batch.to(device)) for batch in batches])

    class_indices = torch.from_numpy(evaluation_set.class_indices).to(device)
    class_counts = np.bincount(evaluation_set.class_indices, minlength=len(CLASS_NAMES))
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32, device=device)

    # Weighting by 1 / class_counts turns each class's summed loss into its mean.
    loss = F.cross_entropy(logits, class_indices, weight=class_weights, reduction="sum")

    return Evaluation(
        loss=loss.item() / len(CLASS_NAMES),
        confusion=metrics.confusion_matrix(
            class_indices=evaluation_set.class_indices,
            predicted_class_indices=logits.argmax(dim=1).cpu().numpy(),
        ),
    )
