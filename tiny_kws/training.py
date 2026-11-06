import math
import subprocess
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import TYPE_CHECKING, Literal, get_args, get_origin

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from tiny_kws import metrics
from tiny_kws.dataset import AugmentationConfig, EvaluationSet
from tiny_kws.labels import CLASS_NAMES
from tiny_kws.models import MODELS, ModelName

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

DeviceName = Literal["cuda", "mps", "cpu"]


def get_device(device: DeviceName | None = None) -> torch.device:
    """
    Return the named device if available,
    or this machine's accelerator if no device is specified.
    Never falls back to the CPU, which is only returned when named.

    Raises RuntimeError if the named device is unavailable,
    or if no device is specified and no accelerator is available.
    """

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("no accelerator available, use cpu instead")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda isn't available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps isn't available")

    return torch.device(device)


@dataclass(frozen=True, kw_only=True)
class GitState:
    """
    The repo's HEAD commit hash and its porcelain git status lines.
    """

    commit: str
    git_status: tuple[str, ...]


def git_state(repo_root: Path) -> GitState:
    """
    Return the HEAD commit hash and porcelain git status lines
    of repo_root, untracked files included.

    Raises CalledProcessError if repo_root isn't in a git repository with any commits,
    FileNotFoundError if git isn't on PATH.
    """
    commit_hash = subprocess.run(
        ["git", "-C", repo_root, "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    status = subprocess.run(
        # --untracked-files=normal overrides any status.showUntrackedFiles config.
        ["git", "-C", repo_root, "status", "--porcelain", "--untracked-files=normal"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout

    return GitState(commit=commit_hash, git_status=tuple(status.splitlines()))


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


def lr_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """
    Return the factor to scale the peak learning rate by at a given step.

    Linear ramp from 1 / warmup_steps to 1 over the first warmup_steps steps,
    then a decay to 0 over the first half period of a unipolar cosine.

    warmup_steps must be less than total_steps.
    """

    if step < warmup_steps:
        # Count from 1 so that the factor at the first step
        # is non-zero and reaches the peak at the last ramp step.
        return (step + 1) / warmup_steps

    phase = math.pi * (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 + 0.5 * math.cos(phase)


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    """
    Defines one training run by its model, seed, number of epochs,
    learning rate schedule, AdamW weight decay and augmentation.

    Raises ValueError if model isn't in MODELS, epochs is less than 1,
    warmup_epochs isn't in [0, epochs), peak_lr isn't greater than 0
    or peak_weight_decay isn't at least 0.
    """

    model: ModelName
    """
    Name of architecture to train, a key of MODELS.
    """

    seed: int
    """
    Seeds the initial weights, dropout and every training data draw.
    Validation and test sets are the same for every seed.
    Tuning runs should use seeds from 100 onwards,
    keeping seeds 0 to 4 for reported results.
    """

    epochs: int
    """
    Number of epochs, each one pass over a TrainSampler epoch plan.
    """

    warmup_epochs: int
    """
    Number of epochs at the start of training over which the learning rate ramps
    linearly up to peak_lr, before a cosine decay to 0 over the remaining epochs.
    0 is no warmup.
    """

    peak_lr: float
    """
    The learning rate at the schedule's peak,
    reached once the warmup ends and before the cosine decay begins.
    """

    peak_weight_decay: float
    """
    Weight decay per step at the schedule's peak, as a fraction of each parameter.
    Like the learning rate, it ramps during the warmup and decays to 0.
    """

    augmentation: AugmentationConfig

    def __post_init__(self) -> None:

        if self.model not in MODELS:
            raise ValueError(
                f"model must be one of {', '.join(sorted(MODELS))}, got {self.model!r}"
            )

        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")

        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError(
                f"warmup_epochs must be in [0, {self.epochs}), got {self.warmup_epochs}"
            )

        if not self.peak_lr > 0:
            raise ValueError(f"peak_lr must be positive, got {self.peak_lr}")

        if not self.peak_weight_decay >= 0:
            raise ValueError(
                f"peak_weight_decay must be non-negative, got {self.peak_weight_decay}"
            )


def config_from_dict[T: DataclassInstance](cls: type[T], data: object) -> T:
    """
    Return an instance of cls built from data,
    a dict in the form of a dataclasses.asdict result read back from JSON.

    A missing field never takes its default and an unknown key is never ignored,
    so the instance holds exactly the values in data.

    Raises TypeError if data isn't a dict, if it has missing or unknown keys,
    if a value doesn't match its field's type or if a field's type is unsupported.
    """

    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} must be a dict, got {type(data).__name__}")

    names = {field.name for field in fields(cls)}
    if missing := names - data.keys():
        raise TypeError(f"{cls.__name__} has missing keys {', '.join(sorted(missing))}")
    if unknown := data.keys() - names:
        raise TypeError(f"{cls.__name__} has unknown keys {', '.join(sorted(unknown))}")

    values = {
        field.name: _convert_value(field.type, data[field.name], field.name)
        for field in fields(cls)
    }
    return cls(**values)


def _convert_value(expected_type: object, value: object, name: str) -> object:
    """
    Convert value from its JSON form to expected_type:
    a dict to a dataclass, a list to a tuple and an int to a float.
    name is the field's name, used purely for error messages.

    Supports int, float, str, Literal of strings,
    fixed-length tuples, X | None and dataclasses.

    Raises TypeError when expected_type isn't supported,
    or if value doesn't match expected_type.
    """

    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return config_from_dict(expected_type, value)

    origin, args = get_origin(expected_type), get_args(expected_type)

    if origin is UnionType and len(args) == 2 and NoneType in args:
        if value is None:
            return None
        inner = args[0] if args[1] is NoneType else args[1]
        return _convert_value(inner, value, name)

    if origin is tuple:
        if not isinstance(value, list) or len(value) != len(args):
            raise TypeError(
                f"{name} must be a list of {len(args)} values, got {value!r}"
            )
        return tuple(_convert_value(arg, item, name) for arg, item in zip(args, value))

    # Whether value is one of Literal's options is left to the dataclass's checks.
    if origin is Literal:
        return _convert_value(str, value, name)

    if expected_type is float and type(value) is int:
        return float(value)

    if isinstance(expected_type, type) and expected_type in (int, float, str):
        if type(value) is not expected_type:
            raise TypeError(
                f"{name} must be {expected_type.__name__}, got {type(value).__name__}"
            )
        return value

    raise TypeError(f"{name} has unsupported type {expected_type}")


def config_template[T: DataclassInstance](cls: type[T]) -> dict[str, object]:
    """
    Return a dict with the keys config_from_dict expects for cls,
    with a nested dict for every dataclass field
    and "" as the placeholder for every other value.
    """

    return {
        field.name: (
            config_template(field.type)
            if isinstance(field.type, type) and is_dataclass(field.type)
            else ""
        )
        for field in fields(cls)
    }
