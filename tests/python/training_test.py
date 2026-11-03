import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tiny_kws import dataset, labels, training

CPU = torch.device("cpu")


def patch_accelerator(monkeypatch: pytest.MonkeyPatch, accelerator: str | None) -> None:
    """
    Simulate a machine with only accelerator available, or none if accelerator is None.
    The real hardware running the tests then never affects the result.
    """

    monkeypatch.setattr(torch.cuda, "is_available", lambda: accelerator == "cuda")
    monkeypatch.setattr(
        torch.backends.mps, "is_available", lambda: accelerator == "mps"
    )


@pytest.mark.parametrize("accelerator", ["cuda", "mps"])
def test_get_device_picks_accelerator(
    monkeypatch: pytest.MonkeyPatch, accelerator: str
) -> None:

    patch_accelerator(monkeypatch, accelerator)

    assert training.get_device() == torch.device(accelerator)


def test_get_device_without_accelerator_raises(monkeypatch: pytest.MonkeyPatch) -> None:

    # A CPU fallback would hide the missing accelerator, so get_device must raise.
    patch_accelerator(monkeypatch, None)

    with pytest.raises(RuntimeError, match="no accelerator available"):
        training.get_device()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_get_device_named_available(
    monkeypatch: pytest.MonkeyPatch, device: training.DeviceName
) -> None:

    patch_accelerator(monkeypatch, device)

    assert training.get_device(device) == torch.device(device)


@pytest.mark.parametrize(("device", "accelerator"), [("cuda", "mps"), ("mps", "cuda")])
def test_get_device_named_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch, device: training.DeviceName, accelerator: str
) -> None:

    # Only the other accelerator is available, so mixing up the checks wouldn't raise.
    patch_accelerator(monkeypatch, accelerator)

    with pytest.raises(RuntimeError, match="isn't available"):
        training.get_device(device)


@pytest.mark.parametrize("accelerator", [None, "mps"])
def test_get_device_named_cpu(
    monkeypatch: pytest.MonkeyPatch, accelerator: str | None
) -> None:

    # With MPS available, preferring an accelerator over the request would return MPS.
    # With none, searching for an accelerator first would raise before seeing "cpu".
    patch_accelerator(monkeypatch, accelerator)

    assert training.get_device("cpu") == CPU


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", repo, *args], check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """
    A git repo with a.txt committed and no changes.
    """

    git(tmp_path, "init")

    git(tmp_path, "config", "user.name", "Joe Bloggs")
    git(tmp_path, "config", "user.email", "joe_bloggs@example.com")

    (tmp_path / "a.txt").write_text("a")

    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-m", "Initial commit")

    return tmp_path


def test_git_state_clean_repo(repo: Path) -> None:

    state = training.git_state(repo)

    # Without strip(), rev-parse's trailing newline would add a character to the hash.
    assert len(state.commit) == 40

    assert state.git_status == ()


def test_git_state_status_lines_verbatim(repo: Path) -> None:

    (repo / "a.txt").write_text("changed")
    (repo / "b.txt").write_text("b")

    # strip() on the status output would drop the leading space of " M a.txt".
    assert training.git_state(repo).git_status == (" M a.txt", "?? b.txt")


def test_git_state_overrides_untracked_config(repo: Path) -> None:

    git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "b.txt").write_text("b")

    # Without --untracked-files=normal, this config would hide b.txt,
    # so a run with an untracked file would pass as clean.
    assert training.git_state(repo).git_status == ("?? b.txt",)


def test_git_state_outside_repo_raises(tmp_path: Path) -> None:

    with pytest.raises(subprocess.CalledProcessError):
        training.git_state(tmp_path)


class PerfectClassifier(torch.nn.Module):
    """
    Predict each clip's true class, which the fixture stores in its first feature.

    The true class's probability grows with the class index,
    so each class has a different loss.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        class_indices = features[:, 0, 0].long()
        margins = (class_indices + 1).float().unsqueeze(1)
        return F.one_hot(class_indices, len(labels.CLASS_NAMES)).float() * margins


@pytest.fixture
def evaluation_set() -> dataset.EvaluationSet:
    """
    Clip counts increase from 1 to 12 in class order (78 in all), shuffled.
    """

    counts = np.arange(1, len(labels.CLASS_NAMES) + 1)
    class_indices = np.repeat(np.arange(len(labels.CLASS_NAMES)), counts)
    np.random.default_rng(seed=0).shuffle(class_indices)

    features = np.zeros((len(class_indices), 32, 98), dtype=np.float32)
    features[:, 0, 0] = class_indices

    return dataset.EvaluationSet(features=features, class_indices=class_indices)


def test_evaluate_confusion_matrix(evaluation_set: dataset.EvaluationSet) -> None:

    # Batches of 32 split the 78 clips into 32, 32 and 14.
    evaluation = training.evaluate(
        PerfectClassifier(), evaluation_set, device=CPU, batch_size=32
    )

    assert np.array_equal(evaluation.confusion, np.diag(np.arange(1, 13)))


def test_evaluate_loss_is_class_balanced(evaluation_set: dataset.EvaluationSet) -> None:

    evaluation = training.evaluate(
        PerfectClassifier(), evaluation_set, device=CPU, batch_size=32
    )

    # Softmax cross-entropy for a true logit m, zeros elsewhere, is log(e^m + 11) - m.
    # Each class has one margin, so the mean over margins weights classes equally.
    # Weighting each class by its clip count would give 0.0785.
    margins = np.arange(1, 13)
    expected = np.mean(np.log(np.exp(margins) + 11) - margins)

    assert evaluation.loss == pytest.approx(expected, rel=1e-6)


def test_evaluate_runs_model_in_eval_mode_without_grad(
    evaluation_set: dataset.EvaluationSet,
) -> None:

    modes = []

    class Recorder(PerfectClassifier):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            modes.append((self.training, torch.is_grad_enabled()))
            return super().forward(features)

    model = Recorder()
    training.evaluate(model, evaluation_set, device=CPU, batch_size=32)

    assert modes == [(False, False)] * 3
    assert not model.training
