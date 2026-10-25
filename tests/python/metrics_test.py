import numpy as np
import pytest

from tiny_kws import metrics


def test_confusion_matrix_contents() -> None:

    # A transposed matrix would count the clip at [5, 2].
    matrix = metrics.confusion_matrix(
        class_indices=np.array([2]), predicted_class_indices=np.array([5])
    )
    expected = np.zeros((12, 12), dtype=np.int64)
    expected[2, 5] = 1

    assert matrix.dtype == np.int64
    assert np.array_equal(matrix, expected)


def test_confusion_matrix_repeated_pairs_each_count() -> None:

    # Buffered addition would count the three clips as one.
    matrix = metrics.confusion_matrix(
        class_indices=np.array([3, 3, 3]), predicted_class_indices=np.array([3, 3, 3])
    )

    assert matrix[3, 3] == 3


def test_confusion_matrix_out_of_range_raises() -> None:
    with pytest.raises(IndexError):
        metrics.confusion_matrix(
            class_indices=np.array([0]), predicted_class_indices=np.array([12])
        )


def test_per_class_recall_contents() -> None:

    # Dividing by column sums would give per-class precision, [1.0, 6 / 7].
    recall = metrics.per_class_recall(np.array([[1, 1], [0, 6]]))

    assert recall.dtype == np.float64
    assert np.array_equal(recall, [0.5, 1.0])


def test_per_class_recall_no_clips_gives_nan() -> None:
    with pytest.warns(RuntimeWarning):
        recall = metrics.per_class_recall(np.array([[1, 1], [0, 0]]))

    assert np.array_equal(recall, [0.5, np.nan], equal_nan=True)


def test_balanced_accuracy_contents() -> None:

    # Weighting each class by its clip count would give accuracy, 0.875.
    balanced_accuracy = metrics.balanced_accuracy(np.array([[1, 1], [0, 6]]))

    assert type(balanced_accuracy) is float
    assert balanced_accuracy == 0.75


def test_balanced_accuracy_no_clips_gives_nan() -> None:

    # Averaging over only the classes with clips would give 0.5.
    with pytest.warns(RuntimeWarning):
        balanced_accuracy = metrics.balanced_accuracy(np.array([[1, 1], [0, 0]]))

    assert np.isnan(balanced_accuracy)


def test_accuracy_contents() -> None:

    # Weighting each class equally would give balanced accuracy, 0.75.
    accuracy = metrics.accuracy(np.array([[1, 1], [0, 6]]))

    assert type(accuracy) is float
    assert accuracy == 0.875


def test_accuracy_class_with_no_clips_does_not_give_nan() -> None:

    # Weighting per-class recalls by clip count would give NaN, since 0 * NaN is NaN.
    accuracy = metrics.accuracy(np.array([[1, 1], [0, 0]]))

    assert accuracy == 0.5


def test_accuracy_all_zero_matrix_gives_nan() -> None:
    with pytest.warns(RuntimeWarning):
        accuracy = metrics.accuracy(np.array([[0, 0], [0, 0]]))

    assert np.isnan(accuracy)
