import numpy as np
from numpy.typing import NDArray

from tiny_kws.labels import CLASS_NAMES


def confusion_matrix(
    *, class_indices: NDArray[np.int64], predicted_class_indices: NDArray[np.int64]
) -> NDArray[np.int64]:
    """
    Clip counts by true class index (rows) and predicted class index (columns).
    One row and one column per class in CLASS_NAMES.

    Raises IndexError if an index is out of range for CLASS_NAMES.
    """
    num_classes = len(CLASS_NAMES)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Unbuffered, so repeated (true, predicted) pairs each count.
    np.add.at(matrix, (class_indices, predicted_class_indices), 1)
    return matrix


def per_class_recall(confusion: NDArray[np.int64]) -> NDArray[np.float64]:
    """
    Fraction of each class's clips predicted correctly.

    A class with no clips gives NaN.
    """
    return np.diag(confusion) / confusion.sum(axis=1)


def balanced_accuracy(confusion: NDArray[np.int64]) -> float:
    """
    Mean of the per-class recalls.
    Each class is weighted equally, however many clips it has.

    A class with no clips gives NaN.
    """
    return float(np.mean(per_class_recall(confusion)))


def accuracy(confusion: NDArray[np.int64]) -> float:
    """
    Fraction of all clips predicted correctly.
    Each class is weighted by its clip count.

    An all-zero matrix gives NaN.
    """
    return float(np.trace(confusion) / confusion.sum())
