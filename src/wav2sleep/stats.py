"""Evalation statistics for sleep staging.

Sleep labels and predictions are passed as matrices where each row contains a night of data.
"""

import numpy as np


def confusion_accuracy(cmat) -> float:
    """Calculate accuracy from confusion matrix."""
    return np.trace(cmat) / np.sum(cmat)


def cohens_kappa(cmat, n_classes: int = 4) -> float:
    """Compute Cohen's kappa from a confusion matrix.

    Based on the Scikit learn implementation:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
    which takes raw outputs and labels as inputs.

    Args:
        conf_mat (array)
    """
    cmat = cmat.astype(float)
    sum0 = np.sum(cmat, axis=0)
    sum1 = np.sum(cmat, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    w_mat = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    k = np.sum(w_mat * cmat) / np.sum(w_mat * expected)
    return 1 - k
