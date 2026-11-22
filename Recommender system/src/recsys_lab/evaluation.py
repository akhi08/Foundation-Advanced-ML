"""Scoring functions for rating predictions.

Promises:
- Perfect predictions score 0 for both RMSE and MAE.
- Results match hand-worked examples.
- Inputs of different lengths raise ValueError with a clear message.

Note: predictions must be in the same row order as the actual values.
scikit-learn compares by position and ignores pandas index labels.
"""
from __future__ import annotations

from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def _check_same_length(actual, predicted) -> None:
    """Raise ValueError if actual and predicted have different lengths."""
    if len(actual) != len(predicted):
        raise ValueError(
            f"actual has {len(actual)} values but predicted has {len(predicted)}"
        )


def rmse(actual, predicted) -> float:
    """Root mean squared error: typical miss, big misses count extra."""
    _check_same_length(actual, predicted)
    return root_mean_squared_error(actual, predicted)


def mae(actual, predicted) -> float:
    """Mean absolute error: average size of the miss."""
    _check_same_length(actual, predicted)
    return mean_absolute_error(actual, predicted)