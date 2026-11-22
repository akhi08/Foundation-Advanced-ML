"""Split the ratings and build the fitting matrix.

Promises:
- The outer split holds out TEST_SIZE as test; the rest is split again into
  fitting and validation (VALID_SIZE). Same seed gives the same split.
- No row appears in two sets.
- Rows whose user or movie is absent from the reference data are dropped.
- The matrix has users as rows, movies as columns, NaN where nothing was rated.
"""
from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from recsys_lab.config import RANDOM_STATE, TEST_SIZE, VALID_SIZE


class Splits(NamedTuple):
    outer_train: pd.DataFrame
    test: pd.DataFrame
    fitting: pd.DataFrame
    validation: pd.DataFrame


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    valid_size: float = VALID_SIZE,
    random_state: int = RANDOM_STATE,
) -> Splits:
    """Split ratings into outer_train/test, then outer_train into fitting/validation."""
    outer_train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    fitting, validation = train_test_split(outer_train, test_size=valid_size, random_state=random_state)
    return Splits(outer_train, test, fitting, validation)


def filter_to_known(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose user AND movie both appear in the reference data."""
    known_user = rows["user_id"].isin(reference["user_id"])
    known_movie = rows["movie_id"].isin(reference["movie_id"])
    return rows[known_user & known_movie].copy()


def build_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """User-movie matrix: users as rows, movies as columns, NaN where not rated."""
    return ratings.pivot(index="user_id", columns="movie_id", values="rating")


# How a caller uses it

# splits = split_ratings(load_ratings())
# validation = filter_to_known(splits.validation, splits.fitting)
# test = filter_to_known(splits.test, splits.outer_train)
# matrix = build_matrix(splits.fitting)