import pandas as pd
import pytest

from recsys_lab.splitting import build_matrix, filter_to_known, split_ratings


@pytest.fixture
def ratings():
    # 100 rows with unique users, so rows are easy to tell apart
    return pd.DataFrame(
        {"user_id": range(100), "movie_id": range(100), "rating": [3] * 100}
    )


def test_split_sizes(ratings):
    splits = split_ratings(ratings)
    assert len(splits.test) == 20
    assert len(splits.outer_train) == 80
    assert len(splits.fitting) == 64
    assert len(splits.validation) == 16


def test_no_row_in_two_sets(ratings):
    splits = split_ratings(ratings)
    fitting = set(splits.fitting["user_id"])
    validation = set(splits.validation["user_id"])
    test = set(splits.test["user_id"])
    assert not fitting & validation
    assert not fitting & test
    assert not validation & test


def test_same_seed_gives_same_split(ratings):
    first = split_ratings(ratings)
    second = split_ratings(ratings)
    assert first.fitting.equals(second.fitting)
    assert first.test.equals(second.test)


def _frame(pairs):
    return pd.DataFrame(pairs, columns=["user_id", "movie_id", "rating"])


def test_filter_keeps_known_and_drops_unseen_user_and_movie():
    reference = _frame([(1, 10, 3), (2, 20, 4)])
    rows = _frame([(1, 20, 5), (3, 10, 4), (1, 30, 2)])
    result = filter_to_known(rows, reference)
    assert result[["user_id", "movie_id"]].values.tolist() == [[1, 20]]


def test_build_matrix_shape_and_missing_cells_are_nan():
    ratings = _frame([(1, 10, 5), (1, 20, 3), (2, 10, 4)])
    matrix = build_matrix(ratings)
    assert matrix.shape == (2, 2)
    assert matrix.loc[1, 10] == 5
    assert pd.isna(matrix.loc[2, 20])