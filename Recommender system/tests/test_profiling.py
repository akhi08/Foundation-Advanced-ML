import pandas as pd
import pytest

from recsys_lab.profiling import density, ratings_per_movie, ratings_per_user, summary


@pytest.fixture
def ratings():
    # 3 users, 2 movies, 4 ratings: easy to check by hand
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "movie_id": [10, 20, 10, 20],
            "rating": [5, 3, 4, 5],
        }
    )


def test_summary_counts(ratings):
    result = summary(ratings)
    assert result["n_ratings"] == 4
    assert result["n_users"] == 3
    assert result["n_movies"] == 2


def test_summary_scale_is_sorted_distinct_values(ratings):
    assert summary(ratings)["scale"] == [3, 4, 5]


def test_ratings_per_user(ratings):
    assert ratings_per_user(ratings).to_dict() == {1: 2, 2: 1, 3: 1}


def test_ratings_per_movie(ratings):
    assert ratings_per_movie(ratings).to_dict() == {10: 2, 20: 2}


def test_density_is_ratings_over_possible_pairs(ratings):
    assert density(ratings) == pytest.approx(4 / (3 * 2))