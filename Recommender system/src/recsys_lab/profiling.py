"""Profile the rating data.

Promises:
- Counts of ratings, users and movies come from the ratings table itself.
- The rating scale is the sorted distinct rating values.
- Density is ratings divided by (users x movies), returned unrounded.
"""
from __future__ import annotations

import pandas as pd


def ratings_per_user(ratings: pd.DataFrame) -> pd.Series:
    """Number of ratings each user gave."""
    return ratings.groupby("user_id")["rating"].count()


def ratings_per_movie(ratings: pd.DataFrame) -> pd.Series:
    """Number of ratings each movie received."""
    return ratings.groupby("movie_id")["rating"].count()


def density(ratings: pd.DataFrame) -> float:
    """Share of user-movie pairs that have a rating (between 0 and 1)."""
    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()
    return len(ratings) / (n_users * n_movies)


def summary(ratings: pd.DataFrame) -> dict:
    """Headline numbers, all counted from the ratings table."""
    return {
        "n_ratings": len(ratings),
        "n_users": ratings["user_id"].nunique(),
        "n_movies": ratings["movie_id"].nunique(),
        "scale": sorted(ratings["rating"].unique().tolist()),
    }
