"""models.py: baseline predictors (global mean, movie mean).

Promises:
- Means are learned from the rows given to fit() only.
- predict() returns one value per row, with the same index as the input.
- Predictions stay inside the rating range.
- predict() before fit() raises a clear error.
"""
import pandas as pd

from recsys_lab.config import RATING_MAX, RATING_MIN


class GlobalMeanBaseline:
    def __init__(self):
        self.global_mean = None

    def fit(self, fitting_rows):
        self.global_mean = fitting_rows["rating"].mean()
        return self

    def predict(self, rows):
        if self.global_mean is None:
            raise RuntimeError("Call fit() before predict().")
        return pd.Series(self.global_mean, index=rows.index).clip(RATING_MIN, RATING_MAX)


class MovieMeanBaseline:
    def __init__(self):
        self.movie_means = None
        self.global_mean = None

    def fit(self, fitting_rows):
        self.movie_means = fitting_rows.groupby("movie_id")["rating"].mean()
        self.global_mean = fitting_rows["rating"].mean()  # fallback for unseen movies
        return self

    def predict(self, rows):
        if self.global_mean is None:
            raise RuntimeError("Call fit() before predict().")
        predictions = rows["movie_id"].map(self.movie_means).fillna(self.global_mean)
        return predictions.clip(RATING_MIN, RATING_MAX)