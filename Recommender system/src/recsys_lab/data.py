"""Load and validate the data
This loads three files CSV files from 'data/raw/':
-ratings.csv (who rated which movie,what score, and when)
-movies.csv (movie titles and genres)
- users.csv (age,gender,occupation ,zip code)
promises:
-A missing file raises FileNotFoundError.
-A file with no rows raises ValueError.
-A file missing a required column raises ValueError.
-The rating timestamp is returned as UTC datetime,not raw seconds.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from recsys_lab.config import Path, PROJECT_ROOT,RATINGS_PATH,RATINGS_COLUMNS,MOVIES_PATH,MOVIES_COLUMNS,USERS_PATH,USERS_COLUMNS



def _read_csv(path:Path,required_columns: tuple[str,...]) -> pd.DataFrame:
    """Read a CSV & check if it exists is not empty and has required columns"""
    if not path.exists():
        raise FileNotFoundError(f"File not found {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Empty file")
    missing=[col for col in required_columns if col not in df.columns]
    if missing:
        raise  ValueError(f"Columns missing {missing}")
    return df

def load_ratings(path:Path = RATINGS_PATH) -> pd.DataFrame:
    """
    Read a rating CSV and converting timestamp as datetime64[ns,UTC]
    """
    rating = _read_csv(path,RATINGS_COLUMNS)
    rating['timestamp'] = pd.to_datetime(rating['timestamp'],unit='s',utc=True)
    return rating


def load_movies(path:Path = MOVIES_PATH) -> pd.DataFrame:
    """
    Read a movies CSV
    """
    movies = _read_csv(path,MOVIES_COLUMNS)
    return movies

def load_users(path:Path = USERS_PATH) -> pd.DataFrame:
    """
    Read a users CSV
    """
    users = _read_csv(path,USERS_COLUMNS)
    return users




    