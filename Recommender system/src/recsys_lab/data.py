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


# Why parents[2]:

# Recommender system/
# └── a/
#     └── b/
#         └── file.py
# - parents[0] is b/, the folder holding the file
# - parents[1] is a/
# - parents[2] is Recommender system/
# The file itself is not counted. Path(__file__) is file.py, and .parents starts at its containing folder.
# You can also check this with .parent, the singular version. Path(__file__).parent is the same as parents[0]. .parent.parent is parents[1], and so on. parents[2] is just a shorter way to write it.

PROJECT_ROOT=Path(__file__).resolve().parents[2]
DATA_DIR=PROJECT_ROOT/"data"/"raw"
RATINGS_PATH=DATA_DIR/"ratings.csv"
MOVIES_PATH=DATA_DIR/"movies.csv"
USERS_PATH=DATA_DIR/"users.csv"
# print(PROJECT_ROOT)
# print(RATINGS_PATH.exists())

# why tuple
# A tuple can't be changed after it's created, and column names are fixed, so a tuple protects them from accidental edits. 
# A tuple is technically a little faster than a list, but for 3 to 5 items the difference is too small to matter.
# Choose it for safety, not speed.
RATINGS_COLUMNS = ("user_id","movie_id","rating","timestamp")
MOVIES_COLUMNS = ("movie_id", "title", "genre")
USERS_COLUMNS = ("user_id", "age", "gender", "occupation", "zip_code")
RATING_MAX = 5
RATING_MIN = 1

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




    