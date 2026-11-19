"""Contains path for different files and different constant parameters"""

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

RATING_MAX = 5
RATING_MIN = 1
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.2
