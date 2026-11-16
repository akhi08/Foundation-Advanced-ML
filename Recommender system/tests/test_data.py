import pandas as pd
import pytest

from recsys_lab.data import _read_csv, load_movies, load_ratings, load_users


# Each test follows the same 3 steps:
#   Arrange: write a tiny fake CSV into tmp_path (a throwaway folder)
#   Act:     call the function
#   Assert:  check the result (or that the right error is raised)


# --- Error cases for _read_csv ---------------------------------------------

def test_missing_file_raises(tmp_path):
    fake = tmp_path / "nope.csv"  # a path to a file we never create
    with pytest.raises(FileNotFoundError):
        _read_csv(fake, ("a", "b"))


def test_empty_file_raises(tmp_path):
    file = tmp_path / "empty.csv"
    file.write_text("a,b\n")  # header only, no data rows
    with pytest.raises(ValueError):
        _read_csv(file, ("a", "b"))


def test_missing_column_raises(tmp_path):
    file = tmp_path / "bad.csv"
    file.write_text("user_id,movie_id\n1,2\n")  # has a data row, so the empty-file check doesn't fire first
    with pytest.raises(ValueError, match="rating"):  # message is "Columns missing ['rating']"
        _read_csv(file, ("user_id", "rating"))  # file has no rating column


# --- Success cases -----------------------------------------------------------

def test_valid_file_returns_dataframe(tmp_path):
    file = tmp_path / "ok.csv"
    file.write_text("user_id,movie_id\n1,2\n")  # header + one row
    result = _read_csv(file, ("user_id", "movie_id"))
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["user_id", "movie_id"]
    assert len(result) == 1


def test_load_ratings_converts_timestamp_to_utc(tmp_path):
    file = tmp_path / "ratings.csv"
    file.write_text("user_id,movie_id,rating,timestamp\n196,242,3,881250949\n")
    result = load_ratings(file)
    # 881250949 seconds after 1 Jan 1970 is 4 Dec 1997, 15:55:49 UTC
    assert result["timestamp"].iloc[0] == pd.Timestamp("1997-12-04 15:55:49", tz="UTC")
    assert str(result["timestamp"].dt.tz) == "UTC"


def test_load_movies_returns_expected_columns(tmp_path):
    file = tmp_path / "movies.csv"
    file.write_text("movie_id,title,genre\n1,Toy Story (1995),Animation\n")
    result = load_movies(file)
    assert list(result.columns) == ["movie_id", "title", "genre"]


def test_load_users_returns_expected_columns(tmp_path):
    file = tmp_path / "users.csv"
    file.write_text("user_id,age,gender,occupation,zip_code\n1,24,M,technician,85711\n")
    result = load_users(file)
    assert list(result.columns) == ["user_id", "age", "gender", "occupation", "zip_code"]
