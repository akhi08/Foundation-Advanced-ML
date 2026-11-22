import pytest
from recsys_lab.evaluation import mae ,rmse

def test_perfect_predictions_score_zero():
    actual=[1,2,3]
    assert rmse(actual,actual)==0
    assert mae(actual,actual)==0

def test_global_mean_example():
    # misses are 1.5 and 1.5
    assert rmse([5,2],[3.5,3.5])==pytest.approx(1.5)
    assert mae([5,2],[3.5,3.5])==pytest.approx(1.5)

def test_movie_mean_example():
    #misses are 1.0 and 0.5
    assert rmse([5,2],[4.0,2.5])==pytest.approx(0.7906,abs=1e-4)
    assert mae([5,2],[4.0,2.5])==pytest.approx(0.75)

def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="3 values but predicted has 2"):
        rmse([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="3 values but predicted has 2"):
        mae([1, 2, 3], [1, 2])    