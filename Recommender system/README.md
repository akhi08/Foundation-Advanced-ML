# Project 1: Movie Rating Prediction (Recommender System Lab)

A small, tested, end-to-end pipeline that predicts how a member would rate a movie
they have not seen yet, then uses those predictions to suggest what to watch next.
Built step by step following [ROADMAP.md](ROADMAP.md).

> Status: folders and roadmap only; code is written by hand, one step at a time. Sections below describe the plan; results get filled in as we build.

## The business problem (in simple words)

A movie-streaming service has thousands of titles and members who rate only a few.
The product team wants to know: *"For this member, which movies they haven't rated
will they probably like?"* so the home screen can show those first.

We do not know the missing ratings, so we **predict** them and compare two ideas:

1. **Simple baselines**: guess the overall average rating, or the movie's average rating.
2. **Collaborative filtering**: "people who rate like you liked this" (user-user) or
   "movies rated like this one were liked by you" (item-item), using similarity between
   members or movies.

The final recommendation is based on two things: how accurate each method is on
ratings it has never seen, and what it costs to run it in practice (coverage,
sparsity, cold start).

## Data (in `data/raw/`, not committed)

MovieLens 100K: `ratings.csv` (~100,000 ratings, 1-5), `movies.csv` (titles, genre),
`users.csv` (age, gender, occupation, zip). Timestamps are Unix seconds (UTC).

## Flow: how we are doing it

```
load + validate  ->  profile  ->  split  ->  fit predictors  ->  tune  ->  select  ->  test once  ->  recommend
   data.py        profiling.py  splitting.py  models.py/       tuning.py  training.py  training.py  inference.py
                                              similarity.py
```

| Step | What we do | Why |
|---|---|---|
| A. Profile | Count ratings/users/movies, ratings per user and movie, matrix density | A missing rating is *unknown*, not zero; sparsity limits every method later |
| B. Split | 80/20 outer train/test, then 80/20 fit/validation (`random_state=42`); drop users/movies unseen in fitting data | Test stays untouched; no leakage of means or similarities |
| C. Fit | 2 baselines + 4 neighborhood models (user-user cosine, user-user Pearson, item-item cosine, item-item adjusted cosine); tune k, min overlap, significance weighting, shrinkage | Compare simple vs. smarter fairly |
| D. Select | Pick the winner on **validation** RMSE (check MAE, coverage); refit on outer training; score test once | Honest final number |
| Use | Top-5 unrated movies for a user, with the rated titles and similarities that explain each | Explainable output for the product team |

Metrics: RMSE, MAE, and coverage (share of rows a neighborhood model could predict
without falling back to the baseline).

## Planned folder structure (files appear as each roadmap step is done)

```
Recommender system/
├── pyproject.toml          # package identity, dependencies, pytest config
├── README.md
├── src/recsys_lab/
│   ├── data.py             # load + validate
│   ├── profiling.py        # Section A
│   ├── splitting.py        # Section B (splits, user-movie matrix)
│   ├── similarity.py       # cosine / Pearson / adjusted cosine
│   ├── models.py           # baselines + neighborhood predictors
│   ├── tuning.py           # parameter sensitivity sweeps
│   ├── evaluation.py       # RMSE, MAE, coverage
│   ├── stats.py            # (stretch) confidence intervals, model comparison tests, A/B test toolkit
│   ├── training.py         # select -> refit -> test once -> save
│   ├── inference.py        # top-N recommendations + explanations
│   └── cli.py              # terminal commands
├── tests/                  # one test file per module
├── notebook/               # reproducible notebook + HTML export (deliverable)
├── data/raw, data/processed   # ignored by git
├── artifacts/              # saved model + run metadata (ignored)
└── reports/, reports/figures  # tables and plots (ignored)
```

## How to run (once built)

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
python -m recsys_lab.cli profile
python -m recsys_lab.cli train
python -m recsys_lab.cli recommend --user 196
```

## Results, learnings, limitations

To be added as we build.
