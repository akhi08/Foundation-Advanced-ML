# Roadmap: build it one step at a time

Rule for every step: **write the code yourself -> write its tests -> run `python -m pytest -v` -> commit.**
Each step lists the file, what to build, and the "promises" to test. Source brief: `Dump/Recommender System Exercise/`.

## Step 0: Setup
- [ ] Create the GitHub repo, link it, first commit (README, ROADMAP, pyproject, .gitignore)
- [ ] `python -m pip install -e ".[dev]"`, confirm `python -m pytest` runs (0 tests is fine)
- [ ] Create `src/recsys_lab/__init__.py` with a docstring and `__version__`

## Step 1: `data.py` (get and validate)
Build: constants at top (paths, column names, rating range); `load_ratings`, `load_movies`, `load_users`.
Promises to test: missing file -> `FileNotFoundError`; missing column -> `ValueError`; empty file -> `ValueError`; timestamp converted to UTC datetime.

## Step 2: `profiling.py` (Section A)
Build: counts (ratings, users, movies, rating scale), ratings per user/movie, matrix density, two distribution plots.
Promises: counts match a tiny hand-made frame; density = ratings / (users x movies); plots saved to `reports/figures`.
Write-up: why a missing rating is not zero; one consequence of sparsity.

## Step 3: `splitting.py` (Section B)
Build: 80/20 outer split, 80/20 fit/validation (`random_state=42`); drop rows whose user or movie is absent from fitting data; user-movie matrix with NaN.
Promises: sizes are ~64/16/20%; no row appears in two sets; no unseen user/movie left in the evaluated rows; matrix built from fitting rows only.
Write-up: random vs chronological holdout.

## Step 4: `evaluation.py` + baselines in `models.py`
Build: RMSE, MAE, coverage; global-mean and movie-mean (global fallback).
Promises: perfect predictions give 0 error; coverage is the share of rows predicted without fallback; unseen movie falls back to global mean.

## Step 5: `similarity.py`
Build: user-user cosine (NaN -> 0), user-user Pearson (`DataFrame.corr`), item-item cosine (NaN -> 0), item-item adjusted cosine (center by user mean).
Promises: diagonal = 1; matrix is symmetric; identical rows give similarity 1; known tiny example matches hand calculation.

## Step 6: neighborhood predictors in `models.py`
Build: min overlap (3), k = 20, non-positive similarity ineligible, centered similarity-weighted average, fallback to movie mean, clip to rating range.
Promises: no qualifying neighbor -> fallback; predictions stay within 1-5; tiny example matches hand calculation.
Optional check: compare against `scikit-surprise`.

## Step 7: `tuning.py`
Build: one-at-a-time sweeps on validation: k (5, 10, 20, 50, 100), overlap (1, 3, 10), significance beta (5, 10, 20, 50), shrinkage beta (0, 1, 2, 5, 10, 20); report RMSE, MAE, coverage.
Promises: sweep returns one row per value; base settings reproduce the Step 6 numbers.

## Step 8: `training.py` (Section D, selection)
Build: compare all six models on identical validation rows, pick the winner without touching test, refit on outer training, score test once; save `reports/` tables and `artifacts/` run metadata.
Promises: test set not used before the final step; saved files exist.

## Step 8b (stretch, if time permits): `stats.py` (statistical learning)
Build, in order of value:
1. Bootstrap confidence intervals for RMSE and MAE of each model on the test rows.
2. Paired comparison of two models on identical rows (paired bootstrap, or a paired test on squared errors): is the gap real or noise?
3. A/B test toolkit: sample size and power for a chosen minimum detectable lift, a two-group significance test, and lift with its confidence interval.
Note: MovieLens has no online traffic, so the A/B part is a simulation that shows the method, not real results. Offline accuracy does not prove business impact; an A/B test is how that gets measured.
Promises: interval contains the point estimate; same seed gives the same interval; identical models give a difference near 0 and a wide-open "no difference" verdict; power rises with sample size.
Notebook: an optional closing statistics section, written independently with numpy/scipy (the notebook never imports this package).

## Step 9: `inference.py`
Build: top-5 unrated movies for a user, with the supporting rated titles and similarity values.
Promises: no already-rated movie returned; exactly 5 results; each recommendation has its supporting neighbors.

## Step 10: `cli.py`
Build: `profile`, `train`, `recommend --user N` commands. Last, because it only connects the other files.

## Step 11: Deliverables
- [ ] Notebook for Sections A-D, executed, plus an HTML export
- [ ] Validation comparison table, sensitivity tables, one final test table (six models)
- [ ] Interpretation of the k / overlap / significance / shrinkage experiments
- [ ] Top-5 recommendations with similarity-based explanation for at least one user
- [ ] 300-500 word summary (validation-based choice, final test result, two limitations)
- [ ] Cold-start explanation with one fallback each (new member, new movie)
- [ ] Fill in README results, learnings, limitations

## Agreed design decisions (code quality: simple, explainable, production-ready)
- `config.py` created at Step 3 (when the seed is first needed): paths, seed, split sizes, rating range, model defaults. Shared constants move there from `data.py`; column names stay in `data.py`.
- Run record (`artifacts/run_metadata.json`): selected model + settings, validation RMSE/MAE/coverage, test scores, row counts per split, seed, package + library versions, timestamp.
- Model dispatcher: one plain dict mapping model name -> builder for the six models (no class hierarchy).
- Cheap deployment items now: absolute paths (done), CLI input arguments, reuse `data.py` validation on new input, pinned versions. API and monitoring deferred to a later project.
- `ruff` (format + check): added to dev extras and `pyproject.toml` when `config.py` is created; learned in depth at project end.
- README: results, learnings, limitations and plain-words metric explanations filled at the end.
