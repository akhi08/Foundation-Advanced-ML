# Foundation & Advanced ML

A learning curriculum, built project by project. Each project is a small, tested,
production-style pipeline that reuses the same blueprint:

```
src/<package>/   data -> features -> models -> training -> evaluation -> inference -> cli
tests/           one test per promise
notebook/        reproducible analysis
pyproject.toml   installable package + pytest config
README.md        problem, flow, results, learnings
```

The path runs from foundations (decision trees, PCA, association rules / apriori)
to advanced topics (recommender systems and beyond).

## Projects

| # | Project | Topic | Status |
|---|---|---|---|
| 1 | Customer Intelligence ML-Lab | Churn classification | to be added |
| 2 | [Recommender system](Recommender%20system/) | Rating prediction, collaborative filtering | scaffolded |

## Conventions

- Reference material (exercise PDFs, original datasets) lives in `Dump/` and is **not** committed.
- Data and generated outputs (`data/raw`, `artifacts`, `reports`) are git-ignored in every project.
- Each project has its own README explaining its business problem and flow.
