# Foundation & Advanced ML

A learning curriculum, built project by project by hand, with Claude as guide and roadmap
maker. Each project is a small, tested, production-style pipeline that reuses the same blueprint:

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
| 1 | [Recommender system](Recommender%20system/) | Rating prediction, collaborative filtering | roadmap ready, code not started |

## Conventions

- `Dump/` holds the reference material (exercise briefs, original datasets) each project is built from. It is **not** committed.
- Data and generated outputs (`data/raw`, `artifacts`, `reports`) are git-ignored in every project.
- Each project has its own README explaining its business problem and flow.

## References

- **Theory and concepts:** *Data Mining for Business Analytics*, Shmueli, Bruce, Gedeck, Patel, Wali.
- **Approach and code patterns:** *Approaching (Almost) Any Machine Learning Problem*, Abhishek Thakur.
- Code written here aims to be production quality and deployment ready. The books are references, not templates, so the code may differ from them.
