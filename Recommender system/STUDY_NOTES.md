# Study notes: formulas, tables, interview questions

Running file. Each step adds: formula (plain text), what each part means, a tiny example, and cross-questions an interviewer might ask.

## 1. Data and sparsity (Section A)
```
density = number_of_ratings / (number_of_users * number_of_movies)
```
MovieLens 100K: 100,000 / (943 * 1682) = 6.3% filled.

| Question | Short answer |
|---|---|
| Why is a missing rating not 0? | 0 would mean "rated it zero". Missing means unknown. |
| One consequence of sparsity? | Two users rarely share many rated movies, so similarity rests on little evidence. |

## 2. Split and leakage (Section B)
| Question | Short answer |
|---|---|
| Train / validation / test purpose? | Train = learn, validation = choose settings, test = one final honest score (mock exams vs real exam). |
| Why drop unseen users/movies from validation? | The model has nothing to learn them from (cold start). Evaluate them separately. |
| Random vs chronological holdout? | Random can train on ratings made after the ones it predicts (future leaks into past). Chronological is closer to real use. We used random (PDF), noted as a limitation. |
| What is leakage? | Using information from validation/test rows when computing means or similarities. |

## 3. Metrics and baselines (Section C)
```
RMSE = sqrt( mean( (actual - predicted)^2 ) )
MAE  = mean( |actual - predicted| )
coverage = rows predicted by the model without fallback / all rows
```
- RMSE punishes large misses more than MAE.
- Global-mean baseline RMSE equals the standard deviation of the ratings.

| Baseline | RMSE | MAE |
|---|---|---|
| Global mean | 1.119 | 0.940 |
| Movie mean | 1.030 | 0.822 |

| Question | Short answer |
|---|---|
| Why is the movie mean noisy? | Movies with 1 or 2 ratings have unreliable means. Shrinkage pulls them toward the global mean. |
| Why must fit() learn only from fitting rows? | Otherwise validation information leaks into the model. |

## 4. User-user cosine similarity
```
sim(u, v) = sum_i( r_ui * r_vi ) / ( sqrt(sum_i r_ui^2) * sqrt(sum_i r_vi^2) )
```
| Part | Meaning |
|---|---|
| r_ui | Rating of user u for movie i (blank counted as 0) |
| sum_i( r_ui * r_vi ) | Multiply the two users' ratings movie by movie, add up |
| sqrt(sum_i r_ui^2) | Length of u's rating list |
| Result | 1 = same pattern, 0 = nothing in common |

Tiny example: Ann (5,4,0), Ben (4,5,0): 40 / (sqrt41 * sqrt41) = 0.98.

| Question | Short answer |
|---|---|
| Diagonal is 1, matrix is symmetric: why? | A user matches themselves exactly. The formula does not depend on order. |
| Why drop the user themselves from their neighbors? | Their own rating of the target movie is what we are trying to predict. |
| Weakness of cosine with blanks as 0? | A blank acts like a dislike, and it ignores how many movies are shared. |

## 5. Overlap (shared rated movies)
```
overlap(u, v) = sum_i( rated_ui * rated_vi )
rated_ui = 1 if user u rated movie i, else 0
```
In code: `rated @ rated.T`. Diagonal = the user's own number of ratings.

| Question | Short answer |
|---|---|
| Why min overlap 3? | A similarity built on 1 or 2 shared movies is weak evidence. |
| If overlap is 0, cosine is? | 0. Nothing multiplies, so the top of the formula is 0. The bottom is each user's own list length, so it is not 0 (NaN only for a user with no ratings). |
| What does min overlap solve? | Two users who each rated one movie, both with 5, get cosine 1.0 (looks perfect) but it is chance. Overlap 1 < 3 removes them. |
| overlap.T @ overlap instead? | That multiplies the other way and counts users shared by two movies: used for item-item models later. |
| Overlap vs "neighbor rated the target movie"? | Different checks. Overlap = movies two users share. The neighbor must also have rated the movie we predict. Both are required. |

## 6. User means and centered ratings
```
user_mean(u)   = average of the ratings user u gave (blanks ignored)
centered(u, i) = r_ui - user_mean(u)
```
| Part | Meaning |
|---|---|
| user_mean(u) | That user's usual level |
| centered(u, i) | Above (+) or below (-) their usual, for movie i |

Example: a 4 from a user with mean 2 is +2 (strong praise). A 5 from a user with mean 4.8 is +0.2 (almost their usual).

| Question | Short answer |
|---|---|
| Why center before averaging neighbors? | Users use the scale differently (generous vs harsh). Centering compares taste, not scale habit. |
| Pandas pitfall? | `table - Series` lines the Series up with columns. Use `table.sub(series, axis=0)` to subtract row-wise. |
| Check that centering worked? | Every user's centered average is ~0. |

## 7. User-user prediction for one (user, movie)
```
prediction(u, m) = user_mean(u) + sum_v( sim(u,v) * centered(v,m) ) / sum_v( sim(u,v) )
```
| Part | Meaning |
|---|---|
| neighbors v | not u, rated m, sim > 0, overlap >= 3, top k = 20 by sim |
| sim(u,v) | Weight: more alike = more say |
| centered(v,m) | How far above/below their usual v rated m |
| user_mean(u) | Starting point: u's usual level |

Example: mean 3.5; neighbors sim 0.4/0.3/0.1 with centered +1.2/-0.4/+0.8 -> adjustment 0.44/0.8 = +0.55 -> 4.05.
First real test (user 864, movie 143): 140 eligible, 20 used, prediction 3.76, actual 4.

| Question | Short answer |
|---|---|
| Why not average neighbors' raw ratings? | Ignores each neighbor's scale habit, ignores who is more similar, ignores the target user's own level. |
| Why divide by sum of similarities? | Keeps it a true weighted average (weights sum to 1). Dividing by the count makes the size depend on how big the similarities are. |
| What if no neighbor qualifies? | 0/0 gives NaN. Fall back to the movie mean (and count it in coverage). |
| Order of filtering and top-k? | Filter first, then take top k. Top k first can leave fewer than k valid neighbors. |

## 8. First model result (validation, 15,964 rows, k=20, min overlap 3)
| Model | RMSE | MAE | Coverage |
|---|---|---|---|
| Global mean | 1.119 | 0.940 | 1.0 |
| Movie mean | 1.030 | 0.822 | 1.0 |
| User-user cosine | 0.943 | 0.739 | ~1.0 (3 fallback rows) |

- Improvement over movie mean: RMSE -0.087, MAE -0.083.
- RMSE is not additive by percent. MSE is: overall MSE = share_A * MSE_A + share_B * MSE_B, then take sqrt.
- Coverage ~100%: with 943 users and 80% of ratings, almost every movie has a qualifying neighbor, so coverage barely separates models here. It matters more with higher min overlap or sparser data.
- Compare models only on identical rows. A gap of 0.087 should later be checked with a paired bootstrap (stats track).

| Question | Short answer |
|---|---|
| What does RMSE 0.94 mean for the business? | Typical miss is a bit under 1 star; big misses count extra. Useful only against a baseline. |
| Why keep a flag per row? | To compute coverage and to score neighbor rows and fallback rows separately. |
| Does lower offline RMSE mean more watching? | Not proven. Offline accuracy is a proxy; an A/B test measures impact. |

## 9. User-user Pearson similarity
u, v = two users; i = a movie both rated (sums run only over those movies)
```
               Σ over shared i of  (r(u,i) − r̄(u)) · (r(v,i) − r̄(v))
sim(u, v) = ──────────────────────────────────────────────────────────
            √( Σ (r(u,i) − r̄(u))² )  ·  √( Σ (r(v,i) − r̄(v))² )
```
| Symbol | Meaning | Data it uses in our code |
|---|---|---|
| sim(u,v) | Pearson similarity, -1 to +1 | cell of `fitting_matrix.T.corr()` |
| I_uv | Movies both users rated | `overlap_df.loc[u, v]` = its size |
| r_ui | Rating of user u on movie i | `fitting_matrix.loc[u, i]` |
| mean_u | u's average over the shared movies | shared columns of `fitting_matrix` row u |

Findings (fitting data): 34% of user pairs are NaN (fewer than 2 shared movies). For user 405 the top-5 Pearson neighbors all had similarity 1.0 but overlap 2 (cosine only ~0.03): a perfect score from 2 movies is chance, not taste. For the cosine top-5 (overlap 71-118), Pearson was lower (0.16-0.35) than cosine (0.33-0.36).

| Question | Short answer |
|---|---|
| What does a negative Pearson mean? | Opposite taste: above-usual for one is below-usual for the other. Our funnel keeps only sim > 0. |
| Why is NaN safe in the Pearson table? | `NaN > 0` is False, so those users drop out of the funnel. |
| Why does Pearson give 1.0 so easily? | With 2 shared movies any two points fit a line perfectly. This is why we need min overlap and significance weighting. |
| Cosine vs Pearson? | Cosine turns blanks into 0 and does not center, so it is always 0-1 and rewards people who just rate high. Pearson centers and uses only shared movies. |

## 10. Validation table so far (k=20, min overlap 3)
| Model | RMSE | MAE | Coverage |
|---|---|---|---|
| Global mean | 1.119 | 0.940 | 1.000 |
| Movie mean | 1.030 | 0.822 | 1.000 |
| User-user cosine | 0.943 | 0.739 | 1.000 |
| User-user Pearson | 0.961 | 0.754 | 0.998 |

- Pearson is slightly worse than cosine here, and coverage 0.998 (about 30 fallback rows) is far too small to explain a 0.018 gap. Rows that fall back are still scored (with the movie mean); they are not dropped.
- Likely reason: Pearson from 3 to 5 shared movies is noisy and can be extreme, so weak neighbors get big weights. Significance weighting and shrinkage (sweeps) target exactly this.
- One split and one setting: do not call Pearson "worse in general". Check with a paired comparison later.

| Question | Short answer |
|---|---|
| Does a worse RMSE prove a worse method? | No. It is one split, one setting, and a small gap. Needs a confidence interval on the paired difference. |
| Are fallback rows excluded from RMSE? | No. They get the movie-mean prediction and are scored with everyone else. |

## Open questions (add the answers as we go)
- Why does shrinkage help for sparse neighbors?
- Why fall back to the movie mean when no neighbor qualifies?
