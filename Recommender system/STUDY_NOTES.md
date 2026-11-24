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

## Open questions (add the answers as we go)
- Why does shrinkage help for sparse neighbors?
- Why fall back to the movie mean when no neighbor qualifies?
