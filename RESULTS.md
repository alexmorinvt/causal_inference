# autostrategy/apr24 — iteration log

Partial-perturbation synthetic benchmark: `n_genes=50`, `n_perturbed=25`, `top_k=1000`.
Train seeds: `[0, 1, 2]`. Test seeds: `[100, 101, 102]`. Both headline metrics are seed-means at `top_k=1000`.

Benchmark harness: `scripts/eval_partial_perturbation.py` (commit `df7b1c5`).

## Headline metrics

- **hidden-source recall** (↑): fraction of true edges whose source is unperturbed that the method recovered in its top-`k`. Both baselines sit at 0 by construction — MD can't score such edges, and `RandomBaseline` draws sources only from the perturbed set.
- **precision@k** (↑): fraction of top-`k` that is a true edge.

Sanity floor: mean W1 on evaluable edges must not drop below `RandomBaseline`'s on the same split. FOR is reported for context.

## Pre-iteration-0 baselines

### Train (seeds 0, 1, 2)

| method              | mean W1 | FOR   | precision@k | hidden-source recall | runtime/seed |
|---------------------|---------|-------|-------------|----------------------|--------------|
| MeanDifferenceModel | 0.2741  | 0.000 | **0.128**   | 0.000                | 0.01s        |
| RandomBaseline      | 0.2457  | 0.353 | 0.116       | 0.000                | 0.08s        |

### Test (seeds 100, 101, 102)

| method              | mean W1 | FOR   | precision@k | hidden-source recall | runtime/seed |
|---------------------|---------|-------|-------------|----------------------|--------------|
| MeanDifferenceModel | 0.2571  | 0.000 | 0.131       | 0.000                | 0.01s        |
| RandomBaseline      | 0.2309  | 0.357 | 0.118       | 0.000                | 0.08s        |

### Current best (on train) — the bar iteration 1 must beat

- **precision@k**: `MeanDifferenceModel` at **0.128**.
- **hidden-source recall**: tied at **0.000** (both baselines). Any positive number beats this.
- W1 sanity floor: must stay above `RandomBaseline`'s 0.2457.

## Iteration log

_(None yet.)_
